"""
OpenAI-compatible server for Inkling-2-Small EXL3 (v3: uniform K4 mcg experts): TP4 attention + EP4 experts over NCCL,
paged global KV + SWA rings, CUDA-graphed DSpark speculative decoding at every batch size (always on, no plain fallback).

    torchrun --nproc_per_node=4 server/i2_server.py --port 8000 [--kv-gb 18]

Rank 0 runs the HTTP server and the scheduler; ranks 1..3 follow rank 0's command stream.
Sampling: temperature (Gumbel-max, speculative sampling with deterministic drafts stays exact); top_p / top_k are ignored.
"""
import os, sys, time, json, math, threading, queue, argparse, asyncio, uuid
import torch, torch.distributed as dist
from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.modules.inkling2_engine import I2Engine, Batch
from exllamav3.modules.inkling2_draft import I2Draft

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL_DIR", "/models/Inkling-2-Small-EXL3-4bpw"))
ap.add_argument("--port", type=int, default=8123)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--slots", type=int, default=16)
ap.add_argument("--kv-gb", type=float, default=0.0, help="KV cache budget per GPU for global-layer pages (0 = auto from free memory)")
ap.add_argument("--reserve-gb", type=float, default=2.0, help="free memory kept for workspace when sizing KV automatically (measured peak extra use: 0.6 GiB)")
ap.add_argument("--max-model-len", type=int, default=524288)
ap.add_argument("--bits-g", type=int, default=4); ap.add_argument("--bits-l", type=int, default=8)
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--served-name", default="inkling-2-small-exl3-v3")
args = ap.parse_args()
SPEC = True               # speculative decoding is always on (every batch size); there is no plain-decode path
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
dist.init_process_group("nccl", device_id=dev)
def log(*x):
    if rank == 0: print(time.strftime("%H:%M:%S"), *x, flush=True)

# ---------------------------------------------------------------------------------------------------------- load
cfg = Config.from_directory(args.model)
model = Model.from_config(cfg, tp_rank=rank, tp_size=world)
t0 = time.time()
model.embed.load(dev)
for L in model.layers: L.load(dev)
model.head.load(dev)
S = args.slots
PS = 64
free0, total = torch.cuda.mem_get_info()
log(f"weights loaded in {time.time()-t0:.0f}s; GPU free {free0/2**30:.1f} GiB")
# KV sizing: bytes per global page on this rank (target 4-bit + draft bf16)
HKL = 8 // world
def page_bytes():
    tgt = 8 * 2 * PS * HKL * ((64 if args.bits_g == 4 else 128 if args.bits_g == 8 else 256) + (2 if args.bits_g != 16 else 0))
    drf = PS * I2Draft.global_bytes_per_token(HKL) if SPEC else 0
    return tgt + drf
fixed = (S + 1) * (40 * 2 * (1024 + 64) * HKL * ((64 if args.bits_l == 4 else 128) + 2) + 48 * 2 * 4096 * 16 * 2 // 1) * 0  # accounted by free-memory probe below
pt_len = args.max_model_len // PS + 2
# measure the fixed per-slot state by constructing the engine with a tiny page pool first
eng = I2Engine(model, dev, rank, world, max_slots=S + 1, num_pages=8, page_size=PS, bits_g=args.bits_g, bits_l=args.bits_l,
               max_pages_per_seq=pt_len)
draft = I2Draft(args.model + "/mtp.safetensors", eng, dev, rank, world, S + 1, 8, PS) if SPEC else None
free1, _ = torch.cuda.mem_get_info()
budget = args.kv_gb * 2**30 if args.kv_gb > 0 else max(0.0, free1 - args.reserve_gb * 2**30)
NP = max(16, int(budget // page_bytes()))
NP = int(torch.tensor(NP, device=dev).min().item()) if world == 1 else int(NP)
t = torch.tensor([NP], device=dev, dtype=torch.long); dist.all_reduce(t, op=dist.ReduceOp.MIN); NP = int(t.item())
# rebuild the page-based tensors at the final size
for i, (kq, vq, ks, vs) in list(eng.kv.items()):
    if cfg.i2.is_global(i):
        shape = (NP, PS, HKL, kq.shape[-1])
        eng.kv[i] = (torch.zeros(shape, dtype=kq.dtype, device=dev), torch.zeros(shape, dtype=vq.dtype, device=dev),
                     None if ks is None else torch.zeros(shape[:-1], dtype=ks.dtype, device=dev),
                     None if vs is None else torch.zeros(shape[:-1], dtype=vs.dtype, device=dev))
if SPEC:
    draft.alloc(NP)
torch.cuda.empty_cache()
log(f"KV pages: {NP} x {PS} = {NP*PS:,} tokens (per GPU {NP*page_bytes()/2**30:.1f} GiB, {page_bytes()/PS:.0f} B/token)")

# ---------------------------------------------------------------------------------------------------------- state
BUCKETS = [b for b in (1, 2, 4, 8, 16) if b <= S]
DUMMY = S                               # scratch slot for padded rows; owns page 0
anchor_all = torch.zeros(S + 1, dtype=torch.long, device=dev)
d_all = torch.zeros((S + 1, 7), dtype=torch.long, device=dev)
P_all = torch.zeros(S + 1, dtype=torch.int32, device=dev)
temp_all = torch.zeros(S + 1, dtype=torch.float32, device=dev)
eng.page_table[DUMMY, :2] = 0
EOS = 200006
V = cfg.i2.unpadded_vocab
ar8 = torch.arange(8, dtype=torch.int32, device=dev)

def sample_rows(z, temps):
    """z [T, Vshard] fp32 logits shard, temps [T] -> token ids [T] (greedy where temp == 0, Gumbel-max otherwise)"""
    if True:
        u = torch.rand_like(z).clamp_(1e-20, 1.0)
        g = -torch.log(-torch.log(u))
        t = temps[:, None]
        z = torch.where(t > 0, z / t.clamp_min(1e-4) + g, z)
    return eng.global_argmax(z)

class Bucket:
    def __init__(self, B, spec):
        self.B = B
        self.spec = spec and SPEC
        self.slots = torch.full((B,), DUMMY, dtype=torch.int32, device=dev)
        self.emit = torch.zeros((B, 8), dtype=torch.long, device=dev)
        self.n = torch.zeros(B, dtype=torch.long, device=dev)
        self.taps = torch.empty((B * 8 if self.spec else B, 8, 4608), dtype=torch.bfloat16, device=dev) if SPEC else None
        self.graph = None

    @torch.no_grad()
    def step(self):
        B = self.B
        sl = self.slots.long()
        slot8 = self.slots.repeat_interleave(8)
        nprev = ar8.repeat(B); nafter = (7 - nprev).to(torch.int32)
        P = P_all[sl]
        anchor = anchor_all[sl]
        temps = temp_all[sl].repeat_interleave(8)
        pos = (P[:, None] + ar8[None]).reshape(-1).to(torch.int32)
        nsplit_g = 8
        if self.spec:
            d = d_all[sl]
            ids = torch.cat([anchor[:, None], d], 1).reshape(-1)
            bt = Batch(slot8, pos, nprev, True, nsplit_g=nsplit_g, nsplit_l=4, nafter=nafter)
            z = eng.forward(ids, bt, taps=self.taps)
            g = sample_rows(z, temps).view(B, 8)
            n = (d == g[:, :7]).int().cumprod(1).sum(1)
            draft.store_context(self.taps, slot8, pos)
            newanchor = g.gather(1, n[:, None])[:, 0]
            emit = torch.cat([d, g[:, :1]], 1)
            emit.scatter_(1, n[:, None], newanchor[:, None])
            self.emit.copy_(emit); self.n.copy_(n)
            Pn = P + (n + 1).to(torch.int32)
            f_last = self.taps.view(B, 8, 8, 4608)[torch.arange(B, device=dev), n]
            dn = draft.draft(newanchor, self.slots, Pn, f_last, nsplit_g=nsplit_g)
            anchor_all[sl] = newanchor; P_all[sl] = Pn; d_all[sl] = dn
        else:
            bt = Batch(self.slots, P, torch.zeros(B, dtype=torch.int32, device=dev), False, nsplit_g=nsplit_g, nsplit_l=4)
            z = eng.forward(anchor, bt, taps=self.taps)
            if SPEC:       # keep the draft's context K/V complete so speculative steps can resume at any time
                draft.store_context(self.taps, self.slots, P)
            tok = sample_rows(z, temp_all[sl])
            self.emit[:, 0] = tok; self.n.zero_()
            anchor_all[sl] = tok; P_all[sl] = P + 1

buckets = {}                                        # no plain-decode graphs: every decode step is speculative
sbuckets = {B: Bucket(B, True) for B in BUCKETS}
pool = None
def capture(b):
    global pool
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2): b.step()
    torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize(); dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        b.step()
    pool = g.pool()
    b.graph = g
    anchor_all.zero_(); P_all.zero_(); d_all.zero_()

t0 = time.time()
with torch.no_grad():
    for B in sbuckets:
        capture(sbuckets[B])
log(f"speculative graphs captured for buckets {BUCKETS} in {time.time()-t0:.0f}s; GPU free {torch.cuda.mem_get_info()[0]/2**30:.1f} GiB")

# ---------------------------------------------------------------------------------------------------------- commands
OP_NOOP, OP_PREFILL, OP_DECODE, OP_STOP = 0, 1, 2, 9
HDR = torch.zeros(8, dtype=torch.long, device=dev)
PAY = torch.zeros(args.max_model_len // PS + 4096 + 64, dtype=torch.long, device=dev)
FPAY = torch.zeros(64, dtype=torch.float32, device=dev)

def bcast(hdr_vals=None, pay=None, fpay=None):
    """rank 0 fills (hdr_vals, pay, fpay); everyone receives; returns (hdr list, payload view, float payload view)"""
    if rank == 0:
        HDR.zero_(); HDR[:len(hdr_vals)] = torch.tensor(hdr_vals, dtype=torch.long, device=dev)
    dist.broadcast(HDR, 0)
    h = HDR.tolist()
    npay, nf = h[6], h[7]
    if npay:
        if rank == 0: PAY[:npay] = torch.tensor(pay, dtype=torch.long, device=dev)
        dist.broadcast(PAY[:npay], 0)
    if nf:
        if rank == 0: FPAY[:nf] = torch.tensor(fpay, dtype=torch.float32, device=dev)
        dist.broadcast(FPAY[:nf], 0)
    return h, PAY[:npay], FPAY[:nf]

@torch.no_grad()
def do_prefill(h, pay, fpay):
    slot, start, n, last, npg, total_len = h[1], h[2], h[3], h[4], h[5], int(h[7] and 0)
    # pay = [token ids (n)] + [page ids (npg)] (page ids only on the first chunk)
    ids = pay[:n]
    if npg:
        eng.page_table[slot, :npg] = pay[n:n + npg].to(torch.int32)
    pos = torch.arange(start, start + n, dtype=torch.int32, device=dev)
    b = Batch(torch.full((n,), slot, dtype=torch.int32, device=dev), pos, torch.arange(n, dtype=torch.int32, device=dev), True,
              nsplit_g=8, nsplit_l=4)
    taps = torch.empty((n, 8, 4608), dtype=torch.bfloat16, device=dev) if SPEC else None
    z = eng.forward(ids, b, taps=taps)
    if SPEC:
        draft.store_context(taps, b.slot, b.pos)
    if last:
        temp = float(fpay[0]) if fpay.numel() else 0.0
        temp_all[slot] = temp
        tok = sample_rows(z[-1:], temp_all[slot:slot + 1])
        anchor_all[slot] = tok[0]; P_all[slot] = start + n
        if SPEC:
            sl = torch.tensor([slot], dtype=torch.int32, device=dev)
            d_all[slot] = draft.draft(tok, sl, P_all[slot:slot + 1], taps[-1:], nsplit_g=8)[0]
        return int(tok[0]) if rank == 0 else None
    return None

@torch.no_grad()
def do_decode(h, pay, fpay):
    B, nact = h[1], h[2]
    b = sbuckets[B] if h[3] else buckets[B]
    sl = pay[:nact].to(torch.int32)
    b.slots.fill_(DUMMY); b.slots[:nact] = sl
    P_all[DUMMY] = 0
    b.graph.replay()
    if rank == 0:
        return b.emit.cpu(), b.n.cpu()
    return None

def worker_loop():
    while True:
        h, pay, fpay = bcast()
        op = h[0]
        if op == OP_STOP: break
        if op == OP_PREFILL: do_prefill(h, pay, fpay)
        elif op == OP_DECODE: do_decode(h, pay, fpay)

if rank != 0:
    worker_loop()
    sys.exit(0)

# ====================================================================================================== rank 0 only
from tokenizers import Tokenizer
import jinja2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

tok = Tokenizer.from_file(args.model + "/tokenizer.json")
def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
def _raise(msg): raise jinja2.TemplateError(msg)
jenv = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=False, lstrip_blocks=False)
jenv.filters["tojson"] = _tojson
jenv.globals["raise_exception"] = _raise
template = jenv.from_string(open(args.model + "/chat_template.jinja").read())

T_MSG_MODEL, T_TEXT, T_THINK, T_END, T_EOS, T_TOOL = 200001, 200004, 200008, 200010, 200006, 200049

pending = queue.Queue()
free_slots = list(range(S))
free_pages = list(range(1, NP))       # page 0 belongs to the dummy slot
active = {}                           # slot -> Req
prefilling = []                       # Reqs with remaining prompt
loop_ref = {}
STATS = {'dec_steps': 0, 'dec_time': 0.0, 'dec_rows': 0, 'dec_tok': 0, 'pre_time': 0.0, 'pre_tok': 0, 'idle_time': 0.0}

class Req:
    def __init__(self, ids, max_tokens, temperature, out_q):
        self.ids, self.max_tokens, self.temperature, self.out_q = ids, max_tokens, temperature, out_q
        self.id = "chatcmpl-" + uuid.uuid4().hex[:24]
        self.slot = None; self.pages = []; self.done_prompt = 0; self.gen = []; self.t_arrive = time.time()
        self.first_token_time = None; self.finish = None

def emit(req, item):
    loop_ref["loop"].call_soon_threadsafe(req.out_q.put_nowait, item)

def try_admit():
    while not pending.empty() and free_slots:
        req = pending.queue[0]
        need = (len(req.ids) + req.max_tokens + 24 + PS - 1) // PS
        if need > len(free_pages) or need > pt_len - 2:
            if need > pt_len - 2 or need > NP - 1:
                pending.get(); emit(req, ("error", "request longer than the KV cache / max_model_len")); continue
            break
        pending.get()
        req.slot = free_slots.pop(0)
        req.pages = [free_pages.pop() for _ in range(need)]
        prefilling.append(req)

def finish(req, reason):
    req.finish = reason
    active.pop(req.slot, None)
    free_slots.append(req.slot); free_pages.extend(req.pages)
    emit(req, ("done", reason))

def scheduler():
    last_hb = time.time()
    while True:
        try_admit()
        did = False
        if prefilling:
            req = prefilling[0]
            n = min(args.chunk, len(req.ids) - req.done_prompt)
            first = req.done_prompt == 0
            last = req.done_prompt + n == len(req.ids)
            pay = req.ids[req.done_prompt:req.done_prompt + n] + (req.pages if first else [])
            h = [OP_PREFILL, req.slot, req.done_prompt, n, int(last), len(req.pages) if first else 0, len(pay), 1 if last else 0]
            tp0 = time.time()
            t, _, _ = bcast(h, pay, [req.temperature] if last else None)
            res = do_prefill(HDR.tolist(), PAY[:len(pay)], FPAY[:1 if last else 0])
            req.done_prompt += n
            STATS['pre_time'] += time.time() - tp0; STATS['pre_tok'] += n
            if last:
                prefilling.pop(0)
                req.first_token_time = time.time()
                active[req.slot] = req
                handle_tokens(req, [res])
            did = True
        if active:
            slots = sorted(active)
            nb = len(slots)
            use_spec = True
            B = next(b for b in (sbuckets if use_spec else buckets) if b >= nb)
            h = [OP_DECODE, B, nb, int(use_spec), 0, 0, nb, 0]
            td0 = time.time()
            bcast(h, slots)
            emit_t, n_t = do_decode(HDR.tolist(), PAY[:nb], None)
            STATS['dec_steps'] += 1; STATS['dec_time'] += time.time() - td0; STATS['dec_rows'] += nb; STATS['dec_tok'] += (int(n_t[:nb].sum()) + nb) if use_spec else nb
            for i, s in enumerate(slots):
                req = active.get(s)
                if req is None: continue
                k = int(n_t[i]) + 1 if use_spec else 1
                handle_tokens(req, emit_t[i, :k].tolist())
            did = True
        if not did:
            time.sleep(0.002)
            if time.time() - last_hb > 1.0:
                bcast([OP_NOOP]); last_hb = time.time()

def handle_tokens(req, toks):
    for t in toks:
        if req.slot not in active or req.finish: return
        if t in (T_EOS, 199999):
            finish(req, "stop"); return
        req.gen.append(t)
        emit(req, ("tok", t))
        if len(req.gen) >= req.max_tokens:
            finish(req, "length"); return

# ---------------------------------------------------------------------------------------------------------- output parsing
class Parser:
    """Token-level parser of Inkling's typed blocks -> OpenAI deltas (reasoning_content / content / tool_calls)"""
    def __init__(self):
        self.mode = None; self.buf = []; self.sent = 0; self.tool_idx = 0
    def _text(self):
        return tok.decode(self.buf, skip_special_tokens=False)
    def feed(self, t):
        out = []
        if t == T_MSG_MODEL:
            self.mode = "hdr"; self.buf = []; self.sent = 0
        elif t in (T_TEXT, T_THINK, T_TOOL) and self.mode in ("hdr", None):
            self.mode = {T_TEXT: "text", T_THINK: "think", T_TOOL: "tool"}[t]; self.buf = []; self.sent = 0
        elif t == T_END:
            if self.mode == "tool":
                body = self._text()
                try:
                    obj = json.loads(body)
                    out.append(("tool", {"index": self.tool_idx, "id": "call_" + uuid.uuid4().hex[:20], "type": "function",
                                         "function": {"name": obj.get("name", ""), "arguments": json.dumps(obj.get("args", {}))}}))
                except Exception:
                    out.append(("content", body))
                self.tool_idx += 1
            else:
                out += self._flush(final=True)
            self.mode = None; self.buf = []; self.sent = 0
        elif self.mode in ("text", "think"):
            self.buf.append(t)
            out += self._flush()
        elif self.mode == "tool":
            self.buf.append(t)
        return out
    def _flush(self, final=False):
        s = self._text()
        if not final and s.endswith("�"):
            return []
        d = s[self.sent:]; self.sent = len(s)
        if not d: return []
        return [("reasoning" if self.mode == "think" else "content", d)]
    def finish(self):
        out = []
        if self.mode in ("text", "think"): out += self._flush(final=True)
        return out

def render_prompt(messages, tools, reasoning_effort):
    kw = dict(messages=messages, tools=tools or None, add_generation_prompt=True)
    if reasoning_effort is not None: kw["reasoning_effort"] = reasoning_effort
    return template.render(**kw)

def encode(text):
    return tok.encode(text, add_special_tokens=False).ids

app = FastAPI()

@app.get("/health")
async def health(): return {"status": "ok", "stats": STATS, "spec": SPEC, "kv_tokens": NP * PS, "active": len(active), "queued": pending.qsize()}

@app.get("/v1/models")
async def models(): return {"object": "list", "data": [{"id": args.served_name, "object": "model", "owned_by": "local"}]}

def norm_messages(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if isinstance(p, dict) and p.get("type") in ("text", "input_text"): parts.append({"type": "text", "text": p.get("text", "")})
                elif isinstance(p, str): parts.append(p)
            m["content"] = parts
        for tc in m.get("tool_calls") or []:
            a = tc["function"].get("arguments")
            if isinstance(a, str):
                try: tc["function"]["arguments"] = json.loads(a)
                except Exception: tc["function"]["arguments"] = {}
        out.append(m)
    return out

async def run_request(body, chat):
    loop = asyncio.get_running_loop()
    loop_ref["loop"] = loop
    if chat:
        prompt = render_prompt(norm_messages(body["messages"]), body.get("tools"), body.get("reasoning_effort"))
    else:
        prompt = body["prompt"] if isinstance(body["prompt"], str) else body["prompt"][0]
    ids = encode(prompt)
    max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 4096)
    temp = float(body.get("temperature", 0.0) or 0.0)
    out_q = asyncio.Queue()
    req = Req(ids, max_tokens, temp, out_q)
    pending.put(req)
    return req, out_q, len(ids)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    req, out_q, nprompt = await run_request(body, True)
    stream = bool(body.get("stream"))
    created = int(time.time())
    parser = Parser()
    def chunk(delta, finish=None):
        return {"id": req.id, "object": "chat.completion.chunk", "created": created, "model": args.served_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    async def events():
        sent_role = False
        has_tool = False
        while True:
            kind, val = await out_q.get()
            if kind == "tok":
                for k, v in parser.feed(val):
                    d = {}
                    if not sent_role: d["role"] = "assistant"; sent_role = True
                    if k == "reasoning": d["reasoning_content"] = v
                    elif k == "content": d["content"] = v
                    elif k == "tool": d["tool_calls"] = [v]; has_tool = True
                    yield d, None
            elif kind == "done":
                for k, v in parser.finish():
                    yield ({"reasoning_content": v} if k == "reasoning" else {"content": v}), None
                yield {}, ("tool_calls" if has_tool else val)
                return
            elif kind == "error":
                yield {"content": ""}, "error"
                return
    if stream:
        async def gen():
            async for d, fin in events():
                yield f"data: {json.dumps(chunk(d, fin))}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    content, reasoning, tools, fin = "", "", [], "stop"
    async for d, f in events():
        content += d.get("content", ""); reasoning += d.get("reasoning_content", "")
        tools += d.get("tool_calls", [])
        if f: fin = f
    msg = {"role": "assistant", "content": content or None}
    if reasoning: msg["reasoning_content"] = reasoning
    if tools: msg["tool_calls"] = tools
    return {"id": req.id, "object": "chat.completion", "created": created, "model": args.served_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
            "usage": {"prompt_tokens": nprompt, "completion_tokens": len(req.gen), "total_tokens": nprompt + len(req.gen)}}

@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    req, out_q, nprompt = await run_request(body, False)
    text, fin = "", "stop"
    while True:
        kind, val = await out_q.get()
        if kind == "tok": pass
        elif kind in ("done", "error"):
            fin = val; break
    text = tok.decode(req.gen, skip_special_tokens=True)
    return {"id": req.id, "object": "text_completion", "created": int(time.time()), "model": args.served_name,
            "choices": [{"index": 0, "text": text, "finish_reason": fin}],
            "usage": {"prompt_tokens": nprompt, "completion_tokens": len(req.gen), "total_tokens": nprompt + len(req.gen)}}

threading.Thread(target=scheduler, daemon=True).start()
log(f"serving on http://{args.host}:{args.port}  (spec decoding {'ON' if SPEC else 'OFF'}, slots {S}, buckets {BUCKETS})")
uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
