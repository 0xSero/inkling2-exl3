"""
OpenAI-compatible server for Inkling-2-Small EXL3 (v3: uniform K4 mcg experts): TP4 attention + EP4 experts over NCCL,
paged global KV + SWA rings, CUDA-graphed DSpark speculative decoding at every batch size (always on, no plain fallback).

    torchrun --nproc_per_node=4 server/i2_server.py --port 8000 [--kv-gb 18]

Rank 0 runs the HTTP server and the scheduler; ranks 1..3 follow rank 0's command stream.
Sampling: temperature (Gumbel-max, speculative sampling with deterministic drafts stays exact); top_p / top_k are ignored.
"""
import os, sys, time, json, math, threading, queue, argparse, asyncio, uuid
import torch, torch.distributed as dist
os.environ.setdefault("I2_DRAFT_RINGD", "2112")      # draft local ring: window 1024 + previous generation (session reuse)
from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.modules.inkling2_engine import I2Engine, Batch
from exllamav3.modules.inkling2_draft import I2Draft

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("MODEL_DIR", "/models/Inkling-2-Small-EXL3-4bpw"))
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--slots", type=int, default=16)
ap.add_argument("--kv-gb", type=float, default=0.0, help="KV cache budget per GPU for global-layer pages (0 = auto from free memory)")
ap.add_argument("--reserve-gb", type=float, default=2.75, help="free memory kept for workspace when sizing KV automatically (4096-row sharded prefill keeps ~2.5 GiB headroom unused at 1.75; 16x380K needs <= 1.9)")
ap.add_argument("--max-model-len", type=int, default=380000)
ap.add_argument("--bits-g", type=int, default=4); ap.add_argument("--bits-l", type=int, default=8)
ap.add_argument("--burst", type=int, default=1, help="max back-to-back speculative steps per scheduler round (1 = one host sync per step)")
ap.add_argument("--spec-k", default="1:4,2:3,4:2,8:2,16:2", help="draft length per decode bucket: one K (1..7) or a map like 1:2,2:2,4:2,8:3,16:4")
ap.add_argument("--kv-rot", type=int, default=1, help="global KV: randomized Hadamard rotation along head_dim before quantizing")
ap.add_argument("--kv-ng", type=int, default=4, help="global KV: fp16 scale groups per 128-vector (1, 2, 4)")
ap.add_argument("--kv-q16", type=int, default=1, help="global 4-bit KV: 16-level symmetric grid")
ap.add_argument("--swa-rows", type=int, default=1024, help="SWA ring headroom (window + this); longer multi-row calls run their SWA layers in sub-chunks of this many rows")
ap.add_argument("--nsplit-g", type=int, default=32, help="KV splits of the global-layer attention in decode graphs (target verify + draft); long contexts need many (8 gave 11 tok/s at 128K)")
ap.add_argument("--sse-batch", type=int, default=1, help="one stream event / SSE chunk per request per scheduler step (instead of per token)")
ap.add_argument("--chunk", type=int, default=4096, help="prefill chunk rows of a pure prefill step (b12x T=1024 with 32-row route blocks: 2.33 us/token/layer vs 3.59 at 512)")
ap.add_argument("--mixed", type=int, default=1, help="chunked prefill rides in the decode step: one forward carries every active slot's verify rows plus prefill chunks of pending prompts (eager); pure decode steps keep the graphs")
ap.add_argument("--prefill-budget", type=int, default=512, help="prefill tokens per mixed step while requests are decoding (TTFT vs inter-token latency), capped by --max-step-rows; the full --chunk when none are")
ap.add_argument("--state-slots", type=int, default=24, help="sequence state slots (rings + pages); beyond --slots they hold parked sessions for reuse")
ap.add_argument("--reuse", type=int, default=0, help="session reuse: a new request extending a parked session's history takes its slot over and prefills only the rest (needs --mixed 1)")
ap.add_argument("--reuse-min", type=int, default=1024, help="minimum reused prefix (tokens)")
ap.add_argument("--max-step-rows", type=int, default=512, help="rows per mixed (decode + prefill) forward: keeps the decoding users' step near a 512-row chunk")
ap.add_argument("--served-name", default="inkling-2-small")
ap.add_argument("--adaptive", type=int, default=1, help="adaptive verification: every round drafts the full 7-token block; inside the graph each request verifies the live width (rows incl. anchor) with the most expected tokens per ms (calibrated draft confidence x measured round cost, shared row budget); one graph per (bucket, physical width)")
ap.add_argument("--av-widths", default="1:2,3,4,5,6,8;2:2,3,4,5,6,8;4:2,3,4,5,6,8;8:2,3,4,6,8;16:2,3,4,6,8", help="captured physical verify widths per bucket (adaptive)")
ap.add_argument("--av-calib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/av_calib.json"), help="confidence -> prefix-acceptance table (serving/av_fit.py)")
ap.add_argument("--av-cost", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/av_cost.json"), help="measured round ms per (B, physical width, live width) (serving/av_offline.py --mode cost)")
ap.add_argument("--av-host-ms", type=float, default=0.8, help="host time per decode round added to the graph cost (server loop overhead)")
ap.add_argument("--av-fixed", type=int, default=0, help="diagnostics: every request verifies this live width (1..8) instead of the policy")
ap.add_argument("--b12x", type=int, default=int(os.environ.get("I2_B12X", "0")), help="routed experts of the speculative verify through the b12x fused Trellis MoE (needs the b12x stack: cutlass-dsl + FlashInfer tree at I2_B12X_PATH)")
args = ap.parse_args()
if args.b12x:
    os.environ["EXL3_LOAD_ARENA"] = "0"     # b12x repack frees the per-expert tensors; arena slabs would pin a second copy
if args.adaptive:
    os.environ["I2_DRAFT_CONF"] = "1"     # the width policy reads the Markov step's draft confidences
SPEC = True               # speculative decoding is always on (every batch size); there is no plain-decode path
_SK = {int(k): int(v) for k, v in (x.split(":") for x in args.spec_k.split(","))} if ":" in args.spec_k else None
def spec_k(B):
    """draft length of decode bucket B (the smallest mapped bucket >= B, else the largest)"""
    if _SK is None: return int(args.spec_k)
    ks = sorted(_SK)
    return _SK[next((k for k in ks if k >= B), ks[-1])]
_AVW = {int(b): [int(x) for x in ws.split(",")] for b, ws in (p.split(":") for p in args.av_widths.split(";"))}
def widths(B):
    """physical verify widths (rows per request) with captured graphs for decode bucket B"""
    if not args.adaptive: return [spec_k(B) + 1]
    ks = sorted(_AVW); return _AVW[next((k for k in ks if k >= B), ks[-1])]
DRAFT_K = lambda K: 7 if args.adaptive else K
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
dist.init_process_group("nccl", device_id=dev)
def log(*x):
    if rank == 0: print(time.strftime("%H:%M:%S"), *x, flush=True)

# ---------------------------------------------------------------------------------------------------------- load
cfg = Config.from_directory(args.model)
model = Model.from_config(cfg, tp_rank=rank, tp_size=world)
if args.b12x:
    from exllamav3.modules.inkling2_b12x import prepare_model, attach_moe
    prepare_model(model)
t0 = time.time()
model.embed.load(dev)
for L in model.layers: L.load(dev)
model.head.load(dev)
S = args.slots
REUSE = bool(args.reuse and args.mixed)
NS = max(S, args.state_slots) if REUSE else S          # state slots (decode concurrency stays S)
PS = 64
free0, total = torch.cuda.mem_get_info()
log(f"weights loaded in {time.time()-t0:.0f}s; GPU free {free0/2**30:.1f} GiB")
# KV sizing: bytes per global page on this rank (target 4-bit + draft bf16)
HKL = 8 // world
def page_bytes():
    tgt = 8 * 2 * PS * HKL * ((64 if args.bits_g == 4 else 128 if args.bits_g == 8 else 256) + (2 * args.kv_ng if args.bits_g != 16 else 0))
    drf = PS * I2Draft.global_bytes_per_token(HKL) if SPEC else 0
    return tgt + drf
fixed = (S + 1) * (40 * 2 * (1024 + 64) * HKL * ((64 if args.bits_l == 4 else 128) + 2) + 48 * 2 * 4096 * 16 * 2 // 1) * 0  # accounted by free-memory probe below
pt_len = args.max_model_len // PS + 2
# measure the fixed per-slot state by constructing the engine with a tiny page pool first
eng = I2Engine(model, dev, rank, world, max_slots=NS + 1, num_pages=8, page_size=PS, bits_g=args.bits_g, bits_l=args.bits_l,
               max_pages_per_seq=pt_len, kv_rot=bool(args.kv_rot), kv_ng=args.kv_ng, kv_q16=bool(args.kv_q16),
               max_rows=args.swa_rows)
draft = I2Draft(args.model + "/mtp.safetensors", eng, dev, rank, world, NS + 1, 8, PS) if SPEC else None
# conv-ring snapshots at prompt end (session reuse restore point (b)): same shapes as the engine's per-slot conv rings
SNAP = {i: tuple(torch.zeros_like(r) for r in rs) for i, rs in eng.rings.items()} if REUSE else None
if args.b12x:
    from exllamav3.modules.inkling2_b12x import B12xMoE
    tb = time.time()
    _vt = sorted(set(w * b for b in (1, 2, 4, 8, 16) if b <= S for w in widths(b)) | ({8} if args.adaptive else set()))
    attach_moe(eng, model, args.model, cfg, dev, rank, world, sizes=_vt)
    log(f"b12x fused Trellis MoE attached for verify T={sorted(eng.b12x.sizes)} in {time.time()-tb:.0f}s (tile {eng.b12x.tile})")
free1, _ = torch.cuda.mem_get_info()
budget = args.kv_gb * 2**30 if args.kv_gb > 0 else max(0.0, free1 - args.reserve_gb * 2**30)
NP = max(16, int(budget // page_bytes()))
NP = int(torch.tensor(NP, device=dev).min().item()) if world == 1 else int(NP)
t = torch.tensor([NP], device=dev, dtype=torch.long); dist.all_reduce(t, op=dist.ReduceOp.MIN); NP = int(t.item())
# rebuild the page-based tensors at the final size
for i, (kq, vq, ks, vs) in list(eng.kv.items()):
    if cfg.i2.is_global(i):
        shape = (NP, PS, HKL, kq.shape[-1])
        sshape = shape[:-1] + ((args.kv_ng,) if args.kv_ng > 1 else ())
        eng.kv[i] = (torch.zeros(shape, dtype=kq.dtype, device=dev), torch.zeros(shape, dtype=vq.dtype, device=dev),
                     None if ks is None else torch.zeros(sshape, dtype=ks.dtype, device=dev),
                     None if vs is None else torch.zeros(sshape, dtype=vs.dtype, device=dev))
if SPEC:
    draft.alloc(NP)
torch.cuda.empty_cache()
log(f"KV pages: {NP} x {PS} = {NP*PS:,} tokens (per GPU {NP*page_bytes()/2**30:.1f} GiB, {page_bytes()/PS:.0f} B/token)")

# ---------------------------------------------------------------------------------------------------------- state
BUCKETS = [b for b in (1, 2, 4, 8, 16) if b <= S]
DUMMY = NS                              # scratch slot for padded rows; owns page 0
PROF_AT = int(os.environ.get("I2_PROFILE_AT", "0")); PROF_STEP = 0
# per burst step, per bucket row: [emit 0..7 | n | live width w | next-round width index (same in every row)]
BURST_OUT = torch.zeros((max(1, args.burst), S + 1, 11), dtype=torch.long, device=dev)
anchor_all = torch.zeros(NS + 1, dtype=torch.long, device=dev)
d_all = torch.zeros((NS + 1, 7), dtype=torch.long, device=dev)
P_all = torch.zeros(NS + 1, dtype=torch.int32, device=dev)
temp_all = torch.zeros(NS + 1, dtype=torch.float32, device=dev)
conf_all = torch.ones((NS + 1, 7), dtype=torch.float32, device=dev)      # draft confidences of d_all (adaptive)
eng.page_table[DUMMY, :2] = 0
ar7 = torch.arange(7, device=dev)
POL = None
if args.adaptive:
    from exllamav3.modules.inkling2_av import AVPolicy
    POL = AVPolicy.from_files(args.av_calib, args.av_cost, dev, {B: widths(B) for B in (1, 2, 4, 8, 16) if B <= S},
                              host_ms=args.av_host_ms)
    if args.av_fixed: POL.fixed = args.av_fixed
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
    """one captured speculative round for decode bucket B at physical verify width W (rows per request, incl. anchor)"""
    def __init__(self, B, spec, W=None):
        self.B = B
        self.spec = spec and SPEC
        self.W = (spec_k(B) + 1) if W is None else W
        R = self.W
        self.slots = torch.full((B,), DUMMY, dtype=torch.int32, device=dev)
        self.out = torch.zeros((B, 11), dtype=torch.long, device=dev)
        self.taps = torch.empty((B * R if self.spec else B, 8, 4608), dtype=torch.bfloat16, device=dev) if SPEC else None
        self.graph = None

    @torch.no_grad()
    def step(self):
        B, R = self.B, self.W
        K = R - 1
        arR = ar8[:R]
        sl = self.slots.long()
        real = self.slots != DUMMY
        slot8 = self.slots.repeat_interleave(R)
        nprev = arR.repeat(B); nafter = (K - nprev).to(torch.int32)
        P = P_all[sl]
        anchor = anchor_all[sl]
        tb = temp_all[sl]
        temps = tb.repeat_interleave(R)
        pos = (P[:, None] + arR[None]).reshape(-1).to(torch.int32)
        nsplit_g = args.nsplit_g
        d = d_all[sl][:, :K]
        if POL is not None:
            # live width per request from the calibrated confidences of its drafts (shared row budget); dead rows skip
            # their routed experts, dummy-slot rows are all dead
            ctx = POL.mean_ctx(P, real)
            w = POL.choose(conf_all[sl], tb, real, B, R, ctx=ctx)
            live = ((arR[None].long() < w[:, None]) & real[:, None]).reshape(-1)
        else:
            w = torch.full((B,), R, dtype=torch.long, device=dev)
            live = None
        ids = torch.cat([anchor[:, None], d], 1).reshape(-1)
        bt = Batch(slot8, pos, nprev, True, nsplit_g=nsplit_g, nsplit_l=4, nafter=nafter)
        eng.live = live
        z = eng.forward(ids, bt, taps=self.taps)
        eng.live = None
        g = sample_rows(z, temps).view(B, R)
        n = ((d == g[:, :K]) & (ar7[None, :K] < (w - 1)[:, None])).int().cumprod(1).sum(1)
        draft.store_context(self.taps, slot8, pos)
        newanchor = g.gather(1, n[:, None])[:, 0]
        emit = torch.cat([d, g[:, :1]], 1)
        emit.scatter_(1, n[:, None], newanchor[:, None])
        Pn = P + (n + 1).to(torch.int32)
        f_last = self.taps.view(B, R, 8, 4608)[torch.arange(B, device=dev), n]
        KD = DRAFT_K(K)
        dn = draft.draft(newanchor, self.slots, Pn, f_last, nsplit_g=nsplit_g, K=KD)
        anchor_all[sl] = newanchor; P_all[sl] = Pn; d_all[sl, :KD] = dn
        self.out[:, :R].copy_(emit); self.out[:, 8].copy_(n); self.out[:, 9].copy_(w)
        if POL is not None:
            conf_all[sl] = draft.last_conf
            self.out[:, 10].copy_(POL.best_width(draft.last_conf, tb, real, B, ctx=POL.mean_ctx(Pn, real)).expand(B))

buckets = {}                                        # no plain-decode graphs: every decode step is speculative
sbuckets = {(B, W): Bucket(B, True, W) for B in BUCKETS for W in widths(B)}
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
    anchor_all.zero_(); P_all.zero_(); d_all.zero_(); conf_all.fill_(1.0)

t0 = time.time()
with torch.no_grad():
    for B in sbuckets:
        capture(sbuckets[B])
log(f"speculative graphs captured for (bucket, width) {sorted(sbuckets)} (draft depth {DRAFT_K(0) if args.adaptive else 'K'}) in {time.time()-t0:.0f}s; GPU free {torch.cuda.mem_get_info()[0]/2**30:.1f} GiB")

# ---------------------------------------------------------------------------------------------------------- commands
OP_NOOP, OP_PREFILL, OP_DECODE, OP_PAGES, OP_MIXED, OP_SNAP, OP_RESTORE, OP_STOP = 0, 1, 2, 3, 4, 5, 6, 9
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
            if POL is not None: conf_all[slot] = draft.last_conf[0]
        return int(tok[0]) if rank == 0 else None
    return None

@torch.no_grad()
def do_decode(h, pay, fpay):
    """h[4] = number of back-to-back speculative steps (burst; one host sync for all of them)"""
    B, nact, nsteps = h[1], h[2], max(1, h[4])
    b = sbuckets[(B, h[5] or widths(B)[-1])]
    sl = pay[:nact].to(torch.int32)
    b.slots.fill_(DUMMY); b.slots[:nact] = sl
    P_all[DUMMY] = 0
    global PROF_STEP
    PROF_STEP += nsteps
    if PROF_AT and PROF_AT <= PROF_STEP < PROF_AT + nsteps:
        # diagnostics: 20 replays on every rank with all rows on the dummy slot (live sequences untouched)
        from torch.profiler import profile, ProfilerActivity
        saved = b.slots.clone(); b.slots.fill_(DUMMY)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(20):
                P_all[DUMMY] = 0; b.graph.replay()
            torch.cuda.synchronize()
        P_all[DUMMY] = 0; b.slots.copy_(saved)
        if rank == 0:
            print(f"PROFILE bucket {B} (20 dummy-slot replays):\n" + pr.key_averages().table(sort_by="cuda_time_total", row_limit=40, max_name_column_width=80), flush=True)
    for i in range(nsteps):
        b.graph.replay()
        if rank == 0:
            BURST_OUT[i, :B].copy_(b.out)
    if rank == 0:
        return BURST_OUT[:nsteps, :B].cpu()
    return None

@torch.no_grad()
def do_pages(h, pay, fpay):
    """page table of a newly admitted request (sent once at admission)"""
    slot, npg = h[1], h[2]
    eng.page_table[slot, :npg] = pay[:npg].to(torch.int32)

@torch.no_grad()
def do_snap(h, pay, fpay):
    """conv rings of slot h[1] -> snapshot (at prompt end); h[2] = 1 restores the snapshot instead"""
    slot = h[1]
    for i, rs in eng.rings.items():
        for r, sn in zip(rs, SNAP[i]):
            (r[slot] if h[2] else sn[slot]).copy_(sn[slot] if h[2] else r[slot])

@torch.no_grad()
def do_mixed(h, pay, fpay):
    """one eager forward over every active slot's verify rows (speculative step as Bucket.step) plus prefill chunks.
    h = [OP_MIXED, nact, nseg, ntok, ...]; pay = decode slots (nact) + nseg x (slot, start, n, last) + chunk token ids;
    fpay = per-segment temperature. Rank 0 returns (emit [nact, R], n [nact], {slot: first token})"""
    nact, nseg, ntok = h[1], h[2], h[3]
    p = pay[:nact + 4 * nseg].tolist()
    dslots = p[:nact]
    segs = [p[nact + 4 * i: nact + 4 * i + 4] for i in range(nseg)]
    ids_p = pay[nact + 4 * nseg: nact + 4 * nseg + ntok]
    i32 = lambda t: t.to(torch.int32)
    parts = []                                     # (ids, slot, pos, nprev, nafter) per block of rows
    if nact:
        Bm = next(b for b in BUCKETS if b >= nact)
        R = h[4] or widths(Bm)[-1]; K = R - 1; arR = ar8[:R]
        sl = torch.tensor(dslots, dtype=torch.int32, device=dev); sll = sl.long()
        P = P_all[sll]; d = d_all[sll][:, :K]
        if POL is not None:
            _real = torch.ones(nact, dtype=torch.bool, device=dev)
            wv = POL.choose(conf_all[sll], temp_all[sll], _real, nact, R, cost_B=Bm, ctx=POL.mean_ctx(P, _real))
        else:
            wv = torch.full((nact,), R, dtype=torch.long, device=dev)
        nprev = arR.repeat(nact)
        parts.append((torch.cat([anchor_all[sll][:, None], d], 1).reshape(-1), sl.repeat_interleave(R),
                      i32((P[:, None] + arR[None]).reshape(-1)), nprev, i32(K - nprev)))
    off = 0
    for slot, start, n, last in segs:
        ar = torch.arange(n, dtype=torch.int32, device=dev)
        parts.append((ids_p[off:off + n], torch.full((n,), slot, dtype=torch.int32, device=dev), i32(ar + start), ar, i32(n - 1 - ar)))
        off += n
    ids, slot_r, pos_r, nprev_r, nafter_r = (torch.cat([q[j] for q in parts]) for j in range(5))
    T = ids.shape[0]
    # one prompt chunk and no decode rows: a single-sequence batch, so the engine can run it token-sharded (I2_PC_SHARD)
    bt = Batch(slot_r, pos_r, nprev_r, True, nsplit_g=8, nsplit_l=4, nafter=None if (nact == 0 and nseg == 1) else nafter_r)
    taps = torch.empty((T, 8, 4608), dtype=torch.bfloat16, device=dev)
    # logits only for the rows that are sampled: every decode row, and the last row of each finishing prompt
    lr, r_ = list(range(nact * R)) if nact else [], (nact * R if nact else 0)
    last_rows = {}
    for slot, start, n, last in segs:
        r_ += n
        if last: last_rows[slot] = len(lr); lr.append(r_ - 1)
    if POL is not None and nact:
        eng.live = torch.cat([(arR[None].long() < wv[:, None]).reshape(-1), torch.ones(T - nact * R, dtype=torch.bool, device=dev)])
    z = eng.forward(ids, bt, taps=taps, logit_rows=torch.tensor(lr, dtype=torch.long, device=dev)) if lr else None
    if z is None:
        eng.forward(ids, bt, taps=taps, return_hidden=True)
    eng.live = None
    if nact:
        g = sample_rows(z[:nact * R], temp_all[sll].repeat_interleave(R)).view(nact, R)
        nacc = ((d == g[:, :K]) & (ar7[None, :K] < (wv - 1)[:, None])).int().cumprod(1).sum(1)
    draft.store_context(taps, slot_r, pos_r)
    emit_n = None
    if nact:
        newanchor = g.gather(1, nacc[:, None])[:, 0]
        emit = torch.cat([d, g[:, :1]], 1)
        emit.scatter_(1, nacc[:, None], newanchor[:, None])
        Pn = P + (nacc + 1).to(torch.int32)
        f_last = taps[:nact * R].view(nact, R, 8, 4608)[torch.arange(nact, device=dev), nacc]
        KD = DRAFT_K(K)
        dn = draft.draft(newanchor, sl, Pn, f_last, nsplit_g=8, K=KD)
        anchor_all[sll] = newanchor; P_all[sll] = Pn; d_all[sll, :KD] = dn
        if POL is not None: conf_all[sll] = draft.last_conf
        emit_n = (emit, nacc, wv)
    first = {}
    row = nact * R if nact else 0
    for i, (slot, start, n, last) in enumerate(segs):
        row += n
        if last:
            temp_all[slot] = fpay[i]
            zi = last_rows[slot]
            tk = sample_rows(z[zi:zi + 1], temp_all[slot:slot + 1])
            anchor_all[slot] = tk[0]; P_all[slot] = start + n
            sl1 = torch.tensor([slot], dtype=torch.int32, device=dev)
            d_all[slot] = draft.draft(tk, sl1, P_all[slot:slot + 1], taps[row - 1:row], nsplit_g=8)[0]
            if POL is not None: conf_all[slot] = draft.last_conf[0]
            first[slot] = tk
    if rank == 0:
        return ((emit_n[0].cpu(), emit_n[1].cpu(), emit_n[2].cpu()) if emit_n else (None, None, None)), {k: int(v[0]) for k, v in first.items()}
    return None

def worker_loop():
    while True:
        h, pay, fpay = bcast()
        op = h[0]
        if op == OP_STOP: break
        if op == OP_PREFILL: do_prefill(h, pay, fpay)
        elif op == OP_DECODE: do_decode(h, pay, fpay)
        elif op == OP_PAGES: do_pages(h, pay, fpay)
        elif op == OP_MIXED: do_mixed(h, pay, fpay)
        elif op in (OP_SNAP, OP_RESTORE): do_snap(h, pay, fpay)

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
free_slots = list(range(NS))
import collections, numpy as np
parked = collections.OrderedDict()  # slot -> parked session (LRU order): hist, P, plen, written, pages, snap
free_pages = list(range(1, NP))       # page 0 belongs to the dummy slot
active = {}                           # slot -> Req
prefilling = []                       # Reqs with remaining prompt
loop_ref = {}
STATS = {'dec_steps': 0, 'dec_time': 0.0, 'dec_rows': 0, 'dec_tok': 0, 'pre_time': 0.0, 'pre_tok': 0, 'idle_time': 0.0, 'stall_time': 0.0, 'tool_ok': 0, 'tool_bad': 0, 'mix_steps': 0, 'mix_time': 0.0, 'mix_pre_tok': 0, 'mix_rows': 0, 'mix_tok': 0, 'mix_dsteps': 0, 'aborted': 0, 'pre_cached_tok': 0, 'reuse_a': 0, 'reuse_b': 0, 'reuse_evict': 0, 'dec_vrows': 0, 'dec_prows': 0}

class WidthChooser:
    """rank 0: physical verify width of the next decode graph. The graph itself reports, after drafting, which captured
    width has the most expected tokens per ms for its slot set's new drafts; that report is used when the next round has
    the same slot set, else the bucket's last choice (the widest width before any report)"""
    def __init__(self):
        self.prev = None; self.last = {}; self.chosen = collections.Counter(); self.live = collections.Counter()
    def choose(self, B, slots):
        ws = widths(B)
        if self.prev is not None and self.prev[0] == (B, tuple(slots)):
            W = ws[self.prev[1]]
        else:
            W = self.last.get(B, ws[-1])
        self.chosen[(B, W)] += 1
        return W
    def update(self, B, slots, W, wi_next, wlive):
        self.prev = ((B, tuple(slots)), int(wi_next)); self.last[B] = widths(B)[int(wi_next)]
        for x in wlive: self.live[int(x)] += 1
CHOOSER = WidthChooser() if args.adaptive else None

class Req:
    def __init__(self, ids, max_tokens, temperature, out_q):
        self.ids, self.max_tokens, self.temperature, self.out_q = ids, max_tokens, temperature, out_q
        self.id = "chatcmpl-" + uuid.uuid4().hex[:24]
        self.slot = None; self.pages = []; self.done_prompt = 0; self.gen = []; self.t_arrive = time.time()
        self.ignore_eos = False
        self.first_token_time = None; self.finish = None

def emit(req, item):
    loop_ref["loop"].call_soon_threadsafe(req.out_q.put_nowait, item)

admitted = []                         # admitted since the last scheduler round (their page tables go out via OP_PAGES)

def evict_one():
    """drop the least recently used parked session; False if none"""
    if not parked: return False
    slot, rec = parked.popitem(last=False)
    free_pages.extend(rec["pages"]); free_slots.append(slot); STATS['reuse_evict'] += 1
    return True

def take_pages(n):
    while len(free_pages) < n and evict_one(): pass
    if len(free_pages) < n: return None
    return [free_pages.pop() for _ in range(n)]

def match_parked(ids):
    """best exact restore point among parked sessions: (slot, p, mode) or None. (a) continuation of the committed history
    (p = common prefix, within 8 of the valid KV length); (b) the previous prompt end (conv snapshot; previous generation
    <= 1000 tokens so the SWA / draft rings still hold the window before it)"""
    if not parked or len(ids) <= args.reuse_min: return None
    arr = np.asarray(ids, dtype=np.int64); best = None
    for slot, rec in parked.items():
        h = rec["hist"]; L = min(len(h), len(arr))
        if L < args.reuse_min: continue
        neq = np.flatnonzero(h[:L] != arr[:L]); c = int(neq[0]) if len(neq) else L
        Pv = min(rec["P"], len(h))
        if c >= Pv - 8 and min(c, Pv) <= len(arr) - 1: p, mode = min(c, Pv), "a"
        elif rec["snap"] and c >= rec["plen"] and rec["plen"] <= len(arr) - 1 and rec["written"] - rec["plen"] <= 1000: p, mode = rec["plen"], "b"
        else: continue
        if p >= args.reuse_min and (best is None or p > best[1]): best = (slot, p, mode)
    return best

def try_admit():
    while not pending.empty() and len(active) + len(prefilling) < S:
        req = pending.queue[0]
        if getattr(req, "cancelled", False):
            pending.get(); STATS['aborted'] += 1; continue
        need = (len(req.ids) + req.max_tokens + 24 + PS - 1) // PS
        if need > pt_len - 2 or need > NP - 1:
            pending.get(); emit(req, ("error", "request longer than the KV cache / max_model_len")); continue
        m = match_parked(req.ids) if REUSE and not getattr(req, "no_reuse", False) else None
        if m is not None:
            slot, p, mode = m
            rec = parked.pop(slot)
            pages = rec["pages"]
            if len(pages) >= need:
                free_pages.extend(pages[need:]); pages = pages[:need]
            else:
                more = take_pages(need - len(pages))
                if more is None:          # not enough KV: put the session back and wait
                    parked[slot] = rec; parked.move_to_end(slot, last=False); break
                pages = pages + more
            pending.get()
            req.slot, req.pages, req.done_prompt, req.restore = slot, pages, p, mode == "b"
            STATS['pre_cached_tok'] += p; STATS['reuse_' + mode] += 1
        else:
            if not free_slots and not evict_one(): break
            pages = take_pages(need)
            if pages is None: break
            pending.get()
            req.slot = free_slots.pop(0); req.pages = pages; req.restore = False
        prefilling.append(req)
        if args.mixed: admitted.append(req)

def finish(req, reason):
    req.finish = reason
    active.pop(req.slot, None)
    if REUSE and req.first_token_time is not None:
        # park the session: its slot keeps rings + pages for a request that extends this history
        hist = np.asarray(req.ids + req.gen, dtype=np.int64)
        P = int(P_all[req.slot].item())
        parked[req.slot] = dict(hist=hist, P=P, plen=len(req.ids), written=P + 8, pages=req.pages, snap=getattr(req, "snapped", False))
    else:
        free_slots.append(req.slot); free_pages.extend(req.pages)
    emit(req, ("done", reason))

def scheduler():
    last_hb = time.time()
    while True:
        try_admit()
        did = False
        if args.mixed:
            while admitted:
                req = admitted.pop(0)
                bcast([OP_PAGES, req.slot, len(req.pages), 0, 0, 0, len(req.pages), 0], req.pages)
                do_pages(HDR.tolist(), PAY[:len(req.pages)], None)
                if getattr(req, "restore", False):          # session reuse at the previous prompt end: conv rings back
                    bcast([OP_RESTORE, req.slot, 1, 0, 0, 0, 0, 0]); do_snap(HDR.tolist(), None, None)
        if args.mixed and prefilling:
            mixed_round(); continue
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
            torch.cuda.synchronize()        # bill the chunk's GPU time to prefill (else the next decode step absorbs it)
            req.done_prompt += n
            STATS['pre_time'] += time.time() - tp0; STATS['pre_tok'] += n
            if active: STATS['stall_time'] += time.time() - tp0       # decoding requests waited for this prefill chunk
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
            B = next(b for b in BUCKETS if b >= nb)
            W = CHOOSER.choose(B, slots) if CHOOSER else spec_k(B) + 1
            # burst of back-to-back steps when nothing waits for admission/prefill; bounded by every active request's
            # page headroom (24 positions past max_tokens) so overshooting steps never write outside its pages
            R = W
            nsteps = 1
            if not prefilling and (pending.empty() or not free_slots) and args.burst > 1:
                head = min((r.max_tokens + 24 - R - len(r.gen)) // R for r in active.values())
                nsteps = max(1, min(args.burst, head))
            h = [OP_DECODE, B, nb, int(use_spec), nsteps, W, nb, 0]
            td0 = time.time()
            bcast(h, slots)
            out_t = do_decode(HDR.tolist(), PAY[:nb], None)
            emit_t, n_t, w_t = out_t[:, :, :8], out_t[:, :, 8], out_t[:, :, 9]
            if CHOOSER: CHOOSER.update(B, slots, W, out_t[-1, 0, 10], w_t[-1, :nb].tolist())
            STATS['dec_vrows'] += int(w_t[:, :nb].sum()); STATS['dec_prows'] += B * W * nsteps
            STATS['dec_steps'] += nsteps; STATS['dec_time'] += time.time() - td0; STATS['dec_rows'] += nb * nsteps
            STATS['dec_tok'] += int(n_t[:, :nb].sum()) + nb * nsteps
            for st in range(nsteps):
                for i, s in enumerate(slots):
                    req = active.get(s)
                    if req is None: continue
                    k = int(n_t[st, i]) + 1
                    handle_tokens(req, emit_t[st, i, :k].tolist())
            did = True
        if not did:
            time.sleep(0.002)
            if time.time() - last_hb > 1.0:
                bcast([OP_NOOP]); last_hb = time.time()

def mixed_round():
    """one mixed step: all active slots' speculative verify + prefill chunks of pending prompts under the token budget"""
    for req in [r for r in prefilling if getattr(r, "cancelled", False)]:     # client gone during its prefill
        prefilling.remove(req); free_slots.append(req.slot); free_pages.extend(req.pages); STATS['aborted'] += 1
    if not prefilling and not active:
        return
    slots = sorted(active)
    nact = len(slots)
    Bm = next(b for b in BUCKETS if b >= nact) if nact else 0
    Km = (widths(Bm)[-1] if CHOOSER else spec_k(Bm) + 1) if nact else 0       # physical rows per decoding slot
    R = Km
    left = min(args.prefill_budget, args.max_step_rows - nact * R) if nact else args.chunk
    segs = []
    for req in prefilling:
        if left <= 0 or len(segs) >= 16: break
        n = min(left, len(req.ids) - req.done_prompt, args.chunk)
        segs.append((req, req.done_prompt, n, req.done_prompt + n == len(req.ids)))
        left -= n
    ids = [t for req, st, n, _ in segs for t in req.ids[st:st + n]]
    pay = slots + [x for req, st, n, last in segs for x in (req.slot, st, n, int(last))] + ids
    h = [OP_MIXED, nact, len(segs), len(ids), Km, 0, len(pay), len(segs)]
    t0 = time.time()
    bcast(h, pay, [req.temperature for req, _, _, _ in segs])
    (emit_t, n_t, _w), first = do_mixed(HDR.tolist(), PAY[:len(pay)], FPAY[:len(segs)])
    dt = time.time() - t0
    STATS['mix_steps'] += 1; STATS['mix_time'] += dt; STATS['mix_pre_tok'] += len(ids)
    STATS['mix_rows'] += nact; STATS['mix_tok'] += (int(n_t.sum()) + nact) if nact else 0
    STATS['mix_dsteps'] += 1 if nact else 0          # mixed steps that also decoded
    if not nact: STATS['pre_time'] += dt; STATS['pre_tok'] += len(ids)
    for i, s in enumerate(slots):
        req = active.get(s)
        if req is not None:
            handle_tokens(req, emit_t[i, :int(n_t[i]) + 1].tolist())
    for req, st, n, last in segs:
        req.done_prompt += n
        if last and REUSE:                                   # conv rings at prompt end (restore point (b) for a successor)
            bcast([OP_SNAP, req.slot, 0, 0, 0, 0, 0, 0]); do_snap(HDR.tolist(), None, None); req.snapped = True
        if last:
            prefilling.remove(req)
            req.first_token_time = time.time()
            active[req.slot] = req
            handle_tokens(req, [first[req.slot]])

def handle_tokens(req, toks):
    if getattr(req, "cancelled", False) and req.slot in active and not req.finish:
        finish(req, "abort"); STATS['aborted'] += 1; return
    out = []
    def flush():
        if out:
            if args.sse_batch: emit(req, ("toks", list(out)))
            else:
                for t_ in out: emit(req, ("tok", t_))
            out.clear()
    for t in toks:
        if req.slot not in active or req.finish: break
        if t in (T_EOS, 199999) and not req.ignore_eos:
            flush(); finish(req, "stop"); return
        req.gen.append(t)
        out.append(t)
        if len(req.gen) >= req.max_tokens:
            flush(); finish(req, "length"); return
    flush()

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
                    STATS['tool_ok'] += 1
                    out.append(("tool", {"index": self.tool_idx, "id": "call_" + uuid.uuid4().hex[:20], "type": "function",
                                         "function": {"name": obj.get("name", ""), "arguments": json.dumps(obj.get("args", {}))}}))
                except Exception as e:
                    STATS['tool_bad'] += 1
                    print(f"TOOL_JSON_BAD body_len={len(body)} err={type(e).__name__}: {str(e)[:80]}", flush=True)   # no content logged
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

_EFFORT_NAMES = {"none": 0.0, "off": 0.0, "minimal": 0.2, "low": 0.4, "medium": 0.6, "high": 0.9, "xhigh": 0.99, "max": 0.99}

def norm_effort(e):
    """Inkling's template wants a float in [0, 0.99]; clients (Pi/OMP) send names or numeric strings."""
    if e is None or isinstance(e, bool): return None
    if isinstance(e, str):
        s = e.strip().lower()
        if s in _EFFORT_NAMES: return _EFFORT_NAMES[s]
        try: e = float(s)
        except ValueError: return None
    try: return min(max(float(e), 0.0), 0.99)
    except (TypeError, ValueError): return None

def render_prompt(messages, tools, reasoning_effort):
    kw = dict(messages=messages, tools=tools or None, add_generation_prompt=True)
    reasoning_effort = norm_effort(reasoning_effort)
    if reasoning_effort is not None: kw["reasoning_effort"] = reasoning_effort
    return template.render(**kw)

def encode(text):
    return tok.encode(text, add_special_tokens=False).ids

app = FastAPI()

@app.get("/health")
async def health(): return {"status": "ok", "stats": STATS, "spec": SPEC, "kv_tokens": NP * PS, "active": len(active), "queued": pending.qsize(), "parked": len(parked)}

@app.get("/v1/models")
async def models(): return {"object": "list", "data": [{"id": args.served_name, "object": "model", "owned_by": "local"}]}

@app.post("/tokenize")
@app.post("/v1/tokenize")
async def tokenize(request: Request):
    """vLLM-style token count: {"prompt": text} or {"messages": [...]} (rendered with the chat template)"""
    body = await request.json()
    if body.get("messages"):
        text = render_prompt(norm_messages(body["messages"]), body.get("tools"), body.get("reasoning_effort"))
    else:
        text = body.get("prompt") or body.get("content") or body.get("text") or ""
    ids = encode(text)
    return {"tokens": ids, "count": len(ids), "max_model_len": args.max_model_len}

def norm_messages(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("role") == "developer":      # OpenAI's newer name for the system message (Pi / OMP send it)
            m["role"] = "system"
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
    if chat and body.get("raw_prompt"):      # extension: pre-rendered prompt in the model's chat format (eval harness)
        prompt = body["raw_prompt"]
    elif chat:
        prompt = render_prompt(norm_messages(body["messages"]), body.get("tools"), body.get("reasoning_effort"))
    else:
        prompt = body["prompt"] if isinstance(body["prompt"], str) else body["prompt"][0]
    ids = encode(prompt)
    max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 4096)
    temp = float(body.get("temperature", 0.0) or 0.0)
    out_q = asyncio.Queue()
    req = Req(ids, max_tokens, temp, out_q)
    req.no_reuse = bool(body.get("no_reuse"))          # extension (tests): force a cold prefill
    req.ignore_eos = bool(body.get("ignore_eos"))          # vLLM extension (benchmarks): decode to max_tokens past EOS
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
    so = body.get("stream_options") or {}
    cont_usage = bool(so.get("continuous_usage_stats"))   # vLLM: cumulative usage on every streamed chunk
    ntok_seen = [0]
    async def events():
        sent_role = False
        has_tool = False
        while True:
            kind, val = await out_q.get()
            if kind in ("tok", "toks"):
                vals = val if kind == "toks" else [val]
                ntok_seen[0] += len(vals)
                out = [kv for t_ in vals for kv in parser.feed(t_)]
                if kind == "toks":          # one merged delta per step: text pieces joined, tool calls kept separate
                    merged = []
                    for k, v in out:
                        if merged and k in ("reasoning", "content") and merged[-1][0] == k: merged[-1] = (k, merged[-1][1] + v)
                        else: merged.append((k, v))
                    out = merged
                if not out and cont_usage:
                    yield {}, None          # tokens without visible text still advance the usage count
                for k, v in out:
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
          try:
            async for d, fin in events():
                c = chunk(d, fin)
                if fin is not None:
                    c["usage"] = {"prompt_tokens": nprompt, "completion_tokens": len(req.gen), "total_tokens": nprompt + len(req.gen)}
                elif cont_usage:
                    c["usage"] = {"prompt_tokens": nprompt, "completion_tokens": ntok_seen[0], "total_tokens": nprompt + ntok_seen[0]}
                yield f"data: {json.dumps(c)}\n\n"
            yield "data: [DONE]\n\n"
          finally:          # client gone (generator closed / cancelled) before the request finished: stop decoding it
            if req.finish is None: req.cancelled = True
        return StreamingResponse(gen(), media_type="text/event-stream")
    content, reasoning, tools, fin = "", "", [], "stop"
    async for d, f in events():
        content += d.get("content", ""); reasoning += d.get("reasoning_content", "")
        tools += d.get("tool_calls", [])
        if f: fin = f
    msg = {"role": "assistant", "content": content or None}
    if reasoning: msg["reasoning_content"] = reasoning
    if tools: msg["tool_calls"] = tools
    out = {"id": req.id, "object": "chat.completion", "created": created, "model": args.served_name,
           "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
           "usage": {"prompt_tokens": nprompt, "completion_tokens": len(req.gen), "total_tokens": nprompt + len(req.gen)}}
    if body.get("return_token_ids"):
        out["token_ids"] = list(req.gen)
    return out

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
