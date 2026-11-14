"""Inkling-2-Small EXL3 decode/prefill speed sweep (TP4 attention + EP4 experts). Launch via scripts/sweep.sh (torchrun, 4 ranks)."""
import os, sys, time, json, argparse, importlib.util
import torch, torch.distributed as dist
from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.modules.inkling2_engine import I2Engine, Batch
from exllamav3.modules.inkling2_draft import I2Draft

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/models/Inkling-2-Small-EXL3-3.48bpw"))
ap.add_argument("--concurrency", default=os.environ.get("CONCURRENCY", "1,4,8,16"))
ap.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "3")))
ap.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "128")), help="tokens generated per sequence")
ap.add_argument("--prompt", type=int, default=int(os.environ.get("PROMPT", "256")))
ap.add_argument("--graph", type=int, default=int(os.environ.get("GRAPH", "1")))
ap.add_argument("--spec", default=os.environ.get("SPEC", "on"), choices=["on", "off"],
                help="speculative decoding (MTP) on/off; default on")
ap.add_argument("--prompt-file", default=os.environ.get("PROMPT_FILE", "/opt/exllamav3/README.md"), help="public text used for prompts")
ap.add_argument("--layers", type=int, default=48)
ap.add_argument("--out", default=os.environ.get("RESULTS_DIR", "/models/results"))
a = ap.parse_args()

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
dist.init_process_group("nccl", device_id=dev)
def log(*x):
    if rank == 0: print(time.strftime("%H:%M:%S"), *x, flush=True)


spec_on = a.spec == "on"
cfg = Config.from_directory(a.model_dir)
model = Model.from_config(cfg, tp_rank=rank, tp_size=world)
model.layers = model.layers[:a.layers]
t0 = time.time()
model.embed.load(dev)
for L in model.layers: L.load(dev)
model.head.load(dev)
log(f"loaded in {time.time()-t0:.0f}s, {torch.cuda.memory_allocated()/2**30:.1f} GB/rank")

Bs = [int(x) for x in a.concurrency.split(",")]
B_MAX = max(Bs)
PP = (a.prompt + a.steps + 8 * 4 + 64) // 64 + 2          # pages per slot
eng = I2Engine(model, dev, rank, world, max_slots=B_MAX, num_pages=B_MAX * PP + 8, bits_g=4, bits_l=8)
for s in range(B_MAX):
    for p in range(PP): eng.page_table[s, p] = s * PP + p
draft = None
if spec_on:
    draft = I2Draft(os.path.join(a.model_dir, "mtp.safetensors"), eng, dev, rank, world, B_MAX, B_MAX * PP + 8, 64)
    log("draft loaded", f"{torch.cuda.memory_allocated()/2**30:.1f} GB/rank")
V = cfg.i2.unpadded_vocab

# Prompts: slices of a public text file, tokenized with the model's tokenizer
from tokenizers import Tokenizer
tk = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
ids_all = torch.tensor([t for t in tk.encode(open(a.prompt_file, encoding="utf-8", errors="ignore").read()).ids if t < V])
assert ids_all.numel() > a.prompt + 64, "prompt file too short"
def prompt_for(s):
    off = (s * 211) % (ids_all.numel() - a.prompt)
    return ids_all[off:off + a.prompt].to(dev)

@torch.no_grad()
def prefill(slot, ids):
    n, CH = ids.shape[0], 512
    last = None
    for s0 in range(0, n, CH):
        c = ids[s0:s0 + CH]; m = c.shape[0]
        pos = torch.arange(s0, s0 + m, dtype=torch.int32, device=dev)
        b = Batch(torch.full((m,), slot, dtype=torch.int32, device=dev), pos, torch.arange(m, dtype=torch.int32, device=dev),
                  True, nsplit_g=4, nsplit_l=4)
        taps = torch.empty((m, 8, 4608), dtype=torch.bfloat16, device=dev)
        z = eng.forward(c, b, taps=taps)
        if spec_on:
            draft.store_context(taps, b.slot, b.pos)
        last = (eng.global_argmax(z[-1:]), taps[-1:].clone())
    return last

def do_prefill(B):
    torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
    res = [prefill(s, prompt_for(s)) for s in range(B)]
    torch.cuda.synchronize(); return res, time.time() - t0

@torch.no_grad()
def decode_plain(B, N, first):
    tok = torch.cat([f[0] for f in first])
    slot = torch.arange(B, dtype=torch.int32, device=dev)
    pos = torch.full((B,), a.prompt, dtype=torch.int32, device=dev)
    bt = Batch(slot, pos, torch.zeros(B, dtype=torch.int32, device=dev), False, nsplit_g=4, nsplit_l=4)
    sid = tok.clone()
    def one(): return eng.global_argmax(eng.forward(sid, bt))
    graph = None
    if a.graph:
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): one()
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize(); dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): sout = one()
    out = tok
    torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
    for _ in range(N - 1):
        sid.copy_(out)
        out = (graph.replay() or sout.clone()) if graph else one()
        pos += 1
    torch.cuda.synchronize(); dt = time.time() - t0
    return dict(steps=N - 1, tokens=B * (N - 1), dt=dt, accept=None)

@torch.no_grad()
def decode_spec(B, N, first):
    slot = torch.arange(B, dtype=torch.int32, device=dev)
    anchor = torch.cat([f[0] for f in first]); feats_last = torch.cat([f[1] for f in first])
    P = torch.full((B,), a.prompt, dtype=torch.int32, device=dev)
    d = draft.draft(anchor, slot, P, feats_last).clone()
    ar8 = torch.arange(8, dtype=torch.int32, device=dev)
    slot8 = slot.repeat_interleave(8); nprev = ar8.repeat(B); nafter = (7 - nprev).to(torch.int32)
    taps = torch.empty((B * 8, 8, 4608), dtype=torch.bfloat16, device=dev)
    arB = torch.arange(B, device=dev)
    n_out = torch.zeros(B, dtype=torch.long, device=dev)
    def step():
        ids = torch.cat([anchor[:, None], d], 1)
        pos = (P[:, None] + ar8[None]).reshape(-1).to(torch.int32)
        bt = Batch(slot8, pos, nprev, True, nsplit_g=4, nsplit_l=4, nafter=nafter)
        z = eng.forward(ids.reshape(-1), bt, taps=taps)
        g = eng.global_argmax(z).view(B, 8)
        n = (d == g[:, :7]).int().cumprod(1).sum(1)
        draft.store_context(taps, slot8, pos)
        newanchor = g.gather(1, n[:, None])[:, 0]
        n_out.copy_(n)
        P.add_((n + 1).to(torch.int32))
        anchor.copy_(newanchor)
        f_last = taps.view(B, 8, 8, 4608)[arB, n]
        d.copy_(draft.draft(anchor, slot, P, f_last))
    graph = None
    if a.graph:
        P0, a0, d0 = P.clone(), anchor.clone(), d.clone()
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2): step()
        torch.cuda.current_stream().wait_stream(st); torch.cuda.synchronize(); dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): step()
        P.copy_(P0); anchor.copy_(a0); d.copy_(d0)
    torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
    steps, emitted = 0, 0
    produced = torch.zeros(B, dtype=torch.long)
    while int(produced.min()) < N:
        if graph: graph.replay()
        else: step()
        gn = n_out.cpu() + 1
        produced += gn; emitted += int(gn.sum()); steps += 1
    torch.cuda.synchronize(); dt = time.time() - t0
    return dict(steps=steps, tokens=emitted, dt=dt, accept=emitted / (steps * B))

results = []
for B in Bs:
    runs = []
    for r in range(a.repeats + 1):               # first run is warmup (kernel autotune), discarded
        first, tp = do_prefill(B)
        d_ = (decode_spec if spec_on else decode_plain)(B, a.steps, first)
        d_["prefill_tok_s"] = B * a.prompt / tp
        d_["decode_tok_s"] = d_["tokens"] / d_["dt"]; d_["steps_s"] = d_["steps"] / d_["dt"]
        if r > 0: runs.append(d_)
        torch.cuda.empty_cache()
    med = lambda k: sorted(x[k] for x in runs)[len(runs) // 2]
    acc = [x["accept"] for x in runs if x["accept"] is not None]
    results.append(dict(concurrency=B, decode_tok_s=med("decode_tok_s"), prefill_tok_s=med("prefill_tok_s"),
                        steps_s=med("steps_s"), accept_len=(sorted(acc)[len(acc) // 2] if acc else None),
                        decode_runs=[x["decode_tok_s"] for x in runs]))
    log(f"c{B}: decode {results[-1]['decode_tok_s']:.1f} tok/s, accept {results[-1]['accept_len']}, prefill {results[-1]['prefill_tok_s']:.0f} tok/s")

if rank == 0:
    meta = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), model_dir_name=os.path.basename(a.model_dir.rstrip("/")),
                prompt=a.prompt, tokens_per_seq=a.steps, repeats=a.repeats, cuda_graphs=bool(a.graph),
                spec=a.spec, gpus=[torch.cuda.get_device_name(i) for i in range(world)])
    lab = "speculative decoding ON (DSpark/MTP)" if spec_on else "SPEC OFF (plain decode)"
    lines = [f"### Inkling-2-Small EXL3 sweep - {lab}", "",
             f"{a.prompt}-token prompts, {a.steps} generated tokens/seq, median of {a.repeats} repeats, CUDA graphs={'on' if a.graph else 'off'}", "",
             "| concurrency | decode tok/s | accept len | steps/s | prefill tok/s |", "|---|---|---|---|---|"]
    for r in results:
        al = f"{r['accept_len']:.2f}" if r["accept_len"] else "n/a"
        lines.append(f"| c{r['concurrency']} | {r['decode_tok_s']:.1f} | {al} | {r['steps_s']:.1f} | {r['prefill_tok_s']:.0f} |")
    print("\n" + "\n".join(lines) + "\n", flush=True)
    try:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, "sweep-" + time.strftime("%Y%m%d-%H%M%S") + f"-spec{a.spec}.json")
        json.dump(dict(meta=meta, results=results), open(fn, "w"), indent=2)
        print("wrote", fn)
    except OSError as e:
        print("could not write results:", e)
dist.barrier(); dist.destroy_process_group()
