"""Inkling-2-Small EXL3 decode/prefill speed sweep (TP4 attention + EP4 experts). Launch via scripts/sweep.sh (torchrun, 4 ranks)."""
import os, sys, time, json, argparse, importlib.util
import torch, torch.distributed as dist
from exllamav3.model.config import Config
from exllamav3.model.model import Model
from exllamav3.modules.inkling2_engine import I2Engine, Batch

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/models/Inkling-2-Small-EXL3-3.48bpw"))
ap.add_argument("--concurrency", default=os.environ.get("CONCURRENCY", "1,4,8,16"))
ap.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "3")))
ap.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "64")))
ap.add_argument("--prompt", type=int, default=int(os.environ.get("PROMPT", "256")))
ap.add_argument("--graph", type=int, default=int(os.environ.get("GRAPH", "0")))
ap.add_argument("--spec", default=os.environ.get("SPEC", "on"), choices=["on", "off"],
                help="speculative decoding (MTP) on/off; default on")
ap.add_argument("--layers", type=int, default=48)
ap.add_argument("--out", default=os.environ.get("RESULTS_DIR", "/models/results"))
a = ap.parse_args()

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
dist.init_process_group("nccl", device_id=dev)
def log(*x):
    if rank == 0: print(time.strftime("%H:%M:%S"), *x, flush=True)

# Speculative decoding hook: the spec-decode module is not part of this snapshot until it lands upstream of the overlay.
spec_available = importlib.util.find_spec("exllamav3.modules.inkling2_spec") is not None
spec_effective = a.spec == "on" and spec_available
if a.spec == "on" and not spec_available:
    log("WARNING: SPEC=on requested but speculative decoding is NOT in this snapshot yet. Running SPEC-OFF; "
        "numbers below are spec-off (no acceptance length).")
if spec_effective:
    log("ERROR: spec module present but sweep is not wired to it yet in this version"); sys.exit(2)

cfg = Config.from_directory(a.model_dir)
model = Model.from_config(cfg, tp_rank=rank, tp_size=world)
model.layers = model.layers[:a.layers]
t0 = time.time()
model.embed.load(dev)
for L in model.layers: L.load(dev)
model.head.load(dev)
log(f"loaded in {time.time()-t0:.0f}s, {torch.cuda.memory_allocated()/2**30:.1f} GB/rank")

Bs = [int(x) for x in a.concurrency.split(",")]
Bmax = max(Bs)
PP = (a.prompt + a.steps + 64 + 3 * 1) // 64 + 1
eng = I2Engine(model, dev, rank, world, max_slots=Bmax, num_pages=Bmax * PP + 8, bits_g=4, bits_l=8)
for s in range(Bmax):
    for p in range(PP):
        eng.page_table[s, p] = s * PP + p
V = cfg.i2.unpadded_vocab
g = torch.Generator().manual_seed(1234)
prompts = [torch.randint(1000, V - 1000, (a.prompt,), generator=g).to(dev) for _ in range(Bmax)]  # synthetic tokens

def gather_argmax(z):
    zs = [torch.empty_like(z) for _ in range(world)]
    dist.all_gather(zs, z.contiguous())
    return torch.cat(zs, -1)[:, :V].argmax(-1)

@torch.no_grad()
def prefill(slot, ids):
    n = ids.shape[0]
    b = Batch(torch.full((n,), slot, dtype=torch.int32, device=dev), torch.arange(n, dtype=torch.int32, device=dev),
              torch.arange(n, dtype=torch.int32, device=dev), True, nsplit_g=4, nsplit_l=4)
    h = eng.embed_tokens(ids); blocks = eng.new_blocks(n)
    for L in eng.layers: h = eng.layer(L, h, blocks, b)
    return gather_argmax(eng.head_logits(h[-1:], blocks[:, -1:]))

def run_once(B):
    torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
    tok = torch.cat([prefill(s, prompts[s]) for s in range(B)])
    torch.cuda.synchronize(); tp = time.time() - t0
    slot = torch.arange(B, dtype=torch.int32, device=dev)
    pos = torch.full((B,), a.prompt, dtype=torch.int32, device=dev)
    nprev = torch.zeros(B, dtype=torch.int32, device=dev)
    bt = Batch(slot, pos, nprev, False, nsplit_g=4, nsplit_l=4)
    def step(t):
        return gather_argmax(eng.forward(t, bt))
    with torch.no_grad():
        for _ in range(3): tok = step(tok); pos += 1
        torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
        for _ in range(a.steps): tok = step(tok); pos += 1
        torch.cuda.synchronize(); dt = time.time() - t0
    return B * a.prompt / tp, B * a.steps / dt, a.steps / dt

results = []
for B in Bs:
    runs = [run_once(B) for _ in range(a.repeats)]
    pre = sorted(r[0] for r in runs)[len(runs) // 2]; dec = sorted(r[1] for r in runs)[len(runs) // 2]
    sps = sorted(r[2] for r in runs)[len(runs) // 2]
    results.append(dict(concurrency=B, prefill_tok_s=pre, decode_tok_s=dec, steps_s=sps, accept_len=None,
                        decode_runs=[r[1] for r in runs], prefill_runs=[r[0] for r in runs]))
    log(f"c{B}: decode {dec:.1f} tok/s, prefill {pre:.0f} tok/s")

if rank == 0:
    meta = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), model_dir_name=os.path.basename(a.model_dir.rstrip("/")),
                prompt=a.prompt, steps=a.steps, repeats=a.repeats, graph=a.graph, spec_requested=a.spec,
                spec_effective=spec_effective, gpus=[torch.cuda.get_device_name(i) for i in range(world)])
    lab = "spec ON" if spec_effective else "SPEC OFF (speculative decoding not active)"
    lines = [f"### Inkling-2-Small EXL3 sweep - {lab}", "",
             f"prompt {a.prompt} synthetic tokens, {a.steps} decode steps, median of {a.repeats} repeats, graphs={a.graph}", "",
             "| concurrency | decode tok/s | accept len | prefill tok/s | steps/s |", "|---|---|---|---|---|"]
    for r in results:
        al = f"{r['accept_len']:.2f}" if r["accept_len"] else "n/a"
        lines.append(f"| c{r['concurrency']} | {r['decode_tok_s']:.1f} | {al} | {r['prefill_tok_s']:.0f} | {r['steps_s']:.1f} |")
    print("\n" + "\n".join(lines) + "\n", flush=True)
    try:
        os.makedirs(a.out, exist_ok=True)
        fn = os.path.join(a.out, "sweep-" + time.strftime("%Y%m%d-%H%M%S") + f"-spec{'on' if spec_effective else 'off'}.json")
        json.dump(dict(meta=meta, results=results), open(fn, "w"), indent=2)
        print("wrote", fn)
    except OSError as e:
        print("could not write results:", e)
dist.barrier(); dist.destroy_process_group()
