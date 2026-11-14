# inkling2-exl3

One-command serving and speed sweeps for [Inkling-2-Small EXL3 4.0 bpw](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-4bpw)
on **4x 96 GB Blackwell GPUs (RTX PRO 6000, SM120)**. It is an [exllamav3](https://github.com/turboderp-org/exllamav3) (v1.6.6) overlay that adds:
- the Inkling-2 architecture;
- a batched engine, with attention split across the 4 GPUs (TP4) and experts spread across them (EP4);
- a paged KV cache that knows about the sliding-window layers, with rotated, grouped 4-bit global-layer KV;
- CUDA-graphed **speculative decoding (DSpark/MTP), always on**;
- the speculative verify step's routed experts through the [b12x](https://github.com/local-inference-lab/b12x) fused Trellis MoE kernels from Local Inference Lab (now in FlashInfer `flashinfer/experimental/b12x`). This is on by default.

## Run

```bash
# OpenAI-compatible server on :8000 (16 slots x 380K context)
docker run --rm --gpus all --ipc=host -p 8000:8000 -v $HOME/models:/models ghcr.io/0xsero/inkling2-exl3:latest serve

# speed sweep (c1/c4/c8/c16 decode tok/s, acceptance length, prefill tok/s)
docker run --rm --gpus all --ipc=host -v $HOME/models:/models ghcr.io/0xsero/inkling2-exl3:latest sweep

# import check, no GPU memory used: exllamav3 extension + b12x kernels
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES= ghcr.io/0xsero/inkling2-exl3:latest check
```

The first run downloads the model (~313 GB, resumable) into `$HOME/models/Inkling-2-Small-EXL3-4bpw`.
- `sweep` prints a markdown table (median of 3 repeats) and writes JSON to `/models/results/`.
- `serve` exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/health`. `/health` also reports the KV token capacity (`kv_tokens`).
- On the first request of each batch size, the b12x kernels JIT-compile through the CUTLASS DSL, so the first requests are slow.

| Env (pass with `-e`) | Default | |
|---|---|---|
| `PORT` | `8000` | server port |
| `I2_B12X` | `1` | verify-step routed experts through b12x; `0` = exllamav3's coop MoE kernel |
| `CONCURRENCY` | `1,4,8,16` | sweep batch sizes |
| `REPEATS` / `STEPS` / `PROMPT` | `3` / `128` / `256` | sweep settings |
| `GRAPH` | `1` | CUDA graphs |
| `KV_ROT` / `KV_NG` / `KV_Q16` | `1` / `4` / `1` | sweep global-KV format (same as the server) |
| `I2_MOE_GROUP` | `64` | tokens per coop MoE call (the paths b12x does not cover) |
| `MODEL_DIR` / `HF_REPO` / `RESULTS_DIR` | `/models/...` | |

Server flags go after `serve`:

| Flag | Default | |
|---|---|---|
| `--max-model-len` | `380000` | |
| `--slots` | `16` | concurrent sequences |
| `--bits-g` / `--bits-l` | `4` / `8` | KV bits for global / sliding-window layers; `--bits-g 8` for quality, at about half the token capacity |
| `--kv-rot` / `--kv-ng` / `--kv-q16` | `1` / `4` / `1` | global 4-bit KV: Hadamard rotation along head_dim, 4 fp16 scale groups per 128-vector, 16-level symmetric grid |
| `--kv-gb` | auto | KV budget per GPU |
| `--b12x` | `$I2_B12X` | |

Speculative decoding has no off switch. It runs at every batch size.

### Without Docker
1. Make a Python 3.12 venv. Install `docker/requirements-engine.lock` with `--extra-index-url https://download.pytorch.org/whl/cu128`, then `docker/requirements-b12x.lock` with `--no-deps`.
   - The second lock deliberately overrides torch's `cuda-bindings` pin. This is the validated combination.
2. Run `scripts/install_overlay.sh <exllamav3 v1.6.6 checkout>`. Then run `pip install --no-build-isolation .` in that checkout with `TORCH_CUDA_ARCH_LIST=12.0`, against the CUDA 12.8 toolkit.
3. Fetch FlashInfer at commit `9c50996794564e1e3b2a4b0683e9aa2bcc528df8`. No build is needed.
4. Set:
   - `I2_B12X_PATH=<tree>`;
   - `PYTHONPATH=<tree>:<tree>/flashinfer/experimental/b12x/_compat`;
   - `EXL3_LOAD_ARENA=0`;
   - `NCCL_CUMEM_ENABLE=0`.
5. Run `scripts/sweep.sh` or `torchrun --nproc_per_node 4 server/i2_server.py --port 8000 --b12x 1`.

FlashInfer's own `b12x` extra asks for torch >= 2.12. The kernels import and run on torch 2.10 here, and that is the validated setup, so the image installs the extra's other dependencies at fixed versions and leaves out the torch floor.

## Requirements
- 4x SM120 GPUs with 96 GB each and NCCL P2P;
- an NVIDIA driver supporting CUDA 12.8 (>= 570). Tested on 595.84;
- ~313 GB free disk;
- Docker with the NVIDIA container toolkit.

## Status (work in progress)
Measured 2026-11-14 on 4x RTX PRO 6000 at the 300 W power cap. Decode is power-bound. All numbers use speculative decoding (DSpark, 7 drafted tokens per step), with CUDA graphs.

**Quant quality.** The model was quantized against the BF16 reference; full results are on the [model card](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-4bpw).

| | Value |
|---|---|
| KLD vs BF16 | 0.0213 |
| Top-1 agreement | 97.35% |

**b12x vs coop MoE**, teacher-forced, 20K tokens, 16-bit KV, MoE at 128 rows:

| MoE kernel | KLD | Top-1 |
|---|---:|---:|
| b12x | 0.01489 | 97.08% |
| coop | 0.01480 | 96.92% |

The two are equivalent. Kernel vs fp32 truth (one layer, T = 1..128): b12x rel err ~8.6e-4, coop ~9.6e-4.

**Server end-to-end, current default** (b12x, 4-bit rotated KV, 16 x 380K). Fixed realistic prompt set: 16 authored chat / code / agentic+tool prompts, up to 512 tokens per reply, at least 2K generated tokens per row.

| concurrency | temp 0: e2e tok/s | temp 0: decode tok/s | temp 0: acceptance | temp 0.7: e2e tok/s | temp 0.7: decode tok/s | temp 0.7: acceptance |
|---|---:|---:|---:|---:|---:|---:|
| c1 | 141 | 149 | 2.94 | 130 | 137 | 2.71 |
| c4 | 222 | 244 | 2.95 | 211 | 230 | 2.76 |
| c8 | 275 | 308 | 2.98 | 251 | 278 | 2.78 |
| c16 | 322 | 369 | 3.01 | 288 | 324 | 2.63 |

- **e2e** includes prefill, scheduling and HTTP.
- **decode** is decode time only, from the server's `/health` counters.
- With replies this short, the c16 run averages only ~8 live rows per step. Batches stay fuller at sustained load.

**Engine bench**, coop -> b12x verify MoE (256-token prompts, 128 generated tokens, distinct prompts):

| concurrency | coop steps/s | b12x steps/s | b12x tok/s | b12x acceptance |
|---|---:|---:|---:|---:|
| c1 | 50.3 | 54.2 | 99 | 1.83 |
| c4 | 21.9 | 25.7 | 249 | 2.42 |
| c8 | 14.4 | 17.4 | 569 | 4.10 |
| c16 | 7.43 | 11.86 | 778 | 4.10 |

- The fair comparison is steps/s. The greedy trajectories differ at near-ties, so acceptance moves between runs.
- Step time c1 / c16: 18.4 / 84.3 ms with b12x, vs 19.9 / 134.6 ms with coop.
- This bench ran just before the 4-bit KV format change. The server numbers above are on the current default.

**KV cache: 16 x 380K.** The server default is:
- 4-bit global-layer KV, Hadamard-rotated, 16-level grid, 4 scale groups;
- 8-bit sliding-window KV;
- int8 draft KV.

That costs 2824 B/token/GPU. The pool holds **6,358,400 tokens**: 16 x 380K = 6.08M fits, with 4.6% to spare. Pages are shared, so any mix of lengths that totals 6.36M fits. The 40 sliding-window layers keep a 1536-token ring and cost no per-length memory.

KLD vs the BF16 teacher by KV format. These are eval subsets (short = 40K tokens, long = 3 x 32K long-context rows), so compare them only with each other:

| global / SWA KV | short | long | long, positions >= 16K | Top-1 short / long |
|---|---:|---:|---:|---:|
| 16 / 16 | 0.01231 | 0.01978 | 0.01494 | 97.33 / 97.81 |
| 8 / 8 | 0.01349 | 0.01952 | 0.01609 | 97.32 / 97.86 |
| 4 plain / 8 (previous default) | 0.01841 | 0.02608 | 0.02111 | 96.35 / 97.24 |
| **4 rotated, q16, 4 groups / 8 (default)** | **0.01698** | **0.02277** | **0.01889** | **96.96 / 97.52** |

- The rotated format recovers about half of the 4-bit KV loss on long context, and about a quarter on short.
- `--bits-g 8` is closest to 16-bit, at 4744 B/token/GPU. That means fewer than 16 x 380K sessions.
- Fixed in this snapshot: the sliding-window ring was too short for 512-row prefill chunks, so early rows of a chunk could read overwritten entries. All numbers above are from after the fix.

**Long context, current default:**

| Test | Result |
|---|---|
| Needle in a haystack, 375,802-token prompt (needle at 20% depth) | Found; 347 s |
| 16 concurrent sessions of ~24K tokens each (385K prompt tokens) | 16/16 needles correct |

An earlier soak on the coop MoE build (8 rounds of 24 mixed requests at c16) went 192/192 OK, with 0 errors. It has not been rerun on this build.

**Next:**
- a draft-length sweep;
- per-round verify depth;
- all-reduce choice at decode message sizes;
- the server scheduler gap between engine and server numbers.

Expect these numbers to move.

**Verification:**
- Verified: image build, and the in-image import check: the exllamav3 extension loads, and the b12x kernels import from the pinned FlashInfer tree.
- **The containerized GPU run is not verified.** The GPUs were busy with other jobs when this image was built, so the GPU smoke test was skipped. The server, engine and KV numbers above come from the same code and the same pinned versions outside Docker.
- Refresh the engine code with `scripts/update_snapshot.sh <fork checkout>`. It also syncs `server/i2_server.py`. The current snapshot's fork commit is in `overlay/FORK_COMMIT`. Run `scripts/audit.sh` before pushing.

## Credits
- [Thinking Machines](https://thinkingmachines.ai): Inkling-2-Small (Apache-2.0).
- [turboderp](https://github.com/turboderp-org): exllamav3 (MIT).
- [Local Inference Lab](https://github.com/local-inference-lab):
  - the [b12x](https://github.com/local-inference-lab/b12x) fused Trellis MoE kernels, now in [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/experimental/b12x) (Apache-2.0). This image runs them for the speculative verify step;
  - their EXL3 serving recipes. The 4.0 bpw quant's mcg codebook and uniform bit layout follow those recipes.

See `NOTICE.md`. This repo: MIT.
