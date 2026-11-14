# inkling2-exl3

One-command speed sweeps (and, soon, serving) for [Inkling-2-Small EXL3 3.48 bpw](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-3.48bpw)
on **4x 96 GB Blackwell GPUs (RTX PRO 6000, SM120)**, using an [exllamav3](https://github.com/turboderp-org/exllamav3)
(v1.6.6) overlay: Inkling-2 architecture, Triton kernels, and a batched TP4 attention / EP4 experts engine.

## Run

```bash
docker run --rm --gpus all --ipc=host -v $HOME/models:/models ghcr.io/0xsero/inkling2-exl3:latest sweep
```

The first run downloads the model (~270 GB) into `$HOME/models/Inkling-2-Small-EXL3-3.48bpw`, then runs c1/c4/c8/c16
decode tok/s and prefill tok/s (median of 3 repeats), prints a markdown table (decode tok/s, acceptance length, steps/s, prefill tok/s), and writes JSON to `/models/results/`.

| Env (pass with `-e`) | Default | |
|---|---|---|
| `SPEC` | `on` | speculative decoding (MTP); `SPEC=off` for A/B |
| `CONCURRENCY` | `1,4,8,16` | |
| `REPEATS` / `STEPS` / `PROMPT` | `3` / `128` / `256` | |
| `GRAPH` | `1` | CUDA graphs |
| `MODEL_DIR` / `RESULTS_DIR` | `/models/...` | |

Without Docker: `scripts/install_overlay.sh <exllamav3 v1.6.6 checkout>`, `pip install .` there with
`TORCH_CUDA_ARCH_LIST=12.0`, then `scripts/sweep.sh`. A `docker-compose.yml` and `Makefile` are provided too.

## Requirements
4x SM120 GPUs with 96 GB each (NCCL P2P), NVIDIA driver supporting CUDA 12.8 (>= 570), ~270 GB free disk, Docker with the NVIDIA container toolkit.

## Status (work in progress)
Measured 2026-11-14 on 4x RTX PRO 6000 (300 W), prompt 256, 4-bit global / 8-bit SWA KV cache. **Work in progress, expect these to move.**

| concurrency | plain decode, CUDA graphs (tok/s) | speculative decoding (tok/s) | acceptance length |
|---|---|---|---|
| c1 | 108 | 114 | 2.10 |
| c8 | 366 | 272 | 3.58 |
| c16 | n/a | 410 | 3.50 |

Earlier eager, spec-off baseline: c1 63.6, c16 467 tok/s. Spec decoding helps single-stream latency and is currently
slower than plain graphs at c8, so `SPEC=off` is worth A/B-ing for batch throughput. Numbers come from the engine's own
bench; this repo's `sweep` uses a different prompt set (slices of a public text file), so expect some drift.

- **Speculative decoding** (DSpark draft from `mtp.safetensors`, 7 drafted tokens per step) is ON by default; `SPEC=off` disables it. CUDA graphs are ON by default (`GRAPH=0` to disable).
- **`serve`** (OpenAI-compatible server): not available yet; the command says so and exits.
- Verified: image build and CPU import of the Inkling2 classes. **Not yet verified: the containerized GPU sweep.**
- Refresh engine code with `scripts/update_snapshot.sh <fork checkout>` (currently snapshot of fork commit in `overlay/FORK_COMMIT`); run `scripts/audit.sh` before pushing.

## Credits
[Thinking Machines](https://thinkingmachines.ai) Inkling-2-Small (Apache-2.0); [turboderp](https://github.com/turboderp-org) exllamav3 (MIT). See `NOTICE.md`. This repo: MIT.
