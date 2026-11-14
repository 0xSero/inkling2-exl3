# inkling2-exl3

One-command serving and speed sweeps for [Inkling-2-Small EXL3 4.0 bpw](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-4bpw)
on **4x 96 GB Blackwell GPUs (RTX PRO 6000, SM120)**. It is an [exllamav3](https://github.com/turboderp-org/exllamav3) (v1.6.6) overlay that adds:
- the Inkling-2 architecture;
- a batched engine, with attention split across the 4 GPUs (TP4) and experts spread across them (EP4);
- a paged KV cache that knows about the sliding-window layers;
- CUDA-graphed **speculative decoding (DSpark/MTP), always on**.

## Run

```bash
# OpenAI-compatible server on :8000
docker run --rm --gpus all --ipc=host -p 8000:8000 -v $HOME/models:/models ghcr.io/0xsero/inkling2-exl3:latest serve

# speed sweep (c1/c4/c8/c16 decode tok/s, acceptance length, prefill tok/s)
docker run --rm --gpus all --ipc=host -v $HOME/models:/models ghcr.io/0xsero/inkling2-exl3:latest sweep
```

The first run downloads the model (~313 GB, resumable) into `$HOME/models/Inkling-2-Small-EXL3-4bpw`.
- `sweep` prints a markdown table (median of 3 repeats) and writes JSON to `/models/results/`.
- `serve` exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/health`. `/health` also reports the KV token capacity.

| Env (pass with `-e`) | Default | |
|---|---|---|
| `PORT` | `8000` | server port |
| `CONCURRENCY` | `1,4,8,16` | sweep batch sizes |
| `REPEATS` / `STEPS` / `PROMPT` | `3` / `128` / `256` | sweep settings |
| `GRAPH` | `1` | CUDA graphs |
| `I2_MOE_GROUP` | `64` | tokens per fused MoE kernel call in the speculative verify step |
| `MODEL_DIR` / `HF_REPO` / `RESULTS_DIR` | `/models/...` | |

Server flags go after `serve`:

| Flag | Default | |
|---|---|---|
| `--max-model-len` | `524288` | |
| `--slots` | `16` | concurrent sequences |
| `--bits-g` / `--bits-l` | `4` / `8` | KV bits for global / sliding-window layers; `--bits-g 8` for quality, at half the token capacity |
| `--kv-gb` | auto | KV budget |

Speculative decoding has no off switch. It runs at every batch size.

Without Docker: `scripts/install_overlay.sh <exllamav3 v1.6.6 checkout>`, then `pip install .` there with `TORCH_CUDA_ARCH_LIST=12.0`. After that, run `scripts/sweep.sh` or `torchrun --nproc_per_node 4 server/i2_server.py --port 8000`.

## Requirements
- 4x SM120 GPUs with 96 GB each and NCCL P2P;
- an NVIDIA driver supporting CUDA 12.8 (>= 570);
- ~313 GB free disk;
- Docker with the NVIDIA container toolkit.

## Status (work in progress)
Measured 2026-11-14 on 4x RTX PRO 6000 at the 300 W power cap. All 4 GPUs sit at the cap under load, so decode is power-bound.

The model was quantized against the BF16 reference; full results are on the [model card](https://huggingface.co/0xSero/Inkling-2-Small-EXL3-4bpw).

| | Value |
|---|---|
| KLD vs BF16 | 0.0213 |
| Top-1 agreement | 97.35% |

**Decode speed, speculative decoding, 7 drafted tokens per step:**

| concurrency | engine bench (tok/s) | acceptance length | server end-to-end (tok/s) |
|---|---:|---:|---:|
| c1 | 101 | 2.0 | 120 |
| c4 | 273 | 3.1 | 216 |
| c8 | 443 | 3.8 | 248 |
| c16 | 475 | 3.9 | 296 |

- The engine bench uses 256-token prompts and 128 generated tokens.
- The server column uses 512-token chat replies at temperature 0, and includes prefill, scheduling and HTTP.
- Acceptance depends on the text: it is 2.5–2.6 on natural chat at c1.

**KV cache:** with the default 4-bit global-layer KV, 8-bit sliding-window KV and 8-bit draft KV, the pool holds 6.9M tokens. That is about 13 concurrent sequences at the full 512K context. Pages are shared, so any mix of lengths that totals 6.9M fits. The 40 sliding-window layers keep a 1024-token ring and cost no per-length memory.

**Stability (server soak):**

| Test | Result |
|---|---|
| Soak: 8 rounds of 24 mixed requests at c16 (128–1024 tokens, temperature 0 / 0.7 / 1.0) | 192/192 OK, 0 errors, 0 empty outputs; server up throughout, memory flat |
| Needle in a haystack, 67K-token prompt | Found; 38 s (prefill 1.76K tok/s) |
| Needle in a haystack, 270K-token prompt | Found; 222 s (prefill 1.21K tok/s) |

**Next:** fused Trellis MoE kernels from [Local Inference Lab](https://github.com/local-inference-lab) for the speculative verify step, a draft-length sweep, and the server scheduler overhead (the gap between the engine and server columns). Expect these numbers to move.

- Verified: image build. The engine and server numbers above come from the same code outside Docker. **A full containerized GPU run is not yet verified.**
- Refresh the engine code with `scripts/update_snapshot.sh <fork checkout>`. The current snapshot's fork commit is in `overlay/FORK_COMMIT`. Run `scripts/audit.sh` before pushing.

## Credits
- [Thinking Machines](https://thinkingmachines.ai): Inkling-2-Small (Apache-2.0).
- [turboderp](https://github.com/turboderp-org): exllamav3 (MIT).
- [Local Inference Lab](https://github.com/local-inference-lab): the [b12x](https://github.com/local-inference-lab/b12x) fused Trellis MoE kernels (now in [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/experimental/b12x), Apache-2.0) and their EXL3 serving recipes. The 4.0 bpw quant's mcg codebook and uniform bit layout follow those recipes.

See `NOTICE.md`. This repo: MIT.
