#!/usr/bin/env bash
# Speed sweep, speculative decoding always on. Env: MODEL_DIR, RESULTS_DIR, CONCURRENCY, REPEATS, STEPS, PROMPT, NGPU
set -euo pipefail
export MODEL_DIR="${MODEL_DIR:-/models/Inkling-2-Small-EXL3-4bpw}"
export RESULTS_DIR="${RESULTS_DIR:-/models/results}"
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_CUMEM_ENABLE=0 I2_MOE_GROUP="${I2_MOE_GROUP:-64}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec torchrun --standalone --nproc_per_node "${NGPU:-4}" "$HERE/bench/i2_sweep.py" "$@"
