#!/usr/bin/env bash
# Speed sweep. Env: MODEL_DIR, RESULTS_DIR, SPEC=on|off (default on), CONCURRENCY, REPEATS, STEPS, PROMPT, NGPU
set -euo pipefail
export MODEL_DIR="${MODEL_DIR:-/models/Inkling-2-Small-EXL3-3.48bpw}"
export RESULTS_DIR="${RESULTS_DIR:-/models/results}"
export SPEC="${SPEC:-on}"
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec torchrun --standalone --nproc_per_node "${NGPU:-4}" "$HERE/bench/i2_sweep.py" "$@"
