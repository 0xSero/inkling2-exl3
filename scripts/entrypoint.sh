#!/usr/bin/env bash
set -euo pipefail
export MODEL_DIR="${MODEL_DIR:-/models/Inkling-2-Small-EXL3-4bpw}"
export HF_REPO="${HF_REPO:-0xSero/Inkling-2-Small-EXL3-4bpw}"
export NCCL_CUMEM_ENABLE=0 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
cmd="${1:-sweep}"; shift || true
fetch() {
  if [ ! -f "$MODEL_DIR/config.json" ] || [ -n "${FORCE_DOWNLOAD:-}" ] || ! ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
    echo "[inkling2] downloading $HF_REPO -> $MODEL_DIR (~313 GB, resumable)"
    mkdir -p "$MODEL_DIR"
    hf download "$HF_REPO" --local-dir "$MODEL_DIR"
  fi
}
case "$cmd" in
  sweep) fetch; exec /opt/inkling2/scripts/sweep.sh "$@" ;;
  serve) fetch; exec torchrun --standalone --nproc_per_node "${NGPU:-4}" /opt/inkling2/server/i2_server.py --model "$MODEL_DIR" --port "${PORT:-8000}" --b12x "${I2_B12X:-1}" "$@" ;;
  download) fetch ;;
  shell|bash) exec bash ;;
  check) exec python /opt/inkling2/scripts/check_imports.py ;;
  *) echo "usage: sweep | serve | download | check | shell   (env: MODEL_DIR PORT CONCURRENCY REPEATS STEPS PROMPT SPEC_K I2_B12X I2_EP_LAYOUT I2_DRAFT_FP8; serve args: --max-model-len --slots --spec-k --burst --bits-g --bits-l --kv-rot --kv-ng --kv-q16 --kv-gb)"; exit 1 ;;
esac
