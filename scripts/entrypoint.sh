#!/usr/bin/env bash
set -euo pipefail
export MODEL_DIR="${MODEL_DIR:-/models/Inkling-2-Small-EXL3-4bpw}"
export HF_REPO="${HF_REPO:-0xSero/Inkling-2-Small-EXL3-4bpw}"
export HF_HUB_ENABLE_HF_TRANSFER=1 NCCL_CUMEM_ENABLE=0 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
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
  serve) fetch; exec torchrun --standalone --nproc_per_node "${NGPU:-4}" /opt/inkling2/server/i2_server.py --model "$MODEL_DIR" --port "${PORT:-8000}" "$@" ;;
  download) fetch ;;
  shell|bash) exec bash ;;
  *) echo "usage: sweep | serve | download | shell   (env: MODEL_DIR PORT CONCURRENCY REPEATS STEPS PROMPT; serve args: --kv-gb --bits-g --bits-l --slots --max-model-len)"; exit 1 ;;
esac
