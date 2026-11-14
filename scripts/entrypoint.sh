#!/usr/bin/env bash
set -euo pipefail
export MODEL_DIR="${MODEL_DIR:-/models/Inkling-2-Small-EXL3-3.48bpw}"
export HF_REPO="${HF_REPO:-0xSero/Inkling-2-Small-EXL3-3.48bpw}"
export HF_HUB_ENABLE_HF_TRANSFER=1
cmd="${1:-sweep}"; shift || true
fetch() {
  if [ ! -f "$MODEL_DIR/config.json" ] || [ -n "${FORCE_DOWNLOAD:-}" ] || ! ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
    echo "[inkling2] downloading $HF_REPO -> $MODEL_DIR (~270 GB, resumable)"
    mkdir -p "$MODEL_DIR"
    hf download "$HF_REPO" --local-dir "$MODEL_DIR"
  fi
}
case "$cmd" in
  sweep) fetch; exec /opt/inkling2/scripts/sweep.sh "$@" ;;
  serve)
    cat <<MSG
[inkling2] 'serve' (OpenAI-compatible server) is NOT AVAILABLE YET in this release.
Status: batched TP4/EP4 engine exists and is benchmarked via 'sweep'; the HTTP server is still to come.
Speculative decoding (SPEC=on by default) will apply to serve once implemented.
MSG
    exit 3 ;;
  download) fetch ;;
  shell|bash) exec bash ;;
  *) echo "usage: sweep | serve | download | shell   (env: SPEC=on|off MODEL_DIR CONCURRENCY REPEATS STEPS PROMPT)"; exit 1 ;;
esac
