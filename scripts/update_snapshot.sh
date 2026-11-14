#!/usr/bin/env bash
# Regenerate overlay/ and patches/ from a local fork checkout (branch with the Inkling-2 work, based on upstream v1.6.6).
# usage: scripts/update_snapshot.sh /path/to/fork [branch=inkling2] [base=v1.6.6]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="${1:?path to fork checkout}"; BR="${2:-inkling2}"; BASE="${3:-v1.6.6}"
NEW="exllamav3/architecture/inkling2.py exllamav3/modules/inkling2.py exllamav3/modules/inkling2_engine.py exllamav3/modules/inkling2_ops.py"
# any additional new inkling2*/spec runtime modules
EXTRA=$(git -C "$FORK" diff --name-only --diff-filter=A "$BASE..$BR" -- 'exllamav3/*.py' | grep -E 'inkling2' || true)
NEW=$(printf '%s\n' $NEW $EXTRA | sort -u)
rm -rf "$HERE/overlay/exllamav3"; mkdir -p "$HERE/overlay"
for f in $NEW; do mkdir -p "$HERE/overlay/$(dirname "$f")"; git -C "$FORK" show "$BR:$f" > "$HERE/overlay/$f"; done
# public paths: pinned FlashInfer tree lives at /opt/flashinfer in the image
sed -i -E 's#os\.environ\.get\("I2_B12X_PATH", "[^"]*"\)#os.environ.get("I2_B12X_PATH", "/opt/flashinfer")#' "$HERE/overlay/exllamav3/modules/inkling2_b12x.py" 2>/dev/null || true
# canonical server (fork serving/i2_server.py) with public defaults: no sys.path hack, MODEL_DIR, port 8000
git -C "$FORK" show "$BR:serving/i2_server.py" \
  | sed -E -e '/^sys\.path\.insert\(0, "[^"]*"\)$/d' \
           -e 's#serving/i2_server\.py --port 8123#server/i2_server.py --port 8000#' \
           -e 's#os\.environ\.get\("I2_MODEL", "[^"]*"\)#os.environ.get("MODEL_DIR", "/models/Inkling-2-Small-EXL3-4bpw")#' \
           -e 's#(add_argument\("--port", type=int, default=)8123#\18000#' \
           -e 's#needs \.venv-eng \+ I2_B12X_PATH#needs the b12x stack: cutlass-dsl + FlashInfer tree at I2_B12X_PATH#' \
  > "$HERE/server/i2_server.py"
MODIFIED=$(git -C "$FORK" diff --name-only --diff-filter=M "$BASE..$BR" -- exllamav3)
git -C "$FORK" diff "$BASE..$BR" -- $MODIFIED > "$HERE/patches/upstream-modifications.patch"
git -C "$FORK" rev-parse --short "$BR" > "$HERE/overlay/FORK_COMMIT"
echo "snapshot of $BR@$(cat "$HERE/overlay/FORK_COMMIT"); NOW run the privacy audit: scripts/audit.sh"
