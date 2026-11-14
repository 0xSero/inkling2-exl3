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
MODIFIED=$(git -C "$FORK" diff --name-only --diff-filter=M "$BASE..$BR" -- exllamav3)
git -C "$FORK" diff "$BASE..$BR" -- $MODIFIED > "$HERE/patches/upstream-modifications.patch"
git -C "$FORK" rev-parse --short "$BR" > "$HERE/overlay/FORK_COMMIT"
echo "snapshot of $BR@$(cat "$HERE/overlay/FORK_COMMIT"); NOW run the privacy audit: scripts/audit.sh"
