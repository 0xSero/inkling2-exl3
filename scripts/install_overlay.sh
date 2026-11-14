#!/usr/bin/env bash
# Apply the Inkling-2 overlay to an exllamav3 v1.6.6 checkout: install_overlay.sh /path/to/exllamav3
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; T="${1:?path to exllamav3 checkout}"
git -C "$T" apply "$HERE/patches/upstream-modifications.patch"
cp -r "$HERE/overlay/exllamav3/." "$T/exllamav3/"
echo "overlay applied to $T"
