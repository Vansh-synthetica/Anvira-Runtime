#!/bin/sh
# Install Anvira Runtime (macOS / Linux). Installs the runtime only - NO models are downloaded.
#   ./scripts/install.sh [path/to/anvira-runtime-1.0.0.zip] [--yes]
set -eu
here="$(cd "$(dirname "$0")/.." && pwd)"
source="${1:-$here}"
PY="$(command -v python3 || command -v python || true)"
[ -n "$PY" ] || { echo "Python 3.10+ is required." >&2; exit 3; }
if [ "${2:-}" != "--yes" ] && [ "${1:-}" != "--yes" ]; then
  printf "Install Anvira Runtime (a shared local AI runtime for Anvira apps)? [y/N] "
  read -r ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "Cancelled. Nothing was changed."; exit 0 ;; esac
fi
[ "$source" = "--yes" ] && source="$here"
PYTHONPATH="$here/sdk/python" "$PY" -m anvira_client install --source "$source"
echo "Done. Next:  anvira runtime start   |   anvira doctor   |   anvira ui"
