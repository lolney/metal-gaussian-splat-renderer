#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/script/package_app.sh"

ARGS=()
for arg in "$@"; do
  if [[ -e "$arg" ]]; then
    ARGS+=("$(cd "$(dirname "$arg")" && pwd)/$(basename "$arg")")
  else
    ARGS+=("$arg")
  fi
done

if [[ ${#ARGS[@]} -gt 0 ]]; then
  open -n "$ROOT/dist/SplatViewer.app" --args "${ARGS[@]}"
else
  open -n "$ROOT/dist/SplatViewer.app"
fi
