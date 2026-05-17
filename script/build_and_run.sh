#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/script/package_app.sh"
open -n "$ROOT/dist/SplatViewer.app" --args "$@"
