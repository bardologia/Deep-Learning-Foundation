#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${1:-8772}"

for candidate in \
  "$HOME/miniconda3/envs/Dune/bin/python" \
  "$HOME/miniconda3/bin/python" \
  "$(command -v python3 || true)"; do
  if [ -x "$candidate" ]; then
    PY="$candidate"
    break
  fi
done

echo "Starting the ML Template control console with: $PY"
exec "$PY" "$HERE/serve.py" --port "$PORT"
