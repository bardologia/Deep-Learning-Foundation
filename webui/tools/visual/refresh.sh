#!/usr/bin/env bash
# Restart the webui server and capture screenshots of the UI routes.
# Usage: refresh.sh [route ...]        (no routes = default page set)
# Env:   PORT (default 8772), OUT (default /tmp/ml-template-ui)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBUI="$(cd "$HERE/../.." && pwd)"

PORT="${PORT:-8772}"
OUT="${OUT:-/tmp/ml-template-ui}"
LOG="/tmp/ml-template-webui-${PORT}.log"

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "PORT must be a number, got '$PORT'" >&2
  exit 1
fi
if [ -z "$OUT" ] || [ "$OUT" = "/" ]; then
  echo "OUT must be a non-root directory, got '$OUT'" >&2
  exit 1
fi

PATTERN="serve\.py --port ${PORT}($| )"

pkill -f "$PATTERN" 2>/dev/null || true
for _ in $(seq 1 20); do
  pgrep -f "$PATTERN" >/dev/null || break
  sleep 0.1
done

nohup "$WEBUI/run.sh" "$PORT" > "$LOG" 2>&1 &

for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/" -o /dev/null 2>/dev/null; then
    break
  fi
  if [ "$i" -eq 60 ]; then
    echo "server did not come up on port $PORT; log tail:" >&2
    tail -20 "$LOG" >&2
    exit 1
  fi
  sleep 0.25
done

rm -f "$OUT"/*.png 2>/dev/null || true
node "$HERE/snap_ui.js" --port "$PORT" --out "$OUT" "$@"
