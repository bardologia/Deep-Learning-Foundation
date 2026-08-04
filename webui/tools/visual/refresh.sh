#!/usr/bin/env bash
# Restart the webui server and capture screenshots of the UI routes.
# Usage: refresh.sh [route ...]        (no routes = default page set)
# Env:   PORT (default 8765), OUT (default /tmp/tomosar-ui)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBUI="$(cd "$HERE/../.." && pwd)"

PORT="${PORT:-8765}"
OUT="${OUT:-/tmp/tomosar-ui}"
LOG="/tmp/tomosar-webui-${PORT}.log"

pkill -f "serve.py --port $PORT" 2>/dev/null || true
for _ in $(seq 1 20); do
  pgrep -f "serve.py --port $PORT" >/dev/null || break
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
