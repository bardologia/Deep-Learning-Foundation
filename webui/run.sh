#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${MLTEMPLATE_MACHINE_PROFILE:-}" ]; then
  ENV_FILE="$(dirname "$MLTEMPLATE_MACHINE_PROFILE")/machine.env"
else
  ENV_FILE="$HOME/.config/ml-template/machine.env"
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "No machine profile found at $ENV_FILE." >&2
  echo "Run ./install.sh from the repository root once; it creates the environment and writes the profile." >&2
  exit 1
fi

source "$ENV_FILE"

PORT="${1:-$MLTEMPLATE_CONSOLE_PORT}"
HOST="${MLTEMPLATE_CONSOLE_HOST}"
PY="${MLTEMPLATE_INTERPRETER}"

if [ ! -x "$PY" ]; then
  echo "The interpreter recorded in the machine profile does not exist: $PY" >&2
  echo "Rerun ./install.sh to rebuild the environment and refresh the profile." >&2
  exit 1
fi

echo "Starting the ML Template control console with: $PY"
echo "Profile: $MLTEMPLATE_MACHINE_PROFILE"
exec "$PY" "$HERE/serve.py" --host "$HOST" --port "$PORT"
