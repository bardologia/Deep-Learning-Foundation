#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_VERSION="3.11"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
TORCH_PIN="$(grep -E '^torch==' "$HERE/requirements.txt")"

ENV_NAME=""
TORCH_INDEX="auto"
ASSUME_YES=0
SKIP_INSTALL=0
WIZARD_ARGS=()

usage() {
  cat <<USAGE
Usage: ./install.sh [options] [wizard options]

Sets up ML-Template on this machine: finds or installs Miniconda, creates the
project environment, installs the pinned dependencies with the torch build that
matches the NVIDIA driver, then runs the guided setup that writes the machine
profile and starts the console. Safe to run again at any time.

Options:
  --env NAME            conda environment name (default: the profile's name, else Dune)
  --torch-index INDEX   cu130, cu128, cpu or auto (default: auto, from the driver)
  --skip-install        do not touch conda or pip, only run the guided setup
  --yes                 unattended: accept every default, install Miniconda if missing
  -h, --help            show this help

Anything else is passed to scripts/setup_machine.py (for example --answers FILE).
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --env)          ENV_NAME="$2"; shift 2 ;;
    --env=*)        ENV_NAME="${1#*=}"; shift ;;
    --torch-index)  TORCH_INDEX="$2"; shift 2 ;;
    --torch-index=*) TORCH_INDEX="${1#*=}"; shift ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --yes)          ASSUME_YES=1; WIZARD_ARGS+=("--yes"); shift ;;
    -h|--help)      usage; exit 0 ;;
    *)              WIZARD_ARGS+=("$1"); shift ;;
  esac
done

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
fail() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

confirm() {
  local question="$1" default="$2" answer
  if [ "$ASSUME_YES" = 1 ]; then return 0; fi
  read -r -p "    $question [$default]: " answer
  answer="${answer:-$default}"
  case "$answer" in y|Y|yes|YES|Yes) return 0 ;; *) return 1 ;; esac
}

say "ML-Template installer"
note "repository: $HERE"

[ "$(uname -s)" = "Linux" ] || fail "this installer supports Linux only (found $(uname -s))"
command -v git >/dev/null 2>&1 || fail "git is not installed; install it with your package manager and run again"

if [ -n "${MLTEMPLATE_MACHINE_PROFILE:-}" ]; then
  ENV_FILE="$(dirname "$MLTEMPLATE_MACHINE_PROFILE")/machine.env"
else
  ENV_FILE="$HOME/.config/ml-template/machine.env"
fi

if [ -f "$ENV_FILE" ]; then
  source "$ENV_FILE"
  note "existing profile found: ${MLTEMPLATE_MACHINE_PROFILE}"
fi

ENV_NAME="${ENV_NAME:-${MLTEMPLATE_ENV_NAME:-Dune}}"

find_conda_root() {
  if [ -n "${MLTEMPLATE_CONDA_ROOT:-}" ] && [ -x "$MLTEMPLATE_CONDA_ROOT/bin/conda" ]; then echo "$MLTEMPLATE_CONDA_ROOT"; return; fi
  if [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then dirname "$(dirname "$CONDA_EXE")"; return; fi
  if command -v conda >/dev/null 2>&1; then dirname "$(dirname "$(readlink -f "$(command -v conda)")")"; return; fi
  local name
  for name in miniconda3 anaconda3 miniforge3 mambaforge; do
    if [ -x "$HOME/$name/bin/conda" ]; then echo "$HOME/$name"; return; fi
  done
  echo ""
}

install_miniconda() {
  local target="$HOME/miniconda3" archive
  archive="$(mktemp -t miniconda-XXXXXX.sh)"
  say "Installing Miniconda into $target"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$MINICONDA_URL" -o "$archive"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$archive" "$MINICONDA_URL"
  else
    fail "neither curl nor wget is available to download Miniconda"
  fi
  bash "$archive" -b -p "$target"
  rm -f "$archive"
  echo "$target"
}

if [ "$SKIP_INSTALL" = 0 ]; then
  say "Step A: conda"
  CONDA_ROOT="$(find_conda_root)"
  if [ -z "$CONDA_ROOT" ]; then
    note "no conda installation found."
    if confirm "Install Miniconda into $HOME/miniconda3 now? (y/n)" "y"; then
      CONDA_ROOT="$(install_miniconda)"
    else
      fail "conda is required; install Miniconda from https://docs.anaconda.com/miniconda/ and run again"
    fi
  fi
  CONDA="$CONDA_ROOT/bin/conda"
  [ -x "$CONDA" ] || CONDA="$CONDA_ROOT/condabin/conda"
  [ -x "$CONDA" ] || fail "conda executable not found under $CONDA_ROOT"
  note "conda: $CONDA"

  say "Step B: environment '$ENV_NAME'"
  ENV_PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
  if [ ! -x "$ENV_PY" ]; then
    note "creating the environment with Python $PYTHON_VERSION (this takes a minute)"
    "$CONDA" create -y -q -n "$ENV_NAME" "python=$PYTHON_VERSION"
  else
    note "environment exists: $ENV_PY"
  fi
  [ -x "$ENV_PY" ] || fail "the environment interpreter did not appear at $ENV_PY"

  say "Step C: torch build"
  if [ "$TORCH_INDEX" = "auto" ]; then
    TORCH_INDEX="$("$ENV_PY" "$HERE/tools/runtime/machine_probe.py" --torch-index)"
    note "NVIDIA driver check chose: $TORCH_INDEX"
  else
    note "requested: $TORCH_INDEX"
  fi
  case "$TORCH_INDEX" in
    cu130|cu128|cpu) ;;
    *) fail "unknown torch index '$TORCH_INDEX'; use cu130, cu128, cpu or auto" ;;
  esac

  say "Step D: dependencies"
  note "installing $TORCH_PIN from https://download.pytorch.org/whl/$TORCH_INDEX"
  "$ENV_PY" -m pip install --quiet --timeout 120 --retries 10 "$TORCH_PIN" --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
  note "installing the pinned requirements"
  "$ENV_PY" -m pip install --quiet --timeout 120 --retries 10 -r "$HERE/requirements.txt"
  EXTRAS_FILE="$(mktemp -t mltemplate-extras-XXXXXX.txt)"
  "$ENV_PY" "$HERE/scripts/extras_requirements.py" "$HERE/pyproject.toml" "$EXTRAS_FILE"
  note "installing the optional extras: $(tr '\n' ' ' < "$EXTRAS_FILE")"
  "$ENV_PY" -m pip install --quiet --timeout 120 --retries 10 -r "$EXTRAS_FILE"
  rm -f "$EXTRAS_FILE"
  note "checking the result"
  "$ENV_PY" -c "import torch; print(f'    torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
else
  say "Skipping conda and pip (--skip-install)"
  ENV_PY="${MLTEMPLATE_INTERPRETER:-}"
  if [ -z "$ENV_PY" ] || [ ! -x "$ENV_PY" ]; then
    CONDA_ROOT="$(find_conda_root)"
    ENV_PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
  fi
  [ -x "$ENV_PY" ] || fail "no interpreter for environment '$ENV_NAME'; run without --skip-install first"
  note "interpreter: $ENV_PY"
fi

say "Step E: guided setup"
cd "$HERE"
exec "$ENV_PY" "$HERE/scripts/setup_machine.py" --env "$ENV_NAME" --interpreter "$ENV_PY" ${WIZARD_ARGS[@]+"${WIZARD_ARGS[@]}"}
