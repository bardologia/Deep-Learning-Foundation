#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

find . -path ./.git -prune -o -type d -name "__pycache__" -exec rm -rf {} +
rm -rf .pytest_cache

echo "Cleaned __pycache__ directories and .pytest_cache."
