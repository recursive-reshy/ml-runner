#!/usr/bin/env bash
# ml-runner gate (TRD §2.3). Runs six steps, stops at the first failure.
#   scripts/gate.sh              run all steps
#   scripts/gate.sh --only 1 4   run only the listed steps
#
# uv flags for steps 5-6 verified locally with uv 0.12.13 (task 1.1):
#   5: --isolated builds a throwaway env on Python 3.11 with the project installed
#   6: --no-project skips the package, so only agent.py (loaded by path) is tested on 3.10
set -uo pipefail
cd "$(dirname "$0")/.."

names=(
  ""
  "ruff check"
  "ruff format --check"
  "pyright"
  "pytest (dev Python)"
  "pytest (Python 3.11, CLI minimum)"
  "pytest tests/vm (Python 3.10, agent minimum)"
)
commands=(
  ""
  "uv run ruff check ."
  "uv run ruff format --check ."
  "uv run pyright"
  "uv run pytest -q"
  "uv run --isolated --python 3.11 --with-editable . --with pytest pytest -q tests"
  "uv run --isolated --python 3.10 --no-project --with pytest pytest -q tests/vm"
)

steps=()
if [[ "${1:-}" == "--only" ]]; then
  shift
  steps=("$@")
else
  steps=(1 2 3 4 5 6)
fi

for n in "${steps[@]}"; do
  if [[ ! "$n" =~ ^[1-6]$ ]]; then
    echo "unknown step: $n (expected 1-6)" >&2
    exit 2
  fi
  log="$(mktemp)"
  if ${commands[$n]} >"$log" 2>&1; then
    echo "✓ $n. ${names[$n]}"
    rm -f "$log"
  else
    echo "✗ $n. ${names[$n]}"
    cat "$log"
    rm -f "$log"
    exit 1
  fi
done
echo "gate green"
