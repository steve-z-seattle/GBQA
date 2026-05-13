#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/logs/agent/gbqa"
VERIFIER_DIR="/logs/verifier"
GROUND_TRUTH="/tests/bugs/dark-castle.json"

mkdir -p "${VERIFIER_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-}:/sandbox"

DEBUG_FLAG=""
if [ "${GBQA_DEBUG:-}" = "1" ]; then
  DEBUG_FLAG="--debug"
fi

if [ -f "${AGENT_DIR}/bugs.json" ]; then
  "${PYTHON_BIN}" /tests/gbqa_verifier.py \
    --bugs "${AGENT_DIR}/bugs.json" \
    --ground-truth "${GROUND_TRUTH}" \
    --out-dir "${VERIFIER_DIR}" \
    ${DEBUG_FLAG}
else
  "${PYTHON_BIN}" /tests/gbqa_verifier.py \
    --bugs /tests/empty_bugs.json \
    --ground-truth "${GROUND_TRUTH}" \
    --out-dir "${VERIFIER_DIR}" \
    ${DEBUG_FLAG}
fi

# Keep the container alive briefly so host-side observers can see the
# verifier result (FOUND) before Harbor tears down the environment.
sleep 5
