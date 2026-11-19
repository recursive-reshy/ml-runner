#!/usr/bin/env bash
set -euo pipefail

MLR_HOME="${MLR_HOME:-/opt/mlr}"
MLR_LOG="${MLR_LOG:-/var/log/mlr-agent.log}"
METADATA_URL="${MLR_METADATA_URL:-http://metadata.google.internal/computeMetadata/v1}"

mkdir -p "$MLR_HOME"

if [[ ! -f "$MLR_HOME/agent.py" ]]; then
    curl -fsS -H "Metadata-Flavor: Google" "$METADATA_URL/instance/attributes/mlr-agent" -o "$MLR_HOME/agent.py"
fi

# The startup script runs again on every boot; a second boot means the first run was cut short.
if [[ -e "$MLR_HOME/started" ]]; then
    python3 "$MLR_HOME/agent.py" finalize --reason "VM restarted" 2>&1 | tee -a "$MLR_LOG"
    exit 0
fi

touch "$MLR_HOME/started"
python3 "$MLR_HOME/agent.py" start 2>&1 | tee -a "$MLR_LOG"
