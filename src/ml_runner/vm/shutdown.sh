#!/usr/bin/env bash
set -euo pipefail

MLR_HOME="${MLR_HOME:-/opt/mlr}"
MLR_LOG="${MLR_LOG:-/var/log/mlr-agent.log}"
# Spot VMs get about 30 seconds after the pre-emption notice.
LIMIT_SECONDS=25

if [[ ! -f "$MLR_HOME/agent.py" ]]; then
    exit 0
fi

# A failing finalize must not hold up the shutdown.
timeout "$LIMIT_SECONDS" python3 "$MLR_HOME/agent.py" finalize 2>&1 | tee -a "$MLR_LOG" || true
