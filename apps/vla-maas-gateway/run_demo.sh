#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# One-command CPU demo: two mock VLA backends behind the MaaS gateway, both
# driven by mock-LIBERO clients at 50 Hz. Open http://localhost:9000/dashboard
# and enter the API key to watch latency and success rate live.
#
#   bash apps/vla-maas-gateway/run_demo.sh [episodes-per-task]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIENTS="$ROOT/examples/online_serving/vla_maas"
GATEWAY="$ROOT/apps/vla-maas-gateway"
EPISODES="${1:-10}"
KEY="${VLA_MAAS_API_KEY:-dev-key-change-me}"
LOGS="$(mktemp -d)"
PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

# Many minimal images (including the SSP CPU images) ship only `python3` and no
# `python` shim, so resolve the interpreter once. Override with PYTHON=...
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found; set PYTHON to a Python 3 interpreter" >&2
  exit 1
fi

"$PYTHON" "$CLIENTS/mock_policy_server.py" --port 8001 --latency-ms 60 >"$LOGS/pi05.log" 2>&1 & PIDS+=($!)
"$PYTHON" "$CLIENTS/mock_policy_server.py" --port 8002 --latency-ms 90 >"$LOGS/internvla_a1.log" 2>&1 & PIDS+=($!)
"$PYTHON" "$GATEWAY/gateway.py" --config "$GATEWAY/gateway.example.yaml" >"$LOGS/gateway.log" 2>&1 & PIDS+=($!)
sleep 2
echo "dashboard: http://localhost:9000/dashboard  (API key: $KEY)  logs: $LOGS"

for model in pi05 internvla_a1; do
  "$PYTHON" "$CLIENTS/libero_eval.py" --mock --port 9000 --path "/v1/robot/$model/openpi" \
    --api-key "$KEY" --report-episodes --episodes-per-task "$EPISODES" \
    --replan-steps 20 --async-prefetch 10 --control-hz 50 \
    --output "$LOGS/$model.json" >"$LOGS/$model.eval.log" 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]:3}"

for model in pi05 internvla_a1; do
  "$PYTHON" -c "import json,sys; s=json.load(open(sys.argv[1])); print(sys.argv[2], 'success', s['success_rate'], 'p50', round(s['inference_p50_ms'],1), 'underruns', s['underruns'])" \
    "$LOGS/$model.json" "$model"
done
if command -v curl >/dev/null 2>&1; then
  echo "metrics: $(curl -s -H "Authorization: Api-Key $KEY" localhost:9000/metrics)"
else
  echo "metrics: curl not found; GET http://localhost:9000/metrics with Authorization: Api-Key $KEY"
fi
