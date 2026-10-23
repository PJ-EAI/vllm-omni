#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# One-command CPU demo: a fleet of concurrent mock robots per model, each with
# its own session and 50 Hz control loop, all hitting two mock VLA backends
# through the MaaS gateway. Open http://localhost:9000/dashboard and enter the
# API key to watch latency, throughput and success rate live.
#
#   bash apps/vla-maas-gateway/run_demo.sh [robots-per-model] [duration-s]
# MODE=serial   backends run one request at a time (max_num_seqs: 1)
# MODE=batched  backends batch concurrent requests (default; max_num_seqs > 1)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIENTS="$ROOT/examples/online_serving/vla_maas"
GATEWAY="$ROOT/apps/vla-maas-gateway"
ROBOTS="${1:-8}"
DURATION="${2:-60}"
MODE="${MODE:-batched}"
KEY="${VLA_MAAS_API_KEY:-dev-key-change-me}"
SCALE=0.005
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

case "$MODE" in
  serial) BATCH=1 ;;
  batched) BATCH=64 ;;
  *) echo "MODE must be serial or batched" >&2; exit 1 ;;
esac

backend() {  # port, latency
  "$PYTHON" "$CLIENTS/mock_policy_server.py" --port "$1" --latency-ms "$2" --max-batch "$BATCH" \
    --max-wait-ms 5 --per-item-ms 4 --action-scale "$SCALE" >"$LOGS/backend-$1.log" 2>&1 &
  PIDS+=($!)
}
backend 8001 60  # pi05
backend 8002 90  # internvla_a1
"$PYTHON" "$GATEWAY/gateway.py" --config "$GATEWAY/gateway.example.yaml" >"$LOGS/gateway.log" 2>&1 & PIDS+=($!)
sleep 2
echo "mode: $MODE  robots per model: $ROBOTS  duration: ${DURATION}s"
echo "dashboard: http://localhost:9000/dashboard  (API key: $KEY)  logs: $LOGS"

FLEETS=()
for model in pi05 internvla_a1; do
  "$PYTHON" "$CLIENTS/fleet_sim.py" --port 9000 --path "/v1/robot/$model/openpi" --api-key "$KEY" \
    --robots "$ROBOTS" --procs 2 --duration-s "$DURATION" --action-scale "$SCALE" \
    --report-episodes --label "$model" --output "$LOGS/fleet.jsonl" >"$LOGS/$model.fleet.log" 2>&1 &
  FLEETS+=($!)
done
wait "${FLEETS[@]}"
cat "$LOGS/fleet.jsonl"
