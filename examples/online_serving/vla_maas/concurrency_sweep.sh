#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Sweep robot-fleet size against three serving layouts, all behind the MaaS
# gateway and all on CPU mocks with the same per-forward cost:
#   serial      1 backend,  max_batch 1   (Pi05Pipeline today: max_num_seqs 1)
#   replicas    N backends, max_batch 1   (replica data parallel, one GPU each)
#   batched     1 backend,  dynamic batching (needs supports_request_batch)
#
#   bash concurrency_sweep.sh [out.jsonl]
# Env overrides: ROBOTS="1 2 4 8 16 32" DURATION=15 LATENCY_MS=60 PER_ITEM_MS=4 REPLICAS=4
#                PYTHON=python3
set -euo pipefail

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found; set PYTHON to a Python 3 interpreter" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY="$HERE/../../../apps/vla-maas-gateway/gateway.py"
OUT="${1:-$HERE/concurrency_sweep.jsonl}"
ROBOTS="${ROBOTS:-1 2 4 8 16 32}"
DURATION="${DURATION:-15}"
LATENCY_MS="${LATENCY_MS:-60}"
PER_ITEM_MS="${PER_ITEM_MS:-4}"
REPLICAS="${REPLICAS:-4}"
SCALE=0.005
KEY=sweep-key
WORK="$(mktemp -d)"
PIDS=()
cleanup() { kill "${PIDS[@]}" 2>/dev/null || true; PIDS=(); }
trap cleanup EXIT
: >"$OUT"

start_layout() {  # name, backend count, max batch
  local name=$1 count=$2 batch=$3 backends=""
  for i in $(seq 1 "$count"); do
    local port=$((8200 + i))
    "$PYTHON" "$HERE/mock_policy_server.py" --port "$port" --latency-ms "$LATENCY_MS" --jitter-ms 5 \
      --max-batch "$batch" --max-wait-ms 5 --per-item-ms "$PER_ITEM_MS" --action-scale "$SCALE" \
      >"$WORK/$name-$i.log" 2>&1 &
    PIDS+=($!)
    backends+="      - ws://127.0.0.1:$port/v1/realtime/robot/openpi"$'\n'
  done
  printf 'api_keys: [%s]\nmodels:\n  pi05:\n    backends:\n%s' "$KEY" "$backends" >"$WORK/$name.yaml"
  "$PYTHON" "$GATEWAY" --config "$WORK/$name.yaml" --port 9100 >"$WORK/$name-gateway.log" 2>&1 &
  PIDS+=($!)
  sleep 2
}

run_layout() {
  local name=$1
  for n in $ROBOTS; do
    "$PYTHON" "$HERE/fleet_sim.py" --port 9100 --path /v1/robot/pi05/openpi --api-key "$KEY" \
      --robots "$n" --duration-s "$DURATION" --action-scale "$SCALE" --label "$name" --output "$OUT"
  done
}

start_layout serial 1 1;               run_layout serial;               cleanup
start_layout "replicas" "$REPLICAS" 1; run_layout "replicas-$REPLICAS"; cleanup
start_layout batched 1 64;             run_layout batched;              cleanup
echo "results: $OUT (logs: $WORK)"
