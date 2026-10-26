#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Simulated multi-arm fleet through the MaaS gateway, with headless video.
#
#   bash examples/online_serving/vla_maas/sim_fleet_demo.sh [scenario.yaml] [duration-s]
#
# Default backends are two CPU mock policies (pi05, internvla_a1 routes of
# apps/vla-maas-gateway/gateway.example.yaml) that sweep the arms so the videos
# show motion. To drive real models, start `vllm serve` yourself and pass a
# gateway config that points at them:
#   GATEWAY_CONFIG=my_gateway.yaml bash sim_fleet_demo.sh scenarios/sim_fleet_libero_mixed.yaml 300
#
# Env: MUJOCO_GL=egl (GPU nodes; osmesa on CPU hosts)  RENDER_GPUS=0,1,...
#      OUT_DIR=<dir>  GW_PORT=9000  PYTHON=python3  VLA_MAAS_API_KEY=dev-key-change-me
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
SCENARIO="${1:-$HERE/scenarios/sim_fleet_robosuite.yaml}"
DURATION="${2:-60}"
PYTHON="${PYTHON:-python3}"
GW_PORT="${GW_PORT:-9000}"
KEY="${VLA_MAAS_API_KEY:-dev-key-change-me}"
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
RENDER_GPUS="${RENDER_GPUS:-0}"
PIDS=()
mkdir -p "$OUT_DIR"
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

if [ -z "${GATEWAY_CONFIG:-}" ]; then
  for spec in 8001:60 8002:90; do  # port:latency-ms for pi05, internvla_a1
    "$PYTHON" "$HERE/mock_policy_server.py" --port "${spec%%:*}" --latency-ms "${spec##*:}" \
      --max-batch 64 --max-wait-ms 5 --per-item-ms 4 --idle-motion 0.6 \
      >"$OUT_DIR/backend-${spec%%:*}.log" 2>&1 &
    PIDS+=($!)
  done
  GATEWAY_CONFIG="$ROOT/apps/vla-maas-gateway/gateway.example.yaml"
fi
"$PYTHON" "$ROOT/apps/vla-maas-gateway/gateway.py" --config "$GATEWAY_CONFIG" --port "$GW_PORT" \
  >"$OUT_DIR/gateway.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 50); do
  curl -sf "http://127.0.0.1:$GW_PORT/healthz" >/dev/null 2>&1 && break
  sleep 0.2
done
echo "scenario: $SCENARIO  duration: ${DURATION}s  MUJOCO_GL=$MUJOCO_GL  render GPUs: $RENDER_GPUS"
echo "dashboard: http://<host>:$GW_PORT/dashboard (key $KEY)  output: $OUT_DIR"

"$PYTHON" "$HERE/sim_fleet.py" --scenario "$SCENARIO" --port "$GW_PORT" --api-key "$KEY" \
  --duration-s "$DURATION" --render-gpus "$RENDER_GPUS" --video-dir "$OUT_DIR/videos" --video-grid \
  --output "$OUT_DIR/summary.jsonl" --episodes-out "$OUT_DIR/episodes.jsonl"
echo "videos: $OUT_DIR/videos  summary: $OUT_DIR/summary.jsonl"
