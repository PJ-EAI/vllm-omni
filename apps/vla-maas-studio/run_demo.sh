#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# One-command CPU demo: a mock policy backend, the MaaS gateway, and the studio.
# Open the printed URL and start a run; no GPU or simulator install is needed.
#
#   bash apps/vla-maas-studio/run_demo.sh
#   PHYSICS=mujoco bash apps/vla-maas-studio/run_demo.sh   # rendered arms
# Env: PYTHON=python3 PHYSICS=mock|mujoco MENAGERIE=~/.cache/mujoco_menagerie
#      BACKEND_PORT=18001 GW_PORT=19000 STUDIO_PORT=19100
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIENTS="$ROOT/examples/online_serving/vla_maas"
GATEWAY="$ROOT/apps/vla-maas-gateway"
STUDIO="$ROOT/apps/vla-maas-studio"
PYTHON="${PYTHON:-python3}"
KEY="${VLA_MAAS_API_KEY:-studio-key}"
PHYSICS="${PHYSICS:-mock}"
MENAGERIE="${MENAGERIE:-$HOME/.cache/mujoco_menagerie}"
BACKEND_PORT="${BACKEND_PORT:-18001}"
GW_PORT="${GW_PORT:-19000}"
STUDIO_PORT="${STUDIO_PORT:-19100}"
LOGS="$(mktemp -d)"
PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

if [ "$PHYSICS" = "mujoco" ] && [ ! -f "$MENAGERIE/franka_emika_panda/panda.xml" ] \
  && [ ! -f "$MENAGERIE/universal_robots_ur5e/ur5e.xml" ] \
  && [ ! -f "$MENAGERIE/ufactory_xarm7/xarm7.xml" ]; then
  echo "error: no MuJoCo Menagerie arms found under $MENAGERIE" >&2
  echo "clone them with:" >&2
  echo "  git clone --depth 1 --filter=blob:none --sparse \\" >&2
  echo "      https://github.com/google-deepmind/mujoco_menagerie ~/.cache/mujoco_menagerie" >&2
  echo "  cd ~/.cache/mujoco_menagerie && git sparse-checkout set \\" >&2
  echo "      franka_emika_panda universal_robots_ur5e ufactory_xarm7" >&2
  exit 1
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found; set PYTHON to a Python 3 interpreter" >&2
  exit 1
fi

"$PYTHON" "$CLIENTS/mock_policy_server.py" --port "$BACKEND_PORT" --latency-ms 20 --jitter-ms 2 \
  --max-batch 8 --max-wait-ms 5 --per-item-ms 2 --action-scale 0.005 \
  >"$LOGS/backend.log" 2>&1 & PIDS+=($!)

cat >"$LOGS/gateway.yaml" <<EOF
api_keys: [$KEY]
request_timeout_s: 5.0
models:
  pi05:
    backends:
      - ws://127.0.0.1:$BACKEND_PORT/v1/realtime/robot/openpi
EOF
"$PYTHON" "$GATEWAY/gateway.py" --config "$LOGS/gateway.yaml" --host 127.0.0.1 --port "$GW_PORT" \
  >"$LOGS/gateway.log" 2>&1 & PIDS+=($!)

for _ in $(seq 1 50); do
  curl -sf "http://127.0.0.1:$GW_PORT/healthz" >/dev/null 2>&1 && break
  sleep 0.2
done
curl -sf "http://127.0.0.1:$GW_PORT/healthz" >/dev/null || {
  echo "error: gateway did not start; logs in $LOGS" >&2
  exit 1
}

"$PYTHON" "$STUDIO/studio.py" --host 127.0.0.1 --port "$STUDIO_PORT" \
  --gateway "http://127.0.0.1:$GW_PORT" --api-key "$KEY" \
  --physics "$PHYSICS" --menagerie "$MENAGERIE" >"$LOGS/studio.log" 2>&1 & PIDS+=($!)

for _ in $(seq 1 50); do
  curl -sf "http://127.0.0.1:$STUDIO_PORT/healthz" >/dev/null 2>&1 && break
  sleep 0.2
done
curl -sf "http://127.0.0.1:$STUDIO_PORT/healthz" >/dev/null || {
  echo "error: studio did not start; logs in $LOGS" >&2
  exit 1
}

echo "studio:  http://127.0.0.1:$STUDIO_PORT  (physics: $PHYSICS)"
echo "gateway: http://127.0.0.1:$GW_PORT  (API key: $KEY)"
echo "logs:    $LOGS"
if [ "$PHYSICS" = "mock" ]; then
  echo "note:    mock physics renders a 2D reach stand-in, not an articulated arm."
  echo "         for rendered arms: PHYSICS=mujoco bash $0"
fi
echo "press Ctrl-C to stop"
wait "${PIDS[2]}"
