#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# One-command CPU demo: a mock policy backend, the MaaS gateway, and the studio.
# Open the printed URL and start a run; no GPU or simulator install is needed.
#
#   bash apps/vla-maas-studio/run_demo.sh
#   PHYSICS=mujoco bash apps/vla-maas-studio/run_demo.sh   # rendered arms
#   PHYSICS=piper bash apps/vla-maas-studio/run_demo.sh    # Piper twin on the standard Robot protocol
# Env: PYTHON=python3 PHYSICS=mock|mujoco|piper MENAGERIE=~/.cache/mujoco_menagerie
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

if [ "$PHYSICS" = "piper" ] && [ ! -f "$MENAGERIE/agilex_piper/piper.xml" ]; then
  echo "error: no MuJoCo Menagerie Piper under $MENAGERIE" >&2
  echo "clone it with:" >&2
  echo "  git clone --depth 1 --filter=blob:none --sparse \\" >&2
  echo "      https://github.com/google-deepmind/mujoco_menagerie ~/.cache/mujoco_menagerie" >&2
  echo "  cd ~/.cache/mujoco_menagerie && git sparse-checkout set agilex_piper" >&2
  exit 1
fi

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

# The Piper twin gets slow joint motion from the mock (it sees no pixel markers).
IDLE_MOTION=0
[ "$PHYSICS" = "piper" ] && IDLE_MOTION=0.04
"$PYTHON" "$CLIENTS/mock_policy_server.py" --port "$BACKEND_PORT" --latency-ms 20 --jitter-ms 2 \
  --max-batch 8 --max-wait-ms 5 --per-item-ms 2 --action-scale 0.005 --idle-motion "$IDLE_MOTION" \
  >"$LOGS/backend.log" 2>&1 & PIDS+=($!)

# pi05_piper: the mock seen through a Piper-shaped /act adapter (delta joints + gripper),
# under the model name packages/vla-maas-robot/robots/piper.yaml uses.
cat >"$LOGS/gateway.yaml" <<EOF
api_keys: [$KEY]
request_timeout_s: 5.0
models:
  pi05:
    backends:
      - ws://127.0.0.1:$BACKEND_PORT/v1/realtime/robot/openpi
  pi05_piper:
    backends:
      - ws://127.0.0.1:$BACKEND_PORT/v1/realtime/robot/openpi
    adapter:
      kind: generic
      images: {front: observation.images.image, wrist: observation.images.image2}
      image_size: [224, 224]
      state: [joint_pos, {field: gripper, range: [-1, 1]}]
      state_dim: 32
      action:
        space: joint_pos
        layout: [{field: joint_pos, mode: delta}, {field: gripper, range: [-1, 1]}]
      control_hz: 20
      embodiments: [piper]
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

STUDIO_ARGS=(--physics "$PHYSICS")
if [ "$PHYSICS" = "piper" ]; then
  # The real robot file, with the twin starting from a ready pose instead of the
  # rest pose (Piper rests on its joint-2/3 limits, where any downward target clamps).
  "$PYTHON" - "$ROOT/packages/vla-maas-robot/robots/piper.yaml" "$LOGS/piper.yaml" <<'EOF'
import sys

import yaml

robot = yaml.safe_load(open(sys.argv[1]))
robot["driver"]["home_joints"] = [0.0, 1.0, -1.0, 0.0, 0.4, 0.0]
yaml.safe_dump(robot, open(sys.argv[2], "w"), sort_keys=False)
EOF
  STUDIO_ARGS=(--robot-config "$LOGS/piper.yaml")
fi
"$PYTHON" "$STUDIO/studio.py" --host 127.0.0.1 --port "$STUDIO_PORT" \
  --gateway "http://127.0.0.1:$GW_PORT" --api-key "$KEY" \
  "${STUDIO_ARGS[@]}" --menagerie "$MENAGERIE" >"$LOGS/studio.log" 2>&1 & PIDS+=($!)

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
if [ "$PHYSICS" = "piper" ]; then
  echo "note:    Piper MuJoCo twin on /v1/robot/pi05_piper/act; the mock policy only sweeps the joints."
  echo "         real arm: studio.py --robot-config <your piper.yaml> --hardware"
elif [ "$PHYSICS" = "mock" ]; then
  echo "note:    mock physics renders a 2D reach stand-in, not an articulated arm."
  echo "         for rendered arms: PHYSICS=mujoco bash $0"
fi
echo "press Ctrl-C to stop"
wait "${PIDS[2]}"
