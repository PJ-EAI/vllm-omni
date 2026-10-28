#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Start only the MuJoCo Piper client: the studio's twin of packages/vla-maas-robot/robots/piper.yaml,
# speaking the standard Robot protocol (/act) to a gateway that is already running. Never drives
# hardware. Needs .venv-maas and the Piper model (ONLY=maas setup_envs.sh). Stop with down.sh.
#
#   bash twin.sh                                                   # gateway from up.sh on this machine
#   GATEWAY=https://maas.example.com API_KEY=<key> bash twin.sh    # any gateway
#
# Env: GATEWAY (default http://127.0.0.1:9000), API_KEY (default: up.sh's operator key, else
#      $VLA_MAAS_API_KEY), MODEL (default pi05_piper), ROBOT_ID (default piper-01),
#      STUDIO_PORT (default 9100), TWIN_MAX_VIOLATIONS (default 100000), RUN_DIR, MENAGERIE, MUJOCO_GL
# A robot key works for its own robot_id and model; several twins at once need an operator key.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
RUN_DIR="${RUN_DIR:-$ROOT/maas_runs/local-pi05}"
GATEWAY="${GATEWAY:-http://127.0.0.1:9000}"
MODEL="${MODEL:-pi05_piper}"
ROBOT_ID="${ROBOT_ID:-piper-01}"
STUDIO_PORT="${STUDIO_PORT:-9100}"
MENAGERIE="${MENAGERIE:-$HOME/.cache/mujoco_menagerie}"
MAAS_PY="$ROOT/.venv-maas/bin/python"
if [ -z "${API_KEY:-}" ] && [ -f "$RUN_DIR/deploy/keys/operator.env" ]; then
  API_KEY="$(sed -n 's/^VLA_MAAS_OPERATOR_KEY=//p' "$RUN_DIR/deploy/keys/operator.env")"
fi
API_KEY="${API_KEY:-${VLA_MAAS_API_KEY:-}}"

[ -x "$MAAS_PY" ] || { echo "missing .venv-maas; run ONLY=maas setup_envs.sh" >&2; exit 1; }
[ -f "$MENAGERIE/agilex_piper/piper.xml" ] || { echo "no Piper model under $MENAGERIE; run ONLY=maas setup_envs.sh" >&2; exit 1; }
[ -n "$API_KEY" ] || { echo "set API_KEY (operator key, or $ROBOT_ID's robot key)" >&2; exit 1; }
curl -sf "$GATEWAY/healthz" >/dev/null || { echo "gateway not reachable at $GATEWAY/healthz" >&2; exit 1; }
code="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Api-Key $API_KEY" "$GATEWAY/v1/models/$MODEL")"
[ "$code" = 200 ] || { echo "GET $GATEWAY/v1/models/$MODEL -> $code (wrong key or model?)" >&2; exit 1; }

# The real robot file; the twin starts from a ready pose instead of the joint-limit rest pose.
mkdir -p "$RUN_DIR/studio"
"$MAAS_PY" - packages/vla-maas-robot/robots/piper.yaml "$RUN_DIR/piper.yaml" "$ROBOT_ID" "$MODEL" \
  "$GATEWAY" "${TWIN_MAX_VIOLATIONS:-100000}" <<'PY'
import sys

import yaml

src, dst, robot_id, model, gateway, max_violations = sys.argv[1:]
robot = yaml.safe_load(open(src))
robot["robot_id"] = robot_id
robot["gateway"].update(url=gateway, model=model)
robot["driver"]["home_joints"] = [0.0, 1.0, -1.0, 0.0, 0.4, 0.0]
# Twin only (the studio runs without --hardware here): a model not trained on Piper, such as
# pi05_base, gets most targets clamped, and the default 5-in-a-row rule would end every episode
# within seconds. Clamping still bounds each step; the real arm keeps the file's own limit.
robot["safety"]["max_violations"] = int(max_violations)
yaml.safe_dump(robot, open(dst, "w"), sort_keys=False)
PY

"$MAAS_PY" apps/vla-maas-studio/studio.py --host 127.0.0.1 --port "$STUDIO_PORT" \
  --robot-config "$RUN_DIR/piper.yaml" --gateway "$GATEWAY" --api-key "$API_KEY" \
  --menagerie "$MENAGERIE" >"$RUN_DIR/studio/studio.log" 2>&1 &
echo $! >"$RUN_DIR/studio/studio.pid"
for _ in $(seq 1 100); do
  if curl -sf "http://127.0.0.1:$STUDIO_PORT/healthz" >/dev/null 2>&1; then
    echo "studio     http://127.0.0.1:$STUDIO_PORT   Piper twin $ROBOT_ID -> $GATEWAY ($MODEL); press 启动 there"
    echo "dashboard  $GATEWAY/dashboard (operator key)"
    exit 0
  fi
  kill -0 "$(cat "$RUN_DIR/studio/studio.pid")" 2>/dev/null || break
  sleep 0.2
done
echo "studio did not start; see $RUN_DIR/studio/studio.log (an EGL error: set MUJOCO_GL=osmesa)" >&2
exit 1
