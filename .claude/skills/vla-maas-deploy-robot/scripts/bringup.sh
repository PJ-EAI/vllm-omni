#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# First bring-up of one arm against the remote gateway. Never moves the arm
# except an optional `home`, which asks for confirmation:
#   CAN up (Piper) -> robotctl check -> [home] -> one dry_run episode -> summary.
#
#   ROBOT=piper-01.yaml KEY_FILE=piper-01.env bash bringup.sh
#   ROBOT=piper-01-mock.yaml KEY_FILE=piper-01.env bash bringup.sh   # CPU rehearsal
#
# Env: ROBOT (required), KEY_FILE (file with VLA_MAAS_API_KEY=..., or export it),
#      INSTRUCTION (default "pick up the cube"), STEPS (default 300), DO_HOME=1 to home first,
#      CAN=can0, CAN_SCRIPT (path to piper_sdk's can_activate.sh), ROBOTCTL (default robotctl),
#      OUT (default ./bringup-<robot_id>.jsonl)
set -euo pipefail

: "${ROBOT:?set ROBOT to the robot yaml from render_robot.py}"
ROBOTCTL="${ROBOTCTL:-robotctl}"
INSTRUCTION="${INSTRUCTION:-pick up the cube}"
STEPS="${STEPS:-300}"
CAN="${CAN:-can0}"
if [ -n "${KEY_FILE:-}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$KEY_FILE"
  set +a
fi
: "${VLA_MAAS_API_KEY:?no robot key: set KEY_FILE or export VLA_MAAS_API_KEY}"
robot_id="$(sed -n 's/^robot_id: *//p' "$ROBOT" | head -1)"
OUT="${OUT:-$PWD/bringup-$robot_id.jsonl}"

if grep -q "kind: piper" "$ROBOT"; then
  if ! ip link show "$CAN" 2>/dev/null | grep -q "state UP"; then
    [ -n "${CAN_SCRIPT:-}" ] || { echo "$CAN is down; run piper_sdk's can_activate.sh $CAN 1000000 or set CAN_SCRIPT" >&2; exit 1; }
    bash "$CAN_SCRIPT" "$CAN" 1000000
  fi
  echo "ok   $CAN up"
fi

echo "== robotctl check (no motion)"
"$ROBOTCTL" check --robot "$ROBOT"

if [ "${DO_HOME:-0}" = 1 ]; then
  echo "== robotctl home (moves the arm; asks for confirmation)"
  "$ROBOTCTL" home --robot "$ROBOT"
fi

echo "== one dry_run episode (commands filtered and counted, not sent)"
"$ROBOTCTL" run --robot "$ROBOT" --instruction "$INSTRUCTION" --max-steps "$STEPS" --mode dry_run --output "$OUT"

python3 - "$OUT" <<'PY'
import json
import sys

ep = json.loads(open(sys.argv[1]).read().splitlines()[-1])
safety = ep.get("safety", {})
print("== summary")
for key in ("inferences", "rtt_p50_ms", "server_p50_ms", "network_p50_ms", "underruns", "dropped_rows", "stopped"):
    print(f"  {key}: {ep.get(key)}")
ticks = ep.get("ticks") or 0
print(f"  safety clamped: {safety.get('clamped')} of {ticks} ticks, worst streak {safety.get('worst_streak')}, reasons {safety.get('reasons')}")
if ep.get("underruns"):
    print("  underruns > 0: raise control.prefetch_threshold or cut server latency (bf16, fewer denoise steps)")
if ticks and (safety.get("clamped") or 0) > 0.2 * ticks:
    print("  many clamps: check units, joint signs, gripper range and the gateway adapter before any motion")
PY
echo "episode log: $OUT"
