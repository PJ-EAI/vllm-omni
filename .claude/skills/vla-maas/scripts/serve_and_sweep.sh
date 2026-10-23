#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# One cluster job: start a policy server, optionally put the MaaS gateway in
# front of it, wait until ready, run a fleet-size sweep, collect everything in
# one result directory, and shut down. Runs the same way under Slurm, a k8s Job
# or a bare node.
#
#   MODEL=/ckpt/pi05_libero DEPLOY_CONFIG=examples/online_serving/vla_maas/pi05_libero.yaml \
#     bash .claude/skills/vla-maas/scripts/serve_and_sweep.sh
#
# Env (defaults in brackets):
#   MODEL            checkpoint path or HF id (required unless SERVER_CMD is set)
#   DEPLOY_CONFIG    [examples/online_serving/vla_maas/pi05_libero.yaml; empty with SERVER_CMD]
#   SERVE_ARGS       extra `vllm serve` args [--enforce-eager --disable-log-stats]
#   SERVER_CMD       full server command; overrides MODEL/DEPLOY_CONFIG/SERVE_ARGS
#                    (e.g. the CPU mock, to dry-run this script)
#   READY            http (poll /health) | tcp (port open) [http]
#   READY_TIMEOUT_S  [1800]
#   PORT             server port [8000]
#   USE_GATEWAY      1 = sweep through the gateway, 0 = direct [1]
#   GW_PORT          [9000]
#   ROBOTS           ["1 2 4 8 16"]
#   DURATION         seconds per point [30]
#   LABEL            sweep label [basename of DEPLOY_CONFIG or "custom"]
#   CLIENT_ARGS      extra fleet_sim args, e.g. camera keys
#   OUT_DIR          [maas_runs/<UTC timestamp>-<LABEL>]
#   PYTHON           [python3]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
CLIENTS="examples/online_serving/vla_maas"
PYTHON="${PYTHON:-python3}"
SERVE_ARGS="${SERVE_ARGS:---enforce-eager --disable-log-stats}"
READY="${READY:-http}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"
PORT="${PORT:-8000}"
USE_GATEWAY="${USE_GATEWAY:-1}"
GW_PORT="${GW_PORT:-9000}"
ROBOTS="${ROBOTS:-1 2 4 8 16}"
DURATION="${DURATION:-30}"
CLIENT_ARGS="${CLIENT_ARGS:-}"
if [ -n "${SERVER_CMD:-}" ]; then
  DEPLOY_CONFIG="${DEPLOY_CONFIG:-}"  # recorded only if the custom command uses one
  LABEL="${LABEL:-custom}"
else
  DEPLOY_CONFIG="${DEPLOY_CONFIG:-$CLIENTS/pi05_libero.yaml}"
  : "${MODEL:?set MODEL (checkpoint) or SERVER_CMD}"
  LABEL="${LABEL:-$(basename "$DEPLOY_CONFIG" .yaml)}"
  SERVER_CMD="vllm serve $MODEL --omni --host 0.0.0.0 --port $PORT --deploy-config $DEPLOY_CONFIG $SERVE_ARGS"
fi
OUT_DIR="${OUT_DIR:-maas_runs/$(date -u +%Y%m%dT%H%M%SZ)-$LABEL}"
KEY="${VLA_MAAS_API_KEY:-job-$RANDOM$RANDOM}"
mkdir -p "$OUT_DIR"
PIDS=()
SERVER_PID=""
# The server runs in its own process group so its engine workers die with it.
cleanup() {
  if [ -n "$SERVER_PID" ]; then kill -TERM -- "-$SERVER_PID" 2>/dev/null || true; fi
  kill "${PIDS[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT

"$PYTHON" - "$OUT_DIR/manifest.json" <<EOF
import json, os, platform, subprocess, sys
def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""
json.dump({
    "git_sha": sh("git rev-parse HEAD"),
    "git_branch": sh("git rev-parse --abbrev-ref HEAD"),
    "host": platform.node(),
    "gpus": sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),
    "server_cmd": """$SERVER_CMD""",
    "deploy_config": "$DEPLOY_CONFIG",
    "use_gateway": "$USE_GATEWAY" == "1",
    "robots": "$ROBOTS".split(),
    "duration_s": float("$DURATION"),
    "label": "$LABEL",
    "client_args": """$CLIENT_ARGS""",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
}, open(sys.argv[1], "w"), indent=2)
EOF
if [ -f "$DEPLOY_CONFIG" ]; then cp "$DEPLOY_CONFIG" "$OUT_DIR/"; fi

echo "server: $SERVER_CMD"
# shellcheck disable=SC2086
setsid $SERVER_CMD >"$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!

ready() {
  if [ "$READY" = http ]; then
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1
  else
    "$PYTHON" -c "import socket,sys; socket.create_connection(('127.0.0.1', $PORT), 1)" 2>/dev/null
  fi
}
deadline=$((SECONDS + READY_TIMEOUT_S))
until ready; do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -50 "$OUT_DIR/server.log" >&2; echo "server exited" >&2; exit 1; }
  [ "$SECONDS" -lt "$deadline" ] || { echo "server not ready after ${READY_TIMEOUT_S}s" >&2; exit 1; }
  sleep 5
done
echo "server ready on :$PORT"

if [ "$USE_GATEWAY" = 1 ]; then
  printf 'api_keys: [%s]\nrequest_timeout_s: 30.0\nmodels:\n  %s:\n    backends:\n      - ws://127.0.0.1:%s/v1/realtime/robot/openpi\n' \
    "$KEY" "$LABEL" "$PORT" >"$OUT_DIR/gateway.yaml"
  "$PYTHON" apps/vla-maas-gateway/gateway.py --config "$OUT_DIR/gateway.yaml" --port "$GW_PORT" \
    >"$OUT_DIR/gateway.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 1 50); do curl -sf "http://127.0.0.1:$GW_PORT/healthz" >/dev/null 2>&1 && break; sleep 0.2; done
  TARGET=(--port "$GW_PORT" --path "/v1/robot/$LABEL/openpi" --api-key "$KEY")
else
  TARGET=(--port "$PORT" --path /v1/realtime/robot/openpi)
fi

for n in $ROBOTS; do
  echo "sweep: robots=$n duration=${DURATION}s"
  # shellcheck disable=SC2086
  "$PYTHON" "$CLIENTS/fleet_sim.py" "${TARGET[@]}" --robots "$n" --duration-s "$DURATION" \
    --label "$LABEL" --output "$OUT_DIR/sweep.jsonl" $CLIENT_ARGS 2>&1 | tee -a "$OUT_DIR/client.log"
done

if [ "$USE_GATEWAY" = 1 ]; then
  curl -sf -H "Authorization: Api-Key $KEY" "http://127.0.0.1:$GW_PORT/metrics" >"$OUT_DIR/gateway_metrics.json" || true
fi
echo "results: $OUT_DIR"
cat "$OUT_DIR/sweep.jsonl"
