#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Start one π0.5 backend per GPU and print the backend URLs for render_gateway.py.
# Backends listen on the cluster network only; never expose these ports publicly.
#
#   CKPT=<hf id or path> DEPLOY=<deploy yaml> GPUS="0 1" bash serve_pi05.sh
#   MOCK=1 GPUS="0 1" bash serve_pi05.sh          # CPU mock backends, same ports and URLs
#
# Env:
#   CKPT        checkpoint (default lerobot/pi05_base: link and latency tests only, dry_run)
#   DEPLOY      deploy yaml (default vllm_omni/deploy/pi05.yaml)
#   GPUS        GPU ids, one backend each (default "0")
#   BASE_PORT   first port; backend i uses BASE_PORT+i (default 8001)
#   HOST        bind address (default 0.0.0.0) and ADVERTISE host in the URLs (default hostname)
#   LOG_DIR     logs and pids (default ./maas-backends)
#   PYTHON      interpreter for MOCK=1 (default python3)
#   VLLM        vllm executable (default vllm)
# Stop: kill $(cat $LOG_DIR/*.pid)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CKPT="${CKPT:-lerobot/pi05_base}"
DEPLOY="${DEPLOY:-$ROOT/vllm_omni/deploy/pi05.yaml}"
GPUS="${GPUS:-0}"
BASE_PORT="${BASE_PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
ADVERTISE="${ADVERTISE:-$(hostname)}"
LOG_DIR="${LOG_DIR:-$PWD/maas-backends}"
PYTHON="${PYTHON:-python3}"
VLLM="${VLLM:-vllm}"
MOCK="${MOCK:-0}"
PATH_WS=/v1/realtime/robot/openpi
mkdir -p "$LOG_DIR"

urls=()
i=0
for gpu in $GPUS; do
  port=$((BASE_PORT + i))
  log="$LOG_DIR/backend-$port.log"
  if [ "$MOCK" = 1 ]; then
    # Mock keys match the Piper adapter's defaults, so the same gateway.yaml works.
    "$PYTHON" "$ROOT/examples/online_serving/vla_maas/mock_policy_server.py" --host "$HOST" --port "$port" \
      --latency-ms 150 --jitter-ms 20 --base-camera-key observation.images.base_0_rgb >"$log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpu" "$VLLM" serve "$CKPT" --omni --host "$HOST" --port "$port" \
      --deploy-config "$DEPLOY" --enforce-eager --disable-log-stats \
      --robot-openpi-idle-timeout 0 >"$log" 2>&1 &
  fi
  echo $! >"$LOG_DIR/backend-$port.pid"
  urls+=("ws://$ADVERTISE:$port$PATH_WS")
  echo "backend gpu=$gpu port=$port pid=$! log=$log"
  i=$((i + 1))
done

# A real checkpoint takes minutes to load; wait until each port accepts TCP connections.
deadline=$((SECONDS + ${READY_TIMEOUT_S:-900}))
i=0
for _ in $GPUS; do
  port=$((BASE_PORT + i))
  until (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
    pid="$(cat "$LOG_DIR/backend-$port.pid")"
    kill -0 "$pid" 2>/dev/null || { echo "backend on $port exited; see $LOG_DIR/backend-$port.log" >&2; exit 1; }
    [ $SECONDS -lt $deadline ] || { echo "backend on $port not ready in time" >&2; exit 1; }
    sleep 2
  done
  echo "ready port=$port"
  i=$((i + 1))
done

joined="$(IFS=,; echo "${urls[*]}")"
echo "$joined" >"$LOG_DIR/backends.txt"
echo "BACKENDS=$joined"
