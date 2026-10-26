#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Start the gateway on localhost behind the TLS proxy, and wait for /healthz.
#
#   CONFIG=deploy/gateway.yaml bash start_gateway.sh
#
# Env: CONFIG (required), GW_HOST (default 127.0.0.1), GW_PORT (default 9000),
#      PYTHON (default python3), LOG_DIR (default ./maas-gateway)
# Stop: kill $(cat $LOG_DIR/gateway.pid)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
: "${CONFIG:?set CONFIG to the gateway yaml from render_gateway.py}"
GW_HOST="${GW_HOST:-127.0.0.1}"
GW_PORT="${GW_PORT:-9000}"
PYTHON="${PYTHON:-python3}"
LOG_DIR="${LOG_DIR:-$PWD/maas-gateway}"
mkdir -p "$LOG_DIR"

"$PYTHON" -c "import websockets, msgpack, yaml, numpy, vla_maas_protocol" \
  || { echo "missing gateway deps: pip install ./packages/vla-maas-protocol websockets msgpack msgpack-numpy pyyaml numpy" >&2; exit 1; }

"$PYTHON" "$ROOT/apps/vla-maas-gateway/gateway.py" --config "$CONFIG" --host "$GW_HOST" --port "$GW_PORT" \
  >"$LOG_DIR/gateway.log" 2>&1 &
echo $! >"$LOG_DIR/gateway.pid"
for _ in $(seq 1 50); do
  curl -sf "http://127.0.0.1:$GW_PORT/healthz" >/dev/null 2>&1 && { echo "gateway up on $GW_HOST:$GW_PORT pid=$(cat "$LOG_DIR/gateway.pid")"; exit 0; }
  kill -0 "$(cat "$LOG_DIR/gateway.pid")" 2>/dev/null || break
  sleep 0.2
done
echo "gateway did not start; see $LOG_DIR/gateway.log" >&2
exit 1
