#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Stop what up.sh started (studio, gateway, backends). Keys and logs stay in RUN_DIR.
#
#   bash down.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/maas_runs/local-pi05}"
for pidfile in "$RUN_DIR"/studio/*.pid "$RUN_DIR"/gateway/*.pid "$RUN_DIR"/backends/*.pid; do
  [ -f "$pidfile" ] || continue
  pid="$(cat "$pidfile")"
  if kill "$pid" 2>/dev/null; then
    echo "stopped $(basename "$pidfile" .pid) pid=$pid"
  fi
  rm -f "$pidfile"
done
