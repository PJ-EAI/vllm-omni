#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# CPU smoke test of the VLA MaaS loop: mock backend -> gateway -> mock LIBERO
# client. Needs only the client extras (websockets, msgpack, msgpack-numpy,
# pyyaml, numpy, openpi-client). Exits non-zero on the first failed check.
#
#   bash .claude/skills/vla-maas/scripts/maas_smoke.sh
# Env: PYTHON=python3 BACKEND_PORT=18001 GW_PORT=19000 OUT_DIR=<tmp>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CLIENTS="$ROOT/examples/online_serving/vla_maas"
PYTHON="${PYTHON:-python3}"
BACKEND_PORT="${BACKEND_PORT:-18001}"
GW_PORT="${GW_PORT:-19000}"
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
KEY=smoke-key
GW="http://127.0.0.1:$GW_PORT"
PIDS=()
mkdir -p "$OUT_DIR"
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

fail() { echo "FAIL: $*" >&2; echo "logs: $OUT_DIR" >&2; exit 1; }
pass() { echo "ok   $*"; }

"$PYTHON" -c "import websockets, msgpack, msgpack_numpy, yaml, numpy, openpi_client.websocket_client_policy" \
  || fail "client extras missing (see references/maas-stack.md, Client dependencies)"

"$PYTHON" "$CLIENTS/mock_policy_server.py" --port "$BACKEND_PORT" --latency-ms 20 --jitter-ms 2 \
  >"$OUT_DIR/backend.log" 2>&1 &
PIDS+=($!)
printf 'api_keys: [%s]\nrequest_timeout_s: 5.0\nmodels:\n  pi05:\n    backends:\n      - ws://127.0.0.1:%s/v1/realtime/robot/openpi\n' \
  "$KEY" "$BACKEND_PORT" >"$OUT_DIR/gateway.yaml"
"$PYTHON" "$ROOT/apps/vla-maas-gateway/gateway.py" --config "$OUT_DIR/gateway.yaml" --host 127.0.0.1 \
  --port "$GW_PORT" >"$OUT_DIR/gateway.log" 2>&1 &
PIDS+=($!)

for _ in $(seq 1 50); do
  curl -sf "$GW/healthz" >/dev/null 2>&1 && break
  sleep 0.2
done
curl -sf "$GW/healthz" >/dev/null || fail "gateway /healthz not reachable on $GW_PORT"
pass "gateway /healthz"

code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Api-Key wrong" "$GW/metrics")
[ "$code" = 401 ] || fail "bad key returned $code, want 401"
pass "bad key -> 401"

code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Api-Key $KEY" "$GW/v1/robot/nope/openpi")
[ "$code" = 404 ] || fail "unknown model returned $code, want 404"
pass "unknown model -> 404"

curl -sf -H "Authorization: Api-Key $KEY" "$GW/v1/models" | grep -q '"pi05"' || fail "/v1/models lacks pi05"
pass "/v1/models lists pi05"

"$PYTHON" "$CLIENTS/libero_eval.py" --mock --port "$GW_PORT" --path /v1/robot/pi05/openpi --api-key "$KEY" \
  --max-tasks 2 --episodes-per-task 2 --replan-steps 20 --async-prefetch 10 --report-episodes \
  --output "$OUT_DIR/eval.json" >"$OUT_DIR/eval.log" 2>&1 || fail "libero_eval through the gateway failed"
pass "mock LIBERO eval through gateway"

curl -sf -H "Authorization: Api-Key $KEY" "$GW/metrics" >"$OUT_DIR/metrics.json" || fail "/metrics failed"
"$PYTHON" - "$OUT_DIR/eval.json" "$OUT_DIR/metrics.json" <<'EOF' || fail "result checks failed"
import json, sys
ev = json.load(open(sys.argv[1]))
m = json.load(open(sys.argv[2]))["pi05"]
assert ev["num_inferences"] > 0, ev
assert ev["success_rate"] > 0, f"mock success_rate is 0: {ev}"
assert m["requests"] >= ev["num_inferences"], m
assert m["errors"] == 0, m
assert m["episodes"] == 4, m
print(f"ok   eval success_rate={ev['success_rate']:.2f} p50={ev['inference_p50_ms']:.1f}ms; "
      f"gateway requests={m['requests']} relay_p50={m['relay_p50_ms']:.1f}ms")
EOF
echo "PASS (artifacts: $OUT_DIR)"
