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

"$PYTHON" -c "import websockets, msgpack, msgpack_numpy, yaml, numpy, openpi_client.websocket_client_policy, vla_maas_protocol" \
  || fail "client extras missing (see references/maas-stack.md, Client dependencies)"

"$PYTHON" "$CLIENTS/mock_policy_server.py" --port "$BACKEND_PORT" --latency-ms 20 --jitter-ms 2 --max-batch 8 \
  >"$OUT_DIR/backend.log" 2>&1 &
PIDS+=($!)
cat >"$OUT_DIR/gateway.yaml" <<YAML
api_keys: [$KEY]
request_timeout_s: 5.0
models:
  pi05:
    backends: [ws://127.0.0.1:$BACKEND_PORT/v1/realtime/robot/openpi]
  pi05_mock:
    backends: [ws://127.0.0.1:$BACKEND_PORT/v1/realtime/robot/openpi]
    adapter:
      kind: generic
      images: {front: observation.images.image, wrist: observation.images.image2}
      state: [joint_pos, {field: gripper, range: [-1, 1]}]
      action: {space: joint_pos, layout: [{field: joint_pos, mode: delta}, {field: gripper, range: [-1, 1]}]}
      control_hz: 20
robots:
  mock-01: {api_key: smoke-robot-key, models: [pi05_mock]}
YAML
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
# Standard Robot protocol: a mock robot through /act with its own key (needs ./packages/vla-maas-robot).
if "$PYTHON" -c "import vla_maas_robot" 2>/dev/null; then
  sed -e "s|ws://127.0.0.1:9000|ws://127.0.0.1:$GW_PORT|" -e "s|api_key_env: VLA_MAAS_API_KEY|api_key: smoke-robot-key|" \
    "$ROOT/packages/vla-maas-robot/robots/mock.yaml" >"$OUT_DIR/robot.yaml"
  "$PYTHON" -m vla_maas_robot.robotctl check --robot "$OUT_DIR/robot.yaml" >"$OUT_DIR/robot.log" 2>&1 \
    || fail "robotctl check through /act failed"
  "$PYTHON" -m vla_maas_robot.robotctl run --robot "$OUT_DIR/robot.yaml" --instruction smoke --max-steps 40 \
    --mode live --yes --output "$OUT_DIR/robot.jsonl" >>"$OUT_DIR/robot.log" 2>&1 || fail "robotctl run through /act failed"
  "$PYTHON" - "$OUT_DIR/robot.jsonl" <<'EOF' || fail "robot protocol checks failed"
import json, sys
r = json.loads(open(sys.argv[1]).read().splitlines()[-1])
assert r["stopped"] is None and r["commands_sent"] > 0 and r["inferences"] > 0, r
print(f"ok   robot protocol: {r['commands_sent']} commands, {r['inferences']} inferences, rtt_p50={r['rtt_p50_ms']}ms")
EOF
else
  echo "skip robot protocol (pip install ./packages/vla-maas-protocol ./packages/vla-maas-robot)"
fi
# Simulated multi-arm fleet (mock envs, two groups); records MP4s when imageio is present.
VIDEO_ARGS=()
if "$PYTHON" -c "import imageio, imageio_ffmpeg" 2>/dev/null; then
  VIDEO_ARGS=(--video-dir "$OUT_DIR/videos" --record-episodes 1)
fi
cat >"$OUT_DIR/scenario.yaml" <<'YAML'
defaults: {control_hz: 50, replan_steps: 25, async_prefetch: 10, start_jitter_s: 0.5}
groups:
  - {name: panda-mock, model: pi05, robots: 2, env: {kind: mock, robot: Panda, resolution: 128}}
  - {name: ur5e-mock, model: pi05, robots: 1, env: {kind: mock, robot: UR5e, resolution: 128}}
YAML
"$PYTHON" "$CLIENTS/sim_fleet.py" --scenario "$OUT_DIR/scenario.yaml" --port "$GW_PORT" --api-key "$KEY" \
  --duration-s 4 --output "$OUT_DIR/sim_fleet.jsonl" "${VIDEO_ARGS[@]}" >"$OUT_DIR/sim_fleet.log" 2>&1 \
  || fail "sim_fleet through the gateway failed"
"$PYTHON" - "$OUT_DIR/sim_fleet.jsonl" "${#VIDEO_ARGS[@]}" <<'EOF' || fail "sim_fleet checks failed"
import json, sys
rows = {r["group"]: r for r in map(json.loads, open(sys.argv[1]))}
assert set(rows) == {"panda-mock", "ur5e-mock", "all"}, rows
assert rows["all"]["errors"] == 0 and rows["all"]["episodes"] > 0, rows["all"]
assert rows["all"]["success_rate"] > 0, rows["all"]
if int(sys.argv[2]):
    assert rows["all"]["videos"] == 3, rows["all"]
print(f"ok   sim_fleet episodes={rows['all']['episodes']} success_rate={rows['all']['success_rate']} "
      f"videos={rows['all']['videos']}")
EOF
echo "PASS (artifacts: $OUT_DIR)"
