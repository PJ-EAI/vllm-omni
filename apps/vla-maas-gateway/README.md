# VLA MaaS gateway

A gateway that exposes vllm-omni VLA policy servers as one API.
It sits in front of one or more `vllm serve ... --omni` instances (or the CPU
mock in `examples/online_serving/vla_maas/mock_policy_server.py`) and adds:

- **API-key auth**: `Authorization: Api-Key <key>` (or `Bearer <key>`), the header
  `openpi-client` sends when constructed with `api_key=`.
- **Model routing**: `ws://<gateway>/v1/robot/<model>/openpi`; each session goes to
  the least-loaded backend configured for that model.
- **Standard Robot protocol**: `wss://<gateway>/v1/robot/<model>/act?robot_id=<id>`
  speaks the model-agnostic protocol in
  [packages/vla-maas-protocol](../../packages/vla-maas-protocol/spec/robot_protocol.md).
  The model's `adapter` (`model_adapters.py`) translates each frame to the
  backend's native OpenPI frame and back, so robots never see model keys and
  model servers do not change. Robots run the
  [vla-maas-robot](../../packages/vla-maas-robot/README.md) client.
- **Transparent OpenPI relay**: `/v1/robot/<model>/openpi` frames pass through
  unchanged; the handshake metadata gains `model` and `backend`.
- **Per-robot keys**: `robots.<id>.api_key` is valid only on `/act`, for that
  `robot_id` and its `models`. `api_keys` are operator keys for every route.
- **Robots**: `GET /v1/robots` lists connected robots;
  `GET /v1/robots/<id>/stop` answers that robot's next observation with `stop`
  (the websockets HTTP server accepts only GET). This is not an e-stop: keep a
  hardware e-stop at every robot. `GET /v1/models/<model>` shows what a model
  requires and provides on `/act`.
- **Dashboard**: `GET /dashboard`, a live fleet view for operators. It shows each
  model's backends and their load, one-second p50/p99 and throughput history,
  and every connected session (`/act` robots and native OpenPI clients). Click a
  session to see the camera frames the model receives, the robot state, the
  latest action chunk, recent latency, events and errors, and to stop an `/act`
  robot. The page holds no data and asks for an operator key.
- **Monitor socket**: `WS /v1/monitor`, the dashboard's data feed (JSON frames).
  Send `{"type": "auth", "key": "<operator key>"}` first, then
  `{"type": "watch", "session_id": ..., "fps": 5}` to receive `detail` frames for
  one session; a `snapshot` arrives every second. Images are JPEG-encoded off the
  event loop and only for watched sessions; see `monitor.py` for the messages.
- **Metrics**: `GET /metrics` returns per-model request/error counts, active
  sessions and relay p50/p99. `GET /v1/models` lists models with their current
  revision, checkpoint and model type; `GET /healthz` is open.
- **Live checkpoint / model-type swap**: edit a model's `backends`,
  `checkpoint`, `model_type` or `adapter` in the YAML, then
  `GET /v1/admin/reload` (operator key) or `kill -HUP` the gateway. It probes the
  new backends first (handshake, adapter, and the checkpoint / model type they
  report against the labels) and changes nothing on any problem. New sessions
  go to the new revision; open sessions finish on the old one (`draining`), or
  get `stop` with `?cutover=stop`. `backends: []` pauses a model (503).
  `GET /v1/admin/revisions` shows each model's revision and the recent swaps.
  [`swap_model.py`](../../.claude/skills/vla-maas-deploy-server/scripts/swap_model.py)
  does the whole blue/green swap: start, reload, drain, stop the old servers.
- **Structured logs**: one event per line (`session.open`, `session.ready`,
  `act.request`, `act.error`, `session.close`, `route.revision`,
  `backend.draining`, `backend.drained`, ...), each tagged with the session,
  robot, model, revision, checkpoint, model type and backend that served it.
  `--log-file logs/gateway.jsonl` writes them as JSON lines; settings are under
  `logging:` (`maas_log.py`). The robot prints the same session id and revision
  when it connects, so `jq 'select(.session == "<id>")' logs/gateway.jsonl`
  shows everything that robot saw; vllm-omni request ids are
  `robot-<session id>-<n>` when the adapter sends the session id.

Errors: 401 for a missing or wrong key, 404 for an unknown model, websocket close
1011 when a backend is unreachable, and an OpenPI `{"type": "error"}` reply when a
backend times out (`request_timeout_s`).

Install the protocol package next to the gateway's other dependencies:
`pip install ./packages/vla-maas-protocol`.

## One-command demo

```bash
bash apps/vla-maas-gateway/run_demo.sh 8 60           # robots per model, seconds
MODE=serial bash apps/vla-maas-gateway/run_demo.sh 8 60
```

The script uses `python3`; set `PYTHON=/path/to/python` to use an interpreter
from a virtualenv instead.

Starts two mock backends (`pi05` at 60 ms, `internvla_a1` at 90 ms per forward),
the gateway, and a concurrent fleet of mock robots per model (own session, own
50 Hz loop). `MODE=batched` (default) lets each backend batch overlapping
requests; `MODE=serial` runs them one at a time, like a replica with `max_num_seqs: 1`. Open
`http://localhost:9000/dashboard` and enter the API key (`dev-key-change-me`) to
watch success rate, p50/p99 relay latency against the 100 ms budget, throughput,
backend load and every connected robot; click a robot to see its camera frames. Robots send
`{"endpoint": "episode_result", "success": bool}` after each episode; the gateway
counts it and does not forward it.

## Run with mock backends

```bash
V=examples/online_serving/vla_maas
python3 $V/mock_policy_server.py --port 8001 --latency-ms 60 &
python3 $V/mock_policy_server.py --port 8002 --latency-ms 90 &
python3 apps/vla-maas-gateway/gateway.py --config apps/vla-maas-gateway/gateway.example.yaml &

python3 $V/libero_eval.py --mock --port 9000 --path /v1/robot/pi05/openpi \
    --api-key dev-key-change-me --replan-steps 20 --async-prefetch 10 --control-hz 50
curl -H "Authorization: Api-Key dev-key-change-me" localhost:9000/metrics
```

For real backends, replace the `backends` URLs in the config with the
`/v1/realtime/robot/openpi` endpoints of the vllm-omni servers.

## A robot on the standard protocol

```bash
pip install ./packages/vla-maas-protocol ./packages/vla-maas-robot
python3 examples/online_serving/vla_maas/mock_policy_server.py --port 8001 --latency-ms 40 --idle-motion 0.05 &
python3 apps/vla-maas-gateway/gateway.py --config apps/vla-maas-gateway/gateway.example.yaml &

export VLA_MAAS_API_KEY=mock-01-key-change-me
robotctl check --robot packages/vla-maas-robot/robots/mock.yaml
robotctl run --robot packages/vla-maas-robot/robots/mock.yaml --instruction "wave" --max-steps 100
curl -H "Authorization: Api-Key dev-key-change-me" localhost:9000/v1/robots
```

A model is available on `/act` once its entry has an `adapter`; see the
docstring of `model_adapters.py` for every key. On public networks put the
gateway behind a TLS-terminating proxy and give every robot its own key.

## Status and roadmap

Checked 2026-10-31 against `maas` `a8ee0ee` and `feat/arx-deltapose-adapter`
`d28daf5`. Update this section when a PR lands or a real-hardware check
finishes; keep claims to what was actually run.

π0.5 runs end to end as a service: vllm-omni backend → gateway → standard Robot
protocol → `vla-maas-robot`, verified on a real AgileX Piper. ARX X5 is
connected and its real-arm link runs; its delta-pose conversion fix is not yet
verified on the arm. The hierarchical layers (System 2 planner, System 0
whole-body control) are design only.

### Models

| Model | OpenPI serving | Request batching | `/act` adapter | Notes |
| --- | --- | --- | --- | --- |
| π0.5 (`Pi05Pipeline`) | yes | yes | yes: `joint_pos`, `eef_pose` | Real Piper run passed; ARX delta-pose in progress |
| π0 (`Pi0Pipeline`) | yes | no | `generic` should fit | Not run through the gateway |
| GR00T N1.7 | yes | no | no | Output is not one action array; needs its own adapter kind |
| DreamZero | yes (stateful) | no | no | Needs its own adapter kind and session stickiness |
| Cosmos3 policy | yes | no | not configured | |
| InternVLA-A1 | no (offline only) | no | no | Needs a deploy yaml with `policy_server_config`; demo route is a mock |

The checklist for adding a model is in
[`.claude/skills/vla-maas/references/models.md`](../../.claude/skills/vla-maas/references/models.md).

### What exists

- Gateway: everything listed at the top of this file (keys, routing, `/act`
  and OpenPI routes, dashboard, live swap, structured logs, metrics).
- Robot client: drivers `mock`, `piper` (real-arm run passed), `arx` (link
  runs on the real arm); safety modes `dry_run` / `slow` / `live`.
- On `feat/arx-deltapose-adapter`, not yet in `maas`: `eef_pose` delta
  adapter options in `render_gateway.py`, pipelined policy requests
  (`LoopConfig.in_flight`, keeps a 30 Hz loop fed across a ~230 ms RTT),
  chunks merged by absolute tick, optional EMA smoothing of targets.
- [Studio](../vla-maas-studio/README.md): browser console with mock, MuJoCo
  (Panda, UR5e, xArm7) and Piper twin physics, plus a real-robot mode.
- Simulation: LIBERO and robosuite fleets with per-episode video
  ([examples](../../examples/online_serving/vla_maas/README.md)).
- Deployment skills: `vla-maas-deploy-server` (one backend per GPU, including
  8× RTX 4090 on one node), `vla-maas-deploy-robot`, `vla-maas-local-pi05`.
- CI: `.github/workflows/vla-maas.yml` (protocol, robot, gateway and studio
  tests, CPU smoke).

### Open items

- [ ] ARX: verify PR #14 (step-wise `eef_pose` deltas in the tool frame) on
      the arm, `dry_run` then `slow`.
- [ ] ARX: confirm whether translation deltas are in the tool frame and
      whether the end point is `link6` or the TCP.
- [ ] Merge `feat/arx-deltapose-adapter` into `maas` once verified.
- [ ] GPU baselines: serial vs batched sweep and latency on real GPUs (only
      mock sweeps exist). Recipe numbers for π0.5 are 156–398 ms client E2E;
      the 100 ms target needs graph capture, fewer denoising steps and bf16.
- [ ] Live swap on GPU backends and real arms (tested on CPU mocks only).
- [ ] InternVLA-A1 online serving.
- [ ] Record per-item Piper results in the
      [studio TODO](../vla-maas-studio/TODO.md).
- [ ] Drivers: Franka (`franky`), single-arm ARX5 (`arx5-sdk`).
- [ ] Gateway: multi-tenancy (tenants, quotas, billing), built-in TLS (use a
      reverse proxy for now), health-checked backend failover, adapters for
      GR00T and DreamZero, a plain HTTP inference endpoint.

### Next: model cards and embodiment matching (planned, not started)

1. Model cards (read-only): `maas-model/1` model cards and embodiment cards,
   loading the current inline `adapter:` unchanged, GET endpoints and
   `ready.card_rev`.
2. Dashboard: model and embodiment tabs, contract view, edit with validation,
   diff and rollback.
3. Robot client: SDK registry, `detect()`, `robotctl detect` / `models`,
   automatic camera count and role detection.
4. Robot console: `robotctl ui`, a local wizard (connect, identify the arm,
   SDK, self-check, pick a model, run).

The protocol only gains additive fields; `maas-robot/1` stays.

### Later: hierarchical embodied systems

| Layer | Rate | Where | Status |
| --- | --- | --- | --- |
| System 2 planner (VLM/LLM, subtasks) | 0.1–1 Hz | cloud, `/v1/planner/<model>/chat/completions` | planned |
| System 1 VLA (action chunks) | 2–10 Hz | cloud, `/v1/robot/<model>/act` and `/openpi` | done |
| System 0 whole-body / low-level control | ≥ 500 Hz | on the robot or edge; MaaS defines the interface only | planned |
| Cross-layer sessions | — | gateway `/v1/sessions` | planned |

The robot-side runtime orchestrates the layers; the cloud layers stay
stateless. Open questions: which System 2 model (external API or
self-hosted), and which body and controller System 0 targets first.
