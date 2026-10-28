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
  sessions and relay p50/p99. `GET /v1/models` lists models, `GET /healthz` is open.

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

## Not yet covered

Multi-tenancy (tenants, quotas, billing), TLS termination (put it behind a
reverse proxy), health-checked backend failover, adapters for models whose
native output is not one action array (GR00T, DreamZero), and a plain HTTP
inference endpoint for clients that cannot hold a websocket.
