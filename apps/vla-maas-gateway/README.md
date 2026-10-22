# VLA MaaS gateway

A single-tenant gateway that exposes vllm-omni VLA policy servers as one API.
It sits in front of one or more `vllm serve ... --omni` instances (or the CPU
mock in `examples/online_serving/vla_maas/mock_policy_server.py`) and adds:

- **API-key auth**: `Authorization: Api-Key <key>` (or `Bearer <key>`), the header
  `openpi-client` sends when constructed with `api_key=`.
- **Model routing**: `ws://<gateway>/v1/robot/<model>/openpi`; each session goes to
  the least-loaded backend configured for that model.
- **Transparent OpenPI relay**: frames pass through unchanged; the handshake
  metadata gains `model` and `backend`.
- **Dashboard**: `GET /dashboard`, a live panel that polls `/metrics` with the key
  typed into it.
- **Metrics**: `GET /metrics` returns per-model request/error counts, active
  sessions and relay p50/p99. `GET /v1/models` lists models, `GET /healthz` is open.

Errors: 401 for a missing or wrong key, 404 for an unknown model, websocket close
1011 when a backend is unreachable, and an OpenPI `{"type": "error"}` reply when a
backend times out (`request_timeout_s`).

## One-command demo

```bash
bash apps/vla-maas-gateway/run_demo.sh 10   # episodes per task
```

The script uses `python3`; set `PYTHON=/path/to/python` to use an interpreter
from a virtualenv instead.

Starts two mock backends (`pi05` at 60 ms, `internvla_a1` at 90 ms), the gateway,
and a mock-LIBERO client per model at 50 Hz. Open `http://localhost:9000/dashboard`
and enter the API key (`dev-key-change-me`) to watch success rate, p50/p99 relay
latency against the 100 ms budget, requests, sessions and errors per model.
Clients run with `--report-episodes`, which sends
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

## Not yet covered

Multi-tenancy (per-tenant keys, quotas, billing), TLS termination (put it behind
a reverse proxy), health-checked backend failover, and a plain HTTP inference
endpoint for clients that cannot hold a websocket.
