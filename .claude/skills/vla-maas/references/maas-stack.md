# MaaS stack reference

## Gateway (`apps/vla-maas-gateway/gateway.py`)

Start: `python3 apps/vla-maas-gateway/gateway.py --config <yaml> [--host 0.0.0.0] [--port 9000]`.

Config:

```yaml
api_keys: [dev-key-change-me]     # or env VLA_MAAS_API_KEY (added to the list)
request_timeout_s: 5.0            # per inference reply; timeout → OpenPI error reply
models:
  pi05:
    backends:                     # several = replicas, least-loaded per session
      - ws://127.0.0.1:8001/v1/realtime/robot/openpi
```

| Endpoint | Auth | Returns |
| --- | --- | --- |
| `GET /healthz` | no | `{"status": "ok"}` |
| `GET /dashboard` | no (page asks for key) | HTML, polls `/metrics` |
| `GET /v1/models` | yes | `{"models": [{"id", "backends"}]}` |
| `GET /metrics` | yes | per model: `requests, errors, active_sessions, episodes, success_rate, relay_p50_ms, relay_p99_ms` |
| `WS /v1/robot/<model>/openpi` | yes | OpenPI relay; handshake metadata gains `model`, `backend` |

Auth header: `Authorization: Api-Key <key>` or `Bearer <key>`.
Errors: 401 bad/missing key, 404 unknown path or model, WS close 1011 when the
backend is unreachable, `{"type": "error", ...}` reply on backend timeout.
Clients may send `{"endpoint": "episode_result", "success": bool}`; the gateway
counts it and answers `{"status": "ok"}` without forwarding.

Routing is per session (websocket), not per request: a session stays on the
backend chosen at connect time.

## OpenPI wire protocol

Full spec: `docs/serving/openpi_api.md`. Summary:

- Connect → server sends msgpack `policy_server_config` dict.
- Infer → client sends msgpack obs dict, server sends msgpack ndarray
  `[action_horizon, action_dim]` (or `{"type": "error", "message"}`).
- Reset → `{"endpoint": "reset"}` → status reply.
- NumPy arrays use openpi-client `msgpack_numpy` markers.
- Idle connections close after `--robot-openpi-idle-timeout` s (default 30, 0 = off).

## Observation (π0.5)

```python
{
    "<camera key>": np.uint8[H, W, 3],   # keys must match checkpoint input_features order
    "state": np.float32[state_dim],      # width from input_features["observation.state"]
    "prompt": "pick up the black bowl",
    "session_id": "<uuid>",              # accepted, ignored by π0.5
    "sampling_params": {"num_inference_steps": 10},  # optional
    "seed": 0,                           # optional
}
```

LIBERO fine-tunes use two cameras; `pi05_base` uses
`observation.images.{base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb}`. Map other
names with `model_config.image_key_map` in the deploy yaml. The clients default
to `observation.images.image` / `observation.images.image2`; override with
`--base-camera-key` / `--wrist-camera-key`.

## Standard Robot protocol (`/act`)

`WS /v1/robot/<model>/act?robot_id=<id>[&embodiment=<e>]`, spec in
`packages/vla-maas-protocol/spec/robot_protocol.md`. Handshake:
`server_hello` (model `requires` cameras by role + state fields, `provides`
space/horizon/control_hz) → `robot_hello` → `ready` or `error` + close 4400.
Then `obs` → `actions` (absolute targets, `server_ms`) | `stop` | `error`.
Only models with an `adapter:` block in the gateway yaml are served here
(404 otherwise). Robot keys (`robots.<id>.api_key`) work only on `/act` for
that robot and its `models`; `api_keys` are operator keys.
`GET /v1/robots`, `GET /v1/robots/<id>/stop`, `GET /v1/models/<model>`.

Robot side: `pip install ./packages/vla-maas-protocol ./packages/vla-maas-robot`,
then `robotctl check|run|home --robot <yaml>` (`packages/vla-maas-robot/robots/mock.yaml`).
`run` defaults to `dry_run`.

## Client tools (`examples/online_serving/vla_maas/`)

All take `--host --port --path --api-key`. Direct to a server:
`--path /v1/realtime/robot/openpi` (default). Through the gateway:
`--port 9000 --path /v1/robot/<model>/openpi --api-key <k>`.

| Tool | Key flags | Output |
| --- | --- | --- |
| `mock_policy_server.py` | `--latency-ms --jitter-ms --max-batch --max-wait-ms --per-item-ms --action-scale` | CPU backend; batch of B costs `latency + (B-1)*per_item` |
| `libero_eval.py` | `--mock --task-suite --max-tasks --episodes-per-task --replan-steps --async-prefetch --control-hz --output` | JSON: `success_rate, per_task, inference_p50_ms, inference_p99_ms, num_inferences, underruns` |
| `fleet_sim.py` | `--robots --procs --duration-s --control-hz --replan-steps --async-prefetch --label --output --report-episodes` | JSONL line: `label, robots, duration_s, inferences, throughput_rps, latency_p50/p90/p99_ms, late_tick_pct, underruns, episodes, success_rate` |
| `latency_bench.py` | `--cameras --state-dim --image-size --num-inference-steps --warmup --iters --output` | JSON latency percentiles vs control budget |
| `concurrency_sweep.sh` | env `ROBOTS DURATION LATENCY_MS PER_ITEM_MS REPLICAS PYTHON` | JSONL over serial / replicas-N / batched |
| `mock_libero.py` | — | Dependency-free LIBERO stand-in |
| `sim_fleet.py` | `--scenario --duration-s --episodes --render-gpus --video-dir --video-grid --record-episodes --output --episodes-out` | JSONL per group + `all`: fleet_sim fields plus `group, model, env, task, robot, sim_step_p50_ms, videos, errors`; per-episode JSONL; MP4s |
| `sim_envs.py` | scenario `env: {kind: libero\|robosuite\|mock, robot, task, suite, task_ids, resolution, max_steps, video_camera}` | Simulator adapters, same obs layout as `libero_eval.py` |
| `video_grid.py` | `<video_dir> --cols --max-clips` | Grid MP4 of a run's clips |
| `sim_fleet_demo.sh` | env `GATEWAY_CONFIG MUJOCO_GL RENDER_GPUS OUT_DIR GW_PORT` | mock backends (or yours) + gateway + `sim_fleet.py` |

`--action-scale` must match between `mock_policy_server.py` and the mock env
(`fleet_sim.py --action-scale`), otherwise mock success drops.

## Studio (`apps/vla-maas-studio/`)

Browser console for the same stack: pick a LIBERO suite/tasks, run 1–64
concurrent sessions through the gateway or a direct backend, and watch camera
thumbnails, client p50/p99, gateway relay p50, underruns and per-session errors.
The studio runs physics server-side and holds the API key; the browser only
talks to the studio.

```bash
bash apps/vla-maas-studio/run_demo.sh   # mock backend + gateway + studio
# http://127.0.0.1:19100
```

`--physics libero` uses the real LIBERO/robosuite simulator; `--physics
mujoco` renders MuJoCo Menagerie arms (Panda, UR5e, xArm 7) in a tabletop
reach task and needs a Menagerie sparse checkout plus EGL. The default mock
physics has no articulated robot by design. It is a debugging console, not a
benchmark: sessions share one process, so use `fleet_sim.py` for timing numbers.

`--robot-config <robot.yaml>` switches the studio to the standard Robot
protocol (`/act`): it runs the robot file through `vla_maas_robot`'s adapter,
safety filter and control loop. The default source is a MuJoCo twin
(`piper_sim.py`, Piper only, spec copied from `PiperDriver`); `--hardware` uses
the file's real driver and cameras, one session, and needs a browser
confirmation for `slow`/`live`. `PHYSICS=piper bash apps/vla-maas-studio/run_demo.sh`
is the CPU demo.

## Latency budget

An async client stalls only if `latency > prefetch_threshold / control_hz`.
At 50 Hz with `--async-prefetch 10`, that is 200 ms. `late_tick_pct` and
`underruns` are the stall signals; throughput alone hides them.

## Client dependencies

```bash
pip install websockets msgpack msgpack-numpy pyyaml numpy typing_extensions
pip install --no-deps openpi-client   # the PyPI package pulls dm-tree, which often fails to build
```

CI pins `openpi-client` to OpenPI revision
`215abfb217dbac7d5f1273282331b9b1866c0479` (see `recipes/lerobot/Pi05.md`).
