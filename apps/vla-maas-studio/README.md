# VLA MaaS Studio

A browser console for debugging a VLA Model-as-a-Service deployment. It runs
the simulators server-side, opens one OpenPI session per robot, and streams
camera frames plus telemetry to the browser.

```text
browser (scene picker, session grid, latency chart)
        │  JSON control + JPEG frames over one websocket
        ▼
apps/vla-maas-studio/studio.py
        │  OpenPI (msgpack) per robot, API key held server-side
        ▼
vla-maas-gateway  →  vllm serve ... --omni   (or mock_policy_server.py)
```

The simulator is only the observation source; the MaaS service is what is under
test. Keeping physics and the policy cameras on the server means the images
match training (LIBERO/robosuite), the browser never needs the API key, and one
process can drive 16–64 sessions.

## Quick start (CPU, no GPU)

```bash
bash apps/vla-maas-studio/run_demo.sh
# open http://127.0.0.1:19100
```

The script starts a mock policy backend, the gateway, and the studio, then
waits. `PYTHON=/path/to/python` overrides the interpreter.

The default `mock` physics is a 2D reach stand-in: the camera shows a grey
background with a red goal and a blue end effector, **not** an articulated
robot. Use the MuJoCo backend below to see real arms.

## See a robot arm (MuJoCo Menagerie)

Sparse-clone the arm assets once, then run the same demo with `PHYSICS=mujoco`:

```bash
pip install mujoco
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie ~/.cache/mujoco_menagerie
cd ~/.cache/mujoco_menagerie && git sparse-checkout set \
    franka_emika_panda universal_robots_ur5e ufactory_xarm7
cd -
PHYSICS=mujoco bash apps/vla-maas-studio/run_demo.sh
```

Pick `panda`, `ur5e`, or `xarm7` in the sidebar. The task is a tabletop reach:
the arm tracks a Cartesian target with damped-least-squares IK, a red sphere is
the target, and the mock policy sees exact red/blue pixel markers so it can
drive the arm from pixels. No GPU checkpoint is needed.

Notes:

- Rendering uses EGL by default (`MUJOCO_GL=egl`). If EGL is unavailable, set
  `MUJOCO_GL=osmesa` or another MuJoCo backend.
- The MuJoCo backend caps at 16 sessions because each session owns an OpenGL
  context.
- The images are a reach scene, not LIBERO. A LIBERO-trained checkpoint is out
  of distribution here; use this mode to inspect arms and the serving loop, not
  to claim policy quality.

## Manual start

```bash
# 1. mock backend (or point the gateway at a real vllm-omni server)
python3 examples/online_serving/vla_maas/mock_policy_server.py \
    --port 18001 --latency-ms 20 --max-batch 8 --max-wait-ms 5 --per-item-ms 2 \
    --action-scale 0.005

# 2. gateway
cat > /tmp/studio-gateway.yaml <<'EOF'
api_keys: [studio-key]
request_timeout_s: 5.0
models:
  pi05:
    backends:
      - ws://127.0.0.1:18001/v1/realtime/robot/openpi
EOF
python3 apps/vla-maas-gateway/gateway.py --config /tmp/studio-gateway.yaml \
    --host 127.0.0.1 --port 19000

# 3. studio
python3 apps/vla-maas-studio/studio.py \
    --gateway http://127.0.0.1:19000 --api-key studio-key
```

Dependencies: `websockets`, `msgpack`, `msgpack-numpy`, `numpy`, `pillow`,
`openpi-client`, and `pyyaml` for the gateway. `--physics libero` needs LIBERO
and robosuite; `--physics mujoco` needs `mujoco` and the Menagerie checkout
described above.

## Using it

- **Target**: either the gateway (`model` picked from `GET /v1/models`) or a
  direct backend websocket. Direct mode skips gateway auth, relay metrics and
  `episode_result` reporting.
- **Scene**: suite and tasks. With `--physics mock|libero` each robot is
  pinned to one LIBERO task for the run and rotates init states between
  episodes; with `--physics mujoco` the suite is `reach` and the selected arm
  is the embodiment. With more robots than selected tasks, several robots
  share a task.
- **Run**: robots (1–64), duration, control rate, actions executed per chunk,
  prefetch threshold, render resolution and frame rate.
- **Grid**: one tile per robot with the last camera frame, episode/success
  counts and per-session p50. Click a tile for full-resolution frames, the last
  action vector and the session error.
- **Metrics**: client-side p50/p99 and throughput, gateway relay p50 from
  `/metrics`, underruns and late ticks. Comparing client p50 with gateway relay
  p50 separates client overhead from gateway/model time.

The studio binds to `127.0.0.1` by default and holds the API key; do not expose
it publicly. The gateway itself is single-tenant and has no TLS.

## Caveats

- This is a debugging console, not a benchmark. All sessions share one Python
  process, so GIL contention can inflate `late_tick_pct`. Use
  `examples/online_serving/vla_maas/fleet_sim.py` for timing numbers.
- The mock backend's success rate (always high against `mock_libero`) says the
  observation/action round trip works. It says nothing about a real checkpoint
  (lessons L002). Use real LIBERO and a real checkpoint for policy quality.
- `frame_fps` is per session. At 64 sessions, start with 1–2 fps and 128 px
  thumbnails; the encoder thread drops frames rather than slowing the control
  loop.
- Real LIBERO creates one robosuite env per session. Start with a few robots
  and a GPU-backed renderer before trying 16–64.
- The MuJoCo reach task is kinematic IK, not physical contact; it is a
  visualization/load source, not a manipulation benchmark.
- Each browser watcher gets thumbnails for every session and full-resolution
  frames only for the session it opened.

## Extending to more arms

`scenes.py` is the extension point. The `mujoco` backend already ships Panda,
UR5e and xArm 7:

1. Add the arm to `MUJOCO_ARMS` with its Menagerie `model` path, `eef_body`,
   `dof`, and camera `view` offset.
2. Add the model directory to the sparse checkout.
3. Bind each arm to a checkpoint that accepts its camera keys, state width and
   action width. A policy trained for one embodiment will not control another;
   the studio validates the selected arm against `--physics` but cannot validate
   it against a checkpoint's `policy_server_config`.

For a new simulator (LIBERO is the existing example), implement the same
`reset`, `step`, `render`, `policy_obs` interface and return it from
`create_scene`. The runner, telemetry, and browser protocol do not change.
