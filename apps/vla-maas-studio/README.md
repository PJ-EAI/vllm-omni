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

## AgileX Piper on the standard Robot protocol

`--robot-config` runs one robot file from
[`packages/vla-maas-robot`](../../packages/vla-maas-robot/README.md) (for example
[`robots/piper.yaml`](../../packages/vla-maas-robot/robots/piper.yaml)) on the
standard Robot protocol, `maas-robot/1` on `/v1/robot/<model>/act`, instead of
the native OpenPI relay. The studio drives it with the robot client's own
`RobotAdapter`, `SafetyFilter`, `PolicyClient` and `ControlLoop`, so a session
in the browser does what `robotctl run` does with that file: the same
handshake, model adapter, safety modes, tick-aligned chunks and hold on stop.
The studio adds only observation (camera frames, joint state, commands, round
trip, safety counters).

```bash
# CPU demo: mock backend behind a Piper-shaped /act adapter, gateway, studio with the twin
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie ~/.cache/mujoco_menagerie
(cd ~/.cache/mujoco_menagerie && git sparse-checkout set agilex_piper)
pip install mujoco ./packages/vla-maas-protocol ./packages/vla-maas-robot
PHYSICS=piper bash apps/vla-maas-studio/run_demo.sh

# Against your gateway, MuJoCo twin first
python3 apps/vla-maas-studio/studio.py --robot-config my-piper.yaml

# Then the real arm (one session; slow/live need the browser confirmation)
pip install "./packages/vla-maas-robot[piper,realsense,uvc]"
python3 apps/vla-maas-studio/studio.py --robot-config my-piper.yaml --hardware
```

- **Gateway and key** come from the robot file (`gateway.url`,
  `gateway.api_key_env`); `--gateway` and `--api-key` override them. A robot
  key can use `/act` but cannot list models, so the model picker then offers
  only the file's `gateway.model`. Only models with an `adapter` are listed.
- **Twin (default)**: `piper_sim.py` builds `PiperSimDriver` from the same
  `driver:` block and copies the real `PiperDriver`'s spec (joint limits,
  gripper stroke 0–70 mm, modes), so the handshake and the safety filter see
  what they will see on hardware. Joints track targets kinematically at
  `speed_pct`, end-pose targets go through IK on `link6`, and the YAML's camera
  roles are rendered (`front` looks at the table, `wrist` rides on `link6`).
  It starts at `home_joints`, where `robotctl home` leaves the real arm. Up to 8
  sessions; with more than one, robot ids get a `-<n>` suffix and need an
  operator key.
- **Hardware (`--hardware`)**: the file's own driver and cameras, one session
  with the file's `robot_id`. The safety mode defaults to the file's
  `safety.start_mode` (`dry_run`: commands are filtered and counted, nothing is
  sent); `slow` and `live` need the confirmation box. **Stop** sets the
  control loop's stop event: the arm holds and the motors stay enabled, as with
  Ctrl+C in `robotctl`. The studio never homes the arm; use `robotctl home`.
- **Detail view**: all camera roles side by side, joint positions, gripper,
  end-effector position, the last command after the safety filter, clamp
  counts and reasons, round trip / server / network p50, and the handshake's
  `requires` / `provides`.
- **Real-arm checks still to do** are tracked in [TODO.md](TODO.md).
- **Order for a first real run**: twin with the real file, then `--hardware`
  in `dry_run`, then `slow`, then `live`.

Caveats:

- The twin is kinematic: no contact, no gravity, no grasping. It checks the
  route, the adapter, units and safety settings, not policy quality (lessons
  L002). Rendered images are not real camera images.
- Piper rests with joints 2 and 3 on their limits (0 rad). With the example
  file's `joint_margin_rad: 0`, any target below those limits is clamped, and
  five clamped commands in a row stop the episode. A delta-joint model started
  from the rest pose can hit this; start from a ready pose (`home_joints` plus
  `robotctl home`). `run_demo.sh` does this for the twin.
- `eef_pose` in the twin is the `link6` frame; the real driver reports the
  SDK's end pose. Check both against each other before relying on
  `eef_pose` actions.
- `success_rate` stays empty: nothing judges success here. Use
  `robotctl run --ask-success` for success counts.

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
