# Robots on the standard protocol: robot client, Piper, studio

How to bring an arm onto the MaaS through `/v1/robot/<model>/act`, check it in
the studio without hardware, then run it for real. Code:
`packages/vla-maas-robot/` (client), `apps/vla-maas-studio/` (browser console).

## Pieces

| Piece | File | Role |
| --- | --- | --- |
| Robot file | `packages/vla-maas-robot/robots/<arm>.yaml` | `robot_id`, `gateway` (url, model, `api_key_env`), `driver`, `cameras` by protocol role, `calibration`, `control`, `safety` |
| Driver | `vla_maas_robot/drivers/<arm>.py`, registered in `driver.DRIVERS` | SI units, absolute targets; vendor SDK imported lazily |
| Safety filter | `vla_maas_robot/safety.py` | joint limits (+ `joint_margin_rad`), per-tick step, workspace box; `dry_run` / `slow` / `live`; `max_violations` consecutive clamps stop the episode |
| Control loop | `vla_maas_robot/loop.py` | fixed rate, tick-aligned chunks, prefetch, optional blend, hold on stop |
| CLI | `robotctl check | run | home --robot <yaml>` | `check` never moves; `run` defaults to the file's `start_mode` |
| Studio robot mode | `apps/vla-maas-studio/{robot_run,piper_sim}.py` | same adapter / safety / loop as `robotctl run`, plus camera frames and telemetry in a browser |

Drivers today: `mock` (CPU) and `piper` (AgileX Piper over CAN, `piper_sdk>=0.6`,
extra `piper`). ARX and Franka are next; each is one driver file, one yaml and
one extra.

## AgileX Piper specifics (`drivers/piper.py`)

- SDK units are 0.001 deg / 0.001 mm / 0.001 rad/s; the driver converts to SI.
  Joint targets go out as `JointCtrl` in MOVE J at `speed_pct`; end-pose
  targets (`modes: [joint_pos, eef_pose]`) as `EndPoseCtrl` in MOVE P. End pose
  is assumed roll-pitch-yaw: verify on the real arm.
- Gripper stroke 0–70 mm, protocol `gripper` 0 = closed, 1 = open.
- `connect` enables the motors without moving; `close` leaves them enabled (a
  disabled Piper drops under gravity) unless `disable_on_close: true`.
- Stale CAN feedback for `feedback_timeout_s` makes `read_state` raise, which
  stops the episode and holds.
- **Rest-pose pitfall:** Piper rests with joints 2 and 3 on their limits (0 rad).
  With the example's `joint_margin_rad: 0`, any target below those limits is
  clamped; five in a row stop the episode. A delta-joint model started at rest
  hits this. Set `home_joints` to a ready pose (e.g. `[0, 1.0, -1.0, 0, 0.4, 0]`,
  check on the arm) and run `robotctl home` first.

## Without hardware: the studio's Piper twin

`piper_sim.py` builds `PiperSimDriver` from the same `driver:` block and copies
`PiperDriver`'s `RobotSpec`, so the handshake, adapter and safety filter see
the real limits, gripper range and modes. Joints follow targets kinematically
at `speed_pct`; `eef_pose` targets go through IK on `link6`; `front` and `wrist`
cameras are rendered. It needs `mujoco` and the Menagerie `agilex_piper`
checkout (`git sparse-checkout set agilex_piper`).

```bash
# Environment with uv (do not `uv sync` at the repo root: that installs all of vllm-omni)
uv venv .venv-studio --python 3.12
uv pip install --python .venv-studio mujoco websockets msgpack msgpack-numpy pillow pyyaml numpy \
    ./packages/vla-maas-protocol ./packages/vla-maas-robot
uv pip install --python .venv-studio --no-deps openpi-client   # its dm-tree dependency often fails to build

# CPU demo: mock backend behind a Piper-shaped /act adapter (pi05_piper), gateway, studio
PYTHON=.venv-studio/bin/python PHYSICS=piper bash apps/vla-maas-studio/run_demo.sh
# -> http://127.0.0.1:19100 ; MUJOCO_GL=osmesa on hosts without a GPU

# Your own gateway / model
.venv-studio/bin/python apps/vla-maas-studio/studio.py --robot-config my-piper.yaml \
    --gateway http://<gw> --api-key <key>
```

The twin checks the route, adapter shapes and units, action direction and
size, safety settings and latency. It does not check contact, grasping or
policy quality on real images (lessons L002), and its `eef_pose` is the
`link6` frame, which may differ from the SDK end pose.

## Real-arm bring-up order

1. Gateway: the model has an `adapter:` (see `apps/vla-maas-gateway/gateway.example.yaml`,
   commented `pi05_piper`) with `embodiments: [piper]`; the robot has its own key
   under `robots.<id>`.
2. Robot file: copy `robots/piper.yaml`, set `gateway.url`, `api_key_env`,
   camera `serial` / `device`, and a ready-pose `home_joints`.
3. Twin: `studio.py --robot-config my-piper.yaml` until there are no clamps and
   actions look right.
4. `bash can_activate.sh can0 1000000`, `robotctl check`, `robotctl home`.
5. `studio.py --robot-config my-piper.yaml --hardware` (or `robotctl run`):
   `dry_run` first (filtered, not sent; compare joint signs, gripper mapping,
   eef pose), then `slow`, then `live`. The browser needs the
   "workspace clear" confirmation for `slow` / `live`. Stop holds with motors
   enabled; the studio never homes the arm.

## Tests

```bash
cd packages/vla-maas-robot && python -m pytest -q      # includes tests/test_piper.py (fake SDK)
cd apps/vla-maas-studio && python -m pytest -q         # robot mode; twin tests skip without mujoco/Menagerie
```
