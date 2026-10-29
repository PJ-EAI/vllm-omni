# vla-maas-robot

The robot-side client of the VLA MaaS. Install it on the PC that drives the
arm; it depends only on `vla-maas-protocol`, never on vllm, torch or the
gateway, and talks to the cloud only through the
[standard Robot protocol](../vla-maas-protocol/spec/robot_protocol.md).

```bash
pip install ./packages/vla-maas-protocol ./packages/vla-maas-robot
pip install "./packages/vla-maas-robot[jpeg,realsense]"   # optional extras per site
```

| Extra | Adds |
| --- | --- |
| `jpeg` | Pillow: JPEG images on slow links, better resizing |
| `uvc` | OpenCV: USB cameras |
| `realsense` | pyrealsense2: Intel RealSense cameras |
| `piper` | piper_sdk: AgileX Piper over CAN |

## Parts

| Module | Role |
| --- | --- |
| `driver.py`, `drivers/` | `RobotDriver` per arm: SI units, absolute targets, vendor SDK behind it |
| `cameras.py` | Threaded cameras keyed by protocol role (`front`, `wrist`, ...) |
| `adapter.py` | Hardware data to and from the protocol: resizing, gripper 0-1, base-frame calibration |
| `safety.py` | Joint limits, per-tick step limits, workspace box, `dry_run` / `slow` / `live` |
| `loop.py` | Fixed-rate loop; chunks aligned to ticks so rows that elapsed during inference are dropped |
| `client.py` | Blocking protocol client (`wss://` supported) |
| `robotctl.py` | `check`, `run`, `home` |

| Driver | Arm | Extra | Example |
| --- | --- | --- | --- |
| `mock` | CPU-only simulated arm | none | [robots/mock.yaml](robots/mock.yaml) |
| `piper` | AgileX Piper over CAN (`piper_sdk`) | `piper` | [robots/piper.yaml](robots/piper.yaml) |
| `arx` | ARX X5 over slcan (realarm's `bimanual` stack, Python 3.11 venv) | none | [robots/arx.yaml](robots/arx.yaml) |

Franka (via `franky`) comes next. Each driver is one file under
`drivers/`, registered in `driver.DRIVERS`.

### AgileX Piper

1. Bring the CAN port up with the SDK's `can_activate.sh can0 1000000`.
2. Install `pip install "./packages/vla-maas-robot[piper,realsense]"`.
3. Copy [robots/piper.yaml](robots/piper.yaml) and set the gateway URL, cameras
   and workspace box.
4. Run `robotctl check`, then `run` in `dry_run`, then `--mode slow`, then `live`.

Notes:

- `connect` enables the motors in MOVE J and does not move the arm or the
  gripper.
- `close` leaves the motors enabled, because a disabled Piper drops under
  gravity.
- If CAN feedback stops for `feedback_timeout_s`, `read_state` raises and the
  episode stops.
- Piper rests at its joint-2 and joint-3 limits (0 rad), so the example sets
  `joint_margin_rad: 0`.
- The driver was tested against a fake SDK only. Check joint signs, the
  end-pose Euler convention and gripper scale on the real arm in `dry_run`
  first.

## Use

One YAML per robot; see [robots/mock.yaml](robots/mock.yaml). It names the
gateway route (`gateway.model`) and nothing else about the model.

```bash
export VLA_MAAS_API_KEY=<this robot's key>
robotctl check --robot robots/mock.yaml        # hardware, cameras, handshake; never moves
robotctl run --robot robots/mock.yaml --instruction "pick up the red cube"            # dry_run
robotctl run --robot robots/mock.yaml --instruction "pick up the red cube" --mode slow
```

`run` starts in the file's `safety.start_mode` (`dry_run` unless set): commands
are filtered and counted but not sent. `slow` and `live` ask for confirmation.
Ctrl+C holds the arm. Each episode prints one JSON line with ticks, commands,
underruns, dropped rows, round trip, server and network time, and safety
counters (`--output` appends them to a file).

The software safety layer does not replace the hardware e-stop.

## Tests

```bash
pip install "./packages/vla-maas-protocol" "./packages/vla-maas-robot[test]"
cd packages/vla-maas-robot && python -m pytest -q
```
