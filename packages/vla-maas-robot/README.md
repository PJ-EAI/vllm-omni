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

Drivers available now: `mock`. Hardware drivers (AgileX Piper, ARX5, Franka via
`franky`) are added one per file under `drivers/` and registered in
`driver.DRIVERS`.

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
