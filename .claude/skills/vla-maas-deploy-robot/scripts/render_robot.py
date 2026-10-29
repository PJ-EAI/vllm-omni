# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Write one robot YAML for a Piper or an ARX X5 on a remote gateway.

    python3 render_robot.py --out piper-01.yaml --robot-id piper-01 \
        --gateway https://maas.example.com --front-serial 123456789 --wrist-device 0

    python3 render_robot.py --kind arx --out arx-01.yaml --robot-id arx-01 \
        --gateway ws://192.168.1.10:9000 --can can1 \
        --bimanual-path /home/me/realarm/src/envs/arx_x5 --front-serial 123456789

    # CPU rehearsal against the same gateway: mock arm and mock cameras
    python3 render_robot.py --out piper-01-mock.yaml --robot-id piper-01 --gateway https://maas.example.com --mock

The key is never written to the file: robotctl reads it from VLA_MAAS_API_KEY
(source the robot's .env from the server operator).
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
ROBOTS = ROOT / "packages/vla-maas-robot/robots"
KINDS = {"piper": "pi05_piper", "arx": "pi05_arx"}
READY_POSE = [0.0, 1.0, -1.0, 0.0, 0.4, 0.0]  # off the joint-2/3 rest limits; confirm on the arm


def render(args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(yaml.safe_load((ROBOTS / f"{args.kind}.yaml").read_text()))
    cfg["robot_id"] = args.robot_id
    cfg["gateway"] = {
        "url": args.gateway,
        "model": args.model,
        "api_key_env": "VLA_MAAS_API_KEY",
        "request_timeout_s": args.request_timeout_s,
    }
    if args.mock:
        cfg["driver"] = {"kind": "mock", "dof": 6, "embodiment": args.kind}
        cfg["cameras"] = {
            "front": {"kind": "mock", "width": 640, "height": 480, "fps": 30},
            "wrist": {"kind": "mock", "width": 640, "height": 480, "fps": 30, "seed": 1},
        }
    else:
        if args.kind == "piper":
            home = args.home_joints if args.home_joints is not None else READY_POSE
            cfg["driver"].update(can_name=args.can, speed_pct=args.speed_pct, home_joints=home)
        else:  # arx: the vendor stack lives on the arm PC, not in any venv
            cfg["driver"].update(can_port=args.can, bimanual_path=args.bimanual_path, arm_type=args.arm_type)
            if args.home_joints is not None:
                cfg["driver"]["home_joints"] = args.home_joints
        front = {"kind": "realsense", "width": 640, "height": 480, "fps": 30}
        if args.front_serial:
            front["serial"] = args.front_serial
        wrist = {"kind": "uvc", "device": args.wrist_device, "width": 640, "height": 480, "fps": 30}
        cfg["cameras"] = {"front": front, "wrist": wrist}
    # Public link: hide one inference round trip behind the actions still queued.
    cfg["control"]["prefetch_threshold"] = args.prefetch
    cfg["control"]["jpeg_quality"] = args.jpeg_quality
    cfg["safety"]["start_mode"] = "dry_run"
    if args.workspace:
        x0, x1, y0, y1, z0, z1 = args.workspace
        cfg["safety"]["workspace"] = {"x": [x0, x1], "y": [y0, y1], "z": [z0, z1]}
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--robot-id", required=True, help="must match robots.<id> in the gateway yaml")
    p.add_argument("--gateway", required=True, help="public gateway URL, https://... or wss://...")
    p.add_argument("--kind", choices=sorted(KINDS), default="piper", help="arm to render (template robots/<kind>.yaml)")
    p.add_argument("--model", default=None, help=f"gateway route; default by kind: {KINDS}")
    p.add_argument("--can", default="can0", help="CAN interface: can_name for piper, can_port for arx")
    p.add_argument(
        "--bimanual-path",
        default=None,
        help="arx only: directory containing the vendor bimanual package (realarm/src/envs/arx_x5)",
    )
    p.add_argument("--arm-type", type=int, default=0, choices=(0, 1, 2), help="arx only: 0 x5, 1 x5_master, 2 x5_2025")
    p.add_argument("--speed-pct", type=int, default=30)
    p.add_argument("--home-joints", type=float, nargs=6, default=None, help="default: piper READY_POSE, else the template's")
    p.add_argument("--front-serial", default=None, help="RealSense serial for the front camera")
    p.add_argument("--wrist-device", default="0", help="UVC index or /dev/video path for the wrist camera")
    p.add_argument("--prefetch", type=int, default=20, help="rows left when the next request starts")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--request-timeout-s", type=float, default=5.0)
    p.add_argument("--workspace", type=float, nargs=6, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"))
    p.add_argument("--mock", action="store_true", help="mock arm and cameras for a CPU rehearsal")
    args = p.parse_args(argv)
    if not args.gateway.startswith(("https://", "wss://", "http://", "ws://")):
        p.error("--gateway must start with https://, wss://, http:// or ws://")
    if args.kind == "arx" and not args.mock and not args.bimanual_path:
        p.error("--bimanual-path is required for --kind arx (realarm's vendor stack on the arm PC)")
    if args.wrist_device.isdigit():
        args.wrist_device = int(args.wrist_device)
    args.model = args.model or KINDS[args.kind]
    cfg = render(args)
    Path(args.out).write_text(
        f"# Generated by .claude/skills/vla-maas-deploy-robot/scripts/render_robot.py from robots/{args.kind}.yaml\n"
        + yaml.safe_dump(cfg, sort_keys=False)
    )
    print(f"wrote {args.out} ({'mock' if args.mock else args.kind} driver, model {args.model}, start_mode dry_run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
