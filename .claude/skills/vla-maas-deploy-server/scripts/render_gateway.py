# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Write a gateway YAML for one Piper-shaped π0.5 route and a fleet of robots.

Generates an operator key and one key per robot (kept if the output files
already exist, so re-running adds robots without rotating keys), then writes:

    <out>/gateway.yaml       the gateway config (mode 600: it holds the keys)
    <out>/keys/<robot>.env   VLA_MAAS_API_KEY=... to copy to that robot PC
    <out>/keys/operator.env  VLA_MAAS_OPERATOR_KEY=... for curl / dashboard

    python3 render_gateway.py --out deploy/ --model pi05_piper \
        --backends ws://gpu-1:8001/v1/realtime/robot/openpi,ws://gpu-1:8002/v1/realtime/robot/openpi \
        --robots piper-01,piper-02 --state-dim 32 --control-hz 30
"""

from __future__ import annotations

import argparse
import os
import secrets
import string
from pathlib import Path

import yaml

ID_CHARS = set(string.ascii_letters + string.digits + "_.-")


def valid_robot_id(rid: str) -> bool:
    return 0 < len(rid) <= 64 and set(rid) <= ID_CHARS and rid[0] not in "_.-"


def _read_key(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return None


def _key(path: Path, name: str) -> str:
    key = _read_key(path, name) or secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{name}={key}\n")
    path.chmod(0o600)
    return key


def piper_adapter(args: argparse.Namespace) -> dict:
    """π0.5 with a front and a wrist camera, 6 joints + gripper, absolute joint actions.

    A LeRobot fine-tune's normalization and relative-to-absolute conversion run
    in the vllm-omni server, so the layout here is absolute.
    """
    adapter = {
        "kind": "generic",
        "images": {"front": args.front_key, "wrist": args.wrist_key},
        "image_size": [224, 224],
        "state": ["joint_pos", {"field": "gripper", "range": args.gripper_range}],
        "action": {
            "space": "joint_pos",
            "layout": ["joint_pos", {"field": "gripper", "range": args.gripper_range}],
        },
        "control_hz": args.control_hz,
        "embodiments": [e.strip() for e in args.embodiments.split(",") if e.strip()],
    }
    if args.state_dim:
        adapter["state_dim"] = args.state_dim
    if args.denoise_steps:
        adapter["native_extra"] = {"sampling_params": {"num_inference_steps": args.denoise_steps}}
    return adapter


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--model", default="pi05_piper", help="gateway route name (robots put it in gateway.model)")
    p.add_argument("--backends", required=True, help="comma-separated ws://host:port/v1/realtime/robot/openpi URLs")
    p.add_argument("--robots", required=True, help="comma-separated robot ids, e.g. piper-01,piper-02")
    p.add_argument(
        "--embodiments",
        default="piper",
        help="comma-separated embodiments the /act handshake allows, e.g. piper or arx",
    )
    p.add_argument("--front-key", default="observation.images.base_0_rgb", help="checkpoint camera key for 'front'")
    p.add_argument("--wrist-key", default="observation.images.left_wrist_0_rgb", help="checkpoint key for 'wrist'")
    p.add_argument("--state-dim", type=int, default=32, help="checkpoint observation.state width; 0 = no padding")
    p.add_argument("--control-hz", type=float, default=30.0, help="rate the training data was recorded at")
    p.add_argument(
        "--gripper-range",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        metavar=("CLOSED", "OPEN"),
        help="gripper values in the training data at closed and open",
    )
    p.add_argument(
        "--denoise-steps", type=int, default=0, help="override num_inference_steps per request; 0 = server default"
    )
    p.add_argument("--request-timeout-s", type=float, default=5.0)
    args = p.parse_args(argv)
    if not [e for e in args.embodiments.split(",") if e.strip()]:
        p.error("--embodiments must list at least one embodiment; an empty list would allow every robot")

    out = Path(args.out)
    robots = [r.strip() for r in args.robots.split(",") if r.strip()]
    bad = [r for r in robots if not valid_robot_id(r)]
    if bad:
        p.error(f"robot ids must be 1-64 letters, digits, '_', '.' or '-': {bad}")
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if not backends or not all(b.startswith(("ws://", "wss://")) for b in backends):
        p.error("--backends must be ws:// or wss:// URLs")

    operator = _key(out / "keys" / "operator.env", "VLA_MAAS_OPERATOR_KEY")
    config = {
        "api_keys": [operator],
        "request_timeout_s": args.request_timeout_s,
        "models": {args.model: {"backends": backends, "adapter": piper_adapter(args)}},
        "robots": {
            rid: {"api_key": _key(out / "keys" / f"{rid}.env", "VLA_MAAS_API_KEY"), "models": [args.model]}
            for rid in robots
        },
    }
    path = out / "gateway.yaml"
    path.write_text(
        "# Generated by .claude/skills/vla-maas-deploy-server/scripts/render_gateway.py. Holds keys: keep it private.\n"
        + yaml.safe_dump(config, sort_keys=False)
    )
    os.chmod(path, 0o600)
    print(f"wrote {path} ({len(robots)} robots, {len(backends)} backends)")
    print(f"robot keys: {out / 'keys'}/<robot_id>.env -> copy each to its robot PC only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
