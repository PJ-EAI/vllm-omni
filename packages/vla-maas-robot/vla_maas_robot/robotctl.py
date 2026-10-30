# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``robotctl``: check, run and home one robot against the VLA MaaS.

    robotctl check --robot robots/mock.yaml
    robotctl run   --robot robots/mock.yaml --instruction "pick up the red cube" --mode dry_run
    robotctl home  --robot robots/mock.yaml --yes

``run`` defaults to the robot file's ``safety.start_mode`` (``dry_run`` unless
set). ``slow`` and ``live`` ask for confirmation unless ``--yes`` is given.
Ctrl+C holds the arm and ends the run.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from contextlib import ExitStack
from pathlib import Path

from vla_maas_protocol import ProtocolError

from vla_maas_robot.adapter import RobotAdapter
from vla_maas_robot.cameras import Camera, make_camera
from vla_maas_robot.client import PolicyClient
from vla_maas_robot.config import RobotConfig, load_robot_config
from vla_maas_robot.driver import RobotDriver, make_driver
from vla_maas_robot.loop import ControlLoop
from vla_maas_robot.safety import MODES, SafetyFilter


def _open(cfg: RobotConfig, stack: ExitStack) -> tuple[RobotDriver, dict[str, Camera]]:
    driver = make_driver(cfg.driver)
    driver.connect()
    stack.callback(driver.close)
    cameras = {}
    for role, cam_cfg in cfg.cameras.items():
        cam = make_camera(cam_cfg)
        cam.start()
        stack.callback(cam.stop)
        cameras[role] = cam
    return driver, cameras


def _connect(cfg: RobotConfig, adapter: RobotAdapter, stack: ExitStack) -> PolicyClient:
    client = PolicyClient(
        cfg.gateway.url,
        cfg.gateway.model,
        cfg.robot_id,
        cfg.gateway.api_key,
        embodiment=adapter.spec.embodiment,
        request_timeout_s=cfg.gateway.request_timeout_s,
    )
    server = client.connect(adapter.hello(cfg.robot_id, jpeg=cfg.control.jpeg_quality is not None))
    stack.callback(client.close)
    adapter.configure(server)
    return client


def cmd_check(cfg: RobotConfig, _: argparse.Namespace) -> int:
    with ExitStack() as stack:
        driver, cameras = _open(cfg, stack)
        state = driver.read_state()
        joints = state.joint_pos.round(3).tolist()
        print(f"driver ok: {driver.spec.embodiment}, dof={driver.spec.dof}, joint_pos={joints}")
        for role, cam in cameras.items():
            _, frame = cam.read()
            print(f"camera {role}: {frame.shape[1]}x{frame.shape[0]}")
        adapter = RobotAdapter(driver.spec, list(cameras), cfg.calibration)
        try:
            client = _connect(cfg, adapter, stack)
        except ProtocolError as exc:
            print(f"gateway rejected this robot for {cfg.gateway.model!r}: {exc}")
            return 2
        s = client.server
        print(f"gateway ok: model={s.model} protocol={s.protocol} session={client.session_id}")
        print(f"  serving revision={client.ready.get('revision')} checkpoint={client.ready.get('checkpoint')}")
        print(f"  requires images={s.requires.images} state={s.requires.state}")
        print(f"  provides space={s.provides.space} horizon={s.provides.horizon} control_hz={s.provides.control_hz}")
    return 0


def _confirm(mode: str, assume_yes: bool) -> bool:
    if mode == "dry_run" or assume_yes:
        return True
    reply = input(f"About to move the arm in {mode!r} mode. Clear the workspace, keep the e-stop at hand. Type yes: ")
    return reply.strip().lower() == "yes"


def cmd_run(cfg: RobotConfig, args: argparse.Namespace) -> int:
    mode = args.mode or cfg.safety.start_mode
    if not _confirm(mode, args.yes):
        print("aborted")
        return 1
    stop = threading.Event()
    out = Path(args.output).open("a") if args.output else None
    failed = False
    with ExitStack() as stack:
        previous = signal.signal(signal.SIGINT, lambda *_: stop.set())
        stack.callback(signal.signal, signal.SIGINT, previous)
        if out:
            stack.callback(out.close)
        driver, cameras = _open(cfg, stack)
        adapter = RobotAdapter(driver.spec, list(cameras), cfg.calibration)
        client = _connect(cfg, adapter, stack)
        loop_hz = cfg.control.control_hz or client.server.provides.control_hz
        safety = SafetyFilter(driver.spec, cfg.safety, loop_hz)
        safety.mode = mode
        loop = ControlLoop(driver, cameras, adapter, safety, client, cfg.control)
        print(
            f"{cfg.robot_id} -> {client.server.model} at {loop.control_hz:g} Hz, mode={mode}, "
            f"session={client.session_id} revision={client.ready.get('revision')}"
        )
        for episode in range(args.episodes):
            if stop.is_set():
                break
            stats = loop.run_episode(args.instruction, args.max_steps, stop)
            success = None
            if args.ask_success and not stop.is_set():
                success = input("success? [y/N] ").strip().lower() == "y"
                client.episode_result(success)
            line = {
                "robot_id": cfg.robot_id,
                "model": client.server.model,
                "session_id": client.session_id,
                "revision": client.ready.get("revision"),
                "checkpoint": client.ready.get("checkpoint"),
                "mode": mode,
                "episode": episode,
                "success": success,
                **stats.summary(),
                "safety": {
                    "clamped": safety.stats.clamped,
                    "worst_streak": safety.stats.worst_streak,
                    "reasons": safety.stats.reasons,
                },
            }
            print(json.dumps(line))
            if out:
                out.write(json.dumps(line) + "\n")
                out.flush()
            failed |= bool(stats.stopped and stats.stopped != "operator")
    return 1 if failed else 0


def cmd_home(cfg: RobotConfig, args: argparse.Namespace) -> int:
    if not _confirm("home", args.yes):
        print("aborted")
        return 1
    with ExitStack() as stack:
        driver, _ = _open(RobotConfig(**{**cfg.__dict__, "cameras": {}}), stack)
        driver.home()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robotctl", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run", "home"):
        p = sub.add_parser(name)
        p.add_argument("--robot", required=True, help="robot YAML")
        p.add_argument("--yes", action="store_true", help="skip the motion confirmation")
    run = sub.choices["run"]
    run.add_argument("--instruction", required=True)
    run.add_argument("--episodes", type=int, default=1)
    run.add_argument("--max-steps", type=int, default=600)
    run.add_argument("--mode", choices=MODES, default=None)
    run.add_argument("--ask-success", action="store_true", help="ask the operator after each episode")
    run.add_argument("--output", default=None, help="append one JSON line per episode")
    args = parser.parse_args(argv)
    cfg = load_robot_config(args.robot)
    return {"check": cmd_check, "run": cmd_run, "home": cmd_home}[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
