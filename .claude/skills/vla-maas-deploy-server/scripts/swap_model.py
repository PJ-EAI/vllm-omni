# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Swap the checkpoint (and optionally the model type and adapter) behind one gateway route, live.

Blue/green by default: start new backends on spare GPUs with serve_pi05.sh,
point the route at them in the gateway YAML, ask the gateway to reload (it
probes the new backends and refuses the swap on any mismatch), wait until the
old backends have no sessions left, then stop the old processes.

    source deploy/keys/operator.env
    python3 swap_model.py --config deploy/gateway.yaml --model pi05_piper \\
        --checkpoint /ckpt/pi05-piper/step-30000 --gpus "4 5 6 7" --base-port 8101

    # another VLA on the same route: new deploy yaml, model type and adapter block
    python3 swap_model.py --config deploy/gateway.yaml --model pi05_piper \\
        --checkpoint InternRobotics/InternVLA-A1-3B --model-type internvla_a1 \\
        --deploy vllm_omni/deploy/internvla_a1.yaml --adapter-file adapter-internvla.yaml \\
        --gpus "4 5 6 7" --base-port 8101

    # no spare GPUs: pause the route, drain and stop the old servers, then start the new ones
    python3 swap_model.py ... --gpus "0 1 2 3" --base-port 8001 --stop-old-first

Old backends are stopped only once drained (``--drain-timeout-s``), and only
those started on this node by serve_pi05.sh (pid files under ``--backends-dir``).
``--cutover stop`` sends ``stop`` to robots still on the old revision instead of
waiting for them to disconnect. Every step is in the gateway's JSON log
(``route.revision``, ``backend.draining``, ``backend.drained``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parent


def _gateway(args: argparse.Namespace, path: str) -> tuple[int, dict]:
    req = urllib.request.Request(args.gateway.rstrip("/") + path, headers={"Authorization": f"Api-Key {args.key}"})
    try:
        with urllib.request.urlopen(req, timeout=args.probe_timeout_s + 30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def _write_config(path: Path, config: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "# Rewritten by .claude/skills/vla-maas-deploy-server/scripts/swap_model.py. Holds keys: keep it private.\n"
        + yaml.safe_dump(config, sort_keys=False)
    )
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def _reload(args: argparse.Namespace) -> dict:
    code, body = _gateway(args, f"/v1/admin/reload?cutover={args.cutover}")
    if code != 200:
        raise RuntimeError(f"gateway refused the reload ({code}): {json.dumps(body, indent=2)}")
    return body


def _wait_drained(args: argparse.Namespace) -> bool:
    deadline = time.time() + args.drain_timeout_s
    while True:
        _, body = _gateway(args, "/v1/admin/revisions")
        left = sum(b["active"] for b in body.get("draining", {}).get(args.model, []))
        if left == 0:
            return True
        if time.time() > deadline:
            print(f"{left} session(s) still on the old revision after {args.drain_timeout_s:.0f} s", file=sys.stderr)
            return False
        print(f"waiting for {left} session(s) on the old revision to finish")
        time.sleep(5)


def _port(url: str) -> int | None:
    match = re.match(r"^wss?://[^/:]+:(\d+)/", url)
    return int(match[1]) if match else None


def _stop_backends(urls: list[str], backends_dir: Path) -> list[str]:
    """Stop the local serve_pi05.sh processes behind ``urls``; returns the URLs not found here."""
    missing = []
    for url in urls:
        port = _port(url)
        pids = []
        for pid_file in backends_dir.rglob(f"backend-{port}.pid") if port else []:
            try:
                pid = int(pid_file.read_text().strip())
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            except (OSError, ValueError):
                continue  # stale pid file
            if str(port).encode() in cmdline:  # not a recycled pid
                pids.append(pid)
        if not pids:
            missing.append(url)
        for pid in pids:
            os.kill(pid, signal.SIGTERM)
            print(f"stopped pid {pid} ({url})")
    return missing


def _start_backends(args: argparse.Namespace, log_dir: Path, model_type: str | None) -> list[str]:
    env = {
        **os.environ,
        "CKPT": args.checkpoint,
        "GPUS": args.gpus,
        "BASE_PORT": str(args.base_port),
        "LOG_DIR": str(log_dir),
        "MOCK": "1" if args.mock else os.environ.get("MOCK", "0"),
    }
    if args.deploy:
        env["DEPLOY"] = str(Path(args.deploy).resolve())
    if model_type:  # what mock backends report; a real backend reports its pipeline class
        env.setdefault("MODEL_TYPE", model_type)
    out = subprocess.run(
        ["bash", str(SCRIPTS / "serve_pi05.sh")], env=env, check=True, capture_output=True, text=True
    ).stdout
    print(out, end="")
    line = next(line for line in out.splitlines() if line.startswith("BACKENDS="))
    return line.removeprefix("BACKENDS=").split(",")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="the gateway YAML the running gateway was started with")
    p.add_argument("--model", required=True, help="route to swap, e.g. pi05_piper")
    p.add_argument("--checkpoint", required=True, help="HF id or path the new backends serve")
    p.add_argument("--model-type", help="new model type label (default: keep the route's label)")
    p.add_argument("--deploy", help="deploy yaml for the new backends (default: serve_pi05.sh's default)")
    p.add_argument("--adapter-file", help="YAML file holding the route's new adapter block")
    p.add_argument("--gpus", required=True, help='GPU ids for the new backends, e.g. "4 5 6 7"')
    p.add_argument("--base-port", type=int, required=True, help="first port of the new backends")
    p.add_argument("--stop-old-first", action="store_true", help="free the GPUs first (the route pauses meanwhile)")
    p.add_argument("--cutover", choices=["drain", "stop"], default="drain")
    p.add_argument("--drain-timeout-s", type=float, default=900.0)
    p.add_argument("--gateway", default="http://127.0.0.1:9000")
    p.add_argument("--key-env", default="VLA_MAAS_OPERATOR_KEY", help="env var holding the operator key")
    p.add_argument("--backends-dir", default="maas-backends", help="where serve_pi05.sh wrote pid files")
    p.add_argument("--probe-timeout-s", type=float, default=15.0)
    p.add_argument("--mock", action="store_true", help="start CPU mock backends (rehearsal)")
    args = p.parse_args(argv)
    args.key = os.environ.get(args.key_env)
    if not args.key:
        p.error(f"set {args.key_env} (source deploy/keys/operator.env)")

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    route = (config.get("models") or {}).get(args.model)
    if route is None:
        p.error(f"{args.model!r} is not a model in {config_path}")
    old_urls = list(route["backends"])
    backup = config_path.with_name(f"{config_path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    backup.write_text(config_path.read_text())
    os.chmod(backup, 0o600)
    print(f"backup: {backup}")
    backends_dir = Path(args.backends_dir)
    log_dir = backends_dir / f"{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"

    if args.stop_old_first:
        route["backends"] = []
        _write_config(config_path, config)
        _reload(args)
        print(f"{args.model} paused; new sessions get 503 until the new backends are up")
        if not _wait_drained(args) and args.cutover != "stop":
            print("old sessions did not finish; re-run with --cutover stop or a longer --drain-timeout-s")
            return 1
        _stop_backends(old_urls, backends_dir)

    try:
        new_urls = _start_backends(args, log_dir, args.model_type or route.get("model_type"))
    except subprocess.CalledProcessError as exc:
        print(exc.stdout, exc.stderr, sep="\n", file=sys.stderr)
        print(f"new backends did not start; the gateway config is unchanged except for --stop-old-first ({backup})")
        return 1
    route["backends"] = new_urls
    route["checkpoint"] = args.checkpoint
    if args.model_type:
        route["model_type"] = args.model_type
    if args.adapter_file:
        route["adapter"] = yaml.safe_load(Path(args.adapter_file).read_text())
    _write_config(config_path, config)
    try:
        body = _reload(args)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        config_path.write_text(backup.read_text())
        os.chmod(config_path, 0o600)
        _stop_backends(new_urls, backends_dir)
        if args.stop_old_first:
            print("restored the config, but the old backends are already stopped: restart them, then reload")
        else:
            print("restored the config; the route still serves the old revision")
        return 1
    change = next(c for c in body["changes"] if c["model"] == args.model)
    print(f"{args.model}: {change['from']['revision']} -> {change['to']['revision']} ({change['to']['checkpoint']})")
    if args.stop_old_first:
        return 0
    if not _wait_drained(args):
        print("old backends left running; stop them once their sessions end")
        return 0
    for url in _stop_backends(old_urls, backends_dir):
        print(f"not started from {backends_dir} on this node; stop it where it runs: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
