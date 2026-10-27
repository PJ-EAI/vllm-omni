---
name: vla-maas-deploy-robot
description: Deploy the VLA MaaS robot side on an arm PC (AgileX Piper today) that reaches a remote gateway over the internet - install the standalone robot client, write the robot YAML, bring CAN up, run robotctl check, home and a dry_run episode, and read the latency and safety summary before any motion. Use for setting up or debugging a real arm against a deployed gateway; use vla-maas-deploy-server for the cluster side.
---

# Deploy the VLA MaaS robot side (one arm PC per robot)

## Purpose

This is the robot half of
[docs/serving/vla_maas_piper_deployment.md](../../../docs/serving/vla_maas_piper_deployment.md).
Each arm PC runs only `vla-maas-protocol` + `vla-maas-robot`: no vllm, no torch,
no gateway. It dials out to the gateway's public `https://` URL, so the arm's
network needs outbound 443 only. The server half is
[vla-maas-deploy-server](../vla-maas-deploy-server/SKILL.md).

## Inputs from the server operator

- Public URL, e.g. `https://maas.example.com`.
- Model route, `pi05_piper` by default.
- This arm's `robot_id`, and its key file `<robot_id>.env`. Keep the key file
  on this PC only.
- Whether the served checkpoint allows motion. `lerobot/pi05_base` is
  `dry_run` only, because its actions are not Piper joint angles.

## Steps

Run from a checkout of the `maas` branch on the arm PC.

1. **Install** the robot client into its own venv:

   ```bash
   bash .claude/skills/vla-maas-deploy-robot/scripts/install_robot.sh   # EXTRAS=jpeg for a mock rehearsal
   export PATH="$PWD/.venv-robot/bin:$PATH"
   ```

   The script fails if vllm or torch gets imported on the robot side.

2. **Write the robot file**:

   ```bash
   python3 .claude/skills/vla-maas-deploy-robot/scripts/render_robot.py --out piper-01.yaml \
       --robot-id piper-01 --gateway https://maas.example.com \
       --front-serial <realsense serial> --wrist-device 0 \
       --workspace 0.05 0.6 -0.4 0.4 0.02 0.5
   ```

   - The output starts in `dry_run`, sets `prefetch_threshold: 20` for the
     public link and uses `home_joints: [0, 1.0, -1.0, 0, 0.4, 0]`. Confirm
     that ready pose on your arm.
   - Measure the workspace box on your table.
   - Camera keys are protocol roles (`front`, `wrist`). The gateway adapter
     maps them to model cameras.
   - `--mock` writes a mock arm with mock cameras (embodiment `piper`). Use it
     to rehearse against the real gateway before touching hardware.

3. **Bring up**:

   ```bash
   ROBOT=piper-01.yaml KEY_FILE=piper-01.env CAN_SCRIPT=<piper_sdk>/can_activate.sh \
       bash .claude/skills/vla-maas-deploy-robot/scripts/bringup.sh
   ```

   - The script runs: CAN up → `robotctl check` (driver, cameras, handshake;
     no motion) → one `dry_run` episode → a summary of `rtt_p50_ms`,
     `server_p50_ms`, `network_p50_ms`, `underruns`, `dropped_rows` and
     safety clamps.
   - `DO_HOME=1` adds `robotctl home` before the episode, which asks for
     confirmation. Home first on a Piper: it rests with joints 2 and 3 on their
     limits, and rest-pose targets get clamped.

4. **Read the summary before any motion**:

   | Signal | Meaning | Action |
   | --- | --- | --- |
   | `check` fails at handshake: 401 | wrong key, or `robot_id` not in the gateway | re-check the key file and `robot_id` |
   | 404 | model has no `adapter:` | ask the server operator |
   | close 4400 + problems | the arm does not match the model's needs (cameras, state, dof) | fix cameras or the adapter |
   | close 4409 | the same `robot_id` is already connected | stop the other session |
   | `underruns > 0` | latency is above `prefetch_threshold / control_hz` | raise `prefetch_threshold`, or ask for bf16 or fewer denoise steps |
   | high `network_p50_ms` | the public link is slow | lower `jpeg_quality`, or move the gateway closer |
   | many safety clamps | wrong units, joint signs, gripper range or adapter | fix before moving; never raise limits to hide it |
   | `stopped: safety ...` | a violation streak or a pose outside the workspace | same as above |

5. **Motion**, only with a fine-tuned checkpoint and the e-stop at hand:

   ```bash
   robotctl run --robot piper-01.yaml --instruction "..." --mode slow     # asks "yes"
   robotctl run --robot piper-01.yaml --instruction "..." --mode live --episodes 10 --ask-success --output eps.jsonl
   ```

   Or use the studio with a browser view:
   `apps/vla-maas-studio/studio.py --robot-config piper-01.yaml --gateway https://maas.example.com --api-key $VLA_MAAS_API_KEY --hardware`.

## Without hardware

`studio.py --robot-config piper-01.yaml --gateway <url> --api-key <key>` runs
the same file on the MuJoCo Piper twin. It checks the route, shapes, units,
action direction and size, safety settings and latency. It does not check
grasping or policy quality on real images. See
`.claude/skills/vla-maas/references/robots.md`.

## Safety notes

- On a stop, timeout or disconnect, the arm holds with motors enabled. A
  disabled Piper drops. `close` never disables unless
  `disable_on_close: true`.
- Gateway stop (`/v1/robots/<id>/stop`) is a software stop. Keep the hardware
  e-stop within reach.
- The Piper driver has not been validated on a real arm yet. On the first run,
  check joint signs, the end-pose Euler convention (assumed roll-pitch-yaw) and
  the gripper scale in `dry_run`. Open checks: `apps/vla-maas-studio/TODO.md`.

## Result contract

Report: commit SHA, `robot_id`, gateway URL, checkpoint (motion allowed or
`dry_run` only), the bring-up summary lines and the episode JSONL path.
