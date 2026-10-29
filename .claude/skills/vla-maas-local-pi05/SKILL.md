---
name: vla-maas-local-pi05
description: Bring up a real π0.5 model, the VLA MaaS gateway with its live dashboard, and a simulated Piper arm (MuJoCo twin in the studio) on one local GPU machine with uv environments, then check the dashboard and tear it down; or start only the MuJoCo Piper client (twin.sh) against a running gateway. Use when someone wants to try or demo the MaaS stack end to end on a single workstation (e.g. an RTX 50-series PC) with a real checkpoint instead of the mock backend, or to connect a simulated Piper to a local or remote gateway; use vla-maas-deploy-server for a cluster serving remote arms.
---

# π0.5 + gateway + Piper twin on one machine

## Purpose

One command per phase, all on localhost:

```text
studio :9100 (Piper MuJoCo twin, /act) ──▶ gateway :9000 (/dashboard, /v1/monitor) ──▶ vllm serve π0.5 :8001 (GPU)
```

The scripts reuse the cluster skill's pieces
([vla-maas-deploy-server](../vla-maas-deploy-server/SKILL.md): `serve_pi05.sh`,
`render_gateway.py`, `start_gateway.sh`) and keep everything under
`maas_runs/local-pi05/` (git-ignored): deploy file, gateway config and keys, logs, pids.

## Before starting

Ask or check, then tell the user what applies:

1. **GPU memory.** `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`.
   π0.5 (`lerobot/pi05_base`) peaks near **8.9 GiB** in bfloat16 at batch 1
   ([recipe](../../../recipes/lerobot/Pi05.md)), so an 8 GB card (RTX 5060) will
   most likely OOM; 16 GB (5060 Ti) and up is fine. `up.sh` refuses below 9500 MiB
   unless `FORCE=1`. With too little memory, serve the model on another GPU host and
   pass `BACKENDS=ws://<host>:<port>/v1/realtime/robot/openpi`.
2. **Driver.** The default vLLM wheel is the CUDA 13 build from
   `docs/getting_started/installation/gpu/cuda.inc.md`; the driver must support CUDA 13.
   Blackwell (sm_120) needs a CUDA 12.8+ build in any case. Override with
   `VLLM_WHEEL=...` / `TORCH_BACKEND=...`.
3. **Hugging Face.** The tokenizer is gated: accept `google/paligemma-3b-pt-224` on
   the Hub, then `.venv-pi05/bin/hf auth login`. The first start downloads about
   14.5 GB of weights.
4. **What the run proves.** `pi05_base` is not trained on Piper and ships no
   normalization stats, so the twin's motion is meaningless. The run checks the
   route, handshake, latency, frames and dashboard, not policy quality (lessons L002).
   A Piper fine-tune: `CKPT=<hf id or path>`; check its camera keys and state width
   against `render_gateway.py`'s flags before claiming behaviour.

## Steps

Run from the repo root on a branch that has the live dashboard (`/v1/monitor`).

1. **Environments** (uv; existing envs are kept, `RECREATE=1` rebuilds):

   ```bash
   bash .claude/skills/vla-maas-local-pi05/scripts/setup_envs.sh
   ```

   - `.venv-pi05`: vLLM + vllm-omni (editable) + torch, only for the backend.
   - `.venv-maas`: gateway, studio, robot client, MuJoCo; no vLLM, no torch.
   - Sparse-clones the Piper model into `~/.cache/mujoco_menagerie` (`MENAGERIE=`).
   - `ONLY=pi05` or `ONLY=maas` builds one env.

2. **Bring it up**:

   ```bash
   bash .claude/skills/vla-maas-local-pi05/scripts/up.sh
   ```

   - Writes `maas_runs/local-pi05/pi05.yaml` from `vllm_omni/deploy/pi05.yaml` with
     `dtype: bfloat16` and `max_num_seqs: 1` (`DTYPE=`, `MAX_NUM_SEQS=`).
   - Starts the backend and waits for its port (minutes on a first load), renders the
     gateway config (`pi05_piper` route, Piper adapter, robot `piper-01`, keys kept
     across runs), starts the gateway and the studio, then prints the URLs and the
     operator key.
   - The studio is started by `twin.sh` (below) with the operator key.
   - `MOCK=1` swaps π0.5 for the CPU mock: a rehearsal with no GPU.
   - No EGL (headless box without an NVIDIA GL stack): `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`.

3. **Drive the twin**: open `http://127.0.0.1:9100`, model `pi05_piper`, enter an
   instruction, press 启动.

4. **Check the dashboard**: `http://127.0.0.1:9000/dashboard` with the printed operator
   key (also in `maas_runs/local-pi05/deploy/keys/operator.env`).
   - The `pi05_piper` card shows real GPU latency. For reference, the recipe measured
     177 ms p50 on an RTX PRO 5000 (bfloat16, 2 cameras); expect a consumer card to be
     slower. The red 100 ms line is the mock budget, not a target for π0.5.
   - Click `piper-01`: front and wrist frames (what the model receives), instruction,
     joint state, the action chunk, latency, events.
   - Press 停止 in the drawer: the twin holds, the studio card ends as 已结束（有错误）
     and `studio.log` says `episode stopped: gateway stop: operator stop (dashboard)`.

5. **Only the MuJoCo Piper client** (the gateway runs already, here or elsewhere):

   ```bash
   bash .claude/skills/vla-maas-local-pi05/scripts/twin.sh                       # up.sh's gateway and operator key
   GATEWAY=https://<domain> API_KEY=<key> bash .claude/skills/vla-maas-local-pi05/scripts/twin.sh
   ```

   - Needs only `.venv-maas` and the Piper model (`ONLY=maas setup_envs.sh`); no GPU.
   - Checks `/healthz` and that the key can read `/v1/models/$MODEL` before starting.
   - The twin file is `packages/vla-maas-robot/robots/piper.yaml` with `robot_id`,
     gateway URL and model set, a ready pose, and `max_violations` raised
     (`TWIN_MAX_VIOLATIONS`): pi05_base's targets are mostly clamped, and the default
     rule would stop every episode within seconds. The studio runs without `--hardware`.
   - A robot key (`maas_runs/local-pi05/deploy/keys/piper-01.env`) runs one twin as its
     own `robot_id`; several twins at once (the studio's 机器人 field, up to 8, ids
     `piper-01-<n>`) need an operator key.
   - `MODEL=`, `ROBOT_ID=`, `STUDIO_PORT=` change the route, id and port.

6. **Stop**:

   ```bash
   bash .claude/skills/vla-maas-local-pi05/scripts/down.sh
   ```

## When something fails

| Symptom | Look at |
| --- | --- |
| `up.sh` exits at the GPU check | memory too small; `BACKENDS=` to a bigger GPU, or `FORCE=1` |
| backend never ready / exits | `maas_runs/local-pi05/backends/backend-8001.log` (OOM, CUDA arch, gated tokenizer) |
| gateway does not serve `pi05_piper` | `maas_runs/local-pi05/gateway/gateway.log` |
| studio exits at start | `maas_runs/local-pi05/studio/studio.log`; an EGL error means set `MUJOCO_GL` |
| dashboard says the key is invalid | it needs the operator key, not `piper-01`'s robot key |
| twin episode stops right away | `TWIN_MAX_VIOLATIONS`, or the clamp reason in the studio detail view |

## Result contract

Report: commit SHA, GPU name and memory, checkpoint and dtype, whether the backend
loaded (or the log lines if not), dashboard p50/p99 for `pi05_piper` after a minute of
one twin session, and that frames and stop worked in the drawer. State that pi05_base
results say nothing about task success.
