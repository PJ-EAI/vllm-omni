---
name: vla-maas-deploy-server
description: Deploy the VLA MaaS server side on a GPU cluster for real robots on another network - start π0.5 backends (one per GPU), generate the gateway config with per-robot keys and a Piper adapter, run the gateway behind a TLS proxy on 443, verify the public path, and swap a route's checkpoint or model type live. Use when setting up or changing the cluster/gateway that serves remote arms; use vla-maas-deploy-robot for the arm PCs.
---

# Deploy the VLA MaaS server (cluster + gateway + public entry)

## Purpose

Bring up, on the latest `maas` line, the server half of
[docs/serving/vla_maas_piper_deployment.md](../../../docs/serving/vla_maas_piper_deployment.md):
π0.5 backends on the cluster network, the gateway on localhost, and a TLS
proxy as the only public entry. The robot half is
[vla-maas-deploy-robot](../vla-maas-deploy-robot/SKILL.md). Read the guide once;
this skill is the runnable version.

```text
robot PCs ──wss:443──▶ TLS proxy ──▶ gateway 127.0.0.1:9000 ──ws (cluster)──▶ vllm serve π0.5 ×K
```

## Before starting, settle the checkpoint

| Checkpoint | Proves | Robots may run |
| --- | --- | --- |
| `lerobot/pi05_base` | link, auth, handshake, shapes, latency, capacity | `dry_run` only: no norm stats, actions are not Piper joint angles |
| your LeRobot π0.5 fine-tune on Piper data | the above plus task behaviour | `dry_run` → `slow` → `live` |

There is no public single-arm Piper π0.5 fine-tune. A fine-tune's
normalization and relative→absolute actions run inside the server, so the
gateway adapter stays absolute. Ask the user which one before telling anyone
the arms may move (lessons L002: a link test is not a policy result).

## Steps

Run from the repo root on the `maas` branch. Each script prints what it
started and where its logs are.

1. **Backends** (one per GPU, cluster network only):

   ```bash
   CKPT=<ckpt> DEPLOY=<deploy.yaml> GPUS="0 1 2 3" bash .claude/skills/vla-maas-deploy-server/scripts/serve_pi05.sh
   # prints BACKENDS=ws://<node>:8001/v1/realtime/robot/openpi,...
   ```

   - Copy `vllm_omni/deploy/pi05.yaml` for the deploy file. Its
     `image_feature_keys` must follow the checkpoint's `input_features` order.
     `dtype: bfloat16` is faster.
   - The script passes `--robot-openpi-idle-timeout 0`. The default 30 s
     closes a robot's session between episodes.
   - Several nodes: run the script on each, then join the `BACKENDS` lines.
   - One node with several GPUs and the gateway on the same node: see
     [One node, several GPUs](#one-node-several-gpus-eg-8-rtx-4090).
   - The script downloads the checkpoint and the gated tokenizer once before
     starting (`PREFETCH=0` skips it) and waits `STAGGER_S` (default 10 s)
     between launches.
   - CPU rehearsal: `MOCK=1` starts mock backends on the same ports.

2. **Gateway config and keys**:

   ```bash
   python3 .claude/skills/vla-maas-deploy-server/scripts/render_gateway.py --out deploy/ \
       --backends "$BACKENDS" --robots piper-01,piper-02,piper-03 \
       --state-dim 32 --control-hz 30 --gripper-range 0 1
   ```

   - The flags must match the training data: camera keys (`--front-key`,
     `--wrist-key`), state width, control rate and the gripper scale. Wrong
     values do not error; they give wrong actions.
   - Output: `deploy/gateway.yaml` (mode 600), `deploy/keys/<robot>.env` (one
     per robot, copy each to its own PC only) and `deploy/keys/operator.env`.
   - Re-running keeps existing keys, so adding robots does not rotate them.
   - For another arm or model shape, edit the `adapter:` block by hand; every
     key is documented in `apps/vla-maas-gateway/model_adapters.py`.

3. **Gateway**:

   ```bash
   pip install ./packages/vla-maas-protocol websockets msgpack msgpack-numpy pyyaml numpy
   CONFIG=deploy/gateway.yaml bash .claude/skills/vla-maas-deploy-server/scripts/start_gateway.sh
   ```

   It binds `127.0.0.1:9000`; never bind it publicly.

4. **TLS proxy on 443**: Caddy or nginx; configs, plus the case where the
   cluster cannot open 443, are in
   [references/tls-proxy.md](references/tls-proxy.md).

5. **Verify from outside**, ideally from the robots' network:

   ```bash
   source deploy/keys/operator.env
   URL=https://<domain> OPERATOR_KEY=$VLA_MAAS_OPERATOR_KEY bash .claude/skills/vla-maas-deploy-server/scripts/verify_server.sh
   ```

   It checks `/healthz`, that a wrong key gets 401, `/v1/models/<model>` and
   `/v1/robots`.

6. Hand each robot operator: the URL, the model name (`pi05_piper`), its
   `robot_id` and its key file. Then follow `vla-maas-deploy-robot`.

## One node, several GPUs (e.g. 8× RTX 4090)

Backends and gateway on one machine; only the TLS proxy is public.

```bash
ONLY=pi05 bash .claude/skills/vla-maas-local-pi05/scripts/setup_envs.sh   # uv env .venv-pi05: vLLM + vllm-omni
ONLY=maas bash .claude/skills/vla-maas-local-pi05/scripts/setup_envs.sh   # uv env .venv-maas: gateway
export HF_HOME=/data/hf_cache                                              # local disk with ~20 GB free
.venv-pi05/bin/hf auth login
cp vllm_omni/deploy/pi05.yaml deploy-pi05.yaml && sed -i 's/^dtype: .*/dtype: bfloat16/' deploy-pi05.yaml
CKPT=<ckpt> DEPLOY=$PWD/deploy-pi05.yaml GPUS="0 1 2 3 4 5 6 7" HOST=127.0.0.1 ADVERTISE=127.0.0.1 \
  VLLM=$PWD/.venv-pi05/bin/vllm bash .claude/skills/vla-maas-deploy-server/scripts/serve_pi05.sh
# then step 2 with that BACKENDS line, and step 3 with PYTHON=$PWD/.venv-maas/bin/python
```

- `HOST=127.0.0.1` keeps ports 8001–8008 off the network; only the gateway on
  the same node reaches them.
- Driver: the default vLLM wheel in `setup_envs.sh` is the CUDA 13 build, so
  `nvidia-smi` must show CUDA 13.0 or newer. With an older driver, install vLLM
  per `docs/getting_started/installation/gpu.md` (`VLLM_WHEEL=`, `TORCH_BACKEND=`).
  The RTX 4090 is sm_89; the π0.5 recipe lists serving as verified on it.
- Memory: π0.5 in bfloat16 peaks near 9 GiB at batch 1, well inside 24 GB. Keep
  `max_num_seqs: 1` until a concurrency sweep on this node
  (`.claude/skills/vla-maas/scripts/serve_and_sweep.sh`) shows that batching
  helps; that is unmeasured on the 4090.
- Host: eight processes load the same checkpoint. The prefetch and the stagger
  spread the disk and RAM peak; watch `free -g` while they start, and raise
  `STAGGER_S` if the host swaps.
- Capacity: by the estimate below, eight backends serve roughly 24 Piper arms
  at 30 Hz with 50-step chunks. Confirm with the robots' `rtt_p50` and
  `underruns` and with the dashboard's per-backend load.
- `nvidia-smi` should show one vLLM process per GPU; the dashboard lists the
  eight backends with their session counts.

## Operating

- **Live view:** `https://<domain>/dashboard` (operator key; per-arm camera
  frames and state through `WS /v1/monitor`), `GET /metrics`
  (`relay_p99_ms`, active sessions per model) and `GET /v1/robots`.
- **Stop one robot:** `curl -H "Authorization: Api-Key $VLA_MAAS_OPERATOR_KEY" https://<domain>/v1/robots/<id>/stop`.
  This is a software stop; every arm still needs its hardware e-stop.
- **Capacity:** a backend is `max_num_seqs: 1`, about 150–400 ms per call by
  the recipe's GPU numbers. A Piper at 30 Hz with 50-step chunks asks about
  once a second, so plan roughly 1 backend per 3 arms. This is an estimate:
  confirm it with the robots' `rtt_p50` / `underruns`.
- **Add a robot:** re-run `render_gateway.py` with the longer `--robots` list,
  then `kill -HUP $(cat maas-gateway/gateway.pid)`; keys and robots reload
  without dropping sessions.
- **Swap the checkpoint or model type live** (to compare checkpoints or chase a
  regression): blue/green onto spare GPUs, old robots finish their session on
  the old revision, new ones get the new one:

  ```bash
  source deploy/keys/operator.env
  python3 .claude/skills/vla-maas-deploy-server/scripts/swap_model.py --config deploy/gateway.yaml \
      --model pi05_piper --checkpoint <new ckpt> --gpus "4 5 6 7" --base-port 8101
  # another VLA: add --model-type <type> --deploy <its deploy yaml> --adapter-file <adapter block yaml>
  # no spare GPUs: --stop-old-first (the route answers 503 while the new servers load)
  # cut over now: --cutover stop (robots on the old revision get stop at their next observation)
  ```

  The gateway probes the new backends and rejects the swap, changing nothing,
  if a handshake fails, the adapter rejects it, or a backend reports another
  checkpoint or model type than the labels; the script then restores the YAML
  and stops what it started. A new checkpoint is a new policy: drop the arms
  back to `dry_run` until it has been checked (lessons L002).
- **Track a problem:** `maas-gateway/gateway.jsonl` has one JSON event per line,
  tagged with `session`, `robot`, `model`, `revision`, `checkpoint`,
  `model_type` and `backend`. `robotctl` prints the session id and revision
  when it connects. For example
  `jq -c 'select(.robot == "piper-01" and .level != "INFO")' maas-gateway/gateway.jsonl`,
  `jq -c 'select(.event | startswith("route."))' maas-gateway/gateway.jsonl` for
  the swap history, or `GET /v1/admin/revisions`. The backend's own log names
  requests `robot-<session>-<n>` for adapters that send the session id.
- **Stop everything:** `kill $(cat maas-gateway/gateway.pid maas-backends/*.pid)`.

## Known limits

- Swaps change what new sessions get; a running session never changes model
  mid-episode. In-process weight reload inside one vllm-omni server is not
  used: a model type change needs a new process anyway.
- One gateway process with no health-checked failover. When a backend dies,
  its sessions drop and are routed elsewhere when they reconnect.
- No tenants or quotas. Keys are the only access control.
- Real public-network latency is unmeasured until the robots run.

## Result contract

Report: commit SHA, checkpoint, deploy yaml (dtype, `num_inference_steps`),
GPU type × backends, public URL, robots configured, and the `verify_server.sh`
output. Say whether the checkpoint allows motion or `dry_run` only.
