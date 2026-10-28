---
name: vla-maas-deploy-server
description: Deploy the VLA MaaS server side on a GPU cluster for real robots on another network - start π0.5 backends (one per GPU), generate the gateway config with per-robot keys and a Piper adapter, run the gateway behind a TLS proxy on 443, and verify the public path. Use when setting up or changing the cluster/gateway that serves remote arms; use vla-maas-deploy-robot for the arm PCs.
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
  then restart the gateway (sessions reconnect).
- **Stop everything:** `kill $(cat maas-gateway/gateway.pid maas-backends/*.pid)`.

## Known limits

- One gateway process with no health-checked failover. When a backend dies,
  its sessions drop and are routed elsewhere when they reconnect.
- No tenants or quotas. Keys are the only access control.
- Real public-network latency is unmeasured until the robots run.

## Result contract

Report: commit SHA, checkpoint, deploy yaml (dtype, `num_inference_steps`),
GPU type × backends, public URL, robots configured, and the `verify_server.sh`
output. Say whether the checkpoint allows motion or `dry_run` only.
