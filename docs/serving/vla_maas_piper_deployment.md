# Serving π0.5 to remote AgileX Piper arms

This guide covers serving a π0.5 policy from a GPU cluster to a fleet of AgileX
Piper arms on a different network, over the VLA MaaS gateway and the
standard Robot protocol (`maas` branch).

The steps can be run with two repository skills:

- **Cluster side:** `.claude/skills/vla-maas-deploy-server`.
- **Each arm PC:** `.claude/skills/vla-maas-deploy-robot`.

```text
Piper PC ×N ──wss:443──▶ TLS proxy ──ws──▶ gateway 127.0.0.1:9000 ──ws (cluster)──▶ vllm serve π0.5 ×K
 vla-maas-robot         public entry       /v1/robot/pi05_piper/act        /v1/realtime/robot/openpi
```

Each arm only dials out to port 443. It needs no public IP and no inbound port.

## 0. Choose the checkpoint first

| Checkpoint | What it proves | What it cannot prove |
| --- | --- | --- |
| `lerobot/pi05_base` | Link, auth, handshake, shapes, latency, capacity, safety layer | Anything about actions. It has no normalization stats and its outputs are not Piper joint angles. **Run it in `dry_run` only.** |
| Your LeRobot π0.5 fine-tune on Piper data | All of the above, plus task behaviour | — |

There is no public single-arm Piper π0.5 fine-tune. Bring the service up with
`pi05_base`, then switch to your fine-tune before any motion. A fine-tune's
normalization stats (the `policy_preprocessor` sidecar) and its
relative→absolute action conversion run inside the vllm-omni server, so the
gateway adapter is configured with absolute actions.

## 1. Cluster: π0.5 backends, one per GPU

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve <checkpoint> --omni --host 0.0.0.0 --port 8001 \
  --deploy-config my_pi05_piper.yaml --enforce-eager --disable-log-stats \
  --robot-openpi-idle-timeout 0
```

- **Idle timeout:** `--robot-openpi-idle-timeout 0` matters. The default of
  30 s closes a robot's upstream session when it pauses between episodes.
- **Deploy file:** copy `vllm_omni/deploy/pi05.yaml`.
    - `image_feature_keys` must follow the checkpoint's `input_features` order.
    For `pi05_base` that is `base_0_rgb`, `left_wrist_0_rgb`,
    `right_wrist_0_rgb`.
    - `dtype: bfloat16` is faster. The recipe measured a client end-to-end p50
    of 177 ms on an RTX PRO 5000 and 301 ms on an H100, with two cameras,
    bf16 and 10 denoising steps.
- **Network:** keep these ports on the cluster network.
- **Script:** `serve_pi05.sh` in the server skill starts one backend per GPU
  and prints the backend URLs.

## 2. Cluster: gateway

```yaml
api_keys: [<operator key>]
request_timeout_s: 5.0
models:
  pi05_piper:
    backends:
      - ws://gpu-node-1:8001/v1/realtime/robot/openpi
      - ws://gpu-node-1:8002/v1/realtime/robot/openpi
    adapter:
      kind: generic
      images: {front: observation.images.base_0_rgb, wrist: observation.images.left_wrist_0_rgb}
      image_size: [224, 224]
      state: [joint_pos, {field: gripper, range: [0, 1]}]
      state_dim: 32                     # the checkpoint's observation.state width
      action:
        space: joint_pos
        layout: [joint_pos, {field: gripper, range: [0, 1]}]
      control_hz: 30                    # the rate the training data was recorded at
      embodiments: [piper]
robots:
  piper-01: {api_key: <piper-01 key>, models: [pi05_piper]}
  piper-02: {api_key: <piper-02 key>, models: [pi05_piper]}
```

```bash
python3 apps/vla-maas-gateway/gateway.py --config gateway.yaml --host 127.0.0.1 --port 9000
```

- **Adapter values must match your training data.** Check the state order,
  the gripper scale (`range`: 0–1 or metres) and `control_hz`. A wrong value
  does not error; it produces wrong actions.
- **Keys:** every robot gets its own key. A robot key works only on `/act`,
  for its own `robot_id` and listed models. Operator keys stay on the cluster
  side.
- **Script:** `render_gateway.py` in the server skill writes this file and
  generates the keys.

## 3. Cluster: public entry (TLS)

The gateway does not terminate TLS. Put Caddy or nginx on 443 in front of it
(for nginx, forward the websocket `Upgrade` headers and use a long
`proxy_read_timeout`). If the cluster cannot open 443, run the proxy on a host
that has a public IP and forward it to the gateway. The configs are in the
server skill's `references/tls-proxy.md`.

## 4. Each Piper PC

```bash
pip install ./packages/vla-maas-protocol "./packages/vla-maas-robot[piper,realsense,uvc,jpeg]"
bash can_activate.sh can0 1000000      # from piper_sdk
```

Start from `packages/vla-maas-robot/robots/piper.yaml` and set:

- `robot_id`, matching the gateway entry.
- `gateway.url: https://<domain>` and `gateway.model: pi05_piper`. The key
  comes from `VLA_MAAS_API_KEY`.
- Camera serials and devices, under the protocol roles `front` and `wrist`.
- A ready pose in `driver.home_joints`, for example
  `[0, 1.0, -1.0, 0, 0.4, 0]` (confirm it on the arm).
- `control.prefetch_threshold: 20`.
- A workspace box measured on your table.

Piper rests with joints 2 and 3 on their 0 rad limits, so run `robotctl home`
before every episode. Otherwise the first targets are clamped into a safety
stop.

```bash
robotctl check --robot piper-01.yaml      # driver, cameras, handshake; no motion
robotctl home  --robot piper-01.yaml
robotctl run   --robot piper-01.yaml --instruction "pick up the cube" --max-steps 300 --output ep.jsonl   # dry_run
robotctl run   --robot piper-01.yaml --instruction "..." --mode slow   # fine-tuned checkpoint only
```

`dry_run` runs the whole loop, including inference and the safety filter, but
never sends a command to the arm. `slow` scales every step by `slow_scale`.
Both `slow` and `live` ask for confirmation.

`render_robot.py` and `bringup.sh` in the robot skill write this file and run
the sequence above. Before touching hardware, `studio.py --robot-config` can
run the same file on the MuJoCo Piper twin.

## 5. Latency and capacity

- **Stall-free condition:** an arm stalls only when a round trip (network,
  gateway and inference) takes longer than `prefetch_threshold / control_hz`.
  At 30 Hz with `prefetch_threshold: 20`, that budget is 667 ms. Inference
  takes about 150–400 ms and a public round trip about 50–150 ms, so the
  budget leaves room.
- **Request rate:** one 50-step chunk lasts about 1.7 s at 30 Hz, so each arm
  sends about one request per second.
- **Capacity (an estimate; measure it):** each backend is `max_num_seqs: 1`,
  so it serves roughly 2–4 arms. Plan about one backend per three arms.
    - The gateway assigns each session to the least-loaded backend.
    - `/metrics` shows `relay_p99_ms` and the number of active sessions.
    - Each `robotctl run` line reports `rtt_p50_ms`, `server_p50_ms`,
    `network_p50_ms` and `underruns`. `underruns > 0` means the arm is
    stalling.
- **Bandwidth:** two 224×224 JPEGs are about 20–40 KB per request.
- **Timeout:** when there is no reply within `request_timeout_s`, the episode
  stops and the arm holds. It does not keep executing stale actions.
- **Disconnect:** the arm holds in place with its motors enabled; it does not
  drop.

## 6. Safety and operations

- A robot key works only for its own robot. A second session with the same
  `robot_id` is refused with close code 4409.
- **Remote stop:**
  `curl -H "Authorization: Api-Key <operator key>" https://<domain>/v1/robots/<id>/stop`.
  This is a software stop; keep a hardware e-stop at every arm.
- **Monitoring:** `GET /v1/robots` lists connected arms, and
  `https://<domain>/dashboard` shows live metrics.

## 7. Known limits

- **Piper driver:** validated only against a fake SDK and the MuJoCo twin. On
  the first real run, check joint signs, the end-pose Euler convention
  (assumed roll-pitch-yaw) and the gripper scale. The open checks are in
  `apps/vla-maas-studio/TODO.md`.
- **Single gateway:** one gateway process with no health-checked failover.
  When a backend dies, its sessions drop and land on another backend when they
  reconnect.
- **Latency numbers:** the figures above are the recipe's GPU measurements
  plus estimates. Take public-network numbers from your own `robotctl run`
  output.
