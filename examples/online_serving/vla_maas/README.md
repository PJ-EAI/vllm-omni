# VLA MaaS — LIBERO baseline harness

Client-side tooling for serving VLAs (π0.5, InternVLA-A1) as a model service over
the OpenPI websocket endpoint `/v1/realtime/robot/openpi`, with a mock path that
runs the whole loop on CPU before GPUs or the LIBERO simulator are involved.

| File | Role |
| --- | --- |
| `vla_client.py` | OpenPI connection helpers and `AsyncChunkExecutor` (background-prefetching action queue for a fixed-rate control loop). |
| `libero_eval.py` | Closed-loop LIBERO evaluation: success rate, inference latency, control-loop stalls. `--mock` swaps in `mock_libero`. |
| `latency_bench.py` | Client E2E round-trip benchmark (p50/p90/p99) with synthetic observations. |
| `fleet_sim.py` | Concurrent robot fleet: N robots, each with its own session, env and real-time 50 Hz loop; reports throughput, latency percentiles, stalls, success. |
| `concurrency_sweep.sh` | Fleet-size sweep over three serving layouts (serial, replica-DP, dynamic batching) behind the gateway. |
| `mock_libero.py` | Dependency-free LIBERO stand-in with the same suite/env API and observation keys, shapes and dtypes. |
| `mock_policy_server.py` | CPU mock of the OpenPI policy endpoint with injected latency; solves `mock_libero` from pixels. |
| `sim_fleet.py` | Simulated multi-arm fleet from a scenario YAML: one process per robot (LIBERO, robosuite or mock), random boot offsets and idle gaps, per-group metrics, headless MP4 per episode. |
| `sim_envs.py` | Simulator adapters (LIBERO Panda, robosuite Panda/UR5e/Sawyer/Kinova3/IIWA/Jaco, mock) behind one observation/action interface. |
| `video_recorder.py`, `video_grid.py` | Streaming captioned MP4 per episode; tile a run's clips into one grid video. |
| `sim_fleet_demo.sh`, `scenarios/` | Mock backends + gateway + `sim_fleet.py` in one command; example scenarios. |
| `pi05_libero.yaml` | π0.5 deploy config for a LIBERO fine-tune: bfloat16, camera keys taken from the checkpoint, request batching (`max_num_seqs: 16`). |

Install the client extras: `pip install websockets msgpack msgpack-numpy` and
`openpi-client` (`pip install --no-deps -e packages/openpi-client` from the
[openpi](https://github.com/Physical-Intelligence/openpi) repo; it only needs the
first three at runtime).

## Mock loop (CPU, no simulator)

```bash
python3 mock_policy_server.py --port 8000 --latency-ms 80 --jitter-ms 10 &

# Sync: execute 5 actions, then block on the next chunk (OpenPI LIBERO protocol).
python3 libero_eval.py --mock --port 8000 --max-tasks 3 --episodes-per-task 5

# Async at 50 Hz: prefetch when 10 actions remain; `underruns` counts stalls.
python3 libero_eval.py --mock --port 8000 --replan-steps 20 --async-prefetch 10 --control-hz 50

python3 latency_bench.py --port 8000 \
    --cameras observation.images.image observation.images.image2 --state-dim 8
```

The mock policy reads the goal and end effector from the agent-view pixels, so a
non-zero success rate shows images, state and actions survive the round trip in
the right layout.

## Many robots at once

A serving stack is only exercised when requests from different robots overlap.
`fleet_sim.py` runs N robots concurrently, and the mock server shares one
simulated accelerator across all connections:

- `--max-batch 1` serves one request at a time, like a replica deployed with
  `max_num_seqs: 1`.
- `--max-batch N --max-wait-ms W --per-item-ms P` batches whatever is queued
  within `W` ms; a batch of B costs `latency + (B - 1) * P`.

```bash
bash concurrency_sweep.sh results.jsonl   # ROBOTS="1 2 4 8 16 32" DURATION=15 ...
```

The per-item cost and batch window are assumptions standing in for a batched
pi0.5 forward; replace them with GPU measurements (below).

### Measuring it on a GPU

`Pi05Pipeline` declares `supports_request_batch`, so one replica batches
concurrent sessions up to `max_num_seqs` (16 in `pi05_libero.yaml`). Compare
serial and batched serving with the same fleet (the mock env only generates
load here; its success rate is meaningless against a real checkpoint):

```bash
# batched (as shipped); for serial, set max_num_seqs: 1 in the yaml
vllm serve /path/to/pi05_libero_checkpoint --omni --port 8000 \
    --deploy-config examples/online_serving/vla_maas/pi05_libero.yaml --enforce-eager

for n in 1 2 4 8 16 32; do
  python3 fleet_sim.py --port 8000 --robots $n --duration-s 30 --label batched \
      --base-camera-key <checkpoint base camera key> \
      --wrist-camera-key <checkpoint wrist camera key> --output gpu_sweep.jsonl
done
```

`--request-batch-max-wait-ms` on `vllm serve` trades first-request latency for
fuller batches; the default 0 batches whatever queued while the GPU was busy.

To go through the MaaS gateway (API key, model routing, metrics), see
[`apps/vla-maas-gateway`](../../../apps/vla-maas-gateway/README.md) and pass
`--path /v1/robot/<model>/openpi --api-key <key>` to either client.

## Real loop (GPU + LIBERO)

```bash
vllm serve /path/to/pi05_libero_checkpoint --omni --port 8000 \
    --served-model-name pi05 --deploy-config examples/online_serving/vla_maas/pi05_libero.yaml \
    --enforce-eager --disable-log-stats

python3 libero_eval.py --port 8000 --task-suite libero_spatial --episodes-per-task 10 \
    --base-camera-key <checkpoint base camera key> --wrist-camera-key <checkpoint wrist camera key>
```

Camera keys must match the checkpoint's `input_features`. `pi05_libero.yaml`
leaves `image_feature_keys` unset so the server derives them from the checkpoint.

## Simulated multi-arm fleet (robosuite / LIBERO)

`sim_fleet.py` replaces the mock env with real MuJoCo simulators and mixes arm
types and models in one run. A scenario lists robot groups:

```yaml
defaults:
  control_hz: 20            # robosuite / LIBERO controller rate
  replan_steps: 5
  start_jitter_s: 3.0       # robots boot at uniform(0, 3) s
  think_time_s: [0.5, 2.0]  # idle between episodes, like a user resetting a task
  record_episodes: 1        # first N episodes of every robot -> mp4
  report_episodes: true     # success counters on the gateway dashboard
groups:
  - name: panda-libero
    model: pi05             # gateway route /v1/robot/pi05/openpi (or `path:`)
    robots: 8
    env: {kind: libero, suite: libero_spatial}
  - name: ur5e-lift
    model: pi05
    robots: 4
    env: {kind: robosuite, robot: UR5e, task: Lift, max_steps: 300, video_camera: frontview}
```

Every robot is a separate process with its own simulator and session. All
robots build their scenes first, then the measured window of `--duration-s`
starts; robots finish the episode in progress when it ends. Group options
override `defaults`; `env` keys are `kind, robot, task, suite, task_ids,
resolution, max_steps, control_freq, video_camera`.

What the success rate means: only Panda on LIBERO matches a π0.5 LIBERO
fine-tune. Other arms and robosuite tasks exercise the service with real
renders and a real control loop; their success rate says nothing about the
policy until a checkpoint is trained for that arm.

### Install

```bash
# robosuite only (any Python >= 3.10):
pip install "robosuite>=1.5" "mujoco>=3.3,<3.4" imageio imageio-ffmpeg pillow pyyaml
# LIBERO + robosuite (LIBERO pins robosuite 1.4.x; sim_envs.py handles both APIs):
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install "robosuite==1.4.1" "mujoco==3.2.3" "bddl==1.0.1" "numpy<2" easydict "gym==0.25.2" \
    future matplotlib torch imageio imageio-ffmpeg pillow pyyaml
export PYTHONPATH=$PWD/LIBERO:$PYTHONPATH
```

LIBERO asks for its data paths on first import; write `~/.libero/config.yaml`
(`benchmark_root, bddl_files, init_states, datasets, assets`, pointing into
`LIBERO/libero/libero`) beforehand in batch jobs. robosuite 1.5 fails on
MuJoCo 3.4+, hence the pin.

### Headless rendering and video on A100 / A800

MuJoCo renders offscreen through EGL with the NVIDIA driver; no X server or
display is needed (A100/A800 have no display outputs, and none is required).

```bash
export MUJOCO_GL=egl            # default in sim_envs.py; osmesa on CPU-only hosts
python3 sim_fleet.py --scenario scenarios/sim_fleet_robosuite.yaml --port 9000 \
    --api-key dev-key-change-me --duration-s 300 --render-gpus 0,1,2,3 \
    --video-dir out/videos --video-grid --output out/summary.jsonl --episodes-out out/episodes.jsonl
```

- `--render-gpus` takes EGL device ids (physical GPU order) and spreads robots
  over them round-robin. The sim processes do no CUDA work and drop
  `CUDA_VISIBLE_DEVICES`, so they can share GPUs with `vllm serve`.
- In containers, the NVIDIA runtime must expose graphics:
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` (otherwise EGL finds no
  device), plus `libegl1` in the image.
- Videos: `out/videos/<group>/r<robot>_ep<n>_<success|fail>.mp4`, captioned with
  arm, model, step and last inference latency, H.264/yuv420p (plays in
  browsers). `--video-grid` adds `out/videos/grid.mp4` with one clip per group
  first. `video_camera: frontview` shows the whole arm instead of the policy's
  agent view. `video_stride` in the scenario thins frames for long runs.
- `sim_step_p50_ms` in the summary is simulator + render time per step. When it
  approaches `1000 / control_hz`, `late_tick_pct` reflects the simulator, not
  the service; add CPU cores or lower the resolution.

### One command

```bash
# CPU mock backends that sweep the arms, gateway, robosuite fleet, videos:
bash sim_fleet_demo.sh scenarios/sim_fleet_robosuite.yaml 120
# Real models: start vllm serve, point a gateway yaml at it, then
GATEWAY_CONFIG=my_gateway.yaml RENDER_GPUS=0,1 bash sim_fleet_demo.sh scenarios/sim_fleet_libero_mixed.yaml 600
# Anywhere, no simulator: MUJOCO_GL is unused by the mock scenario.
bash sim_fleet_demo.sh scenarios/sim_fleet_mock.yaml 30
```

`mock_policy_server.py --idle-motion A` returns a slow sweep for non-mock
scenes so the videos confirm actions reach every arm; it is not a policy.

## Latency budget

An async client stalls only when inference outlasts the actions it has left:
`latency < prefetch_threshold / control_hz`. At 50 Hz, a 100 ms inference needs
at least 5 actions of headroom. The current π0.5 numbers in
`recipes/lerobot/Pi05.md` (eager, 10 denoising steps) are 156–398 ms client E2E,
so meeting a 100 ms target needs bfloat16, graph capture/compilation and fewer
denoising steps (`sampling_params.num_inference_steps`).
