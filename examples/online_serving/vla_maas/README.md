# VLA MaaS — LIBERO baseline harness

Client-side tooling for serving VLAs (π0.5, InternVLA-A1) as a model service over
the OpenPI websocket endpoint `/v1/realtime/robot/openpi`, with a mock path that
runs the whole loop on CPU before GPUs or the LIBERO simulator are involved.

| File | Role |
|---|---|
| `vla_client.py` | OpenPI connection helpers and `AsyncChunkExecutor` (background-prefetching action queue for a fixed-rate control loop). |
| `libero_eval.py` | Closed-loop LIBERO evaluation: success rate, inference latency, control-loop stalls. `--mock` swaps in `mock_libero`. |
| `latency_bench.py` | Client E2E round-trip benchmark (p50/p90/p99) with synthetic observations. |
| `fleet_sim.py` | Concurrent robot fleet: N robots, each with its own session, env and real-time 50 Hz loop; reports throughput, latency percentiles, stalls, success. |
| `concurrency_sweep.sh` | Fleet-size sweep over three serving layouts (serial, replica-DP, dynamic batching) behind the gateway. |
| `mock_libero.py` | Dependency-free LIBERO stand-in with the same suite/env API and observation keys, shapes and dtypes. |
| `mock_policy_server.py` | CPU mock of the OpenPI policy endpoint with injected latency; solves `mock_libero` from pixels. |
| `pi05_libero.yaml` | π0.5 deploy config for a LIBERO fine-tune: bfloat16, camera keys taken from the checkpoint. |

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

- `--max-batch 1` serves one request at a time, as `Pi05Pipeline` does today
  (it does not declare `supports_request_batch`, so the engine requires
  `max_num_seqs: 1`).
- `--max-batch N --max-wait-ms W --per-item-ms P` batches whatever is queued
  within `W` ms; a batch of B costs `latency + (B - 1) * P`.

```bash
bash concurrency_sweep.sh results.jsonl   # ROBOTS="1 2 4 8 16 32" DURATION=15 ...
```

The per-item cost and batch window are assumptions standing in for a batched
pi0.5 forward; replace them with A800 measurements once that exists.

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

## Latency budget

An async client stalls only when inference outlasts the actions it has left:
`latency < prefetch_threshold / control_hz`. At 50 Hz, a 100 ms inference needs
at least 5 actions of headroom. The current π0.5 numbers in
`recipes/lerobot/Pi05.md` (eager, 10 denoising steps) are 156–398 ms client E2E,
so meeting a 100 ms target needs bfloat16, graph capture/compilation and fewer
denoising steps (`sampling_params.num_inference_steps`).
