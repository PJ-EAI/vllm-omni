# Architecture for VLA serving

How a robot observation becomes an action chunk, and which file owns each
step. Verified against `maas` at `895574f`; re-check paths with `rg` if they
have moved.

## Layers

| Layer | Owns | Files |
| --- | --- | --- |
| Robot protocol | Model-agnostic messages, vocabulary, codec, version (`maas-robot/1`) | `packages/vla-maas-protocol/` (spec in `spec/robot_protocol.md`) |
| Robot client | Drivers, cameras, safety filter, tick-aligned control loop, `robotctl`; installs without vllm/torch | `packages/vla-maas-robot/` |
| Client | Observation building, control loop, async prefetch | `examples/online_serving/vla_maas/{vla_client,libero_eval,fleet_sim,latency_bench}.py`; simulators and video: `{sim_envs,sim_fleet,video_recorder,video_grid}.py`; `apps/vla-maas-studio/` (browser console) |
| MaaS gateway | Operator and per-robot keys, model → backend routing, `/act` translation through model adapters, OpenPI relay, metrics, dashboard | `apps/vla-maas-gateway/{gateway.py,model_adapters.py,dashboard.html,gateway.example.yaml}` |
| CLI | `vllm serve ... --omni` flags (`--deploy-config`, `--request-batch-max-wait-ms`, `--robot-openpi-idle-timeout`) | `vllm_omni/entrypoints/cli/serve.py` |
| HTTP/WS app | Routes, `/health`, `/v1/models`, `WS /v1/realtime/robot/openpi` | `vllm_omni/entrypoints/openai/api_server.py` |
| OpenPI protocol | Handshake, msgpack-numpy codec, reset, idle timeout, session ids | `vllm_omni/entrypoints/openpi/connection.py` |
| OpenPI serving | obs → `OmniDiffusionRequest` (`extra_args.robot_obs`), result → actions | `vllm_omni/entrypoints/openpi/serving.py` |
| Engine client | Multi-stage orchestration, `generate()` | `vllm_omni/entrypoints/async_omni.py` |
| Pipeline topology | `pipeline: <name>` in the deploy yaml → stage graph | `vllm_omni/config/pipeline_registry.py`, `vllm_omni/diffusion/models/<m>_pipeline_config.py` |
| Deploy config | Devices, dtype, `max_num_seqs`, `model_config`, `policy_server_config` | `vllm_omni/deploy/*.yaml`, `examples/online_serving/vla_maas/pi05_libero.yaml` |
| Scheduler | Admission and request-wave batching up to `max_num_seqs` | `vllm_omni/diffusion/sched/request_scheduler.py` |
| Worker/runner | Builds `DiffusionRequestBatch`, calls `pipeline.forward` | `vllm_omni/diffusion/worker/{diffusion_model_runner,request_batch}.py` |
| Model registry | Class name → module | `vllm_omni/diffusion/registry.py` |
| Model pipeline | Preprocess, denoise, postprocess, per-request errors | `vllm_omni/diffusion/models/<m>/pipeline_<m>.py` (+ `processor_*`, `modeling_*`, `config.py`) |

## Request path (π0.5)

1. Client opens `ws://host:port/v1/realtime/robot/openpi`. The server replies
   with the `policy_server_config` block from the deploy yaml as the first
   frame. No `policy_server_config` → endpoint disabled.
2. Each observation frame is decoded by `connection.py` and handed to
   `ServingRealtimeRobotOpenPI.infer`. `sampling_params` (e.g.
   `num_inference_steps`) and `seed` are pulled out; everything else becomes
   `extra_args["robot_obs"]`.
3. `AsyncOmni.generate` routes the request to the single diffusion stage.
4. The request scheduler groups waiting requests into a wave of up to
   `max_num_seqs`. `--request-batch-max-wait-ms` (default 0) lets it wait for
   more requests before starting a wave; 0 batches whatever queued while the
   GPU was busy.
5. If the pipeline declares `supports_request_batch = True`, the runner calls
   `forward(DiffusionRequestBatch)`; otherwise one request at a time.
6. `Pi05Pipeline._forward_requests` groups by `num_inference_steps`, stacks
   inputs, runs one `sample_actions`, and returns one `DiffusionOutput` per
   request. Invalid obs fail only their own request.
7. `serving._extract_actions` reads `multimodal_output["actions"]` and the
   connection sends the `[horizon, action_dim]` array back.

## Batching pattern for another VLA pipeline

Copy from `vllm_omni/diffusion/models/pi05/pipeline_pi05.py`:

- Class attribute `supports_request_batch = True`.
- `forward(req)` accepts both `OmniDiffusionRequest` and `DiffusionRequestBatch`;
  a single request raises on bad input, a batch converts it to
  `DiffusionOutput.from_exception` for that row.
- Keep a dummy-warmup path for requests without `robot_obs` (prompt
  `"dummy run"` or `num_inference_steps == 1`) returning zeros.
- Per-request noise generators (`_batch_generators`) so a request's actions
  do not depend on its batch peers.
- Group by anything that changes the compute graph (step count, camera count).
- Unit test: mirror `test_pipeline_batches_concurrent_requests` in
  `tests/diffusion/models/pi05/test_pi05_units.py` (CPU, monkeypatched model).
- Ship a deploy yaml with `max_num_seqs > 1` and say so in the recipe.

Stateful policies (DreamZero keeps per-session state, `needs_session_id: true`)
need session-aware batching; check `drop_session` / session state handling
before enabling it.

## Where new work goes

| You want to | Put it in |
| --- | --- |
| New auth, quotas, tenants, routing policy, failover | `apps/vla-maas-gateway/gateway.py` (+ README "Not yet covered") |
| New client workload or benchmark | `examples/online_serving/vla_maas/` |
| New VLA model | `vllm_omni/diffusion/models/<m>/`, registry entry, `<m>_pipeline_config.py`, `vllm_omni/deploy/<m>.yaml`, recipe; see the `add-diffusion-model` skill |
| Robot-protocol change | `packages/vla-maas-protocol/` (spec + schema + version bump if breaking); keep the previous major accepted |
| New real arm | `packages/vla-maas-robot/vla_maas_robot/drivers/<arm>.py`, register in `driver.DRIVERS`, a `robots/<arm>.yaml`, an extra for its SDK |
| Studio twin for an arm | a `RobotDriver` + `Camera` pair like `apps/vla-maas-studio/piper_sim.py` that copies the real driver's spec, listed in `SIM_TWINS` and built in `ActRun._driver_and_cameras` (`robot_run.py`) |
| A model on `/act` | `adapter:` under the model in the gateway yaml; a new `kind` in `model_adapters.ADAPTERS` only when the native output is not one action array |
| Native OpenPI change | `entrypoints/openpi/` plus `docs/serving/openpi_api.md` |
| Non-policy level (planner, world model) | New endpoint family under `vllm_omni/entrypoints/`, gateway route, orchestration in `apps/` |

## Decoupling rules (robot side vs cloud side)

- `vla-maas-robot` imports only `vla_maas_protocol` and its own modules; never
  `vllm`, `vllm_omni`, `torch` or the gateway. CI (`.github/workflows/vla-maas.yml`)
  runs its tests in an environment with only those two packages.
- The gateway imports `vla_maas_protocol`, never `vla_maas_robot` (tests may).
- Robots and the gateway share no config; a robot yaml names only the gateway
  route (`gateway.model`). Model keys live in the gateway's `adapter` blocks.
