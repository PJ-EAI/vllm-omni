# VLA models in vllm-omni

Status at `maas` `895574f`. Re-check with
`rg -l policy_server_config vllm_omni/deploy` and
`rg -n supports_request_batch vllm_omni/diffusion/models`.

| Model | Pipeline class | Deploy yaml | OpenPI serving | Request batching | Recipe / docs | E2E test |
| --- | --- | --- | --- | --- | --- | --- |
| π0.5 | `Pi05Pipeline` | `vllm_omni/deploy/pi05.yaml` (fp32, `max_num_seqs: 1`); `examples/online_serving/vla_maas/pi05_libero.yaml` (bf16, `max_num_seqs: 16`) | yes | **yes** | `recipes/lerobot/Pi05.md` | `tests/e2e/online_serving/test_pi05_expansion.py` |
| π0 | `Pi0Pipeline` | `vllm_omni/deploy/pi0.yaml` | yes | no | `recipes/lerobot/Pi0.md` | `test_pi0_expansion.py` |
| GR00T N1.7 | `Gr00tN1d7Pipeline` | `vllm_omni/deploy/Gr00tN1d7.yaml` | yes | no | `recipes/NVIDIA/GR00T-N1.7.md` | `test_gr00t_openpi_expansion.py` |
| DreamZero | `DreamZeroPipeline` | `vllm_omni/deploy/dreamzero.yaml`, `dreamzero_tp1_cfg2.yaml` | yes (stateful, `needs_session_id`) | no | `docs/serving/openpi_api.md` | `test_dreamzero_expansion.py` |
| Cosmos3 policy | `Cosmos3OmniDiffusersPipeline` | `vllm_omni/deploy/cosmos3_policy_droid.yaml` | yes | no | `recipes/cosmos3/Cosmos3-Nano.md` | — |
| InternVLA-A1 | `InternVLAA1Pipeline` | none | **no** (offline only) | no | `docs/user_guide/examples/offline_inference/internvla_a1.md` | — |

Notes:

- The MaaS demo's `internvla_a1` route points at a **mock** backend. Real
  InternVLA-A1 serving needs a deploy yaml with `policy_server_config` first.
- Checkpoints: `lerobot/pi05_base`, `lerobot/pi0_base`, `nvidia/GR00T-N1.7-3B`,
  `GEAR-Dreams/DreamZero-DROID`. π0/π0.5 need the gated
  `google/paligemma-3b-pt-224` tokenizer (`hf auth login`).
- π0.5 rejects MEM and RTC checkpoints at load; relative-action checkpoints
  need `action_feature_names`.
- π0.5 recipe numbers (eager, 10 steps) are 156–398 ms client E2E; a 100 ms
  budget needs bf16, graph capture/compilation and fewer steps.

## Adapting a new VLA to MaaS (checklist)

1. Pipeline serves through OpenPI: deploy yaml has `policy_server_config`,
   pipeline reads `sampling_params.extra_args["robot_obs"]`, returns
   `DiffusionOutput(output={"actions": ...})`.
2. CPU unit tests for config, preprocessing and output shape (pattern:
   `tests/diffusion/models/pi05/test_pi05_units.py`, marks `core_model`, `cpu`).
3. GPU e2e test driving the websocket (pattern: `test_pi05_expansion.py`).
4. Optional: request batching (see `architecture.md`).
5. Add the model to a gateway yaml and run `serve_and_sweep.sh` for serial
   vs batched numbers.
6. Recipe under `recipes/<vendor>/` (use the `add-recipe` skill).
