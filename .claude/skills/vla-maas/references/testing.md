# Testing and benchmarking ladder

Run the lowest rung that covers the change, then climb only as far as the
claim needs. Each rung lists what it proves and what it does not.

| Rung | Needs | Command | Proves |
| --- | --- | --- | --- |
| T0 CPU smoke | client extras only | `bash .claude/skills/vla-maas/scripts/maas_smoke.sh` | gateway auth/routing/metrics, protocol round trip, mock eval loop |
| T1 CPU unit | repo env (torch, vllm) | see below | config, preprocessing, batching logic, protocol codec |
| T2 CPU load model | client extras | `concurrency_sweep.sh`, `run_demo.sh` | queueing/batching behaviour **under assumed costs** |
| T3 GPU serve + latency | 1 GPU, checkpoint | `serve_and_sweep.sh` with `ROBOTS=1` or `latency_bench.py` | real per-request latency |
| T4 GPU serve + fleet sweep | 1+ GPU, checkpoint | `serve_and_sweep.sh` (serial vs batched yaml) | real throughput/latency vs concurrency |
| T5 GPU e2e test | H100, checkpoint | `pytest tests/e2e/online_serving/test_pi05_expansion.py` | CI-equivalent wire path |
| T6 Task success | GPU + LIBERO sim | `libero_eval.py` without `--mock` | policy quality through the service |
| T7 Multi-arm sim fleet | robosuite (+ LIBERO), EGL GPU for speed | `sim_fleet.py --scenario scenarios/sim_fleet_libero_mixed.yaml` | serving under mixed arms/models with real renders; videos; success only for Panda/LIBERO |

## Change → minimum rung

| Changed | Run |
| --- | --- |
| `apps/vla-maas-gateway/*` | `cd apps/vla-maas-gateway && pytest -q`, T0 (+ T2 if routing/load behaviour changed) |
| `packages/vla-maas-protocol/*`, `packages/vla-maas-robot/*` | `pytest -q` inside each package (only those two packages installed), gateway tests, T0 |
| `apps/vla-maas-studio/*` | `cd apps/vla-maas-studio && pytest -q` (twin tests need `mujoco` + Menagerie `agilex_piper`), the edited mode's `run_demo.sh` (`PHYSICS=mock|mujoco|piper`) |
| `examples/online_serving/vla_maas/*` | T0 and the edited tool's own mock run |
| `sim_envs.py`, `sim_fleet.py`, video tools | T0 (mock sim stage), plus a short robosuite run (`MUJOCO_GL=osmesa` works on CPU) |
| `vllm_omni/entrypoints/openpi/*` | T1 openpi tests, then T5 |
| `vllm_omni/diffusion/models/pi05/*` | T1 pi05 units, then T3/T4 for perf claims, T5 for wire path |
| Deploy yaml / batching knobs | T4 serial vs batched, same fleet and duration |
| Model numerics | `test_pi05_parity.py` in a separate `lerobot[pi]` venv (see the file header) |

## T1 commands

```bash
pytest tests/diffusion/models/pi05/test_pi05_units.py -v -m "core_model and cpu"
pytest tests/diffusion/models/pi05/test_pi05_units.py::test_pipeline_batches_concurrent_requests -v
pytest tests/entrypoints/openai_api/test_openpi_serving.py tests/entrypoints/openai_api/test_openpi_connection.py -v
pytest tests/rl_rollout/test_serving_rl_rollout.py -v
```

These need the full repo environment (`pip install -e .` with torch and
vllm). They do not need a GPU.

## T2 commands

```bash
bash apps/vla-maas-gateway/run_demo.sh 8 60            # batched mock backends
MODE=serial bash apps/vla-maas-gateway/run_demo.sh 8 60
ROBOTS="1 2 4 8 16 32" DURATION=15 bash examples/online_serving/vla_maas/concurrency_sweep.sh sweep.jsonl
```

Reference CPU run (4 robots/model, 15 s, batched, 4 vCPU container): pi05 mock
p50 77 ms, p99 144 ms, 12.3 rps, success 1.0. This only checks the harness.

## T5 command

```bash
pytest -sv tests/e2e/online_serving/test_pi05_expansion.py
```

Marked `slow`, `diffusion`, `hardware_test(cuda=H100)`; boots
`vllm serve lerobot/pi05_base` with `vllm_omni/deploy/pi05.yaml`. Same pattern
for `test_pi0_expansion.py`, `test_gr00t_openpi_expansion.py`,
`test_dreamzero_expansion.py`.

## Reading results

- Batched beats serial when `throughput_rps` rises with `robots` while
  `latency_p99_ms` stays under the control budget and `late_tick_pct` stays
  near 0. Report where each layout crosses the budget.
- `success_rate` from `fleet_sim.py` is only meaningful with the mock backend
  and mock env together.
- Variance: run each point at least twice or for at least 30 s on GPU, and
  report the spread if runs disagree by more than 10%.
