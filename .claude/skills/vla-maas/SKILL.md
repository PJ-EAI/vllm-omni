---
name: vla-maas
description: Orient in vllm-omni and develop, run, test and benchmark the VLA MaaS stack (OpenPI policy serving, Pi05 request batching, the API-key gateway, the standard Robot protocol and robot client with the AgileX Piper driver, the studio browser console with its MuJoCo Piper twin, mock backends, fleet/LIBERO clients), locally or as cluster jobs. Use for any VLA serving, MaaS gateway, real-robot or robot-policy benchmark or embodied-model task in this repo.
---

# VLA MaaS on vllm-omni

## Purpose

Give an agent (interactive or running inside a cluster job) enough of the
framework to change VLA serving code safely, and a fixed set of commands to
run, test and benchmark the MaaS application without re-deriving them.

Scope today: the VLA Model-as-a-Service path. Later work (hierarchical
embodied systems: planner + VLA + world model) builds on the same layers; see
"Extending" below.

Link convention: paths are relative to the repo root. Skill-local files are
under `.claude/skills/vla-maas/`.

## Read first, by task

| Task | Read |
| --- | --- |
| Understand how a request flows, where to change what | [references/architecture.md](references/architecture.md) |
| Gateway API, OpenPI wire protocol, observation schema, client tools | [references/maas-stack.md](references/maas-stack.md) |
| Connect a real arm (robot client, Piper), test it in the studio twin, bring-up order | [references/robots.md](references/robots.md) |
| Which VLA models exist, their deploy configs and status | [references/models.md](references/models.md) |
| Which test or benchmark proves a claim, exact commands | [references/testing.md](references/testing.md) |
| Running any of this as a batch/cluster job | [references/cluster-jobs.md](references/cluster-jobs.md) |

Also read `docs/contributing/ci/lessons.md` before code changes (repo rule),
and use `vllm-omni-test` when adding tests, `precheck-pr` before a PR.

## The stack in one picture

```text
real robot (packages/vla-maas-robot, robotctl)      fleet_sim / libero_eval (openpi-client)
apps/vla-maas-studio --robot-config (twin/real)      apps/vla-maas-studio (OpenPI scenes)
  │  wss://gw/v1/robot/<model>/act?robot_id=..       │  ws://gw:9000/v1/robot/<model>/openpi
  │  standard Robot protocol (packages/vla-maas-protocol), model-agnostic
  ▼                                                  ▼
apps/vla-maas-gateway/gateway.py   keys · least-loaded routing · model_adapters (/act) · relay · /metrics
        │  ws://host:port/v1/realtime/robot/openpi   (one entry per backend in gateway yaml)
        ▼
vllm serve <ckpt> --omni --deploy-config <yaml>      or  mock_policy_server.py (CPU)
  entrypoints/openai/api_server.py  → entrypoints/openpi/{connection,serving}.py
  → AsyncOmni → diffusion stage → request scheduler (batches up to max_num_seqs)
  → DiffusionModelRunner → <Model>Pipeline.forward(req | DiffusionRequestBatch)
```

## Default workflow

1. **Locate the layer.** Use the table in `references/architecture.md` to find
   which file owns the behaviour. Do not put model logic in the gateway or
   serving layer; do not put routing/auth in the model.
2. **Prove the loop on CPU first.** Run the smoke test; it needs no GPU,
   torch or vllm, only the client extras:

   ```bash
   pip install websockets msgpack msgpack-numpy pyyaml numpy typing_extensions
   pip install --no-deps openpi-client
   bash .claude/skills/vla-maas/scripts/maas_smoke.sh
   ```

   It starts a mock backend and the gateway, checks `/healthz`, 401 on a bad
   key, 404 on an unknown model, `/v1/models`, a mock LIBERO eval through the
   gateway, and `/metrics` counts. Exit code 0 means the loop works.
3. **Change code**, then run the narrowest test that covers it
   (`references/testing.md` maps change → test).
4. **GPU / scale evidence** goes through
   `.claude/skills/vla-maas/scripts/serve_and_sweep.sh` (one job = one server +
   one sweep + a result directory). See `references/cluster-jobs.md`.
5. **Report** with the result contract below.

## Common tasks

- **Run the live demo** (two mock models, fleets, dashboard):
  `bash apps/vla-maas-gateway/run_demo.sh 8 60`, `MODE=serial` for the
  unbatched baseline. Dashboard at `http://<host>:9000/dashboard`, key
  `dev-key-change-me`.
- **Open the browser debugging console** (scene/task picker, 1–64 sessions,
  live camera grid, client vs gateway latency): `bash
  apps/vla-maas-studio/run_demo.sh` → `http://127.0.0.1:19100`. Add
  `--physics libero` to `studio.py` for the real simulator; see
  `apps/vla-maas-studio/README.md`. For a robot file on the standard Robot
  protocol (Piper twin or real arm): `PHYSICS=piper bash
  apps/vla-maas-studio/run_demo.sh`, or `studio.py --robot-config <robot.yaml>
  [--hardware]`.
- **Simulated multi-arm fleet with video** (robosuite/LIBERO, headless EGL):
  `bash examples/online_serving/vla_maas/sim_fleet_demo.sh <scenario.yaml> <duration-s>`
  or `sim_fleet.py --scenario ... --render-gpus 0,1 --video-dir out/videos --video-grid`.
  Scenarios in `examples/online_serving/vla_maas/scenarios/`; install and EGL
  notes in that README ("Simulated multi-arm fleet").
- **Serving-layout sweep on CPU mocks** (serial vs replicas vs batched):
  `ROBOTS="1 4 16" DURATION=10 bash examples/online_serving/vla_maas/concurrency_sweep.sh out.jsonl`.
- **Serve a real π0.5 checkpoint**: `vllm serve <ckpt> --omni --port 8000
  --deploy-config examples/online_serving/vla_maas/pi05_libero.yaml --enforce-eager`.
  Camera keys sent by clients must match the checkpoint's `input_features`.
- **Connect a robot (any arm) through the standard protocol**: driver in
  `packages/vla-maas-robot/vla_maas_robot/drivers/` (`mock`, `piper` today),
  robot yaml, `robotctl check` then `robotctl run` (dry_run first). The model
  needs an `adapter:` block in the gateway yaml
  (`apps/vla-maas-gateway/model_adapters.py` docstring). Without hardware, run
  the same yaml in the studio's MuJoCo twin first; full order and the Piper
  rest-pose pitfall are in `references/robots.md`.
- **Put a model behind the gateway**: add `models.<name>.backends: [ws://host:port/v1/realtime/robot/openpi, ...]`
  to a gateway yaml (copy `apps/vla-maas-gateway/gateway.example.yaml`).
  Several backends = replica data parallel, least-loaded per session.
- **Enable batching for another VLA pipeline**: follow the π0.5 pattern
  (`supports_request_batch = True`, `forward` accepting `DiffusionRequestBatch`,
  per-request generators and per-request error isolation). Details and the
  unit test to copy are in `references/architecture.md`.

## Guardrails

- The mock backend's latency/batch costs are assumptions. Never report mock
  numbers as GPU performance; label them `mock` (lessons L002).
- Benchmark claims name model, checkpoint, deploy yaml, `max_num_seqs`,
  dtype, `num_inference_steps`, GPU type and count, git SHA and client
  command. Compare serial vs batched with the same fleet and duration.
- Only Panda on real LIBERO (`kind: libero`) gives a meaningful π0.5 LIBERO
  success rate; other arms/robosuite tasks in `sim_fleet.py` measure serving
  under real rendering, not policy quality.
- Real-checkpoint success rate needs the real LIBERO simulator. `fleet_sim.py`
  against a real checkpoint uses the mock env, so its success rate is
  meaningless there; read only latency/throughput.
- One port set per job. Never assume 8000/9000 are free on a shared node;
  scripts take `PORT`/`GW_PORT` overrides.
- Gateway is single-tenant: no TLS, no quotas, no health-checked failover
  (listed in `apps/vla-maas-gateway/README.md`). Do not expose it publicly.
- Work on the `maas` branch line of `PJ-EAI/vllm-omni` unless told otherwise.

## Result contract

End every run or benchmark with:

1. What ran: command(s), git SHA, host/GPU, deploy yaml, mock or real.
2. Where results are: the JSONL/JSON paths (and the job result dir).
3. A short table of the key fields (`robots`, `throughput_rps`,
   `latency_p50_ms`, `latency_p99_ms`, `late_tick_pct`, `underruns`,
   `success_rate` when meaningful).
4. What it does and does not show, in one or two sentences.

## Extending to hierarchical embodied systems

Treat each level as its own served model with its own deploy yaml and
gateway route (`/v1/robot/<model>/openpi` for action policies). A planner or
world model that is not an OpenPI policy needs a new endpoint family in
`vllm_omni/entrypoints/` and a matching gateway route; keep the orchestration
(which level calls which, at what rate) in a client or app under `apps/`,
not inside a pipeline. Update this skill's references when such a layer lands.
