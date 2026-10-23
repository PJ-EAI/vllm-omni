# Running MaaS work as cluster jobs

The unit of work is one job = one server (or mock) + optional gateway + one
client workload + one result directory. `scripts/serve_and_sweep.sh`
implements this; wrap it in whatever scheduler the cluster uses.

## Job contract

Every job must:

1. Pin the code: check out a commit, record it (`manifest.json.git_sha`).
2. Use job-unique ports (`PORT`, `GW_PORT`) when nodes are shared. Derive
   them from the job id, e.g. `PORT=$((20000 + SLURM_JOB_ID % 10000))`,
   `GW_PORT=$((PORT + 1))`.
3. Wait for readiness (`/health` for vllm serve; model load can take minutes,
   default timeout 1800 s) before sending load.
4. Write all outputs under one `OUT_DIR` on shared storage: `manifest.json`,
   the deploy yaml copy, `server.log`, `gateway.log`, `client.log`,
   `sweep.jsonl`, `gateway_metrics.json`.
5. Tear down the server process group on exit (the script does this).

## Environment setup on a GPU node

```bash
git clone -b maas https://github.com/PJ-EAI/vllm-omni.git && cd vllm-omni
pip install -e .                              # torch + vllm per docs/getting_started
pip install websockets msgpack msgpack-numpy pyyaml typing_extensions
pip install --no-deps openpi-client
hf auth login                                  # gated paligemma tokenizer for pi0/pi05
export HF_HOME=/shared/hf_cache                # avoid re-downloading per job
```

A container image built from `docker/Dockerfile.cuda` plus the client extras
avoids per-job installs.

## Slurm template

```bash
#!/usr/bin/env bash
#SBATCH --job-name=maas-pi05-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=maas_runs/slurm-%j.out
set -euo pipefail
cd /shared/vllm-omni
export PORT=$((20000 + SLURM_JOB_ID % 10000)) GW_PORT=$((PORT + 1))
export MODEL=/shared/ckpt/pi05_libero
export DEPLOY_CONFIG=examples/online_serving/vla_maas/pi05_libero.yaml
export CLIENT_ARGS="--base-camera-key observation.images.image --wrist-camera-key observation.images.image2"
export OUT_DIR=/shared/maas_runs/$SLURM_JOB_ID-batched
bash .claude/skills/vla-maas/scripts/serve_and_sweep.sh
```

`fleet_sim.py` uses CPU threads/processes for the robots; keep
`--cpus-per-task` at least `--procs` (default 3) plus a few for the server.

## Serial vs batched (the core MaaS comparison)

Submit two jobs that differ only in `max_num_seqs`:

```bash
sed 's/max_num_seqs: 16/max_num_seqs: 1/' examples/online_serving/vla_maas/pi05_libero.yaml \
  > /shared/maas_runs/pi05_libero_serial.yaml
# job A: DEPLOY_CONFIG=examples/online_serving/vla_maas/pi05_libero.yaml LABEL=batched
# job B: DEPLOY_CONFIG=/shared/maas_runs/pi05_libero_serial.yaml          LABEL=serial
```

Same GPU type, `ROBOTS`, `DURATION`, `CLIENT_ARGS`. Optionally a third job
with `SERVE_ARGS="--enforce-eager --disable-log-stats --request-batch-max-wait-ms 5"`.

Replica data parallel: run N single-GPU servers (one job each, or one job with
N GPUs and N ports) and list all of them under one model in a gateway yaml;
start the gateway and `fleet_sim.py` yourself (see `concurrency_sweep.sh` for
the shape).

## Dry run without a GPU

The same script drives the CPU mock, which checks the job plumbing before
spending GPU time:

```bash
SERVER_CMD="python3 examples/online_serving/vla_maas/mock_policy_server.py --port 18100 --max-batch 16 --action-scale 0.005" \
READY=tcp PORT=18100 GW_PORT=19100 ROBOTS="1 4" DURATION=8 LABEL=mock-batched \
CLIENT_ARGS="--action-scale 0.005" bash .claude/skills/vla-maas/scripts/serve_and_sweep.sh
```

## Agent loop for a batch of jobs

1. Plan the matrix (model × deploy yaml × fleet sizes) and write it down.
2. Dry-run one cell with the mock.
3. Submit; record job ids next to the matrix.
4. On completion, read each `OUT_DIR/manifest.json` and `sweep.jsonl`; on
   failure read the tail of `server.log` first (load errors, OOM, gated
   tokenizer), then `client.log`.
5. Merge the JSONL files and report per the result contract in `SKILL.md`.
   Keep raw files; do not overwrite earlier runs.
