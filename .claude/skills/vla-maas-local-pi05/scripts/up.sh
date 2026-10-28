#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# One machine: π0.5 backend (vllm serve, bfloat16) -> gateway (+ /dashboard) -> studio with the
# Piper MuJoCo twin on /act. Run setup_envs.sh first. Stop with down.sh.
#
#   bash up.sh
#   BACKENDS=ws://gpu-box:8001/v1/realtime/robot/openpi bash up.sh   # model served elsewhere
#   MOCK=1 bash up.sh                                                 # CPU mock instead of π0.5
#
# Env: CKPT (default lerobot/pi05_base), DTYPE (default bfloat16), MAX_NUM_SEQS (default 1),
#      DENOISE_STEPS (0 = deploy default), BACKEND_PORT (8001), GW_PORT (9000), STUDIO_PORT (9100),
#      ROBOTS (default piper-01), RUN_DIR (default maas_runs/local-pi05), MIN_GPU_MIB (default 9500),
#      FORCE=1 (skip the GPU memory check), MENAGERIE, SKIP_STUDIO=1,
#      MUJOCO_GL (twin rendering; egl by default, osmesa on machines without EGL),
#      TWIN_MAX_VIOLATIONS (default 100000: clamped steps in a row before the twin's episode stops)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
SERVER="$ROOT/.claude/skills/vla-maas-deploy-server/scripts"
CKPT="${CKPT:-lerobot/pi05_base}"
DTYPE="${DTYPE:-bfloat16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
DENOISE_STEPS="${DENOISE_STEPS:-0}"
BACKEND_PORT="${BACKEND_PORT:-8001}"
GW_PORT="${GW_PORT:-9000}"
STUDIO_PORT="${STUDIO_PORT:-9100}"
ROBOTS="${ROBOTS:-piper-01}"
RUN_DIR="${RUN_DIR:-$ROOT/maas_runs/local-pi05}"
MIN_GPU_MIB="${MIN_GPU_MIB:-9500}"  # π0.5 bf16 batch 1 peaks near 8.9 GiB (recipes/lerobot/Pi05.md)
MENAGERIE="${MENAGERIE:-$HOME/.cache/mujoco_menagerie}"
MOCK="${MOCK:-0}"
MAAS_PY="$ROOT/.venv-maas/bin/python"
mkdir -p "$RUN_DIR"

[ -x "$MAAS_PY" ] || { echo "missing .venv-maas; run setup_envs.sh" >&2; exit 1; }

# 1. Backend ------------------------------------------------------------------
if [ -n "${BACKENDS:-}" ]; then
  echo "using remote backends: $BACKENDS"
else
  if [ "$MOCK" != 1 ]; then
    [ -x "$ROOT/.venv-pi05/bin/vllm" ] || { echo "missing .venv-pi05; run setup_envs.sh" >&2; exit 1; }
    mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | tr -d ' ')"
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
    echo "GPU 0: $name, $mib MiB"
    if [ "$mib" -lt "$MIN_GPU_MIB" ] && [ "${FORCE:-0}" != 1 ]; then
      echo "π0.5 in $DTYPE needs about 8.9 GiB at batch 1; $mib MiB will most likely OOM." >&2
      echo "Serve it on a bigger GPU and pass BACKENDS=ws://<host>:<port>/v1/realtime/robot/openpi, or FORCE=1 to try anyway." >&2
      exit 1
    fi
    # The shipped deploy file with dtype and batching changed; comments and keys stay as shipped.
    sed -E "s/^dtype: .*/dtype: $DTYPE/; s/^( *)max_num_seqs: .*/\1max_num_seqs: $MAX_NUM_SEQS/" \
      vllm_omni/deploy/pi05.yaml >"$RUN_DIR/pi05.yaml"
    grep -qE "^dtype: $DTYPE" "$RUN_DIR/pi05.yaml" || { echo "could not set dtype in $RUN_DIR/pi05.yaml" >&2; exit 1; }
  fi
  # A real checkpoint downloads and loads for minutes; serve_pi05.sh waits until the port is up.
  CKPT="$CKPT" DEPLOY="$RUN_DIR/pi05.yaml" GPUS=0 BASE_PORT="$BACKEND_PORT" HOST=127.0.0.1 \
    ADVERTISE=127.0.0.1 LOG_DIR="$RUN_DIR/backends" VLLM="$ROOT/.venv-pi05/bin/vllm" PYTHON="$MAAS_PY" \
    MOCK="$MOCK" bash "$SERVER/serve_pi05.sh" | tee "$RUN_DIR/serve.out"
  BACKENDS="$(sed -n 's/^BACKENDS=//p' "$RUN_DIR/serve.out" | tail -1)"
  [ -n "$BACKENDS" ] || { echo "backend did not start; see $RUN_DIR/backends/" >&2; exit 1; }
fi

# 2. Gateway (keys are kept across runs) ---------------------------------------
"$MAAS_PY" "$SERVER/render_gateway.py" --out "$RUN_DIR/deploy" --model pi05_piper --backends "$BACKENDS" \
  --robots "$ROBOTS" --state-dim 32 --control-hz 30 --gripper-range 0 1 --denoise-steps "$DENOISE_STEPS"
CONFIG="$RUN_DIR/deploy/gateway.yaml" GW_PORT="$GW_PORT" PYTHON="$MAAS_PY" LOG_DIR="$RUN_DIR/gateway" \
  bash "$SERVER/start_gateway.sh"
# shellcheck disable=SC1091
source "$RUN_DIR/deploy/keys/operator.env"
curl -sf -H "Authorization: Api-Key $VLA_MAAS_OPERATOR_KEY" "http://127.0.0.1:$GW_PORT/v1/models/pi05_piper" >/dev/null \
  || { echo "gateway does not serve pi05_piper; see $RUN_DIR/gateway/gateway.log" >&2; exit 1; }

# 3. Studio with the Piper twin -------------------------------------------------
if [ "${SKIP_STUDIO:-0}" != 1 ]; then
  GATEWAY="http://127.0.0.1:$GW_PORT" API_KEY="$VLA_MAAS_OPERATOR_KEY" ROBOT_ID="${ROBOTS%%,*}" \
    STUDIO_PORT="$STUDIO_PORT" RUN_DIR="$RUN_DIR" MENAGERIE="$MENAGERIE" bash "$(dirname "${BASH_SOURCE[0]}")/twin.sh"
fi

cat <<INFO

backend    $BACKENDS  ($( [ "$MOCK" = 1 ] && echo "CPU mock" || echo "$CKPT, $DTYPE" ))
dashboard  http://127.0.0.1:$GW_PORT/dashboard   operator key: $VLA_MAAS_OPERATOR_KEY
studio     http://127.0.0.1:$STUDIO_PORT          model pi05_piper, robot ${ROBOTS%%,*} (MuJoCo twin)
logs       $RUN_DIR/{backends,gateway,studio}
stop       bash $(dirname "${BASH_SOURCE[0]}")/down.sh
INFO
