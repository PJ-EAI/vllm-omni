#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Create the two uv environments for a one-machine π0.5 + gateway + Piper twin run:
#   .venv-pi05  vllm + vllm-omni (+ torch) for the GPU backend
#   .venv-maas  gateway, studio and robot client; no vllm, no torch
# and sparse-clone the Piper model from MuJoCo Menagerie. Existing envs are kept.
#
#   bash setup_envs.sh
#
# Env: ONLY=pi05|maas (one env), VLLM_WHEEL (vLLM wheel or spec), TORCH_BACKEND (default cu130),
#      PY (default 3.12), MENAGERIE (default ~/.cache/mujoco_menagerie), RECREATE=1 (rebuild envs)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
PY="${PY:-3.12}"
ONLY="${ONLY:-}"
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
# The tested vLLM 0.32.0 build from docs/getting_started/installation/gpu/cuda.inc.md (CUDA 13 driver).
VLLM_WHEEL="${VLLM_WHEEL:-vllm @ https://wheels.vllm.ai/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl}"
MENAGERIE="${MENAGERIE:-$HOME/.cache/mujoco_menagerie}"
command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }

fresh() {  # venv dir -> 0 if it must be (re)created
  [ "${RECREATE:-0}" = 1 ] && rm -rf "$1"
  [ ! -x "$1/bin/python" ]
}

if [ -z "$ONLY" ] || [ "$ONLY" = pi05 ]; then
  if fresh .venv-pi05; then
    uv venv .venv-pi05 --python "$PY"
  fi
  VIRTUAL_ENV="$ROOT/.venv-pi05" uv pip install --torch-backend="$TORCH_BACKEND" "$VLLM_WHEEL"
  VIRTUAL_ENV="$ROOT/.venv-pi05" uv pip install -e .
  .venv-pi05/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available() and torch.cuda.get_device_name(0))"
  echo ".venv-pi05 ready ($(.venv-pi05/bin/vllm --version 2>/dev/null | tail -1))"
fi

if [ -z "$ONLY" ] || [ "$ONLY" = maas ]; then
  if fresh .venv-maas; then
    uv venv .venv-maas --python "$PY"
  fi
  VIRTUAL_ENV="$ROOT/.venv-maas" uv pip install websockets msgpack msgpack-numpy pyyaml numpy typing_extensions pillow mujoco
  VIRTUAL_ENV="$ROOT/.venv-maas" uv pip install --no-deps openpi-client
  VIRTUAL_ENV="$ROOT/.venv-maas" uv pip install ./packages/vla-maas-protocol ./packages/vla-maas-robot
  .venv-maas/bin/python -c "import websockets, mujoco, vla_maas_protocol, vla_maas_robot, openpi_client" && echo ".venv-maas ready"

  if [ ! -f "$MENAGERIE/agilex_piper/piper.xml" ]; then
    [ -d "$MENAGERIE/.git" ] || git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/google-deepmind/mujoco_menagerie "$MENAGERIE"
    git -C "$MENAGERIE" sparse-checkout add agilex_piper
  fi
  echo "Piper model: $MENAGERIE/agilex_piper"
fi
