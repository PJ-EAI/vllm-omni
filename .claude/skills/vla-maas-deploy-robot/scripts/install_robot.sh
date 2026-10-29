#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Install the robot client on an arm PC into its own venv: protocol + robot
# packages and the SDK extras only. Never installs vllm, torch or the gateway.
#
#   bash install_robot.sh                     # Piper + RealSense + UVC + JPEG
#   EXTRAS=jpeg bash install_robot.sh         # CPU rehearsal (mock arm), no SDKs
#   PYTHON=python3.11 bash install_robot.sh   # ARX X5: the venv MUST be 3.11
#                                             # (vendor binding arx_x5_python.cpython-311)
#
# Env: VENV (default .venv-robot), PYTHON (default python3), EXTRAS (default piper,realsense,uvc,jpeg)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
VENV="${VENV:-$ROOT/.venv-robot}"
PYTHON="${PYTHON:-python3}"
EXTRAS="${EXTRAS:-piper,realsense,uvc,jpeg}"

"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "$ROOT/packages/vla-maas-protocol" "$ROOT/packages/vla-maas-robot[$EXTRAS]"
"$VENV/bin/python" - <<'PY'
import sys

import vla_maas_robot.robotctl  # noqa: F401

leaked = {"vllm", "vllm_omni", "torch"} & set(sys.modules)
assert not leaked, f"server-side modules imported on the robot: {leaked}"
print("robot client ok:", sys.executable)
PY
echo "robotctl: $VENV/bin/robotctl"
echo "ARX X5 arms: this venv must be Python 3.11 (PYTHON=python3.11 above); the arx driver's"
echo "vendor binding is built for CPython 3.11 and fails to import on any other version."
