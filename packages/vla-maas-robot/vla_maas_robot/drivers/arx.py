# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ARX X5 (方舟 X5) arm + parallel gripper over slcan, through ``bimanual``.

Wraps the vendor stack already present on the arm PC (``realarm``): the
``bimanual`` package and its pybind11 binding ``arx_x5_python`` talk to the
X5 controller over a SocketCAN port brought up with ``slcand`` (CANable2 in
slcan mode, 1 Mbps). Nothing here is pip-installable, so the driver loads it
lazily from ``bimanual_path``. The vendor stack also needs a sourced ROS 2
environment (``AMENT_PREFIX_PATH``) and its ``.so`` directories on
``LD_LIBRARY_PATH`` — the libraries carry no SONAME, so a wrapper script
around ``robotctl`` exporting both is the supported launch path.

``SingleArm`` reports seven joint values, the seventh being the gripper;
``get_ee_pose`` returns ``[x, y, z, qw, qx, qy, qz]`` while the protocol
quaternion order is ``xyzw`` — this driver converts both ways.

The X5 URDF ships placeholder limits, so ``joint_limits`` defaults to a
conservative +/-2.9 rad until the real values are confirmed from the arm's
manual and set in the robot yaml. Keep unverified arms in ``dry_run``.

There is no speed knob: the vendor API exposes no speed argument (the
undocumented ``arx_x(500, 2000, 10)`` gain triple is the only candidate and
its semantics are unknown), so motion between targets runs at the vendor's
default speed. The safety filter's per-tick step limit bounds the control
loop, but ``home()`` sends its whole target at once — first live sessions
must use ``--mode slow`` and a cleared workspace.
"""

from __future__ import annotations

import ctypes
import importlib
import sys
import threading
import time
from pathlib import Path

import numpy as np

from vla_maas_robot.driver import Command, RobotDriver, RobotSpec, RobotState

# Placeholder until the manual's values land in the robot yaml (URDF says +/-10).
JOINT_LIMITS = np.tile([-2.9, 2.9], (6, 1))
HOME_JOINTS = [-0.0, 0.17, 0.23, -0.06, 0.0, 0.0]  # realarm's env_reset_joint
# Best-effort preload by absolute path; the shipped libs have no SONAME, so
# this cannot satisfy the binding's NEEDED entries — LD_LIBRARY_PATH in a
# robotctl wrapper is the supported launch path (see the module docstring).
PRELOAD_LIBS = ("libkdl_parser.so", "libx5_kinematic_solver.so", "libarx_x5_src.so")


class ArxDriver(RobotDriver):
    def __init__(
        self,
        can_port: str = "can1",
        bimanual_path: str | None = None,
        lib_paths: list[str] | None = None,
        arm_type: int = 0,
        embodiment: str = "arx",
        dof: int = 6,
        modes: tuple[str, ...] | list[str] = ("joint_pos", "eef_pose"),
        max_joint_vel: float = 2.0,
        joint_limits: list[list[float]] | None = None,
        gripper_range: tuple[float, float] = (0.0, 1.0),
        home_joints: list[float] | None = None,
        home_timeout_s: float = 10.0,
        home_tol_rad: float = 0.02,
        protect_on_close: bool = False,
    ) -> None:
        if dof != 6:
            raise ValueError(f"ARX X5 has 6 joints, got dof={dof}")
        if arm_type not in (0, 1, 2):
            raise ValueError(f"arm_type must be 0 (x5), 1 (x5_master) or 2 (x5_2025), got {arm_type}")
        unknown = set(modes) - {"joint_pos", "eef_pose"}
        if unknown:
            raise ValueError(f"ArxDriver modes are joint_pos and eef_pose, got {sorted(unknown)}")
        limits = np.asarray(joint_limits if joint_limits is not None else JOINT_LIMITS, dtype=np.float64)
        if limits.shape != (6, 2) or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError(f"joint_limits must be six [lo, hi] rad pairs, got {limits.tolist()}")
        self.spec = RobotSpec(
            embodiment=embodiment,
            dof=6,
            joint_limits=limits,
            max_joint_vel=np.full(6, max_joint_vel),
            modes=tuple(modes),
            gripper_range=gripper_range,
            native_hz=200.0,
        )
        self.can_port = can_port
        self.bimanual_path = str(Path(bimanual_path).expanduser()) if bimanual_path else None
        base = [str(Path(p).expanduser()) for p in lib_paths] if lib_paths else None
        if base is None and bimanual_path:
            root = Path(bimanual_path).expanduser() / "bimanual"  # the api/ tree sits inside the package
            base = [str(root / "api" / "arx_x5_src"), str(root / "api"), "/opt/ros/humble/lib"]
        self.lib_paths = base or []
        self.arm_type = int(arm_type)
        self.home_joints = np.asarray(home_joints if home_joints is not None else HOME_JOINTS, dtype=np.float64)
        self.home_timeout_s = home_timeout_s
        self.home_tol_rad = home_tol_rad
        self.protect_on_close = protect_on_close
        self._arm = None
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------------

    def _import_singlearm(self):
        """Return ``bimanual.SingleArm``, loading the vendor stack on demand."""
        for name in PRELOAD_LIBS:
            for directory in self.lib_paths:
                lib = Path(directory) / name
                if lib.is_file():
                    try:
                        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass  # the import below reports what actually failed
                    break
        if self.bimanual_path and self.bimanual_path not in sys.path:
            sys.path.insert(0, self.bimanual_path)
        try:
            return importlib.import_module("bimanual").SingleArm
        except ImportError as exc:
            py = f"{sys.version_info.major}.{sys.version_info.minor}"
            raise ImportError(
                f"ArxDriver needs the vendor 'bimanual' package (realarm's ARX X5 stack): "
                f"set bimanual_path to the directory containing it (tried sys.path + "
                f"{self.bimanual_path!r}). This venv is Python {py}; the binding is built "
                f"for CPython 3.11 (arx_x5_python.cpython-311), so a 3.10 or 3.12 venv "
                f"fails here no matter the path. The stack also needs a ROS 2 environment "
                f"(source /opt/ros/humble/setup.bash; the vendor lib requires "
                f"AMENT_PREFIX_PATH) and its .so directories on LD_LIBRARY_PATH — the "
                f"libs have no SONAME, so wrap robotctl in a script that exports it "
                f"(see 'ARX X5 specifics' in the vla-maas skill); {exc}"
            ) from exc

    def connect(self) -> None:
        single_arm = self._import_singlearm()
        self._arm = single_arm({"can_port": self.can_port, "type": self.arm_type})

    def close(self) -> None:
        if self._arm is None:
            return
        try:
            if self.protect_on_close:
                self._arm.protect_mode()
        finally:
            self._arm = None

    # -- state ------------------------------------------------------------------

    def _require(self):
        if self._arm is None:
            raise RuntimeError("ArxDriver is not connected")
        return self._arm

    def read_state(self) -> RobotState:
        arm = self._require()
        now = time.monotonic()
        q7 = np.asarray(arm.get_joint_positions(), dtype=np.float64)
        qd7 = np.asarray(arm.get_joint_velocities(), dtype=np.float64)
        pose = np.asarray(arm.get_ee_pose(), dtype=np.float64)  # [x, y, z, qw, qx, qy, qz]
        quat_xyzw = pose[[4, 5, 6, 3]]  # SDK [w, x, y, z] -> protocol [x, y, z, w]
        return RobotState(
            t=now,
            joint_pos=q7[:6],
            joint_vel=qd7[:6],
            eef_pose=np.concatenate((pose[:3], quat_xyzw)),
            gripper=float(q7[6]),
        )

    # -- commands ---------------------------------------------------------------

    def command(self, cmd: Command) -> None:
        if cmd.mode not in self.spec.modes:
            raise ValueError(f"ArxDriver is configured for {self.spec.modes}, got {cmd.mode}")
        with self._lock:
            arm = self._require()
            if cmd.mode == "joint_pos":
                arm.set_joint_positions(np.asarray(cmd.target, dtype=np.float64))
            else:
                pose = np.asarray(cmd.target, dtype=np.float64)
                quat_wxyz = pose[[6, 3, 4, 5]]  # protocol xyzw -> SDK wxyz
                arm.set_ee_pose(pose[:3], quat_wxyz)
            if cmd.gripper is not None:
                lo, hi = self.spec.gripper_range
                arm.set_catch_pos(float(np.clip(cmd.gripper, lo, hi)))

    def hold(self) -> None:
        with self._lock:
            self._require().set_joint_positions(self.read_state().joint_pos)

    def home(self) -> None:
        """Move to ``home_joints`` by commanding them (same semantics as Piper).

        The vendor ``go_home()`` target is undocumented, so it is not used:
        the configured joints are both commanded and validated.
        """
        with self._lock:
            self._require().set_joint_positions(self.home_joints)
        deadline = time.monotonic() + self.home_timeout_s
        while time.monotonic() < deadline:
            if np.max(np.abs(self.read_state().joint_pos - self.home_joints)) < self.home_tol_rad:
                return
            time.sleep(0.05)
        raise RuntimeError(f"ARX X5 did not reach home within {self.home_timeout_s}s")
