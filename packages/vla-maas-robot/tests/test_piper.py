# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PiperDriver against a fake ``piper_sdk``; no CAN hardware needed."""

import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace as Ns

import numpy as np
import pytest
from vla_maas_protocol import geometry as geo
from vla_maas_robot.config import load_robot_config
from vla_maas_robot.driver import Command, make_driver
from vla_maas_robot.drivers.piper import MDEG_PER_RAD, PiperDriver
from vla_maas_robot.safety import SafetyConfig, SafetyFilter

ROBOTS = Path(__file__).parents[1] / "robots"


class FakePiper:
    """Records SDK calls; feedback is whatever the test puts in ``joints`` (0.001 deg)."""

    def __init__(self, can_name="can0"):
        self.can_name, self.calls, self.enable_after, self.stamp = can_name, [], 2, 1.0
        self.joints = [0] * 6
        self.speeds = [0] * 6
        self.pose = [300_000, 0, 200_000, 0, 90_000, 0]  # 0.3 m, 0, 0.2 m; RY = 90 deg
        self.gripper = 35_000  # 35 mm

    def __getattr__(self, name):
        if name[0].isupper():
            return lambda *a: self.calls.append((name, a))
        raise AttributeError(name)

    def EnablePiper(self):
        self.calls.append(("EnablePiper", ()))
        self.enable_after -= 1
        return self.enable_after <= 0

    def GetArmJointMsgs(self):
        self.stamp += 0.005
        return Ns(time_stamp=self.stamp, joint_state=Ns(**{f"joint_{i + 1}": v for i, v in enumerate(self.joints)}))

    def GetArmHighSpdInfoMsgs(self):
        return Ns(**{f"motor_{i + 1}": Ns(motor_speed=v) for i, v in enumerate(self.speeds)})

    def GetArmEndPoseMsgs(self):
        keys = ["X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis"]
        return Ns(end_pose=Ns(**dict(zip(keys, self.pose))))

    def GetArmGripperMsgs(self):
        return Ns(gripper_state=Ns(grippers_angle=self.gripper))

    def sent(self, name):
        return [a for n, a in self.calls if n == name]


@pytest.fixture
def sdk(monkeypatch):
    arms = []
    module = types.ModuleType("piper_sdk")
    module.C_PiperInterface_V2 = lambda can_name="can0": arms.append(FakePiper(can_name)) or arms[-1]
    monkeypatch.setitem(sys.modules, "piper_sdk", module)
    return arms


def _driver(sdk, **kw):
    driver = PiperDriver(**kw)
    driver.connect()
    return driver, sdk[-1]


def test_connect_enables_and_never_moves(sdk):
    driver, arm = _driver(sdk, can_name="can1", speed_pct=20)
    assert arm.can_name == "can1"
    assert [n for n, _ in arm.calls] == ["ConnectPort", "EnablePiper", "EnablePiper", "MotionCtrl_2"]
    assert arm.sent("MotionCtrl_2") == [(0x01, 0x01, 20, 0x00)]  # CAN control, MOVE J
    driver.close()
    assert "DisablePiper" not in [n for n, _ in arm.calls]  # a disabled Piper drops


def test_state_is_si(sdk):
    driver, arm = _driver(sdk)
    arm.joints = [90_000, 45_000, -30_000, 0, 10_000, -120_000]
    arm.speeds = [500, 0, 0, 0, 0, -250]
    s = driver.read_state()
    np.testing.assert_allclose(s.joint_pos, np.deg2rad([90, 45, -30, 0, 10, -120]))
    np.testing.assert_allclose(s.joint_vel[[0, 5]], [0.5, -0.25])
    np.testing.assert_allclose(s.eef_pose[:3], [0.3, 0.0, 0.2])
    np.testing.assert_allclose(geo.quat_to_rpy(s.eef_pose[3:]), [0, np.pi / 2, 0], atol=1e-6)
    assert s.gripper == pytest.approx(0.035)


def test_joint_and_gripper_commands(sdk):
    driver, arm = _driver(sdk)
    driver.command(Command("joint_pos", np.deg2rad([10, 20, -30, 0, 5, 0]), gripper=0.5))
    assert arm.sent("JointCtrl") == [(10_000, 20_000, -30_000, 0, 5_000, 0)]
    assert arm.sent("GripperCtrl") == [(70_000, 1000, 0x01, 0)]  # clipped to the 70 mm stroke
    with pytest.raises(ValueError):
        driver.command(Command("eef_pose", np.zeros(7)))  # not in modes


def test_eef_commands_switch_move_mode(sdk):
    driver, arm = _driver(sdk, modes=["joint_pos", "eef_pose"])
    pose = np.concatenate(([0.25, -0.1, 0.3], geo.rpy_to_quat(np.deg2rad([180, 0, 90]))))
    driver.command(Command("eef_pose", pose))
    driver.command(Command("eef_pose", pose))
    x, y, z, rx, ry, rz = arm.sent("EndPoseCtrl")[0]
    assert (x, y, z) == (250_000, -100_000, 300_000)
    assert abs(abs(rx) - 180_000) <= 1 and abs(ry) <= 1 and abs(rz - 90_000) <= 1
    driver.command(Command("joint_pos", np.zeros(6)))
    modes = [a[1] for a in arm.sent("MotionCtrl_2")]
    assert modes == [0x01, 0x00, 0x01]  # MOVE J, MOVE P once, back to MOVE J


def test_hold_home_and_stale_feedback(sdk):
    driver, arm = _driver(sdk, home_joints=[0, 0.5, -0.5, 0, 0, 0], home_timeout_s=0.2, feedback_timeout_s=0.05)
    arm.joints = [1_000] * 6
    driver.hold()
    assert arm.sent("JointCtrl")[-1] == (1_000,) * 6
    arm.joints = [round(v * MDEG_PER_RAD) for v in [0, 0.5, -0.5, 0, 0, 0]]
    driver.home()
    assert arm.sent("JointCtrl")[-1] == (0, 28_648, -28_648, 0, 0, 0)
    arm.joints = [0] * 6
    with pytest.raises(RuntimeError, match="home"):
        driver.home()
    arm.GetArmJointMsgs = lambda: Ns(time_stamp=5.0, joint_state=Ns(**{f"joint_{i}": 0 for i in range(1, 7)}))
    driver.read_state()
    time.sleep(0.08)
    with pytest.raises(RuntimeError, match="feedback"):
        driver.read_state()


def test_enable_timeout_and_config(sdk, monkeypatch):
    monkeypatch.setattr(FakePiper, "EnablePiper", lambda self: False)
    with pytest.raises(RuntimeError, match="enable"):
        PiperDriver(enable_timeout_s=0.05).connect()
    with pytest.raises(ValueError):
        PiperDriver(dof=7)
    cfg = load_robot_config(ROBOTS / "piper.yaml")
    driver = make_driver(cfg.driver)
    assert isinstance(driver, PiperDriver) and driver.spec.gripper_range == (0.0, 0.07)
    assert cfg.calibration.gripper_raw_range == (0.0, 0.07)


def test_rest_pose_passes_example_safety(sdk):
    """Piper rests at its joint-2/3 limits; the example config must not clamp rest-pose targets."""
    cfg = load_robot_config(ROBOTS / "piper.yaml")
    driver, _ = _driver(sdk)
    safety = SafetyFilter(driver.spec, SafetyConfig(**{**cfg.safety.__dict__, "start_mode": "live"}), 20.0)
    state = driver.read_state()
    state.eef_pose[:3] = [0.3, 0.0, 0.2]
    safety.check_state(state)
    for _ in range(10):
        safety.filter(Command("joint_pos", np.zeros(6), gripper=0.07), state)
    assert safety.stats.clamped == 0
