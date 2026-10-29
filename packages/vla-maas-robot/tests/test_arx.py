# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ArxDriver against a fake ``bimanual``; no CAN hardware needed."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from vla_maas_robot.config import load_robot_config
from vla_maas_robot.driver import Command, make_driver
from vla_maas_robot.drivers.arx import HOME_JOINTS, ArxDriver

ROBOTS = Path(__file__).parents[1] / "robots"


class FakeSingleArm:
    """Records SDK calls; motion is instant, feedback is whatever the test sets."""

    def __init__(self, config):
        self.config = dict(config)
        self.calls = []
        self.joints = list(HOME_JOINTS) + [0.0]  # 7 values, last is the gripper
        self.speeds = [0.0] * 7
        self.pose = [0.3, -0.1, 0.2, 1.0, 0.0, 0.0, 0.0]  # xyz + quat wxyz
        self.freeze = False  # when True, commands are recorded but the arm does not move

    def _record(self, name, *args):
        self.calls.append((name, args))

    def set_joint_positions(self, positions):
        self._record("set_joint_positions", tuple(np.asarray(positions, dtype=float)))
        if not self.freeze:
            self.joints[:6] = list(np.asarray(positions, dtype=float))

    def set_ee_pose(self, pos, quat):
        self._record("set_ee_pose", tuple(pos), tuple(quat))

    def set_catch_pos(self, pos):
        self._record("set_catch_pos", float(pos))
        self.joints[6] = float(pos)

    def go_home(self):
        self._record("go_home")
        if self.reach_home:
            self.joints[:6] = list(self.home_pose)

    def protect_mode(self):
        self._record("protect_mode")

    def gravity_compensation(self):
        self._record("gravity_compensation")

    def get_joint_positions(self):
        return list(self.joints)

    def get_joint_velocities(self):
        return list(self.speeds)

    def get_ee_pose(self):
        return list(self.pose)

    def sent(self, name):
        return [a for n, a in self.calls if n == name]


@pytest.fixture
def sdk(monkeypatch):
    arms = []
    module = types.ModuleType("bimanual")
    module.SingleArm = lambda config: arms.append(FakeSingleArm(config)) or arms[-1]
    monkeypatch.setitem(sys.modules, "bimanual", module)
    return arms


def _driver(sdk, **kw):
    driver = ArxDriver(bimanual_path="/nonexistent", **kw)
    driver.connect()
    return driver, sdk[-1]


def test_registry_and_example_config(sdk):
    assert isinstance(make_driver({"kind": "arx", "bimanual_path": "/x"}), ArxDriver)
    cfg = load_robot_config(ROBOTS / "arx.yaml")
    assert isinstance(make_driver(cfg.driver), ArxDriver)
    driver = make_driver(cfg.driver)
    assert driver.spec.embodiment == "arx" and driver.spec.dof == 6
    assert driver.spec.modes == ("joint_pos",) and driver.spec.gripper_range == (0.0, 1.0)
    assert cfg.calibration.gripper_raw_range == (0.0, 1.0)


def test_connect_passes_config_and_missing_sdk(sdk, monkeypatch):
    driver, arm = _driver(sdk, can_port="can3", arm_type=2)
    assert arm.config == {"can_port": "can3", "type": 2}
    monkeypatch.delitem(sys.modules, "bimanual", raising=False)
    with pytest.raises(ImportError, match="bimanual"):
        ArxDriver().connect()


def test_state_splits_seven_values_and_reorders_quat(sdk):
    driver, arm = _driver(sdk)
    arm.joints = [0.1, 0.2, 0.3, -0.1, -0.2, 0.0, 0.8]
    arm.speeds = [1.0] * 6 + [9.9]
    arm.pose = [0.3, -0.1, 0.2, 0.0, 1.0, 0.0, 0.0]  # 180 deg about x, wxyz
    s = driver.read_state()
    np.testing.assert_allclose(s.joint_pos, [0.1, 0.2, 0.3, -0.1, -0.2, 0.0])
    np.testing.assert_allclose(s.joint_vel, [1.0] * 6)  # the 7th value stays out
    np.testing.assert_allclose(s.eef_pose, [0.3, -0.1, 0.2, 1.0, 0.0, 0.0, 0.0])  # quat -> xyzw
    assert s.gripper == pytest.approx(0.8)


def test_joint_and_gripper_commands(sdk):
    driver, arm = _driver(sdk)
    driver.command(Command("joint_pos", np.array([0.1, -0.2, 0.3, 0.0, 0.5, 0.0]), gripper=5.0))
    assert arm.sent("set_joint_positions") == [((0.1, -0.2, 0.3, 0.0, 0.5, 0.0),)]
    assert arm.sent("set_catch_pos") == [(1.0,)]  # clipped to gripper_range
    driver.close()
    assert "protect_mode" not in [n for n, _ in arm.calls]


def test_eef_command_reorders_quat(sdk):
    driver, arm = _driver(sdk)
    pose = np.array([0.25, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0])  # protocol quat xyzw = [x, y, z, w]
    driver.command(Command("eef_pose", pose))
    pos, quat = arm.sent("set_ee_pose")[0]
    assert pos == (0.25, 0.0, 0.3)
    assert quat == (0.0, 0.0, 1.0, 0.0)  # SDK wxyz = [w, x, y, z]


def test_mode_validation(sdk):
    driver, _ = _driver(sdk, modes=["joint_pos"])
    with pytest.raises(ValueError, match="configured for"):
        driver.command(Command("eef_pose", np.zeros(7)))


def test_hold_home_and_timeout(sdk):
    driver, arm = _driver(sdk, home_timeout_s=0.2)
    arm.joints = [0.1, 0.2, 0.3, -0.1, -0.2, 0.0, 0.0]
    driver.hold()
    assert arm.sent("set_joint_positions")[-1] == ((0.1, 0.2, 0.3, -0.1, -0.2, 0.0),)
    driver.home()  # commands the configured home joints, then waits for them
    assert arm.sent("set_joint_positions")[-1] == (tuple(HOME_JOINTS),)
    assert not [n for n, _ in arm.calls if n == "go_home"]
    arm.freeze = True  # commands recorded, arm parked away from home
    arm.joints = [1.0] * 7
    with pytest.raises(RuntimeError, match="home"):
        driver.home()


def test_home_commands_the_configured_joints(sdk):
    """The yaml's home_joints is what the arm is told to move to (not vendor go_home)."""
    custom = [0.1, 0.2, 0.3, -0.1, -0.2, 0.5]
    driver, arm = _driver(sdk, home_joints=custom, home_timeout_s=0.2)
    arm.joints = [1.0] * 7
    driver.home()
    assert arm.sent("set_joint_positions")[-1] == (tuple(custom),)


def test_bimanual_path_expands_user(sdk):
    driver = ArxDriver(bimanual_path="~/realarm/src/envs/arx_x5")
    assert driver.bimanual_path == str(Path("~/realarm/src/envs/arx_x5").expanduser())
    assert all(p.startswith("/") for p in driver.lib_paths)  # preload paths absolute too
    assert driver.lib_paths[0].endswith("/bimanual/api/arx_x5_src")  # api tree sits inside the package


def test_protect_on_close(sdk):
    driver, arm = _driver(sdk, protect_on_close=True)
    driver.close()
    assert ("protect_mode", ()) in arm.calls


def test_config_validation(sdk):
    with pytest.raises(ValueError, match="6 joints"):
        ArxDriver(dof=7)
    with pytest.raises(ValueError, match="arm_type"):
        ArxDriver(arm_type=5)
    with pytest.raises(ValueError, match="joint_limits"):
        ArxDriver(joint_limits=[[0, 1]] * 5)
