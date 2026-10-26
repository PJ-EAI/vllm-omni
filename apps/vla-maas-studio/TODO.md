# VLA MaaS Studio TODO

## AgileX Piper: real-arm checks (not run yet)

The Piper driver (`packages/vla-maas-robot/vla_maas_robot/drivers/piper.py`) and
the studio's robot mode were tested only against a fake `piper_sdk` and the
MuJoCo twin. Check these on the real arm with
`studio.py --robot-config <piper.yaml> --hardware`, in `dry_run` first, then
`slow`, then `live`. Record the result and the commit next to each item.

- [ ] CAN and enable: `bash can_activate.sh can0 1000000`; `robotctl check`
      connects, enables the motors without motion and reads all six joints.
- [ ] Rest pose and home: from rest, `robotctl home` reaches the ready-pose
      `home_joints` without clamping; joint 2/3 limits match `JOINT_LIMITS`.
- [ ] Joint signs: in `dry_run`, each joint in the studio detail view moves in
      the same direction as the arm when moved by hand.
- [ ] Gripper: 0–70 mm maps to protocol 0 (closed) – 1 (open); check
      `calibration.gripper.raw_range` and the effort setting.
- [ ] End pose: SDK end pose versus the studio's `eef xyz`; confirm the
      roll-pitch-yaw convention before enabling `eef_pose` actions. The twin
      reports `link6`, which may differ from the SDK's end-effector frame.
- [ ] Cameras: `front` (RealSense) and `wrist` (UVC) open, roles are right,
      and the frame skew warning stays at 0.
- [ ] Safety: `max_joint_step_rad`, `workspace` and `slow_scale` feel right in
      `slow`; a deliberate out-of-workspace target is clamped and five in a row
      stop the episode with the arm holding.
- [ ] Stop and faults: the studio Stop button and a gateway stop hold the arm
      with motors enabled; unplugging CAN triggers the stale-feedback stop
      within `feedback_timeout_s`.
- [ ] Speed: `speed_pct` and the model's `control_hz` give smooth MOVE J
      motion without underruns or late ticks.
- [ ] Twin versus real: the same robot file and model show matching action
      direction and size in the twin and in `dry_run` on the arm.
