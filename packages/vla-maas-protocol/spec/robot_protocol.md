# Standard Robot protocol (`maas-robot/1`)

The only interface between a robot and the VLA MaaS. It is language neutral:
any client that speaks WebSocket + MessagePack can implement it. The Python
reference implementation is the `vla-maas-protocol` package; the client is
`vla-maas-robot`.

Robots never see model-specific keys, shapes or normalization. The gateway
translates every frame to and from each model's native format with a
per-model adapter.

## Transport

- `WS /v1/robot/<model>/act?robot_id=<id>` on the gateway (`wss://` on public
  networks; TLS is terminated by a reverse proxy in front of the gateway).
- `Authorization: Api-Key <key>` (or `Bearer <key>`) on the upgrade request.
- Binary frames, one MessagePack map per frame, strictly request/response after
  the handshake: the robot sends one message, the gateway answers one.
- NumPy arrays are encoded as maps
  `{"__ndarray__": true, "data": <bytes>, "dtype": "<f4", "shape": [..]}`
  (the `openpi-client` convention).

## Units and vocabulary

| Kind | Values |
| --- | --- |
| Camera roles | `front`, `top`, `left`, `right`, `wrist`, `wrist_left`, `wrist_right` |
| State / action fields | `joint_pos` (rad, `[dof]`), `joint_vel` (rad/s, `[dof]`), `eef_pose` (`[7]`: xyz in m + quaternion x, y, z, w, robot base frame), `gripper` (`[1]`, 0 closed, 1 open) |
| Action spaces | `joint_pos`, `eef_pose` |

## Sequence

```text
robot                                   gateway
  | --- WS upgrade (key, robot_id) --->  |   401 bad key, 404 unknown model
  | <-- server_hello ------------------   |
  | --- robot_hello ------------------->  |
  | <-- ready | error + close 4400 -----   |
  | --- obs --------------------------->  |
  | <-- actions | stop | error --------   |
  | --- episode_result | reset -------->  |
  | <-- ok ----------------------------   |
```

### `server_hello` (gateway to robot)

```python
{
    "type": "server_hello",
    "protocol": "maas-robot/1",
    "model": "pi05_piper",
    "requires": {
        "images": {"front": [224, 224], "wrist": [224, 224]},  # role -> [H, W]
        "state": ["joint_pos", "gripper"],
    },
    "provides": {"space": "joint_pos", "fields": ["joint_pos", "gripper"], "horizon": 50, "control_hz": 30.0},
}
```

The robot should resize images to the requested size before sending them.

### `robot_hello` (robot to gateway)

```python
{
    "type": "robot_hello",
    "protocol": "maas-robot/1",
    "robot_id": "piper-01",
    "embodiment": "piper",
    "dof": 6,
    "images": ["front", "wrist"],
    "state": ["joint_pos", "eef_pose", "gripper"],
    "action_spaces": ["joint_pos", "eef_pose"],
    "image_encodings": ["raw", "jpeg"],
}
```

The gateway answers `{"type": "ready", "session_id": ...}` (optionally with
`revision`, `checkpoint` and `model_type`: what serves this session, for the
robot's logs; robots must ignore keys they do not know) or an `error` with
`code: "incompatible"` and the list of `problems`, then closes with 4400.
Compatible means: every required camera role and state field is offered, and
the model's action space is in `action_spaces`. A server accepts its own
protocol major version and the previous one.

### `obs` (robot to gateway)

```python
{
    "type": "obs",
    "seq": 17,
    "t": 1234.567,
    "episode_id": "...",
    "instruction": "pick up the red cube",
    "images": {"front": uint8[224, 224, 3], "wrist": {"enc": "jpeg", "data": b"..."}},
    "state": {"joint_pos": float32[6], "eef_pose": float32[7], "gripper": float32[1]},
    "extra": {},
}  # optional, passed to the model adapter untouched
```

`t` is the robot's monotonic clock; it is echoed back as `t_obs`.

### `actions` (gateway to robot)

```python
{
    "type": "actions",
    "seq": 17,
    "t_obs": 1234.567,
    "space": "joint_pos",
    "mode": "absolute",
    "control_hz": 30.0,
    "server_ms": 84.2,
    "actions": {"joint_pos": float32[50, 6], "gripper": float32[50, 1]},
}
```

Targets are always absolute; a model's delta outputs are converted by the
adapter against the observation's state. Row `i` is the target for
`i / control_hz` seconds after the observation. `server_ms` is time spent in the
gateway and model, so `round_trip - server_ms` is network time.

### Control messages

| Robot sends | Gateway answers |
| --- | --- |
| `{"type": "reset"}` | `{"type": "ok"}` (forwarded to stateful models) |
| `{"type": "episode_result", "success": true}` | `{"type": "ok"}` (counted in metrics) |

Instead of `actions`, the gateway may answer an `obs` with
`{"type": "stop", "reason": ...}` after an operator stop; the robot must hold
its pose and end the episode. Errors are
`{"type": "error", "code": ..., "message": ..., "problems": [...]}`.

## Errors and close codes

The upgrade request is refused with HTTP 400 (no `robot_id`), 401 (key not
valid for this robot or model) or 404 (unknown model, or a model without an
adapter). After the upgrade the gateway closes with:

| Code | Meaning |
| --- | --- |
| 4400 | Incompatible robot and model (see `problems`) |
| 4409 | Another session with the same `robot_id` is connected |
| 1011 | Model backend unreachable |
