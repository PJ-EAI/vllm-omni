# vla-maas-protocol

Message types, vocabulary and MessagePack codec of the VLA MaaS standard Robot
protocol (`maas-robot/1`). Both the robot client (`vla-maas-robot`) and the
cloud gateway depend on this package and on nothing of each other.

```bash
pip install ./packages/vla-maas-protocol          # numpy + msgpack only
pip install "./packages/vla-maas-protocol[jpeg]"  # adds Pillow for JPEG images
```

The language-neutral specification is
[spec/robot_protocol.md](spec/robot_protocol.md).

```python
from vla_maas_protocol import RobotHello, ServerHello, compatibility_problems, packb, unpackb
```

Tests run without the rest of the repository:

```bash
pip install "./packages/vla-maas-protocol[test]"
pytest packages/vla-maas-protocol/tests
```
