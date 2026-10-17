import mujoco
import numpy as np

from rlarm.arm import load_model
from rlarm.common import ROOT, load_config
from rlarm.kinematics import fk, ik

CFG = load_config(ROOT / "configs/arm.yaml")


def test_model_elements():
    m = load_model(CFG)
    assert m.nu == 6 and m.nq == 3
    assert m.site("tip").id >= 0 and m.body_mocapid[m.body("target").id] >= 0
    assert m.camera("main").id >= 0


def test_actuator_variant_switch():
    for kind, enabled in (("position", "yaw_pos"), ("motor", "yaw_motor")):
        m = load_model({"arm": {**CFG["arm"], "actuators": kind}})
        d = mujoco.MjData(m)
        d.ctrl[:] = 1.0
        mujoco.mj_forward(m, d)
        assert d.actuator_force[m.actuator(enabled).id] != 0
        other = "yaw_motor" if kind == "position" else "yaw_pos"
        assert d.actuator_force[m.actuator(other).id] == 0


def test_ik_roundtrip_and_unreachable():
    q = np.array([0.7, 1.0, -0.6])
    sols = ik(fk(q))
    assert len(sols) == 4
    assert any(np.allclose(s, q, atol=1e-9) for s in sols)
    assert np.abs(fk(sols) - fk(q)).max() < 1e-9
    assert ik(np.array([1.0, 0.0, 0.1])).shape == (0, 3)
    assert len(ik(np.array([0.55, 0.0, 0.1]))) == 2  # fully stretched: singular
