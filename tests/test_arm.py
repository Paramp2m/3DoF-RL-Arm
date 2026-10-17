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


def test_fk_matches_mujoco():
    m = load_model(CFG)
    d = mujoco.MjData(m)
    for q in np.random.default_rng(0).uniform(m.jnt_range[:, 0], m.jnt_range[:, 1], (50, 3)):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        assert np.abs(fk(q) - d.site("tip").xpos).max() < 1e-9


def test_config_overrides_reach_model():
    arm = {**CFG["arm"], "kp": 123.0, "kv": 4.0, "tau_max": 7.0, "joint_damping": 0.9}
    m = load_model({"arm": arm})
    pos = m.actuator("shoulder_pos").id
    assert np.allclose(m.actuator_biasprm[pos, :3], [0, -123.0, -4.0])
    assert np.allclose(m.actuator_ctrlrange[pos], m.jnt_range[m.joint("shoulder_pitch").id])
    assert np.allclose(m.actuator_ctrlrange[m.actuator("elbow_motor").id], [-7.0, 7.0])
    assert np.allclose(m.dof_damping, 0.9)
    motor = load_model({"arm": {**arm, "actuators": "motor"}})
    d = mujoco.MjData(motor)
    d.ctrl[:] = 100.0  # out of range: must be clipped to tau_max
    mujoco.mj_forward(motor, d)
    assert d.actuator_force[motor.actuator("yaw_motor").id] == 7.0


def test_position_variant_holds_pose():
    m = load_model(CFG)
    d = mujoco.MjData(m)
    goal = np.array([0.5, 0.8, -0.6])
    d.ctrl[:3] = goal
    for _ in range(500):  # 1 s
        mujoco.mj_step(m, d)
    assert np.degrees(np.abs(d.qpos - goal).max()) < 2.0
