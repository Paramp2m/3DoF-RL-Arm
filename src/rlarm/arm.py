"""Load the RRR arm MJCF and apply the config's actuator variant and physics overrides."""

from __future__ import annotations

import mujoco

from rlarm.common import ROOT

POS, MOTOR = 0, 1  # actuator groups in arm.xml


def load_model(cfg: dict) -> mujoco.MjModel:
    """Build the model (overrides apply to the 3 arm dofs only; task bodies keep their own): enable one actuator set, set gains / damping / torque limit from cfg['arm']."""
    a = cfg["arm"]
    m = mujoco.MjModel.from_xml_path(str(ROOT / a["xml"]))
    use = {"position": POS, "motor": MOTOR}[a["actuators"]]
    m.opt.disableactuator = 1 << (MOTOR if use == POS else POS)
    m.actuator_ctrllimited[:] = 1
    for i in range(m.nu):
        if m.actuator_group[i] == POS:  # ctrl is a joint angle: clip to that joint's limits
            m.actuator_gainprm[i, 0] = a["kp"]
            m.actuator_biasprm[i, :3] = [0, -a["kp"], -a["kv"]]
            m.actuator_ctrlrange[i] = m.jnt_range[m.actuator_trnid[i, 0]]
            m.actuator_forcerange[i] = [-a["tau_max"], a["tau_max"]]
            m.actuator_forcelimited[i] = 1
        else:
            m.actuator_ctrlrange[i] = [-a["tau_max"], a["tau_max"]]
    m.dof_damping[:3] = a["joint_damping"]
    m.dof_armature[:3] = a["armature"]
    m.dof_frictionloss[:3] = a.get("frictionloss", 0.0)
    for body, mass in a.get("body_mass", {}).items():
        b = m.body(body).id
        m.body_inertia[b] *= mass / m.body_mass[b]  # same shape, new mass
        m.body_mass[b] = mass
    return m
