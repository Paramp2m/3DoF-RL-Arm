"""Load the RRR arm MJCF and apply the config's actuator variant and physics overrides."""

from __future__ import annotations

import mujoco
import numpy as np

from rlarm.common import ROOT

POS, MOTOR = 0, 1  # actuator groups in arm.xml


def load_model(cfg: dict) -> mujoco.MjModel:
    """Build the model: enable one actuator set, set gains / damping / torque limit from cfg['arm']."""
    a = cfg["arm"]
    m = mujoco.MjModel.from_xml_path(str(ROOT / a["xml"]))
    use = {"position": POS, "motor": MOTOR}[a["actuators"]]
    m.opt.disableactuator = 1 << (MOTOR if use == POS else POS)
    for i in range(m.nu):
        if m.actuator_group[i] == POS:
            m.actuator_gainprm[i, 0] = a["kp"]
            m.actuator_biasprm[i, :3] = [0, -a["kp"], -a["kv"]]
            m.actuator_forcerange[i] = [-a["tau_max"], a["tau_max"]]
            m.actuator_forcelimited[i] = 1
        else:
            m.actuator_ctrlrange[i] = [-a["tau_max"], a["tau_max"]]
    m.dof_damping[:] = a["joint_damping"]
    m.dof_armature[:] = a["armature"]
    return m


def joint_range(m: mujoco.MjModel) -> np.ndarray:
    """(3, 2) joint limits [rad]."""
    return m.jnt_range.copy()
