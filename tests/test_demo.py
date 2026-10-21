import sys

import numpy as np

from rlarm.common import ROOT
from rlarm.kinematics import fk

sys.path.insert(0, str(ROOT / "scripts"))
from demo import target_path

LIM = np.array([[-3.1, 3.1], [0.4, 1.7], [-1.1, 1.4]])


def test_target_path_is_reachable_smooth_and_deterministic():
    script = {"seed": 0, "max_joint_speed": 0.5, "dwell_s": 2.0}
    p = target_path(LIM, 0.04, 600, script)
    assert p.shape == (600, 3)
    np.testing.assert_array_equal(p, target_path(LIM, 0.04, 600, script))
    # dwell: the first 50 steps (2 s) hold still, and the path moves after that
    assert np.ptp(p[:50], axis=0).max() == 0 and np.ptp(p, axis=0).max() > 0.05
    # tip speed bound: joint speed 0.5 rad/s (x pi/2 for the cosine ease) times the max lever arm
    reach = np.linalg.norm(fk(np.zeros(3)))
    assert np.linalg.norm(np.diff(p, axis=0), axis=1).max() <= 0.5 * np.pi / 2 * 0.04 * 3 * reach
