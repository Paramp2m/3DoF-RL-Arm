"""Reach task: static target sampled in joint space and passed through FK, so it is always reachable."""

from __future__ import annotations

import numpy as np

from rlarm.envs.base import ArmGoalEnv
from rlarm.kinematics import fk


class ReachEnv(ArmGoalEnv):
    def _sample_goal(self) -> np.ndarray:
        return fk(self.np_random.uniform(self.lim[:, 0], self.lim[:, 1]))
