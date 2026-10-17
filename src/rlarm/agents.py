"""Non-learning baselines with SB3's `predict(obs, deterministic)` signature."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rlarm.kinematics import ik


class RandomAgent:
    """Uniform random actions from the env's (seedable) action space."""

    def __init__(self, env: gym.Env):
        self.space = env.action_space

    def predict(self, obs, deterministic: bool = True):
        return self.space.sample(), None


class IKAgent:
    """delta_pos only: step joints toward the in-limit IK solution nearest the current pose.

    Feedback on measured q (read from the sin/cos obs) gives integral action on target_q, so gravity
    sag of the position servos is removed.
    """

    def __init__(self, env: gym.Env, gain: float = 0.5):
        u = env.unwrapped
        assert u.mode == "delta_pos", "IKAgent drives delta_pos only"
        self.lim, self.k = u.lim, gain / u.ecfg["max_delta"]

    def predict(self, obs, deterministic: bool = True):
        o = obs["observation"].astype(float)
        q = np.arctan2(o[:3], o[3:6])
        sols = ik(obs["desired_goal"])
        ok = sols[np.all((sols >= self.lim[:, 0] - 1e-6) & (sols <= self.lim[:, 1] + 1e-6), axis=1)]
        goal = ok[np.abs(ok - q).sum(axis=1).argmin()] if len(ok) else q
        return np.clip((goal - q) * self.k, -1, 1).astype(np.float32), None
