"""Feature extractors for SB3 policies (selected by class name in config algo.kwargs.policy_kwargs)."""

from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GoalDiffExtractor(BaseFeaturesExtractor):
    """[observation, achieved_goal, desired_goal, (desired - achieved) * scale].

    The scaled tip->target vector is built from the goal entries inside the network, so HER-relabelled
    goals stay consistent with it (pair with env.obs_err: false).
    """

    def __init__(self, observation_space: gym.spaces.Dict, scale: float = 1.0):
        n = (
            observation_space["observation"].shape[0]
            + 3 * observation_space["desired_goal"].shape[0]
        )
        super().__init__(observation_space, features_dim=n)
        self.scale = scale

    def forward(self, obs: dict) -> th.Tensor:
        ag, dg = obs["achieved_goal"], obs["desired_goal"]
        return th.cat([obs["observation"], ag, dg, (dg - ag) * self.scale], dim=1)
