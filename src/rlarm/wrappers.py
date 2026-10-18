"""Training-only wrappers (never applied to eval / val envs, which always run at full difficulty)."""

from __future__ import annotations

from collections import deque

import gymnasium as gym


class Curriculum(gym.Wrapper):
    """Raise env.<param> from `start` towards `end` by `step` each time the success rate over the last
    `window` episodes reaches `up_at` (the window then restarts). The task env reads `param` at reset."""

    def __init__(
        self,
        env: gym.Env,
        param: str,
        start: float,
        end: float,
        step: float,
        up_at: float,
        window: int,
    ):
        super().__init__(env)
        self.param, self.end, self.step_size, self.up_at = param, end, step, up_at
        self.hist: deque = deque(maxlen=window)
        self.level = start
        setattr(env.unwrapped, param, start)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Forwarded: SB3's HER buffer calls it through the vec env."""
        return self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if term or trunc:
            self.hist.append(bool(info["is_success"]))
            if len(self.hist) == self.hist.maxlen and sum(self.hist) / len(self.hist) >= self.up_at:
                move = self.step_size
                new = self.level + move
                self.level = min(new, self.end) if move > 0 else max(new, self.end)
                setattr(self.env.unwrapped, self.param, self.level)
                self.hist.clear()
        return obs, r, term, trunc, info | {"difficulty": self.level}
