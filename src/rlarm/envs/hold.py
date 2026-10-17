"""Hold task: reach, then stay. Success = tip within success_dist for the final hold_steps consecutive steps."""

from __future__ import annotations

from rlarm.envs.reach import ReachEnv


class HoldEnv(ReachEnv):
    """Same goals/reward as reach; `is_success` needs a streak of >= env.hold_steps steps in the zone."""

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.streak = 0
        obs, info = super().reset(seed=seed, options=options)
        return obs, info | {"is_success": False}

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        self.streak = self.streak + 1 if info["distance"] < self.ecfg["success_dist"] else 0
        return obs, r, term, trunc, info | {"is_success": self.streak >= self.ecfg["hold_steps"]}
