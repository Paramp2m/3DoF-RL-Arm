"""Push task: slide a free cube on the floor to a marker with the flat pad on the tip (phase 5, task E)."""

from __future__ import annotations

import mujoco
import numpy as np

from rlarm.envs.base import MAX_TRIES, ArmGoalEnv

CUBE_Z = 0.025  # cube centre height (5 cm cube on the floor)


class PushEnv(ArmGoalEnv):
    """achieved_goal = cube centre, desired_goal = floor marker (at cube-centre height); success = within
    env.success_dist (3 cm). Cube starts at radius env.cube_r (lo, hi) from the base, within +-env.cube_yaw.
    Goal = cube + a step of length U(0.04, 0.04 + difficulty*(env.push_max - 0.04)) in a random direction,
    kept inside the pad's reachable annulus. `difficulty` (default 1 = full) is what the Curriculum raises.
    Start pose: arm above the cube area (home + noise). obs extras = cube pos, cube velocity * 5, tip->cube.
    Needs models/arm/arm_push.xml (elbow lower limit -2.2 rad)."""

    difficulty = 1.0

    def __init__(self, cfg: dict, **kw):
        super().__init__(cfg, **kw)
        e = cfg["env"]
        self.cube_r, self.cube_yaw = e["cube_r"], e["cube_yaw"]
        self.push_max, self.home = e["push_max"], np.array(cfg["arm"]["home"], float)
        self.reach_r = e.get("goal_r", e["cube_r"])  # goal annulus (lo, hi)
        self.cid = self.model.body("cube").id
        gid = self.model.geom("cube").id
        self.model.geom_friction[gid, 0] = e.get("cube_friction", 0.5)
        self.model.dof_damping[3:] = e.get(
            "cube_damping", 0.0
        )  # default joint damping would drag the cube
        self.model.dof_armature[3:] = 0.0
        self.qadr = self.model.body_dofadr[
            self.cid
        ]  # cube dofs; qpos adr == dof adr here (arm first)

    def _extras_dim(self) -> int:
        return 9

    def _cube(self) -> np.ndarray:
        return self.data.xpos[self.cid]

    def _extras(self) -> np.ndarray:
        tip = self.data.site("tip").xpos
        v = self.data.qvel[self.qadr : self.qadr + 3]
        return np.r_[self._cube(), np.clip(v * 5, -10, 10), self._cube() - tip]

    def _obs(self) -> dict:
        obs = super()._obs()
        obs["achieved_goal"] = self._cube().astype(np.float32)
        return obs

    def _info_extra(self) -> dict:
        return {"reach_dist": float(np.linalg.norm(self._cube() - self.data.site("tip").xpos))}

    def _in_band(self, p: np.ndarray) -> bool:
        return self.reach_r[0] <= np.hypot(p[0], p[1]) <= self.reach_r[1]

    def _sample_goal(self) -> np.ndarray:
        rng = self.np_random
        r, yaw = rng.uniform(*self.cube_r), rng.uniform(-self.cube_yaw, self.cube_yaw)
        self.cube0 = np.array([r * np.cos(yaw), r * np.sin(yaw), CUBE_Z])
        hi = 0.04 + self.difficulty * (self.push_max - 0.04)
        for _ in range(MAX_TRIES):
            d, a = rng.uniform(0.04, max(hi, 0.0401)), rng.uniform(0, 2 * np.pi)
            g = self.cube0 + np.array([d * np.cos(a), d * np.sin(a), 0.0])
            if self._in_band(g):
                return g
        raise RuntimeError(f"no push goal in goal_r after {MAX_TRIES} tries: check cube_r/goal_r")

    def _place_task(self) -> None:
        q = (
            self.home
            + np.r_[self.np_random.uniform(-np.pi, np.pi), self.np_random.normal(0, 0.15, 2)]
        )
        q = np.clip(q, self.lim[:, 0], self.lim[:, 1])
        self.data.qpos[:3] = q
        self.target_q = q.copy()
        a = self.qadr
        self.data.qpos[a : a + 7] = np.r_[self.cube0, 1.0, 0, 0, 0]
        self.data.qvel[a : a + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
