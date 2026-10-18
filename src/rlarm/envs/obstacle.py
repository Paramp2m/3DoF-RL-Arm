"""Obstacle task: reach a target around a box that sits between the start pose and the target (phase 5, task D)."""

from __future__ import annotations

import mujoco
import numpy as np

from rlarm.envs.reach import ReachEnv
from rlarm.kinematics import fk, ik


class ObstacleEnv(ReachEnv):
    """Reach with a mocap box (env.box_half, half-sizes in its own frame, x = thickness) centred on the
    start->target segment and turned to face it. Any arm-box contact during a control step sets
    info["collision"]; episodes never terminate on it. obs extras = box centre.
    A reset is rejected unless: start tip and target are >= env.min_span apart, the arm is not touching
    the box, the target is >= 4 cm from it, and some in-limit IK pose at the target is contact-free."""

    def __init__(self, cfg: dict, **kw):
        super().__init__(cfg, **kw)
        e = cfg["env"]
        self.half = np.array(e["box_half"], float)
        self.min_span = e.get("min_span", 0.2)
        self.gid = self.model.geom("obstacle").id
        self.model.geom_size[self.gid] = self.half
        self.model.geom_rbound[self.gid] = np.linalg.norm(
            self.half
        )  # broadphase bounds follow the size
        self.model.geom_aabb[self.gid] = np.r_[np.zeros(3), self.half]
        self.omocap = self.model.body_mocapid[self.model.body("obstacle").id]
        self._hit = False

    def _extras_dim(self) -> int:
        return 3

    def _extras(self) -> np.ndarray:
        return self.data.mocap_pos[self.omocap].copy()

    def _touching(self) -> bool:
        """Is any contact between the box and another geom active right now?"""
        c = self.data.contact
        return bool(
            (
                (c.geom1[: self.data.ncon] == self.gid) | (c.geom2[: self.data.ncon] == self.gid)
            ).any()
        )

    def _substep(self) -> None:
        self._hit = self._hit or self._touching()

    def _info_extra(self) -> dict:
        return {"collision": self._hit}

    def _box_dist(self, p: np.ndarray) -> float:
        """Distance from point p to the box surface (0 inside)."""
        R = self.data.mocap_quat[self.omocap]
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, R)
        loc = m.reshape(3, 3).T @ (p - self.data.mocap_pos[self.omocap])
        return float(np.linalg.norm(np.maximum(np.abs(loc) - self.half, 0)))

    def _goal_pose_free(self) -> bool:
        """Some in-limit IK pose at the goal does not touch the box."""
        q0 = self.data.qpos[:3].copy()
        ok = False
        for q in ik(self.goal, lim=self.lim):
            self.data.qpos[:3] = q
            mujoco.mj_forward(self.model, self.data)
            if not self._touching():
                ok = True
                break
        self.data.qpos[:3] = q0
        mujoco.mj_forward(self.model, self.data)
        return ok

    def _place_task(self) -> None:
        q = self.data.qpos[:3].copy()
        while True:
            tip, goal = fk(q), self.goal
            if np.linalg.norm(goal - tip) >= self.min_span:
                t = self.np_random.uniform(0.35, 0.65)
                self.data.mocap_pos[self.omocap] = (
                    tip + t * (goal - tip) + self.np_random.normal(0, 0.01, 3)
                )
                d = goal - tip
                yaw = np.arctan2(d[1], d[0]) if np.hypot(d[0], d[1]) > 1e-6 else 0.0
                self.data.mocap_quat[self.omocap] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                mujoco.mj_forward(self.model, self.data)
                if not self._touching() and self._box_dist(goal) >= 0.04 and self._goal_pose_free():
                    return
            # resample start pose and target (same distributions as reach)
            q = self.np_random.uniform(self.lim[:, 0], self.lim[:, 1])
            self.data.qpos[:3] = q
            self.target_q = q.copy()
            self.goal = self._sample_goal()
            self.data.mocap_pos[self.mocap] = self.goal

    def step(self, action):
        self._hit = False
        return super().step(action)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._hit = False
        return super().reset(seed=seed, options=options)
