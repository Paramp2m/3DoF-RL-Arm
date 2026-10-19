"""Obstacle task: reach a target around a box that sits between the start pose and the target (phase 5, task D)."""

from __future__ import annotations

import mujoco
import numpy as np

from rlarm.envs.reach import ReachEnv
from rlarm.kinematics import fk, ik


class ObstacleEnv(ReachEnv):
    """Reach with a mocap box (env.box_half, half-sizes in its own frame, x = thickness) centred on the
    start->target segment and turned to face it. `difficulty` (default 1) scales the box half-sizes at reset
    (what the Curriculum raises; eval always runs at 1). Any arm-box contact during a control step sets
    info["collision"]; episodes never terminate on it. obs extras = box centre.
    A reset is rejected unless: start tip and target are >= env.min_span apart, the arm is not touching
    the box, the target is >= 4 cm from it, and some in-limit IK pose at the target is contact-free."""

    difficulty = 1.0

    def __init__(self, cfg: dict, **kw):
        self.obs_yaw = cfg["env"].get("obs_box_yaw", False)
        self.obs_clear = cfg["env"].get("obs_clearance", False)
        self.clear_margin = cfg["env"].get("clear_margin", 0.04)
        super().__init__(cfg, **kw)
        e = cfg["env"]
        self.half = np.array(e["box_half"], float)
        self.min_span = e.get("min_span", 0.2)
        self.clearance = e.get("start_clearance", 0.0)  # min gap [m] between arm and box at reset
        self.gid = self.model.geom("obstacle").id
        self.full_half = self.half.copy()
        self._set_half(self.full_half)
        self.omocap = self.model.body_mocapid[self.model.body("obstacle").id]
        self._hit = False

    def _set_half(self, half: np.ndarray) -> None:
        """Resize the box; mujoco's broadphase bounds are precomputed so they are updated too."""
        self.half = half
        self.model.geom_size[self.gid] = half
        self.model.geom_rbound[self.gid] = np.linalg.norm(half)
        self.model.geom_aabb[self.gid] = np.r_[np.zeros(3), half]

    def _extras_dim(self) -> int:
        return (5 if self.obs_yaw else 3) + (6 if self.obs_clear else 0)

    def _link_gaps(self) -> list[tuple[float, np.ndarray]]:
        """(gap, vector link->box) for the upper arm and forearm; gap capped at 0.5 m, vector 0 if farther."""
        out, fromto = [], np.zeros(6)
        for n in ("upper_arm", "forearm"):
            d = mujoco.mj_geomDistance(
                self.model, self.data, self.model.geom(n).id, self.gid, 0.5, fromto
            )
            out.append((max(d, 0.0), (fromto[3:] - fromto[:3]) if d < 0.5 else np.zeros(3)))
        return out

    def _extras(self) -> np.ndarray:
        e = self.data.mocap_pos[self.omocap].copy()
        if self.obs_yaw:  # box heading, else the policy cannot know which way the box faces
            q = self.data.mocap_quat[self.omocap]
            yaw = 2 * np.arctan2(q[3], q[0])
            e = np.r_[e, np.cos(yaw), np.sin(yaw)]
        if (
            self.obs_clear
        ):  # link->box vectors (norm = gap): lets the policy steer the whole arm round the box
            e = np.r_[e, *(v * 10 for _, v in self._link_gaps())]
        return e

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
        out = {"collision": self._hit}
        if self.ecfg["reward"].get(
            "w_clear"
        ):  # sum over links of how far inside clear_margin the link is
            out["clear_prox"] = float(
                sum(np.clip(1 - g / self.clear_margin, 0, 1) for g, _ in self._link_gaps())
            )
        return out

    def _arm_gap(self) -> float:
        """Smallest distance from any arm link to the box (capped at 0.5 m)."""
        fromto = np.zeros(6)
        gaps = [
            mujoco.mj_geomDistance(
                self.model, self.data, self.model.geom(n).id, self.gid, 0.5, fromto
            )
            for n in ("upper_arm", "forearm")
        ]
        return min(gaps)

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
                if (
                    not self._touching()
                    and self._arm_gap() >= self.clearance
                    and self._box_dist(goal) >= 0.04
                    and self._goal_pose_free()
                ):
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
        self._set_half(self.full_half * self.difficulty)
        return super().reset(seed=seed, options=options)
