"""Obstacle task: reach a target around a box that sits between the start pose and the target (phase 5, task D)."""

from __future__ import annotations

import mujoco
import numpy as np

from rlarm.envs.reach import ReachEnv
from rlarm.kinematics import fk, ik

# retry sizes of a commanded delta; negatives back the arm out of the box when merely holding still still
# touches it (momentum of the previous step). The scaled action is clipped to [-1, 1] by the base step, so
# -2.0 only differs from -1.0 for |a| < 0.5.
SAFE_SCALES = (1.0, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0, -2.0)


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
        self.obs_wp = cfg["env"].get("obs_waypoint", False)
        self.obs_blocked = cfg["env"].get("obs_blocked", False)
        self.safe = cfg["env"].get("safe_action", False)
        # residual mode: action = clip(waypoint follower + residual * policy action); needs obs_waypoint
        self.residual = cfg["env"].get("residual_waypoint", 0.0)
        # require_plan: reject episodes the straight-segment planner (rlarm.expert) cannot solve, so every
        # episode is solvable; the planner is only used here and as a training teacher, never to act
        self.require_plan = cfg["env"].get("require_plan", False)
        self._expert = None
        super().__init__(cfg, **kw)
        e = cfg["env"]
        if self.safe and self.mode != "delta_pos":
            raise ValueError("safe_action needs action_mode delta_pos")
        self.half = np.array(e["box_half"], float)
        self.min_span = e.get("min_span", 0.2)
        self.clearance = e.get("start_clearance", 0.0)  # min gap [m] between arm and box at reset
        self.gid = self.model.geom("obstacle").id
        self.full_half = self.half.copy()
        self._set_half(self.full_half)
        self.omocap = self.model.body_mocapid[self.model.body("obstacle").id]
        self._hit, self._blocked = False, False

    def _set_half(self, half: np.ndarray) -> None:
        """Resize the box; mujoco's broadphase bounds are precomputed so they are updated too."""
        self.half = half
        self.model.geom_size[self.gid] = half
        self.model.geom_rbound[self.gid] = np.linalg.norm(half)
        self.model.geom_aabb[self.gid] = np.r_[np.zeros(3), half]

    def _extras_dim(self) -> int:
        return (
            (5 if self.obs_yaw else 3)
            + (6 if self.obs_clear else 0)
            + (4 if self.obs_wp else 0)
            + (1 if self.obs_blocked else 0)
        )

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
        if (
            self.obs_wp
        ):  # joint-space vector to the current planner waypoint (rad) + "last waypoint" flag
            e = np.r_[
                e, self.path[self.wp] - self.data.qpos[:3], float(self.wp == len(self.path) - 1)
            ]
        if self.obs_blocked:  # did the safe filter cut the previous move?
            e = np.r_[e, float(self._blocked)]
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

    def _after_substeps(self) -> None:
        """mj_step detects contacts before it integrates, so the last substep can end in contact unseen: refresh."""
        mujoco.mj_forward(self.model, self.data)
        self._hit = self._hit or self._touching()

    def _info_extra(self) -> dict:
        out = {"collision": self._hit, "blocked": self._blocked}
        if self.ecfg["reward"].get("w_wp"):  # joint-space distance to the current waypoint
            out["wp_dist"] = float(np.linalg.norm(self.path[self.wp] - self.data.qpos[:3]))
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
                    and (not self.obs_wp or self._plan(q))
                    and (not self.require_plan or self._solvable(q))
                ):
                    return
            # resample start pose and target (same distributions as reach)
            q = self.np_random.uniform(self.lim[:, 0], self.lim[:, 1])
            self.data.qpos[:3] = q
            self.target_q = q.copy()
            self.goal = self._sample_goal()
            self.data.mocap_pos[self.mocap] = self.goal

    def _free(self, q: np.ndarray) -> bool:
        self.data.qpos[:3] = q
        mujoco.mj_forward(self.model, self.data)
        return not self._touching()

    def _seg_free(self, a: np.ndarray, b: np.ndarray, n: int = 40) -> bool:
        return all(self._free(a + (b - a) * k / n) for k in range(n + 1))

    def _plan(self, q0: np.ndarray, tries: int = 300) -> bool:
        """Collision-free joint-space path q0 -> a contact-free IK pose at the goal: straight, else through
        one random via-point. Sets self.path / self.wp; False (episode rejected) if none is found.
        Uses the box's true pose (privileged), so it is a helper the policy gets as an observation."""
        saved = self.data.qpos[:3].copy()
        goals = [g for g in ik(self.goal, lim=self.lim) if self._free(g)]
        path = None
        for g in goals:
            if self._seg_free(q0, g):
                path = [q0, g]
                break
        for _ in range(0 if path else tries):
            v = self.np_random.uniform(self.lim[:, 0], self.lim[:, 1])
            if self._free(v) and self._seg_free(q0, v):
                g = next((g for g in goals if self._seg_free(v, g)), None)
                if g is not None:
                    path = [q0, v, g]
                    break
        self.data.qpos[:3] = saved
        mujoco.mj_forward(self.model, self.data)
        if path is None:
            return False
        self.path, self.wp = np.array(path), 1
        return True

    def _solvable(self, q: np.ndarray) -> bool:
        if self._expert is None:
            from rlarm.expert import (
                PlannerExpert,  # local: the expert module imports nothing from envs
            )

            self._expert = PlannerExpert(self)
        return self._expert.plan(q) is not None

    def _update_goal(self) -> None:
        """Advance to the next waypoint once within 0.15 rad of the current one."""
        near = (
            np.linalg.norm(self.path[self.wp] - self.data.qpos[:3]) < 0.15 if self.obs_wp else False
        )
        if near and self.wp < len(self.path) - 1:
            self.wp += 1

    def step(self, action):
        if self.residual:  # planner waypoint follower (joint P-step toward the current waypoint) + learned residual
            base = np.clip((self.path[self.wp] - self.target_q) / self.ecfg["max_delta"], -1, 1)
            action = np.clip(
                base + self.residual * np.clip(np.asarray(action, float), -1, 1), -1, 1
            )
        if not self.safe:
            self._hit, self._blocked = False, False
            return super().step(action)
        # contact-safe filter: if the commanded move touches the box, roll the sim back and retry it at half /
        # quarter / zero size, then reversed. info["collision"] counts contacts of the *executed* move only (a
        # rolled-back attempt never happens); info["intended_contact"] says whether the policy's own move would
        # have touched the box. The rollback needs a simulator (privileged): a robot would need a model to predict.
        state = np.zeros(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        keep = (
            self.target_q.copy(),
            self.steps,
            self._prev_a.copy(),
            self.action_delta,
            self.wp if self.obs_wp else 0,
        )
        a = np.clip(np.asarray(action, float), -1, 1)
        intended = None
        for k, scale in enumerate(SAFE_SCALES):
            self._hit, self._blocked = False, k > 0
            out = super().step(a * scale)
            intended = self._hit if intended is None else intended
            if not self._hit or scale == SAFE_SCALES[-1]:
                out[4]["intended_contact"] = intended
                return out
            mujoco.mj_setState(self.model, self.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
            self.data.qvel[:3] = (
                0.0  # retries start from rest, with the servo target = the actual pose:
            )
            mujoco.mj_forward(
                self.model, self.data
            )  # momentum or a lagging target must not re-touch the box
            self.target_q = self.data.qpos[:3].copy()
            self.steps, self._prev_a, self.action_delta = keep[1:4]
            if self.obs_wp:
                self.wp = keep[4]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._hit, self._blocked = False, False
        self._set_half(self.full_half * self.difficulty)
        return super().reset(seed=seed, options=options)
