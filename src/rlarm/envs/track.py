"""Track task: the target moves on a horizontal circle; the tip must follow it (phase 5, task C)."""

from __future__ import annotations

import numpy as np

from rlarm.envs.base import ArmGoalEnv
from rlarm.kinematics import fk, ik


class TrackEnv(ArmGoalEnv):
    """Circle of env.radius [m] and env.period [s] around a centre sampled through FK. Every point on
    the circle has an in-limit IK pose (resampled otherwise). Phase and direction are random.
    obs extras = target velocity [m/s] * env.vel_obs_scale.
    The arm starts near the circle point (see _place_task): with max_delta 0.05 rad/step a random pose
    can need 60 steps just for the yaw, which would make the >50-step check unfair to any controller."""

    def __init__(self, cfg: dict, **kw):
        e = cfg["env"]
        self.R, self.period = e["radius"], e["period"]
        self.vel_obs_scale = e.get("vel_obs_scale", 10.0)
        self.start_noise = e.get(
            "start_noise", 0.5
        )  # rad; initial pose = IK pose of the start point + noise
        super().__init__(cfg, **kw)
        self.dt = e["frame_skip"] * self.model.opt.timestep
        self.omega = 2 * np.pi / self.period  # rad/s

    def _extras_dim(self) -> int:
        return 3

    def _point(self, step: int) -> np.ndarray:
        phi = self.phi0 + self.dir * self.omega * step * self.dt
        return self.centre + self.R * np.array([np.cos(phi), np.sin(phi), 0.0])

    def _extras(self) -> np.ndarray:
        phi = self.phi0 + self.dir * self.omega * self.steps * self.dt
        v = self.dir * self.omega * self.R * np.array([-np.sin(phi), np.cos(phi), 0.0])
        return v * self.vel_obs_scale

    def _branch(self, centre: np.ndarray) -> np.ndarray | None:
        """In-limit joint solution for 64 circle points forming one continuous closed loop (jumps
        < 0.1 rad, i.e. <= 0.6 max_delta per step), or None. Without it the circle would need an IK-branch flip mid-episode."""
        a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        pts = centre + self.R * np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)
        sols = [ik(p, lim=self.lim) for p in pts]
        if not all(len(x) for x in sols):
            return None
        for q0 in sols[0]:
            path = [q0]
            for x in sols[1:] + [q0[None]]:
                j = np.abs(x - path[-1]).max(axis=1)
                if j.min() > 0.1:
                    break
                path.append(x[j.argmin()])
            else:
                return np.array(path[:-1])
        return None

    def _sample_goal(self) -> np.ndarray:
        while True:
            c = fk(self.np_random.uniform(self.lim[:, 0], self.lim[:, 1]))
            path = self._branch(c)
            if path is not None:
                break
        self.centre, self.path = c, path
        self.phi0 = self.np_random.uniform(0, 2 * np.pi)  # the circle is traced from angle phi0
        self.dir = self.np_random.choice([-1.0, 1.0])
        self.steps = 0
        return self._point(0)

    def _place_task(self) -> None:
        """Start near the target (within start_noise rad per joint) so step 50 is past the acquisition."""
        n = len(self.path)
        q = self.path[round(self.phi0 / (2 * np.pi) * n) % n]  # on the loop at the start phase
        q = np.clip(
            q + self.np_random.normal(0, self.start_noise, 3), self.lim[:, 0], self.lim[:, 1]
        )
        self.data.qpos[:3] = q
        self.target_q = q.copy()

    def _update_goal(self) -> None:
        self.goal = self._point(self.steps)
        self.data.mocap_pos[self.mocap] = self.goal
