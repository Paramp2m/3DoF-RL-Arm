"""Training-time teacher for the obstacle task: a deterministic joint-space planner with a clearance margin
plus a path follower. It reads the box pose and the goal from the env (privileged), so it is only used to
make demonstrations and DAgger labels; the trained network drives the arm on its own at test time."""

from __future__ import annotations

import itertools

import mujoco
import numpy as np

from rlarm.kinematics import ik

LINKS = ("upper_arm", "forearm")


class PlannerExpert:
    """`predict(obs)` like the other agents. Plans from the current state at the first step of an episode and
    whenever the arm is > `replan` rad (max-norm) off its path; straight joint-space segments, each kept
    >= `margin` m from the box; via-points from a fixed grid, the cheapest (fewest control steps) first."""

    def __init__(
        self,
        env,
        margin: float = 0.015,
        grid: tuple[int, int, int] = (16, 5, 6),
        gain: float = 0.5,
        replan: float = 0.2,
        markov: bool = False,
        consistent: float = 0.0,
    ):
        self.e = env.unwrapped
        self.m = self.e.model
        self.d = mujoco.MjData(self.m)  # private scratch data: planning never touches the env state
        self.margin, self.gain, self.replan = margin, gain, replan
        self.markov = markov  # replan every step: the label then depends on the current state only (for DAgger)
        # > 0: prefer (per rad, max-norm) the via-point / goal pose of a reference plan that depends on the goal and
        # box only, so the detour side does not flip between nearby states (a student would average the two sides)
        self.consistent, self.pref, self.pref_key = consistent, None, None
        self.links = [self.m.geom(n).id for n in LINKS]
        lo, hi = self.e.lim[:, 0], self.e.lim[:, 1]
        axes = [np.linspace(lo[i] + 0.05, hi[i] - 0.05, grid[i]) for i in range(3)]
        self.grid = np.array(list(itertools.product(*axes)))
        self.max_delta = self.e.ecfg["max_delta"]
        self.path, self.key, self.planned = None, None, False

    # --- geometry on the scratch data ---
    def _sync(self) -> None:
        self.d.mocap_pos[:] = self.e.data.mocap_pos
        self.d.mocap_quat[:] = self.e.data.mocap_quat

    def gap(self, q: np.ndarray) -> float:
        """Smallest distance from the arm links to the box at joint pose q (capped at 0.1 m)."""
        self.d.qpos[:3] = q
        mujoco.mj_kinematics(self.m, self.d)
        ft = np.zeros(6)
        return min(
            mujoco.mj_geomDistance(self.m, self.d, g, self.e.gid, 0.1, ft) for g in self.links
        )

    def seg_ok(self, a: np.ndarray, b: np.ndarray, margin: float) -> bool:
        n = max(1, int(np.ceil(np.abs(b - a).max() / 0.03)))
        return all(self.gap(a + (b - a) * k / n) >= margin for k in range(1, n + 1))

    # --- planning ---
    def plan(self, q0: np.ndarray) -> list[np.ndarray] | None:
        """Waypoints after q0 ending at a goal pose, or None. Deterministic for a given state. Tries the full
        clearance margin first, then a third of it, then bare contact-freedom (some starts are tight)."""
        self._sync()
        if self.consistent and self.pref_key != self._key():
            self.pref_key, self.pref = self._key(), None
            self.pref = self._reference()
        for margin in (self.margin, self.margin / 3, 1e-4):
            path = self._plan(q0, margin)
            if path is not None:
                return path
        return None

    def _key(self) -> tuple[bytes, bytes]:
        return self.e.goal.tobytes(), self.e.data.mocap_pos.tobytes()

    def _reference(self) -> tuple[np.ndarray | None, np.ndarray] | None:
        """(via or None, goal pose) of the plan from a fixed pose behind the box (goal mirrored through the box
        centre, pulled in until reachable and clear): a function of the goal and box only."""
        c = self.e.data.mocap_pos[self.e.omocap]
        for s in (1.0, 0.6, 0.3):
            for q in ik(c + s * (c - self.e.goal), lim=self.e.lim):
                if self.gap(q) >= self.margin:
                    path = self._plan(q, self.margin)
                    if path is not None:
                        return (path[0] if len(path) > 1 else None), path[-1]
        return None

    def _bias(self, v: np.ndarray | None, g: np.ndarray) -> float:
        if not self.pref:
            return 0.0
        pv, pg = self.pref
        return self.consistent * (
            np.abs(g - pg).max() + (np.abs(v - pv).max() if v is not None and pv is not None else 0)
        )

    def _plan(self, q0: np.ndarray, margin: float) -> list[np.ndarray] | None:
        m0 = max(
            min(margin, 0.9 * self.gap(q0)), 1e-4
        )  # starting tight to the box: don't demand more
        sols = ik(self.e.goal, lim=self.e.lim)
        goals = sorted(
            (g for g in sols if self.gap(g) >= margin),
            key=lambda g: np.abs(g - q0).max() + self._bias(None, g),
        )
        for g in goals:
            if self.seg_ok(q0, g, m0):
                return [g]
        vias = [v for v in self.grid if self.gap(v) >= margin]
        cost = lambda v, g: np.abs(v - q0).max() + np.abs(g - v).max() + self._bias(v, g)
        pairs = sorted(itertools.product(vias, goals), key=lambda p: cost(*p))
        for v, g in pairs:
            if self.seg_ok(q0, v, m0) and self.seg_ok(v, g, margin):
                return [v, g]
        first = sorted(
            (v for v in vias if self.seg_ok(q0, v, m0)), key=lambda v: np.abs(v - q0).max()
        )
        for v1 in first[:8]:  # two via-points (rare): bounded search
            for v2, g in pairs[:300]:
                if self.seg_ok(v1, v2, margin) and self.seg_ok(v2, g, margin):
                    return [v1, v2, g]
        return None

    def _off_path(self, q: np.ndarray) -> bool:
        """Max-norm distance from q to the polyline start->waypoints > replan."""
        pts = [self.start, *self.path]
        best = np.inf
        for a, b in itertools.pairwise(pts):
            ab = b - a
            t = np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            best = min(best, np.abs(q - (a + t * ab)).max())
        return best > self.replan

    def predict(self, obs, deterministic: bool = True):
        e = self.e
        q, tq = e.data.qpos[:3].copy(), e.target_q.copy()
        key = (e.goal.tobytes(), e.data.mocap_pos.tobytes())
        new = key != self.key
        if self.markov or self.path is None or new or (self.planned and self._off_path(q)):
            self.key, self.start = key, tq.copy()
            path = self.plan(tq)
            self.planned = (
                path is not None
            )  # no plan: head for the goal pose, don't replan every step
            self.path = path or [ik(e.goal, lim=e.lim)[0]]
        while len(self.path) > 1 and np.abs(self.path[0] - tq).max() < 1e-3:
            self.start = self.path.pop(0)  # waypoint reached by the servo target: next segment
        wp = self.path[0]
        if len(self.path) == 1 and np.abs(wp - tq).max() < 0.05:
            # final approach: closed loop on the measured pose removes the servo's gravity sag
            a = (wp - q) * self.gain / self.max_delta
        else:
            a = (wp - tq) / self.max_delta
        a = a / max(
            1.0, np.abs(a).max()
        )  # scale, don't clip per joint: keeps the straight joint segment
        return a.astype(np.float32), None
