import numpy as np

from rlarm.common import ROOT, load_config, make_env
from rlarm.kinematics import ik


def env():
    return make_env(load_config(ROOT / "configs/track_pilot.yaml"))


def test_circle_reachable_and_speed():
    e = env()
    for seed in range(5):
        e.reset(seed=seed)
        pts = [e._point(s) for s in range(201)]
        assert all(len(ik(p, lim=e.lim)) for p in pts[::10])
        step = np.linalg.norm(pts[1] - pts[0])
        want = 2 * np.pi * 0.08 / 8.0 * e.dt  # arc per control step
        assert abs(step / want - 1) < 0.01


def test_goal_moves_each_step_and_obs_shape():
    e = env()
    obs, _ = e.reset(seed=0)
    g0 = obs["desired_goal"].copy()
    obs, *_ = e.step(np.zeros(3))
    assert not np.allclose(g0, obs["desired_goal"])
    assert obs["observation"].shape == e.observation_space["observation"].shape


def test_branch_oracle_tracks_within_check_threshold():
    """Feasibility of ROADMAP 5.1: a scripted controller on the env's own IK branch gets ~5 mm."""
    e = env()
    errs = []
    for ep in range(3):
        e.reset(seed=10_000 + ep)
        d = []
        for _ in range(200):
            n = len(e.path)
            phi = (e.phi0 + e.dir * e.omega * e.steps * e.dt) % (2 * np.pi)
            ref = e.path[round(phi / (2 * np.pi) * n) % n]
            sols = ik(e._point(e.steps), lim=e.lim)
            qd = sols[np.abs(sols - ref).sum(1).argmin()]
            d.append(e.step(np.clip((qd - e.target_q) / 0.05, -1, 1))[4]["distance"] * 1000)
        errs.append(np.mean(d[50:]))
    assert max(errs) < 15
