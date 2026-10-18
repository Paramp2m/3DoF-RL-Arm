import numpy as np

from rlarm.common import ROOT, load_config, make_env
from rlarm.kinematics import ik


def env():
    return make_env(load_config(ROOT / "configs/push_pilot.yaml"))


def goto(e, path):
    """Scripted delta_pos controller: follow tip waypoints, IK pose nearest the current one."""
    for p in path:
        for _ in range(8):
            q = e.data.qpos[:3].copy()
            sols = ik(p, lim=e.lim)
            if not len(sols):
                raise AssertionError(f"waypoint {p} unreachable")
            qd = sols[np.abs(sols - q).sum(1).argmin()]
            e.step(np.clip((qd - e.target_q) / 0.05, -1, 1))


def push_path(cube, goal, z=0.04, back=0.075, lift=0.20):
    """Above-behind the cube, down, then through it to just short of the goal."""
    u = (goal - cube)[:2] / np.linalg.norm((goal - cube)[:2])
    pre = cube[:2] - u * back
    end = goal[:2] - u * 0.055  # pad radius + cube half-size
    P = lambda xy, h: np.r_[xy, h]
    seg = lambda a, b, n: [a + (b - a) * k / n for k in range(1, n + 1)]
    path = seg(P(pre, lift), P(pre, lift), 1) + seg(P(pre, lift), P(pre, z), 8)
    return path + seg(P(pre, z), P(end, z), int(np.linalg.norm(end - pre) / 0.004))


def test_cube_rests_with_arm_parked():
    e = env()
    e.reset(seed=0)
    c0 = e._cube().copy()
    for _ in range(250):
        e.step(np.zeros(3))
    assert np.linalg.norm(e._cube() - c0) < 1e-3


def test_scripted_push_reaches_goal():
    """Feasibility gate for task E: without RL, a scripted IK push moves the cube >= 5 cm and lands it
    within 3 cm of a nearby goal (otherwise the model must change)."""
    done = 0
    for seed in range(6):
        e = env()
        e.reset(seed=seed)
        c0, g = e._cube().copy(), e.goal.copy()
        goto(e, push_path(c0, g))
        moved = np.linalg.norm(e._cube() - c0)
        done += np.linalg.norm(e._cube() - g) < 0.03 and moved >= 0.04
    assert done >= 4, done


def test_achieved_goal_is_cube_and_success_radius():
    e = env()
    obs, _ = e.reset(seed=1)
    assert np.allclose(obs["achieved_goal"], e._cube(), atol=1e-6)
    assert obs["observation"].shape == e.observation_space["observation"].shape
    a = e.model.body_dofadr[e.cid]
    e.data.qpos[a : a + 3] = e.goal + [0.02, 0, 0]  # inside 3 cm: success
    *_, info = e.step(np.zeros(3))
    assert info["is_success"]
    e.data.qpos[a : a + 3] = e.goal + [0.05, 0, 0]
    *_, info = e.step(np.zeros(3))
    assert not info["is_success"]


def test_her_relabel_reward_matches_env_reward():
    e = env()
    e.reset(seed=2)
    *_, info = e.step(np.zeros(3))
    ag = e._cube().astype(np.float32)
    goals = np.stack([e.goal, ag + [0.01, 0, 0], ag + [0.2, 0, 0]]).astype(np.float32)
    infos = np.array([info] * 3)
    r = e.compute_reward(np.tile(ag, (3, 1)), goals, infos)
    assert r.shape == (3,) and r[1] > r[2]  # closer relabelled goal pays more
    assert np.isclose(r[0], e.compute_reward(ag, e.goal.astype(np.float32), info))


def test_curriculum_difficulty_scales_goal_distance():
    e = env()
    for diff, hi in ((0.0, 0.0401), (1.0, 0.15)):
        e.difficulty = diff
        d = [(e.reset(seed=s), np.linalg.norm(e.goal - e.cube0))[1] for s in range(30)]
        assert max(d) <= hi + 1e-9
    assert max(d) > 0.1
