import mujoco
import numpy as np
import pytest

from rlarm.agents import IKAgent
from rlarm.common import ROOT, evaluate_goal, load_config, make_env
from rlarm.kinematics import ik


def env():
    return make_env(load_config(ROOT / "configs/obstacle_pilot.yaml"))


def seg_hits_box(e, a, b) -> bool:
    """Does the segment a->b pass through the box? (sampled in the box frame)"""
    box, quat = e.data.mocap_pos[e.omocap], e.data.mocap_quat[e.omocap]
    yaw = 2 * np.arctan2(quat[3], quat[0])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    pts = a + np.linspace(0, 1, 200)[:, None] * (b - a)
    loc = (pts - box) @ R
    return bool((np.abs(loc) <= e.half).all(axis=1).any())


def test_reset_invariants_200_seeds():
    e = env()
    for seed in range(200):
        obs, info = e.reset(seed=seed)
        assert not info["collision"] and not e._touching()
        assert e._box_dist(e.goal) >= 0.04
        assert e._goal_pose_free() and len(ik(e.goal, lim=e.lim))
        tip = obs["achieved_goal"].astype(float)
        assert seg_hits_box(e, tip, e.goal)
        assert obs["observation"].shape == e.observation_space["observation"].shape


def test_ik_baseline_is_blocked_and_collision_reported():
    e = env()
    ev = evaluate_goal(IKAgent(e), e, 30)
    assert ev["collision_rate"] > 0.2


def test_collision_reward_term_in_info():
    e = env()
    e.reset(seed=0)
    *_, info = e.step(np.zeros(3))
    assert "r_collision" in info and info["r_collision"] == -float(info["collision"])


def test_a3_start_clearance_and_yaw_obs():
    e = make_env(load_config(ROOT / "configs/obstacle_a3.yaml"))
    for seed in range(40):
        obs, _ = e.reset(seed=seed)
        assert e._arm_gap() >= 0.05
        assert obs["observation"].shape == e.observation_space["observation"].shape
    c, s = obs["observation"][-2:]
    assert np.isclose(c * c + s * s, 1, atol=1e-5)


def test_clearance_obs_reward_and_curriculum_scale():
    cfg = load_config(ROOT / "configs/obstacle_a3.yaml")
    cfg["env"].update(obs_clearance=True, clear_margin=0.04)
    cfg["env"]["reward"]["w_clear"] = 0.5
    e = make_env(cfg)
    obs, info = e.reset(seed=0)
    assert obs["observation"].shape == e.observation_space["observation"].shape
    assert info["r_clear"] <= 0 and info["r_clear"] == -0.5 * info["clear_prox"]
    gaps = [g for g, _ in e._link_gaps()]
    assert min(gaps) >= 0.05  # start clearance > margin: no penalty at reset
    assert info["clear_prox"] == 0
    e.data.mocap_pos[e.omocap] = e.data.site("tip").xpos  # box on the forearm: full penalty
    mujoco.mj_forward(e.model, e.data)
    assert e._info_extra()["clear_prox"] >= 1.0
    # HER-style batched call uses info["clear_prox"]
    *_, sinfo = e.step(np.zeros(3))
    r = e.compute_reward(np.zeros((2, 3)), np.ones((2, 3)), np.array([sinfo] * 2))
    assert r.shape == (2,)
    e.difficulty = 0.3
    e.reset(seed=1)
    assert np.allclose(e.half, e.full_half * 0.3)


def _cfg_v(**env):
    cfg = load_config(ROOT / "configs/obstacle_c1.yaml")
    cfg["env"].update(env)
    return cfg


def test_safe_action_filter_blocks_contact_and_keeps_state_consistent():
    e = make_env(_cfg_v(safe_action=True))
    hits = blocked = steps = 0
    for seed in range(8):
        e.reset(seed=seed)
        for t in range(
            120
        ):  # drive hard toward the goal pose, straight into the box when it is in the way
            sol = ik(e.goal, lim=e.lim)
            qd = sol[np.abs(sol - e.data.qpos[:3]).sum(1).argmin()]
            *_, info = e.step(np.clip((qd - e.target_q) / 0.05, -1, 1))
            hits += info["collision"]
            blocked += info["blocked"]
            steps += 1
    plain = make_env(_cfg_v())
    plain_hits = 0
    for seed in range(8):
        plain.reset(seed=seed)
        for t in range(120):
            sol = ik(plain.goal, lim=plain.lim)
            qd = sol[np.abs(sol - plain.data.qpos[:3]).sum(1).argmin()]
            *_, info = plain.step(np.clip((qd - plain.target_q) / 0.05, -1, 1))
            plain_hits += info["collision"]
    assert blocked > 0 and hits <= 2 and hits < plain_hits * 0.05, (hits, plain_hits, blocked)


def test_planner_waypoint_path_is_collision_free_and_obs_shape():
    e = make_env(_cfg_v(obs_waypoint=True))
    for seed in range(15):
        obs, _ = e.reset(seed=seed)
        assert obs["observation"].shape == e.observation_space["observation"].shape
        assert e.wp == 1 and len(e.path) in (2, 3)
        assert np.allclose(obs["observation"][-4:-1], e.path[1] - e.path[0], atol=1e-5)
        assert all(e._seg_free(a, b) for a, b in zip(e.path[:-1], e.path[1:], strict=True))
    e.reset(seed=0)
    for _ in range(300):  # following the path with IK-style joint steps reaches the last waypoint
        e.step(np.clip((e.path[e.wp] - e.target_q) / 0.05, -1, 1))
    assert e.wp == len(e.path) - 1


def test_waypoint_reward_term():
    e = make_env(
        _cfg_v(
            obs_waypoint=True,
            safe_action=True,
            reward={**load_config(ROOT / "configs/obstacle_c1.yaml")["env"]["reward"], "w_wp": 0.3},
        )
    )
    e.reset(seed=1)
    *_, info = e.step(np.zeros(3))
    assert np.isclose(info["r_waypoint"], -0.3 * info["wp_dist"])


def test_safe_filter_from_free_pose_never_ends_in_contact():
    """Hold-still retries from rest + target=pose: a pre-contact-free step must not end with a hit."""
    e = make_env(_cfg_v(safe_action=True))
    rng = np.random.default_rng(0)
    bad = 0
    for seed in range(6):
        e.reset(seed=seed)
        for _ in range(200):
            free_before = not e._touching()
            *_, info = e.step(rng.uniform(-1, 1, 3) * (1 if rng.random() < 0.5 else 0.3))
            bad += free_before and info["collision"]
    assert bad == 0, bad


def test_residual_mode_zero_action_follows_planner():
    e = make_env(load_config(ROOT / "configs/obstacle_g2.yaml"))
    e.reset(seed=0)
    for _ in range(300):
        e.step(np.zeros(3))
    assert e.wp == len(e.path) - 1


def test_waypoint_options_need_obs_waypoint():
    cfg = load_config(ROOT / "configs/obstacle_g2.yaml")
    cfg["env"]["obs_waypoint"] = False
    with pytest.raises(ValueError, match="obs_waypoint"):
        make_env(cfg)
