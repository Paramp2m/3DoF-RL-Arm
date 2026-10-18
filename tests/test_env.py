import numpy as np
import pytest

from rlarm import rewards
from rlarm.agents import IKAgent, RandomAgent
from rlarm.common import ROOT, evaluate_goal, load_config, make_env
from rlarm.envs.base import MODES


def env_for(mode="delta_pos", **arm):
    cfg = load_config(ROOT / "configs/reach.yaml")
    cfg["env"]["action_mode"] = mode
    cfg["arm"].update(arm)
    return make_env(cfg)


def test_reward_terms_batched_and_values():
    w = {"w_distance": 2.0, "w_precision": 1.0, "sigma": 0.03, "w_success": 5.0}
    ag = np.array([[0.0, 0, 0], [0.01, 0, 0], [0.3, 0, 0]])
    t = rewards.terms(ag, np.zeros((3, 3)), w, 0.02)
    assert np.allclose(t["distance"], [0, -0.02, -0.6])
    assert np.allclose(t["precision"], np.exp(-((np.array([0, 0.01, 0.3]) / 0.03) ** 2)))
    assert t["success"].tolist() == [5.0, 5.0, 0.0]


def test_compute_reward_matches_steps_and_info_terms():
    env = env_for("torque")
    env.action_space.seed(0)
    obs, _ = env.reset(seed=0)
    ag, dg, rs = [], [], []
    for _ in range(20):
        obs, r, _, _, info = env.step(env.action_space.sample())
        ag.append(obs["achieved_goal"]), dg.append(obs["desired_goal"]), rs.append(r)
        assert abs(sum(info[f"r_{k}"] for k in ("distance", "precision", "success")) - r) < 1e-12
    assert np.array_equal(env.compute_reward(np.array(ag), np.array(dg), None), rs)


def test_physics_params_from_config():
    m = env_for(body_mass={"forearm": 1.2}, joint_damping=0.9, frictionloss=0.05, kp=300.0).model
    base = env_for().model
    f = m.body("forearm").id
    assert m.body_mass[f] == 1.2 and np.allclose(m.body_inertia[f], 2 * base.body_inertia[f])
    assert (m.dof_damping == 0.9).all() and (m.dof_frictionloss == 0.05).all()
    assert m.actuator_gainprm[0, 0] == 300.0


def test_action_modes():
    with pytest.raises(ValueError):
        env_for("velocity")
    e = env_for("abs_pos")
    e.reset(seed=0)
    e.step(np.ones(3, np.float32))
    assert np.allclose(e.data.ctrl[:3], e.lim[:, 1])
    e = env_for("delta_pos")
    e.reset(seed=0)
    for _ in range(100):
        e.step(np.ones(3, np.float32))
    assert np.allclose(e.target_q, e.lim[:, 1])  # clipped to joint limits
    # zero torque: gravity compensation holds the pose, plain torque mode sags
    drift = {}
    for mode in ("torque", "torque_gravcomp"):
        e = env_for(mode)
        e.reset(seed=3)
        q0 = e.data.qpos.copy()
        for _ in range(10):
            e.step(np.zeros(3, np.float32))
        drift[mode] = np.abs(e.data.qpos - q0).max()
    assert drift["torque_gravcomp"] < 0.02 < drift["torque"]


def test_same_seed_same_trajectory():
    for mode in MODES:
        runs = []
        for _ in range(2):
            e = env_for(mode)
            rng = np.random.default_rng(1)
            o, _ = e.reset(seed=5)
            for _ in range(30):
                o, *_ = e.step(rng.uniform(-1, 1, 3).astype(np.float32))
            runs.append(o["observation"].tobytes())
        assert runs[0] == runs[1]


def test_agents_and_eval_schema():
    e = env_for()
    ik = evaluate_goal(IKAgent(e), e, 3)
    rnd = evaluate_goal(RandomAgent(e), e, 3)
    assert ik["success_rate"] == 1.0 and ik["final_dist_mm_median"] < 20
    assert rnd["success_rate"] == 0.0
    assert evaluate_goal(RandomAgent(e), e, 3) == rnd  # seeded
    assert set(ik) >= {
        "success_rate",
        "final_dist_mm_mean",
        "steps_to_success_mean",
        "smoothness_mean",
    }


def test_abs_pos_and_torque_scaling():
    e = env_for("abs_pos")
    e.reset(seed=0)
    for a, want in ((-1.0, e.lim[:, 0]), (0.0, e.lim.mean(axis=1))):
        e.step(np.full(3, a, np.float32))
        assert np.allclose(e.data.ctrl[:3], want)
    e = env_for("torque")
    e.reset(seed=0)
    e.step(np.array([1.0, -0.5, 0.0], np.float32))
    assert np.allclose(e.data.ctrl[3:6], [e.tau_max, -e.tau_max / 2, 0])


def test_delta_pos_zero_action_holds_pose():
    e = env_for("delta_pos")
    e.reset(seed=4)
    q0 = e.data.qpos.copy()
    for _ in range(10):
        e.step(np.zeros(3, np.float32))
    assert np.abs(e.data.qpos - q0).max() < 0.02


def test_obs_layout_and_truncation():
    e = env_for()
    o, _ = e.reset(seed=1)
    q = e.data.qpos
    assert np.allclose(o["observation"][:6], np.r_[np.sin(q), np.cos(q)], atol=1e-6)
    assert np.allclose(o["observation"][9:12], o["achieved_goal"], atol=1e-6)
    assert np.allclose(o["observation"][12:15], o["desired_goal"] - o["achieved_goal"], atol=1e-6)
    n = e.ecfg["episode_steps"]
    flags = [e.step(np.zeros(3, np.float32))[3] for _ in range(n)]
    assert not any(flags[:-1]) and flags[-1]


def test_energy_matches_manual_integration():
    import mujoco

    a = np.array([0.3, -0.2, 0.5], np.float32)
    e, ref = env_for("torque"), env_for("torque")
    e.reset(seed=2), ref.reset(seed=2)
    info = e.step(a)[4]
    want = 0.0
    for _ in range(ref.ecfg["frame_skip"]):
        ref.data.ctrl[3:6] = a * ref.tau_max
        mujoco.mj_step(ref.model, ref.data)
        want += np.abs(ref.data.actuator_force[3:6] * ref.data.qvel).sum() * ref.model.opt.timestep
    assert info["energy"] == pytest.approx(want) and want > 0
    assert evaluate_goal(RandomAgent(env_for()), env_for(), 1)["energy_mean"] is None


def test_eval_metrics_against_independent_rollout():
    class Zero:
        def predict(self, obs, deterministic=True):
            return np.zeros(3, np.float32), None

    e = env_for()
    ev = evaluate_goal(Zero(), e, 4, seed=50)
    d = []
    for i in range(4):
        e.reset(seed=50 + i)
        for _ in range(e.ecfg["episode_steps"]):
            info = e.step(np.zeros(3, np.float32))[4]
        d.append(info["distance"] * 1000)
    assert ev["final_dist_mm_median"] == pytest.approx(np.median(d))
    assert ev["final_dist_mm_mean"] == pytest.approx(np.mean(d))
    assert ev["smoothness_mean"] == 0.0 and ev["ever_success_rate"] >= ev["success_rate"]
    ik = evaluate_goal(IKAgent(e), e, 5)
    assert ik["ever_success_rate"] == 1.0 and 5 < ik["steps_to_success_mean"] < 80


def test_goal_resampled_each_reset_and_reachable():
    from rlarm.kinematics import ik

    e = env_for()
    goals = []
    for seed in (1, 2, 1):
        o, _ = e.reset(seed=seed)
        goals.append(o["desired_goal"])
        assert np.allclose(e.data.mocap_pos[e.mocap], e.goal)
        sols = ik(e.goal)
        assert np.all((sols >= e.lim[:, 0] - 1e-9) & (sols <= e.lim[:, 1] + 1e-9), axis=1).any()
    assert not np.allclose(goals[0], goals[1]) and np.array_equal(goals[0], goals[2])


def test_smooth_term_penalises_action_change_only_when_enabled():
    cfg = load_config(ROOT / "configs/reach_final.yaml")
    cfg["env"]["reward"]["w_smooth"] = 0.1
    env = make_env(cfg)
    env.reset(seed=0)
    _, _, _, _, i1 = env.step(np.array([1.0, 0, 0]))
    _, _, _, _, i2 = env.step(np.array([1.0, 0, 0]))
    assert np.isclose(i1["r_smooth"], -0.1) and i2["r_smooth"] == 0
    assert (
        "r_smooth" not in make_env(load_config(ROOT / "configs/reach_final.yaml")).reset(seed=0)[1]
    )
