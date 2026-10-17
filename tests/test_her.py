import numpy as np
import torch

from rlarm.common import ROOT, algo_class, algo_kwargs, load_config, load_policy, make_env
from rlarm.policies import GoalDiffExtractor


def cfg_for(name):
    return load_config(ROOT / f"configs/{name}.yaml")


def test_obs_err_false_observation_is_goal_independent():
    e = make_env(cfg_for("reach_her")).unwrapped
    obs, _ = e.reset(seed=0)
    assert obs["observation"].shape == e.observation_space["observation"].shape == (12,)
    e.goal = e.goal + 0.1  # what HER does to a stored transition: new goal, same state
    assert np.array_equal(e._obs()["observation"], obs["observation"])


def test_goal_diff_extractor():
    e = make_env(cfg_for("reach_her"))
    ext = GoalDiffExtractor(e.observation_space, scale=30.0)
    obs, _ = e.reset(seed=1)
    t = {k: torch.as_tensor(v)[None] for k, v in obs.items()}
    f = ext(t)[0].numpy()
    assert f.shape == (ext.features_dim,) == (21,)
    assert np.allclose(f[-3:], (obs["desired_goal"] - obs["achieved_goal"]) * 30, atol=1e-5)


def test_settle_term_and_her_style_batched_reward():
    e = make_env(cfg_for("hold_her"))
    u = e.unwrapped
    e.reset(seed=0)
    u.goal = u.data.site("tip").xpos.copy()
    e.action_space.seed(0)
    ag, dg, infos, rs = [], [], [], []
    for t in range(30):  # small random moves around the goal: in and out of the zone, nonzero speed
        a = e.action_space.sample() * (0.3 if t % 2 else -0.3)
        obs, r, _, _, info = e.step(a)
        ag.append(obs["achieved_goal"]), dg.append(obs["desired_goal"]), infos.append(info), rs.append(r)
        inside = info["distance"] < 0.02
        assert np.isclose(info["r_settle"], -1.0 * info["joint_speed"] ** 2 * inside)
    assert any(i["r_settle"] < 0 for i in infos)
    batched = u.compute_reward(np.array(ag), np.array(dg), np.array(infos, dtype=object))
    assert np.allclose(batched, rs, atol=1e-6)


def test_her_train_and_load_roundtrip(tmp_path):
    cfg = cfg_for("reach_her")
    cfg["algo"]["kwargs"].update(learning_starts=200, buffer_size=2000)
    m = algo_class(cfg)(cfg["algo"]["policy"], make_env(cfg), seed=0, device="cpu", **algo_kwargs(cfg))
    m.learn(400)
    m.save(tmp_path / "m.zip")
    obs, _ = make_env(cfg).reset(seed=3)
    back = load_policy(cfg, tmp_path / "m.zip")
    assert np.allclose(m.predict(obs, deterministic=True)[0], back.predict(obs, deterministic=True)[0])
