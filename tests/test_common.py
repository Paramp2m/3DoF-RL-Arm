import torch

from rlarm.common import ROOT, apply_threads, evaluate, load_config, make_env


def test_threads_follow_config():
    cfg = load_config(ROOT / "configs/reacher_sac.yaml")
    cfg["torch_threads"] = 3
    assert apply_threads(cfg) == 3 == torch.get_num_threads()


def test_env_and_evaluate():
    cfg = load_config(ROOT / "configs/reacher_sac.yaml")
    env = make_env(cfg)

    class Zero:
        def predict(self, obs, deterministic=True):
            return env.action_space.sample() * 0, None

    rets = evaluate(Zero(), env, 2)
    assert rets.shape == (2,) and (rets < 0).all()
