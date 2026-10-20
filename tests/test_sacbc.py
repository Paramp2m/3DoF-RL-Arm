import numpy as np
import torch as th

from rlarm.common import ROOT, algo_class, algo_kwargs, load_config, make_env


def model_and_demos():
    cfg = load_config(ROOT / "configs/obstacle_h1.yaml")
    cfg["algo"]["kwargs"].update(
        learning_starts=300, buffer_size=5000, actor_freeze=40, bc_decay=10_000
    )
    env = make_env(cfg)
    m = algo_class(cfg)(cfg["algo"]["policy"], env, seed=0, device="cpu", **algo_kwargs(cfg))
    obs, _ = env.reset(seed=0)
    X = {k: np.repeat(v[None], 64, 0) for k, v in obs.items()}
    m.set_demos(X, np.full((64, 3), 0.7, np.float32))
    return m, X


def test_sacbc_freezes_actor_then_clones_demos():
    m, X = model_and_demos()
    before = [p.detach().clone() for p in m.actor.parameters()]
    m.learn(
        340
    )  # HER needs a full episode first; the 40 gradient steps are all inside actor_freeze
    assert all(th.equal(a, b) for a, b in zip(before, m.actor.parameters(), strict=True))
    xt = {k: th.as_tensor(v) for k, v in X.items()}
    err0 = (m.actor(xt, deterministic=True) - 0.7).abs().mean().item()
    m.learn(300, reset_num_timesteps=False)
    err1 = (m.actor(xt, deterministic=True) - 0.7).abs().mean().item()
    assert err1 < err0
    assert 0 < m.bc_lambda() < 1
