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


def test_early_stop_reason_rules():
    from rlarm.common import early_stop_reason

    es = {"min_frac": 0.3, "min_points": 5, "patience": 4, "delta": 0.01, "floor": 0.45}
    rise = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert early_stop_reason(rise, 800, 1000, es) is None  # still improving
    assert early_stop_reason([0.1, 0.2, 0.3, 0.3], 400, 1000, es) is None  # too few val points
    flat_good = [0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    assert (
        early_stop_reason(flat_good, 800, 1000, es) == "plateau"
    )  # good plateau: finalise and keep
    assert early_stop_reason(flat_good, 200, 1000, es) is None  # before min_frac
    low = [0.2, 0.3, 0.3, 0.3, 0.3, 0.3]
    assert early_stop_reason(low, 600, 1000, es) == "stuck_low"  # stuck under the floor: useless
    assert (
        early_stop_reason(low, 600, 1000, {**es, "floor": None}) == "plateau"
    )  # no floor: plain plateau
    assert (
        early_stop_reason(low[:5], 600, 1000, {**es, "floor": None}) is None
    )  # patience not reached
    wobble = [0.6, 0.605, 0.6, 0.607, 0.6, 0.6]
    assert (
        early_stop_reason(wobble, 900, 1000, es) == "plateau"
    )  # noise under delta is not progress


def test_early_stop_creeping_run_that_cannot_reach_floor():
    from rlarm.common import early_stop_reason

    es = {"min_frac": 0.3, "min_points": 5, "patience": 6, "delta": 0.01, "floor": 0.45}
    creep = [
        -0.22,
        -0.14,
        -0.10,
        -0.17,
        -0.09,
        -0.06,
    ]  # d2: rises ~0.03/pt, 14 pts left -> ~0.3 < floor
    assert early_stop_reason(creep, 300, 1000, es) == "stuck_low"
    fast = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5][:5]  # +0.1/pt reaches the floor easily
    assert early_stop_reason(fast, 500, 1000, es) is None


def test_early_stop_flat_trend_with_noisy_new_best_is_plateau():
    from rlarm.common import early_stop_reason

    es = {"min_frac": 0.3, "min_points": 5, "patience": 6, "delta": 0.01, "floor": 0.3}
    noisy = [0.5, 0.52, 0.49, 0.53, 0.5, 0.56, 0.51]  # lucky new best 0.56, trend flat
    assert early_stop_reason(noisy, 900, 1500, es) == "plateau"
    assert (
        early_stop_reason([0.3, 0.36, 0.42, 0.48, 0.54, 0.6], 600, 1500, es) is None
    )  # real climb
