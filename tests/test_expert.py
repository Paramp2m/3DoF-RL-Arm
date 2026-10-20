import itertools

import numpy as np

from rlarm.common import ROOT, evaluate_goal, load_config, make_env
from rlarm.expert import PlannerExpert


def env():
    return make_env(load_config(ROOT / "configs/obstacle_h1.yaml"))


def test_plan_is_deterministic_and_keeps_margin():
    e = env()
    ex = PlannerExpert(e)
    for seed in range(5):
        e.reset(seed=seed)
        q0 = e.target_q.copy()
        p1, p2 = ex.plan(q0), ex.plan(q0)
        assert p1 is not None and all(np.array_equal(a, b) for a, b in zip(p1, p2, strict=True))
        pts = [q0, *p1]
        for a, b in itertools.pairwise(pts[1:]):  # segments after the first keep the full margin
            assert ex.seg_ok(a, b, ex.margin - 1e-6)


def test_expert_solves_without_filter():
    e = env()
    ev = evaluate_goal(PlannerExpert(e), e, 10)
    assert ev["success_rate"] >= 0.9 and ev["collision_rate"] <= 0.1


def test_require_plan_resets_are_plannable():
    e = env()
    ex = PlannerExpert(e)
    for seed in range(10):
        e.reset(seed=100 + seed)
        assert ex.plan(e.target_q.copy()) is not None


def test_markov_label_ignores_history():
    """A markov expert that has followed an episode labels each state like a fresh one."""
    e = env()
    ex, fresh = PlannerExpert(e, markov=True), PlannerExpert(e, markov=True)
    obs, _ = e.reset(seed=3)
    for _ in range(30):
        a = ex.predict(obs)[0]
        fresh.path = None
        assert np.array_equal(a, fresh.predict(obs)[0])
        obs = e.step(a)[0]
