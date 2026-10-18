import sys

from rlarm.agents import RandomAgent
from rlarm.common import EVAL_SCHEMA, ROOT, evaluate_goal, experiments, load_config, make_env

sys.path.insert(0, str(ROOT / "scripts"))
from make_results import _join, _ms


def test_eval_schema_matches_evaluate_goal():
    e = make_env(load_config(ROOT / "configs/e1_torque.yaml"))
    assert tuple(evaluate_goal(RandomAgent(e), e, 1)) == EVAL_SCHEMA


def test_manifests_point_at_configs_and_three_seeds():
    exps = experiments()
    assert {s[:2] for s in exps} == {"e1", "e2", "e3", "e4"}
    for exp in exps.values():
        for arm in exp["arms"].values():
            assert (ROOT / arm["config"]).exists()
            assert len(arm["seeds"]) >= 3 or load_config(ROOT / arm["config"]).get("agent") == "ik"


def test_formatting():
    assert _ms([0.9, 1.0], 100) == "95.0 ± 7.1"
    assert _ms([0.5]) == "0.5" and _ms([None, 1.0]) == "n/a"
    assert _join({"a": "1", "b": "1"}) == "1" and _join({"a": "1", "b": "2"}) == "a: 1; b: 2"
