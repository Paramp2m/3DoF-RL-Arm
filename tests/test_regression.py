"""Phase 5 refactor guard: reach/hold obs, reward and info must stay bit-identical (golden sums)."""

import numpy as np
import pytest

from rlarm.common import ROOT, load_config, make_env

GOLDEN = {  # recorded before the phase-5 base-env refactor
    "reach": [75.69555179200647, 75.69555179200647],
    "hold": [75.69555179200647, 75.69555179200647],
}


def rollout(cfg_name: str, task: str | None = None) -> list[float]:
    cfg = load_config(ROOT / "configs" / cfg_name)
    if task:
        cfg["task"] = task
    env = make_env(cfg)
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed=3)
    tot = [float(obs["observation"].sum()), float(obs["desired_goal"].sum())]
    for _ in range(30):
        obs, r, _, _, info = env.step(rng.uniform(-1, 1, 3))
        tot += [float(obs["observation"].sum()), r, info["distance"], info["limit_prox"]]
    return [float(np.sum(tot)), float(np.sum(np.abs(tot)))]


@pytest.mark.parametrize(
    "name,cfg,task", [("reach", "reach_final.yaml", None), ("hold", "hold_final.yaml", None)]
)
def test_reach_hold_bit_identical(name, cfg, task):
    assert rollout(cfg, task) == GOLDEN[name]
