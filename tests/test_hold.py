import numpy as np

from rlarm.agents import IKAgent, RandomAgent
from rlarm.common import ROOT, evaluate_goal, load_config, make_env


def hold_env(**env):
    cfg = load_config(ROOT / "configs/hold_sac.yaml")
    cfg["env"].update(env)
    return make_env(cfg)


def test_hold_needs_streak_and_resets_it():
    e = hold_env(hold_steps=5)
    e.reset(seed=0)
    q = e.unwrapped
    q.goal = q.data.site("tip").xpos.copy()  # tip starts on the goal; zero action keeps it there
    flags = [e.step(np.zeros(3))[4]["is_success"] for _ in range(6)]
    assert flags == [False] * 4 + [True, True]  # streak of 5 reached on step 5
    q.goal = q.goal + 0.1  # leave the zone: streak must drop to 0
    assert not e.step(np.zeros(3))[4]["is_success"] and q.streak == 0
    _, info = e.reset(seed=1)
    assert q.streak == 0 and info["is_success"] is False


def test_hold_episode_length_and_ik_vs_random():
    e = hold_env()
    assert e.unwrapped.ecfg["episode_steps"] == 250
    ik = evaluate_goal(IKAgent(e), e, 5)
    rnd = evaluate_goal(RandomAgent(e), e, 5)
    assert ik["success_rate"] == 1.0 and rnd["success_rate"] == 0.0
    assert ik["final_joint_speed_mean"] < 0.05 < rnd["final_joint_speed_mean"]
