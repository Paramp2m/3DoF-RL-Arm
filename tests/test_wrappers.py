import gymnasium as gym
import numpy as np

from rlarm.wrappers import Curriculum


class Stub(gym.Env):
    """1-step episodes whose success is set by the test."""

    observation_space = gym.spaces.Box(-1, 1, (1,))
    action_space = gym.spaces.Box(-1, 1, (1,))
    difficulty = 1.0
    win = True

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, np.float32), {}

    def step(self, a):
        return np.zeros(1, np.float32), 0.0, False, True, {"is_success": self.win}


def run(env, n):
    for _ in range(n):
        env.reset()
        env.step(env.action_space.sample())


def test_curriculum_rises_after_window_and_caps():
    base = Stub()
    env = Curriculum(base, "difficulty", 0.0, 0.2, 0.1, 0.5, 2)
    assert base.difficulty == 0.0
    run(env, 1)
    assert base.difficulty == 0.0  # window not full yet
    run(env, 1)
    assert np.isclose(base.difficulty, 0.1)
    run(env, 20)
    assert np.isclose(base.difficulty, 0.2)  # never past `end`


def test_curriculum_stays_when_failing():
    base = Stub()
    base.win = False
    env = Curriculum(base, "difficulty", 0.0, 1.0, 0.1, 0.5, 2)
    run(env, 10)
    assert base.difficulty == 0.0
