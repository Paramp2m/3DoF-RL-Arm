"""Arm task envs; `make` picks one by cfg['task']."""

from __future__ import annotations

from rlarm.envs.base import ArmGoalEnv
from rlarm.envs.hold import HoldEnv
from rlarm.envs.obstacle import ObstacleEnv
from rlarm.envs.push import PushEnv
from rlarm.envs.reach import ReachEnv
from rlarm.envs.track import TrackEnv

TASKS = {
    "reach": ReachEnv,
    "hold": HoldEnv,
    "track": TrackEnv,
    "obstacle": ObstacleEnv,
    "push": PushEnv,
}


def make(cfg: dict, render_mode: str | None = None, **kw) -> ArmGoalEnv:
    """Build the task env named in cfg['task']."""
    return TASKS[cfg["task"]](cfg, render_mode=render_mode, **kw)
