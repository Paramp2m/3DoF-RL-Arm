"""Arm task envs; `make` picks one by cfg['task']."""

from __future__ import annotations

from rlarm.envs.base import ArmGoalEnv
from rlarm.envs.hold import HoldEnv
from rlarm.envs.reach import ReachEnv

TASKS = {"reach": ReachEnv, "hold": HoldEnv}


def make(cfg: dict, render_mode: str | None = None, **kw) -> ArmGoalEnv:
    """Build the task env named in cfg['task']."""
    return TASKS[cfg["task"]](cfg, render_mode=render_mode, **kw)
