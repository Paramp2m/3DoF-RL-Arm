"""Reward terms (SPEC §5). Vectorised over leading batch dims; weights come from config env.reward.

distance + precision + success, plus settle (only when w_settle is set) for hold-task chatter and
joint limits (only when w_limit is set) for reach episodes that end pinned at a limit.
SPEC says add terms when a logged problem calls for them.
"""

from __future__ import annotations

import numpy as np


def terms(
    achieved: np.ndarray,
    desired: np.ndarray,
    w: dict,
    success_dist: float,
    speed=None,
    limit_prox=None,
) -> dict:
    """Each reward term, shape (...,), keyed by name. speed = ‖q̇‖ (rad/s), needed only for settle;
    limit_prox = Σ_j max(0, 1 - gap_j / margin) over joints (see ArmGoalEnv), needed only for w_limit."""
    d = np.linalg.norm(np.asarray(achieved, float) - np.asarray(desired, float), axis=-1)
    t = {
        "distance": -w["w_distance"] * d,
        "precision": w["w_precision"] * np.exp(-((d / w["sigma"]) ** 2)),
        "success": w["w_success"] * (d < success_dist),
    }
    # np.asarray(None, float) is nan, so a missing input would silently poison the reward
    assert (speed is not None or not w.get("w_settle")) and (
        limit_prox is not None or not w.get("w_limit")
    ), "w_settle needs speed, w_limit needs limit_prox"
    if w.get("w_settle"):
        t["settle"] = -w["w_settle"] * np.asarray(speed, float) ** 2 * (d < success_dist)
    if w.get("w_limit"):
        t["limits"] = -w["w_limit"] * np.asarray(limit_prox, float)
    return t
