"""Reward terms (SPEC §5). Vectorised over leading batch dims; weights come from config env.reward.

distance + precision + success, plus settle (only when w_settle is set) for hold-task chatter.
SPEC says add terms when a logged problem calls for them.
"""

from __future__ import annotations

import numpy as np


def terms(
    achieved: np.ndarray, desired: np.ndarray, w: dict, success_dist: float, speed=None
) -> dict:
    """Each reward term, shape (...,), keyed by name. speed = ‖q̇‖ (rad/s), needed only for settle."""
    d = np.linalg.norm(np.asarray(achieved, float) - np.asarray(desired, float), axis=-1)
    t = {
        "distance": -w["w_distance"] * d,
        "precision": w["w_precision"] * np.exp(-((d / w["sigma"]) ** 2)),
        "success": w["w_success"] * (d < success_dist),
    }
    if w.get("w_settle"):
        t["settle"] = -w["w_settle"] * np.asarray(speed, float) ** 2 * (d < success_dist)
    return t
