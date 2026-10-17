"""Reward terms (SPEC §5). Vectorised over leading batch dims; weights come from config env.reward.

Only distance + precision + success for now: SPEC says add terms when a logged problem calls for them.
"""

from __future__ import annotations

import numpy as np


def terms(achieved: np.ndarray, desired: np.ndarray, w: dict, success_dist: float) -> dict:
    """Each reward term, shape (...,), keyed by name."""
    d = np.linalg.norm(np.asarray(achieved, float) - np.asarray(desired, float), axis=-1)
    return {
        "distance": -w["w_distance"] * d,
        "precision": w["w_precision"] * np.exp(-((d / w["sigma"]) ** 2)),
        "success": w["w_success"] * (d < success_dist),
    }
