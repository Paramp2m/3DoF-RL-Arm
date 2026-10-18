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
    collision=None,
    reach_dist=None,
    action_delta=None,
) -> dict:
    """Each reward term, shape (...,), keyed by name. speed = ‖q̇‖ (rad/s), needed only for settle;
    limit_prox = Σ_j max(0, 1 - gap_j / margin) over joints (see ArmGoalEnv), needed only for w_limit.
    collision = 1 if the arm touched the obstacle during the step, needed only for w_collision.
    reach_dist = pad->cube distance (push), needed only for w_reach (goal independent, so HER-safe)."""
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
    assert collision is not None or not w.get("w_collision"), "w_collision needs collision"
    assert reach_dist is not None or not w.get("w_reach"), "w_reach needs reach_dist"
    assert action_delta is not None or not w.get("w_smooth"), "w_smooth needs action_delta"
    if w.get("w_settle"):
        t["settle"] = -w["w_settle"] * np.asarray(speed, float) ** 2 * (d < success_dist)
    if w.get("w_limit"):
        t["limits"] = -w["w_limit"] * np.asarray(limit_prox, float)
    if w.get("w_smooth"):
        t["smooth"] = -w["w_smooth"] * np.asarray(action_delta, float)
    if w.get("w_reach"):
        t["reach"] = -w["w_reach"] * np.asarray(reach_dist, float)
    if w.get("w_collision"):
        t["collision"] = -w["w_collision"] * np.asarray(collision, float)
    return t
