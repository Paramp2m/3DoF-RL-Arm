"""Forward / inverse kinematics of the RRR arm (lengths must match models/arm/arm.xml)."""

from __future__ import annotations

import numpy as np

H, L1, L2 = 0.10, 0.30, 0.25  # column height, upper arm, forearm [m]
# Modified (Craig) DH rows: (alpha_{i-1}, a_{i-1}, d_i); theta_i is the joint angle.
# alpha=pi/2 turns joint 2's z axis into -y of the yaw frame, i.e. positive pitch raises the arm.
_DH = ((0.0, 0.0, H), (np.pi / 2, 0.0, 0.0), (0.0, L1, 0.0))


def _dh(alpha: float, a: float, theta: np.ndarray, d: float) -> np.ndarray:
    """Batched Rx(alpha) Tx(a) Rz(theta) Tz(d); theta has shape (...,)."""
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    T = np.zeros(theta.shape + (4, 4))
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 3] = ct, -st, a
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2], T[..., 1, 3] = st * ca, ct * ca, -sa, -sa * d
    T[..., 2, 0], T[..., 2, 1], T[..., 2, 2], T[..., 2, 3] = st * sa, ct * sa, ca, ca * d
    T[..., 3, 3] = 1.0
    return T


def fk(q: np.ndarray) -> np.ndarray:
    """Tip position (...,3) from joint angles (...,3)."""
    q = np.asarray(q, dtype=float)
    T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4))
    for i, (alpha, a, d) in enumerate(_DH):
        T = T @ _dh(alpha, a, q[..., i], d)
    return (T @ np.array([L2, 0.0, 0.0, 1.0]))[..., :3]


def _wrap(x: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return np.pi - np.mod(np.pi - x, 2 * np.pi)


def ik(p: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """All joint solutions (n,3) reaching tip position p; n == 0 means unreachable.

    Up to 4: {yaw, yaw+pi} x {elbow up, elbow down}; joint limits are NOT applied (see in_limits).
    On the yaw axis (x=y=0) yaw is free and 0 is returned. Straight arm (singular) gives 2 solutions.
    """
    x, y, z = np.asarray(p, dtype=float)
    r, w = np.hypot(x, y), z - H
    D = (r * r + w * w - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(D) > 1 + tol:
        return np.empty((0, 3))
    s = np.sqrt(max(0.0, 1 - min(D, 1.0) ** 2))
    yaw = np.arctan2(y, x) if r > tol else 0.0
    sols = []
    for flip in (False, True):
        rr = -r if flip else r
        for sgn in (1, -1) if s > tol else (1,):
            q3 = np.arctan2(sgn * s, D)
            q2 = np.arctan2(w, rr) - np.arctan2(L2 * np.sin(q3), L1 + L2 * np.cos(q3))
            sols.append((yaw + np.pi * flip, q2, q3))
    return _wrap(np.array(sols))


def in_limits(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Rows of q (n,3) inside [lo, hi]."""
    ok = np.all((q >= lo) & (q <= hi), axis=-1)
    return q[ok]
