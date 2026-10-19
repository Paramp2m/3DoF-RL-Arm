"""Obstacle diagnosis: split episodes by whether a straight joint-space move to a goal pose is collision-free
(direct) or a detour is needed, and report collision rate / success for each. Usage: obstacle_diag.py RUN [episodes]."""

import sys

import mujoco
import numpy as np

from rlarm.common import load_config, load_policy, make_env
from rlarm.kinematics import ik

run, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
cfg = load_config(f"{run}/config.yaml")
cfg.pop("curriculum", None)
e, pol = make_env(cfg), load_policy(cfg, f"{run}/model.zip")


def free(q):
    e.data.qpos[:3] = q
    mujoco.mj_forward(e.model, e.data)
    return not e._touching()


def seg_ok(a, b, k=40):
    return all(free(a + (b - a) * i / k) for i in range(k + 1))


tab = {}
for ep in range(n):
    e.reset(seed=10_000 + ep)
    q0 = e.data.qpos[:3].copy()
    direct = any(seg_ok(q0, s) for s in ik(e.goal, lim=e.lim) if free(s))
    o, _ = e.reset(seed=10_000 + ep)
    done, hit = False, False
    while not done:
        o, _, te, tr, i = e.step(pol.predict(o, deterministic=True)[0])
        done, hit = te or tr, hit or i["collision"]
    a = tab.setdefault("direct" if direct else "detour", [0, 0, 0])
    a[0] += 1
    a[1] += hit
    a[2] += i["is_success"]
tot = np.sum(list(tab.values()), axis=0)
for k, (m, h, s) in {**tab, "all": tot}.items():
    print(f"{k:7s} {m:3d} eps  collided {h / m:4.0%}  success {s / m:4.0%}")
