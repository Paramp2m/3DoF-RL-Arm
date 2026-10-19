"""Mean action change per step once the tip is within success_dist of the goal (the jitter metric), 30 eval episodes."""

import sys

import numpy as np

from rlarm.common import load_config, load_policy, make_env

for r in sys.argv[1:]:
    cfg = load_config(f"{r}/config.yaml")
    pol, e, d = load_policy(cfg, f"{r}/model.zip"), make_env(cfg), []
    for ep in range(30):
        o, _ = e.reset(seed=10_000 + ep)
        prev, done = None, False
        while not done:
            a = np.asarray(pol.predict(o, deterministic=True)[0], float)
            if (
                prev is not None
                and np.linalg.norm(o["achieved_goal"] - o["desired_goal"])
                < cfg["env"]["success_dist"]
            ):
                d.append(np.linalg.norm(a - prev))
            prev = a
            o, _, t, tr, _ = e.step(a)
            done = t or tr
    print(f"{r.split('/')[-1]:32s} settled |da| = {np.mean(d):.2f} (n={len(d)})")
