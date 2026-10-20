"""Obstacle runs with the safe filter: fraction of episodes in which the policy's own (unfiltered) move would
have touched the box at least once, vs the executed collision rate. Usage: intended_contacts.py RUN..."""

import sys

import numpy as np

from rlarm.common import load_config, load_policy, make_env

for run in sys.argv[1:]:
    cfg = load_config(f"{run}/config.yaml")
    e, pol = make_env(cfg), load_policy(cfg, f"{run}/model.zip")
    hit, want = [], []
    for ep in range(100):
        o, _ = e.reset(seed=10_000 + ep)
        done, h, w = False, False, False
        while not done:
            o, _, te, tr, i = e.step(pol.predict(o, deterministic=True)[0])
            done, h, w = te or tr, h or i["collision"], w or i.get("intended_contact", False)
        hit.append(h)
        want.append(w)
    print(
        f"{run.split('/')[-1]}: executed collisions {np.mean(hit):.0%}, intended contact {np.mean(want):.0%}"
    )
