"""Diagnostic eval on fresh goals (default seeds 100_000+, disjoint from eval 10_000+ and val 50_000+).

Prints one JSON line: success, median mm, and success split by the joint-space move the goal needs
(Chebyshev distance from start pose to the nearest in-limit IK solution; 'long' = >= 2 rad).
Usage: fresh_eval.py --run runs/<id> [--episodes 1000] [--model model.zip]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from rlarm.common import apply_threads, load_policy, make_env
from rlarm.kinematics import ik


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=100_000)
    p.add_argument("--model", default="model.zip")
    a = p.parse_args()
    run = Path(a.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    apply_threads(cfg)
    model, env = load_policy(cfg, run / a.model), make_env(cfg)
    u = env.unwrapped
    lo, hi = u.lim.T
    succ, dist, long_ = [], [], []
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=a.seed + ep)
        sols = ik(u.goal)
        ok = sols[np.all((sols >= lo - 1e-6) & (sols <= hi + 1e-6), axis=1)]
        long_.append(len(ok) > 0 and np.abs(ok - u.data.qpos).max(1).min() >= 2.0)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            done = term or trunc
        succ.append(info["is_success"])
        dist.append(info["distance"] * 1000)
    s, L = np.array(succ), np.array(long_)
    print(
        json.dumps(
            {
                "run": run.name,
                "model": a.model,
                "episodes": a.episodes,
                "success": round(float(s.mean()), 4),
                "median_mm": round(float(np.median(dist)), 2),
                "success_long": round(float(s[L].mean()), 4),
                "success_short": round(float(s[~L].mean()), 4),
                "n_long": int(L.sum()),
            }
        )
    )


if __name__ == "__main__":
    main()
