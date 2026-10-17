"""Watch a policy act in a live MuJoCo window. Safe to run at any time, even mid-training.

Loads runs/<id>/model.zip, or checkpoint.zip while the run is still training (and reloads it
between episodes when a newer one appears). With no model yet, it acts randomly.
Close the window or press Ctrl+C to stop.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "glfw"  # a window needs GLFW; the project default (egl) is headless

import glfw
import yaml

from rlarm.common import ROOT, algo_class, apply_threads, make_env


def latest_run() -> Path:
    """Most recently modified run folder (skips runs/_dry and runs/_logs)."""
    runs = [p.parent for p in (ROOT / "runs").glob("*/config.yaml")]
    if not runs:
        sys.exit("no runs yet: pass --config to watch a random policy")
    return max(runs, key=lambda p: (p / "summary.json").stat().st_mtime)


def model_file(run: Path | None) -> Path | None:
    """Final model if training finished, else the latest checkpoint, else None."""
    if run is None:
        return None
    return next((run / f for f in ("model.zip", "checkpoint.zip") if (run / f).exists()), None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", help="run folder (default: most recent run)")
    p.add_argument("--config", help="watch a random policy on this config instead of a run")
    p.add_argument("--episodes", type=int, default=0, help="0 = loop until the window closes")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    a = p.parse_args()

    run = None if a.config else Path(a.run) if a.run else latest_run()
    cfg = yaml.safe_load(Path(a.config or run / "config.yaml").read_text())
    apply_threads({**cfg, "torch_threads": 1})  # stay out of training's way
    env = make_env(cfg, "human")
    model, loaded, ep = None, None, 0
    while not a.episodes or ep < a.episodes:
        f = model_file(run)
        if f and (f, f.stat().st_mtime) != loaded:
            model, loaded = algo_class(cfg).load(f, device="cpu"), (f, f.stat().st_mtime)
        steps = json.loads((run / "summary.json").read_text())["steps"] if run else 0
        label = f"{f.name} @ {steps} steps" if f else "random policy"
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            t0 = time.perf_counter()
            act = model.predict(obs, deterministic=not a.stochastic)[0] if model else None
            obs, r, term, trunc, _ = env.step(env.action_space.sample() if act is None else act)
            total, done = total + r, term or trunc
            win = env.unwrapped.mujoco_renderer.viewer.window
            if glfw.window_should_close(win):
                return env.close()
            glfw.set_window_title(win, f"rl-arm watch | {run.name if run else 'random'} | {label}")
            time.sleep(max(0.0, env.unwrapped.dt - (time.perf_counter() - t0)))  # real time
        ep += 1
        print(f"episode {ep}: return {total:.2f} ({label})")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
