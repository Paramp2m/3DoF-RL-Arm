"""Evaluate a run: writes eval.json and a video GIF into the run folder."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import yaml

from rlarm.agents import IKAgent
from rlarm.common import apply_threads, evaluate, evaluate_goal, load_policy, make_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--no-video", action="store_true", help="only (re)write eval.json")
    a = p.parse_args()
    run = Path(a.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    apply_threads(cfg)
    model = (
        IKAgent(make_env(cfg)) if cfg.get("agent") == "ik" else load_policy(cfg, run / "model.zip")
    )
    if "task" in cfg:  # arm tasks: SPEC §8 schema
        out = evaluate_goal(model, make_env(cfg), cfg["eval_episodes"])
    else:
        rets = evaluate(model, make_env(cfg), cfg["eval_episodes"])
        out = {
            "episodes": len(rets),
            "return_mean": float(rets.mean()),
            "return_std": float(rets.std()),
        }
    (run / "eval.json").write_text(json.dumps(out, indent=1))
    if a.no_video:
        print(f"{run.name}: eval.json written")
        return

    env, frames = make_env(cfg, "rgb_array", width=320, height=320), []
    for ep in range(cfg["gif_episodes"]):
        obs, _ = env.reset(seed=20_000 + ep)
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
            frames.append(env.render())
            done = term or trunc
    imageio.mimsave(run / "video.gif", frames[::2], fps=25)
    print(
        f"{run.name}: return_mean={out['return_mean']:.2f} ± {out['return_std']:.2f} (n={out['episodes']})"
    )


if __name__ == "__main__":
    main()
