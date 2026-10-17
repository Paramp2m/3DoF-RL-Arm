"""Train an SB3 policy from a YAML config. Prints <=5 lines; writes runs/<run_id>/."""

import argparse
import json
import os
import shutil
import time
from datetime import date

import yaml
from stable_baselines3.common.callbacks import BaseCallback

from rlarm.common import ROOT, algo_class, apply_threads, evaluate, git_hash, load_config, make_env


class SummaryCallback(BaseCallback):
    """Every ~10% of training: rewrite summary.json and save checkpoint.zip (for watch.py)."""

    def __init__(self, write, total, run):
        super().__init__()
        self.write, self.every, self.run = write, max(1, total // 10), run

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            # save-then-rename so watch.py never loads a half-written checkpoint
            self.model.save(self.run / "checkpoint_tmp.zip")
            os.replace(self.run / "checkpoint_tmp.zip", self.run / "checkpoint.zip")
            self.write("running", self.num_timesteps)
        return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    threads = apply_threads(cfg)
    total = cfg["dry_steps"] if a.dry_run else cfg["total_steps"]
    run_id = f"{date.today()}_{cfg['name']}_s{cfg['seed']}" + ("_dry" if a.dry_run else "")
    run = ROOT / "runs" / ("_dry" if a.dry_run else "") / run_id
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run / "git_hash.txt").write_text(git_hash() + "\n")
    t0 = time.time()

    def write(status, steps, final_eval=None):
        wall = time.time() - t0
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "steps": steps,
                    "wall_s": round(wall, 1),
                    "steps_per_s": round(steps / wall, 1) if wall else 0,
                    "torch_threads": threads,
                    "final_eval": final_eval,
                    "best_checkpoint": "model.zip",
                },
                indent=1,
            )
        )

    write("running", 0)
    try:
        env = make_env(cfg)
        model = algo_class(cfg)(
            "MlpPolicy",
            env,
            seed=cfg["seed"],
            device="cpu",
            tensorboard_log=str(run / "tb"),
            **cfg["algo"]["kwargs"],
        )
        model.learn(total, callback=SummaryCallback(write, total, run), progress_bar=False)
        model.save(run / "model.zip")
        rets = evaluate(
            model, make_env(cfg), cfg["dry_eval_episodes"] if a.dry_run else cfg["eval_episodes"]
        )
        write("done", total, {"return_mean": float(rets.mean()), "return_std": float(rets.std())})
    except BaseException:
        write("crashed", model.num_timesteps if "model" in locals() else 0)
        raise
    sps = total / (time.time() - t0)
    print(f"run={run_id} status=done steps={total} steps_per_s={sps:.0f}")
    if a.dry_run:
        print(
            f"ETA full budget ({cfg['total_steps']} steps): {cfg['total_steps'] / sps / 60:.0f} min"
        )


if __name__ == "__main__":
    main()
