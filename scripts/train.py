"""Train an SB3 policy from a YAML config. Prints <=5 lines; writes runs/<run_id>/."""

import argparse
import json
import os
import shutil
import time
from datetime import date

import yaml
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from rlarm.common import (
    ROOT,
    algo_class,
    algo_kwargs,
    apply_threads,
    evaluate,
    evaluate_goal,
    git_hash,
    load_config,
    load_policy,
    make_env,
)


class SummaryCallback(BaseCallback):
    """Every ~10% of training: rewrite summary.json and save checkpoint.zip (for watch.py)."""

    def __init__(self, write, total, run):
        super().__init__()
        self.write, self.every, self.run = write, max(1, total // 10), run

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):  # every reward term to TensorBoard
            for k, v in info.items():
                if k.startswith("r_"):
                    self.logger.record_mean(f"reward/{k[2:]}", v)
        if self.num_timesteps % self.every == 0:
            # save-then-rename so watch.py never loads a half-written checkpoint
            self.model.save(self.run / "checkpoint_tmp.zip")
            os.replace(self.run / "checkpoint_tmp.zip", self.run / "checkpoint.zip")
            self.write("running", self.num_timesteps)
        return True


class ValCallback(BaseCallback):
    """Every `every` steps: evaluate on validation goals (seeds disjoint from the eval seeds) and keep
    the best policy as model_best.zip (higher success, then lower median distance). History -> val.json."""

    def __init__(self, cfg: dict, run, every: int, episodes: int, seed: int):
        super().__init__()
        self.env, self.run, self.every, self.n, self.seed = make_env(cfg), run, every, episodes, seed
        self.best, self.hist = None, []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            ev = evaluate_goal(self.model, self.env, self.n, seed=self.seed)
            key = (ev["success_rate"], -ev["final_dist_mm_median"])
            self.logger.record("val/success", ev["success_rate"])
            self.logger.record("val/median_mm", ev["final_dist_mm_median"])
            self.hist.append({"step": self.num_timesteps, "success": key[0], "median_mm": -key[1]})
            if self.best is None or key > self.best[0]:
                self.best = (key, self.num_timesteps)
                self.model.save(self.run / "model_best_tmp.zip")
                os.replace(self.run / "model_best_tmp.zip", self.run / "model_best.zip")
            (self.run / "val.json").write_text(
                json.dumps({"best_step": self.best[1], "history": self.hist}, indent=1)
            )
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

    best = "model.zip"

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
                    "best_checkpoint": best,
                },
                indent=1,
            )
        )

    write("running", 0)
    try:
        env = make_env(cfg)
        model = algo_class(cfg)(
            cfg["algo"]["policy"],
            env,
            seed=cfg["seed"],
            device="cpu",
            tensorboard_log=str(run / "tb"),
            **algo_kwargs(cfg),
        )
        cbs = [SummaryCallback(write, total, run)]
        v = cfg.get("val")  # optional best-checkpoint selection on validation goals
        if v:
            every, n_val = (total // 2, 2) if a.dry_run else (v["every"], v["episodes"])
            cbs.append(ValCallback(cfg, run, every, n_val, v["seed"]))
        model.learn(total, callback=CallbackList(cbs), progress_bar=False)
        if v and (run / "model_best.zip").exists():
            model.save(run / "final.zip")  # last policy kept for comparison; model.zip = val-best
            os.replace(run / "model_best.zip", run / "model.zip")
            best = f"model.zip (val best @ {cbs[1].best[1]} steps; final.zip = last)"
            model = load_policy(cfg, run / "model.zip")
        else:
            model.save(run / "model.zip")
        n_eval = cfg["dry_eval_episodes"] if a.dry_run else cfg["eval_episodes"]
        if "task" in cfg:
            ev = evaluate_goal(model, make_env(cfg), n_eval)
            final = {
                "success": ev["success_rate"],
                "median_mm": ev["final_dist_mm_median"],
                "return_mean": ev["return_mean"],
            }
        else:
            rets = evaluate(model, make_env(cfg), n_eval)
            final = {"return_mean": float(rets.mean()), "return_std": float(rets.std())}
        write("done", total, final)
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
