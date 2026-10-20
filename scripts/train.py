"""Train an SB3 policy from a YAML config. Prints <=5 lines; writes runs/<run_id>/."""

import argparse
import json
import os
import shutil
import time
from datetime import date

import yaml
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from rlarm.agents import IKAgent
from rlarm.common import (
    ROOT,
    algo_class,
    algo_kwargs,
    apply_threads,
    early_stop_reason,
    evaluate,
    evaluate_goal,
    git_hash,
    load_config,
    load_policy,
    make_env,
)
from rlarm.wrappers import Curriculum


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
                elif k == "difficulty":
                    self.logger.record_mean("curriculum/difficulty", v)
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
        self.clean = cfg["val"].get("clean", False)
        self.env, self.run, self.every, self.n, self.seed = (
            make_env(cfg),
            run,
            every,
            episodes,
            seed,
        )
        self.best, self.hist, self.stop_reason = None, [], None
        self.es, self.total = cfg["val"].get("early_stop"), cfg["total_steps"]
        self.at_start = cfg["val"].get("at_start", False)

    def _on_step(self) -> bool:
        if self.num_timesteps % 5000 == 0 and (self.run / "STOP").exists():
            self.stop_reason = (
                "manual"  # `touch runs/<id>/STOP` ends the run; val-best is still saved + evaluated
            )
            return False
        if self.num_timesteps % self.every == 0:
            return self._validate()
        return True

    def _on_training_start(self) -> None:
        # val.at_start: score the starting policy (e.g. a behaviour-cloned init) as step 0
        if self.at_start:
            self._validate()

    def _validate(self) -> bool:
        """Evaluate on the val goals, keep the best policy, update val.json; False = early stop."""
        ev = evaluate_goal(self.model, self.env, self.n, seed=self.seed)
        score = ev["success_rate"]
        if self.clean and ev["collision_rate"] is not None:  # val.clean: best collision-free policy
            score -= ev["collision_rate"]
        key = (score, -ev["final_dist_mm_median"])
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
        if self.es:  # config val.early_stop: stop plateaued / stuck runs by themselves
            self.stop_reason = early_stop_reason(
                [h["success"] for h in self.hist], self.num_timesteps, self.total, self.es
            )
            if self.stop_reason:
                return False
        return True


def init_policy(model, cfg: dict) -> None:
    """cfg init_from: <imitate run>/bc.zip. Copy its policy weights; give SACBC the demos (demos.npz) for the
    BC term and prefill the replay buffer with the expert episodes (expert_eps.pkl) next to it."""
    import pickle
    from pathlib import Path

    import numpy as np

    src = Path(cfg["init_from"])
    model.policy.load_state_dict(load_policy(cfg, src).policy.state_dict())
    d = np.load(src.parent / "demos.npz")
    if hasattr(model, "set_demos"):
        model.set_demos({k[4:]: d[k] for k in d.files if k.startswith("obs_")}, d["actions"])
    eps = src.parent / "expert_eps.pkl"
    if eps.exists():
        for ep in pickle.loads(eps.read_bytes()):
            for obs, nxt, act, rew, done, info in ep:
                model.replay_buffer.add(
                    {k: v[None] for k, v in obs.items()},
                    {k: v[None] for k, v in nxt.items()},
                    act[None],
                    np.array([rew]),
                    np.array([done]),
                    [info],
                )


def dry_kwargs(kw: dict, total: int, dry: bool) -> dict:
    """Dry run: start learning after 1/3 of the steps so the ETA includes gradient updates (with the config's
    learning_starts >= dry_steps the ETA measured env stepping only and was ~10x optimistic for big nets)."""
    if dry and "learning_starts" in kw:
        kw = {**kw, "learning_starts": min(kw["learning_starts"], total // 3)}
    return kw


def make_train_env(cfg: dict):
    """Training env: the task env, wrapped in a Curriculum when cfg has `curriculum:` (eval/val never are)."""
    env = make_env(cfg)
    c = cfg.get("curriculum")
    return Curriculum(env, **c) if c else env


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
    if a.dry_run:  # real runs are never overwritten: a repeat launch fails at mkdir
        shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run / "git_hash.txt").write_text(git_hash() + "\n")
    t0 = time.time()

    best, extra = "model.zip", {}

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
                    **extra,
                },
                indent=1,
            )
        )

    write("running", 0)
    try:
        v = cfg.get("val")  # optional best-checkpoint selection on validation goals
        if cfg.get("agent") == "ik":  # classical baseline (Phase 4 E4): nothing to learn
            model, total, v, best = IKAgent(make_env(cfg)), 0, None, "none (agent: ik)"
        else:
            n_envs = cfg["algo"].get("n_envs", 1)  # PPO: SubprocVecEnv workers (SPEC §6)
            env = (
                make_vec_env(
                    lambda: make_train_env(cfg), n_envs, cfg["seed"], vec_env_cls=SubprocVecEnv
                )
                if n_envs > 1
                else make_train_env(cfg)
            )
            model = algo_class(cfg)(
                cfg["algo"]["policy"],
                env,
                seed=cfg["seed"],
                device="cpu",
                tensorboard_log=str(run / "tb"),
                **dry_kwargs(algo_kwargs(cfg), total, a.dry_run),
            )
            if cfg.get("init_from"):
                init_policy(model, cfg)
            cbs = [SummaryCallback(write, total, run)]
            if v:
                every, n_val = (total // 2, 2) if a.dry_run else (v["every"], v["episodes"])
                cbs.append(ValCallback(cfg, run, every, n_val, v["seed"]))
            model.learn(total, callback=CallbackList(cbs), progress_bar=False)
            if (
                v and cbs[1].stop_reason
            ):  # ended early: record why and where (usable unless "stuck_low")
                total = model.num_timesteps
                extra["stopped_early"] = cbs[1].stop_reason
                extra["budget"] = cfg["total_steps"]
                extra["usable"] = cbs[1].stop_reason != "stuck_low"
        if v and (run / "model_best.zip").exists():
            model.save(run / "final.zip")  # last policy kept for comparison; model.zip = val-best
            os.replace(run / "model_best.zip", run / "model.zip")
            best = f"model.zip (val best @ {cbs[1].best[1]} steps; final.zip = last)"
            model = load_policy(cfg, run / "model.zip")
        elif cfg.get("agent") != "ik":
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
    sps = total / (
        time.time() - t0
    )  # dry run: includes 2/3 training steps, so the ETA is slightly optimistic
    print(f"run={run_id} status=done steps={total} steps_per_s={sps:.0f}")
    if a.dry_run:
        print(
            f"ETA full budget ({cfg['total_steps']} steps): {cfg['total_steps'] / sps / 60:.0f} min"
        )


if __name__ == "__main__":
    main()
