"""Shared helpers for the train / eval / render scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    """Read a YAML experiment config."""
    return yaml.safe_load(Path(path).read_text())


def apply_threads(cfg: dict) -> int:
    """Set torch's thread count from config and return the value in effect."""
    torch.set_num_threads(int(cfg["torch_threads"]))
    return torch.get_num_threads()


def make_env(cfg: dict, render_mode: str | None = None, **kw) -> gym.Env:
    """Build the arm task env (cfg['task']) or the Gymnasium env named in cfg['env']['id']."""
    if "task" in cfg:
        from rlarm.envs import make

        return make(cfg, render_mode=render_mode, **kw)
    return gym.make(cfg["env"]["id"], render_mode=render_mode, **kw)


def git_hash() -> str:
    """Current commit hash (with -dirty suffix if the tree has changes)."""
    run = lambda *a: subprocess.run(
        ["git", *a], check=False, cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    return run("rev-parse", "HEAD") + (
        "-dirty" if run("status", "--porcelain", "--untracked-files=no") else ""
    )


def algo_class(cfg: dict):
    """SB3 algorithm class named in the config."""
    return getattr(sb3, cfg["algo"]["name"])


def algo_kwargs(cfg: dict) -> dict:
    """cfg algo.kwargs with class names resolved: replay_buffer_class (SB3), policy_kwargs
    activation_fn (torch.nn) and features_extractor_class (rlarm.policies)."""
    from rlarm import policies

    kw = dict(cfg["algo"]["kwargs"])
    if "replay_buffer_class" in kw:
        kw["replay_buffer_class"] = getattr(sb3, kw["replay_buffer_class"])
    pk = dict(kw.get("policy_kwargs", {}))
    if "activation_fn" in pk:
        pk["activation_fn"] = getattr(torch.nn, pk["activation_fn"])
    if "features_extractor_class" in pk:
        pk["features_extractor_class"] = getattr(policies, pk["features_extractor_class"])
    if pk:
        kw["policy_kwargs"] = pk
    return kw


def load_policy(cfg: dict, path: str | Path):
    """Load a saved model for inference (HER models need an env to rebuild their buffer)."""
    her = "replay_buffer_class" in cfg["algo"]["kwargs"]
    return algo_class(cfg).load(path, env=make_env(cfg) if her else None, device="cpu")


def evaluate(model, env: gym.Env, episodes: int, seed: int = 10_000) -> np.ndarray:
    """Deterministic rollouts; returns the per-episode return."""
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
            total += r
            done = term or trunc
        rets.append(total)
    return np.array(rets)


EVAL_SCHEMA = (  # eval.json keys for arm tasks (SPEC §8), in the order evaluate_goal writes them
    "episodes",
    "success_rate",
    "ever_success_rate",
    "final_dist_mm_mean",
    "final_dist_mm_median",
    "steps_to_success_mean",
    "final_joint_speed_mean",
    "smoothness_mean",
    "energy_mean",
    "return_mean",
    "return_std",
    "track_err_mm_mean",
    "collision_rate",
)


TRACK_SKIP = 50  # tracking error ignores the first steps (ROADMAP 5.1)


def evaluate_goal(
    policy, env: gym.Env, episodes: int, seed: int = 10_000, speed_window: int = 50
) -> dict:
    """SPEC §8 eval.json metrics for an arm task env.

    success_rate: tip within success_dist at the last step. ever_success_rate: within it at any step.
    track_err_mm_mean: mean tip-goal distance over steps > 50. collision_rate: fraction of episodes with
    an arm-obstacle contact (None when the env reports no `collision`).
    steps_to_success: first step within it, over episodes that ever got there.
    final_joint_speed_mean: mean over episodes of the mean ‖q̇‖ (rad/s) over the last speed_window steps.
    """
    env.action_space.seed(seed)  # makes RandomAgent reproducible; no effect on SB3 policies
    succ, dist, to_succ, smooth, energy, rets, speed = [], [], [], [], [], [], []
    track, hit = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, t, first, prev, da, en, ret, sp = False, 0, None, None, [], 0.0, 0.0, []
        err, hits = [], False
        while not done:
            a = np.asarray(policy.predict(obs, deterministic=True)[0], float)
            obs, r, term, trunc, info = env.step(a)
            t, ret, en = t + 1, ret + r, en + info["energy"]
            if prev is not None:
                da.append(np.linalg.norm(a - prev))
            prev = a
            sp.append(info["joint_speed"])
            if t > TRACK_SKIP:
                err.append(info["distance"] * 1000)
            hits = hits or bool(info.get("collision", False))
            if first is None and info["is_success"]:
                first = t
            done = term or trunc
        succ.append(info["is_success"])
        dist.append(info["distance"] * 1000)
        smooth.append(np.mean(da) if da else 0.0)
        energy.append(en)
        speed.append(np.mean(sp[-speed_window:]))
        rets.append(ret)
        track.append(np.mean(err))
        hit.append(hits)
        if first is not None:
            to_succ.append(first)
    return {
        "episodes": episodes,
        "success_rate": float(np.mean(succ)),
        "ever_success_rate": len(to_succ) / episodes,
        "final_dist_mm_mean": float(np.mean(dist)),
        "final_dist_mm_median": float(np.median(dist)),
        "steps_to_success_mean": float(np.mean(to_succ)) if to_succ else None,
        "final_joint_speed_mean": float(np.mean(speed)),
        "smoothness_mean": float(np.mean(smooth)),
        "energy_mean": float(np.mean(energy)) if env.unwrapped.mode.startswith("torque") else None,
        "return_mean": float(np.mean(rets)),
        "return_std": float(np.std(rets)),
        "track_err_mm_mean": float(np.mean(track)),
        "collision_rate": float(np.mean(hit)) if "collision" in info else None,
    }


def experiments() -> dict[str, dict]:
    """Phase 4 manifests: configs/experiments/<stem>.yaml -> {title, question, arms}."""
    return {p.stem: load_config(p) for p in sorted((ROOT / "configs/experiments").glob("*.yaml"))}


def arm_runs(arm: dict) -> list[Path]:
    """Run folder for each seed of a manifest arm: newest runs/<date>_<name>_s<seed> (dates sort)."""
    name = load_config(ROOT / arm["config"])["name"]
    out = []
    for s in arm["seeds"]:
        found = sorted((ROOT / "runs").glob(f"????-??-??_{name}_s{s}"))
        if not found:
            raise FileNotFoundError(f"no run folder for {name} seed {s}")
        out.append(found[-1])
    return out
