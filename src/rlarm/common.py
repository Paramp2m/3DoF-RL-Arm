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
    return run("rev-parse", "HEAD") + ("-dirty" if run("status", "--porcelain") else "")


def algo_class(cfg: dict):
    """SB3 algorithm class named in the config."""
    return getattr(sb3, cfg["algo"]["name"])


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


def evaluate_goal(policy, env: gym.Env, episodes: int, seed: int = 10_000) -> dict:
    """SPEC §8 eval.json metrics for an arm task env.

    success_rate: tip within success_dist at the last step. ever_success_rate: within it at any step.
    steps_to_success: first step within it, over episodes that ever got there.
    """
    env.action_space.seed(seed)  # makes RandomAgent reproducible; no effect on SB3 policies
    succ, dist, to_succ, smooth, energy, rets = [], [], [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, t, first, prev, da, en, ret = False, 0, None, None, [], 0.0, 0.0
        while not done:
            a = np.asarray(policy.predict(obs, deterministic=True)[0], float)
            obs, r, term, trunc, info = env.step(a)
            t, ret, en = t + 1, ret + r, en + info["energy"]
            if prev is not None:
                da.append(np.linalg.norm(a - prev))
            prev = a
            if first is None and info["is_success"]:
                first = t
            done = term or trunc
        succ.append(info["is_success"])
        dist.append(info["distance"] * 1000)
        smooth.append(np.mean(da) if da else 0.0)
        energy.append(en)
        rets.append(ret)
        if first is not None:
            to_succ.append(first)
    return {
        "episodes": episodes,
        "success_rate": float(np.mean(succ)),
        "ever_success_rate": len(to_succ) / episodes,
        "final_dist_mm_mean": float(np.mean(dist)),
        "final_dist_mm_median": float(np.median(dist)),
        "steps_to_success_mean": float(np.mean(to_succ)) if to_succ else None,
        "smoothness_mean": float(np.mean(smooth)),
        "energy_mean": float(np.mean(energy)) if env.unwrapped.mode.startswith("torque") else None,
        "return_mean": float(np.mean(rets)),
        "return_std": float(np.std(rets)),
    }
