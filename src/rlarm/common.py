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
    """Build the Gymnasium env named in the config."""
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
