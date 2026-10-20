"""Stage 2 of the network-only obstacle plan: behaviour cloning + DAgger from rlarm.expert.PlannerExpert into the
SAC actor that scripts/train.py later fine-tunes (init_from). Prints <= 5 lines; writes runs/<date>_<name>_imitate/:
bc.zip (full SAC model, cloned actor), demos.npz (all labelled (obs, action) pairs), expert_eps.pkl (the first
`prefill_episodes` expert episodes as transitions, for the replay buffer) and imitate.json (val score per round)."""

import argparse
import json
import multiprocessing as mp
import pickle
import time
from datetime import date

import numpy as np
import torch as th
import yaml

from rlarm.common import (
    ROOT,
    algo_class,
    algo_kwargs,
    apply_threads,
    evaluate_goal,
    git_hash,
    load_config,
    make_env,
)
from rlarm.expert import PlannerExpert


def rollout(env, expert, student, beta: float, seed: int, rng, keep: bool, settle: int = 0):
    """One episode; the expert acts with probability beta per step (student otherwise), every visited state is
    labelled with the expert's action. Unless `keep`, the episode is cut after `settle` steps in a row at the goal
    (0 = never), so the data is not dominated by holding still. Returns (obs, labels, transitions or None, success)."""
    obs, _ = env.reset(seed=seed)
    expert.path = None  # new episode: plan from scratch
    xs, ys, tr, done, held = [], [], [], False, 0
    while not done and not (settle and not keep and held >= settle):
        label = expert.predict(obs)[0]
        act = (
            label
            if student is None or rng.random() < beta
            else student.predict(obs, deterministic=True)[0]
        )
        nxt, r, term, trunc, info = env.step(act)
        done = term or trunc
        xs.append(obs)
        ys.append(label)
        if keep:
            tr.append(
                (
                    obs,
                    nxt,
                    np.asarray(act, np.float32),
                    r,
                    done,
                    info | {"TimeLimit.truncated": trunc},
                )
            )
        obs = nxt
        held = held + 1 if info["is_success"] else 0
    return xs, ys, tr if keep else None, bool(info["is_success"])


def fit(model, X: dict, Y: np.ndarray, epochs: int, batch: int, lr: float) -> float:
    """MSE between the actor's deterministic (squashed) action and the expert label. Returns the last epoch loss."""
    actor = model.actor
    opt = th.optim.Adam(actor.parameters(), lr=lr)
    Xt = {k: th.as_tensor(v) for k, v in X.items()}
    Yt = th.as_tensor(Y)
    n, last = len(Yt), 0.0
    actor.set_training_mode(True)
    for _ in range(epochs):
        perm, tot = th.randperm(n), 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            loss = th.nn.functional.mse_loss(
                actor({k: v[idx] for k, v in Xt.items()}, deterministic=True), Yt[idx]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        last = tot / n
    actor.set_training_mode(False)
    return last


def stack(obs_list: list[dict]) -> dict:
    return {k: np.stack([o[k] for o in obs_list]).astype(np.float32) for k in obs_list[0]}


_W: dict = {}  # per worker process: its own env, expert and student policy copy


def _init(cfg: dict, policy) -> None:
    th.set_num_threads(1)
    env = make_env(cfg)
    _W.update(env=env, expert=PlannerExpert(env, **cfg.get("expert", {})), student=policy)


def _episode(seed: int, beta: float, keep: bool, settle: int):
    # rng per episode seed: the data does not depend on how episodes are spread over workers
    rng = np.random.default_rng(seed)
    return rollout(_W["env"], _W["expert"], _W["student"], beta, seed, rng, keep, settle)


def collect(cfg: dict, policy, jobs: list[tuple], workers: int) -> list:
    """Run rollout jobs (seed, beta, keep, settle) on `workers` processes; results in job order."""
    with mp.get_context("spawn").Pool(workers, _init, (cfg, policy)) as pool:
        return pool.starmap(_episode, jobs, chunksize=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--quick", action="store_true", help="tiny smoke run (tests)")
    p.add_argument("--workers", type=int, default=8, help="rollout processes (compute only)")
    a = p.parse_args()
    cfg = load_config(a.config)
    ic = dict(cfg["imitate"])
    if a.quick:
        ic.update(
            expert_episodes=3,
            dagger_iters=1,
            dagger_episodes=2,
            epochs=1,
            dagger_epochs=1,
            val_episodes=2,
            prefill_episodes=2,
        )
    apply_threads(cfg)
    th.set_num_threads(a.workers)  # the BC fit runs while the rollout workers are idle
    run = ROOT / "runs" / ("_dry" if a.quick else "") / f"{date.today()}_{cfg['name']}_imitate"
    run.mkdir(parents=True, exist_ok=a.quick)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run / "git_hash.txt").write_text(git_hash() + "\n")
    t0 = time.time()
    env = make_env(cfg)
    kw = {
        k: v
        for k, v in algo_kwargs(cfg).items()
        if k not in ("bc_weight", "bc_decay", "actor_freeze")
    }
    model = algo_class(cfg)(cfg["algo"]["policy"], env, seed=cfg["seed"], device="cpu", **kw)
    val_env, hist = make_env(cfg), []

    def score(tag: str, loss: float) -> None:
        ev = evaluate_goal(model, val_env, ic["val_episodes"], seed=cfg["val"]["seed"])
        hist.append(
            {
                "round": tag,
                "samples": len(Y),
                "bc_loss": loss,
                "success": ev["success_rate"],
                "collision": ev["collision_rate"],
                "wall_s": round(time.time() - t0),
            }
        )
        (run / "imitate.json").write_text(json.dumps(hist, indent=1))

    xs, Y, eps, seed, ok, settle = [], [], [], ic["seed_base"], 0, ic.get("settle", 0)
    jobs = [
        (seed + i, 1.0, i < ic["prefill_episodes"], settle) for i in range(ic["expert_episodes"])
    ]
    for x, y, tr, s in collect(cfg, None, jobs, a.workers):
        xs += x
        Y += y
        ok += s
        if tr:
            eps.append(tr)
    seed += len(jobs)
    (run / "expert_eps.pkl").write_bytes(pickle.dumps(eps))
    Y_arr = np.asarray(Y, np.float32)
    loss = fit(model, stack(xs), Y_arr, ic["epochs"], ic["batch_size"], ic["lr"])
    score("bc", loss)
    model.save(run / "bc.zip")
    for it, beta in enumerate(ic["betas"][: ic["dagger_iters"]]):
        jobs = [(seed + i, beta, False, settle) for i in range(ic["dagger_episodes"])]
        for x, y, _, _ in collect(cfg, model.policy, jobs, a.workers):
            xs += x
            Y += y
        seed += len(jobs)
        Y_arr = np.asarray(Y, np.float32)
        loss = fit(model, stack(xs), Y_arr, ic["dagger_epochs"], ic["batch_size"], ic["lr"])
        score(f"dagger{it + 1}", loss)
        model.save(run / "bc.zip")  # every round, so a stopped run still leaves its latest student
    X = stack(xs)
    np.savez_compressed(run / "demos.npz", actions=Y_arr, **{f"obs_{k}": v for k, v in X.items()})
    print(f"run={run.name} expert_success={ok / ic['expert_episodes']:.0%} samples={len(Y)}")
    print(
        "val per round: "
        + ", ".join(f"{h['round']} {h['success']:.0%}/{h['collision']:.0%}" for h in hist)
    )


if __name__ == "__main__":
    main()
