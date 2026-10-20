"""Stage 2 of the network-only obstacle plan: behaviour cloning + DAgger from rlarm.expert.PlannerExpert into the
SAC actor that scripts/train.py later fine-tunes (init_from). Prints <= 5 lines; writes runs/<date>_<name>_imitate/:
bc.zip (full SAC model, cloned actor), demos.npz (all labelled (obs, action) pairs), expert_eps.pkl (the first
`prefill_episodes` expert episodes as transitions, for the replay buffer) and imitate.json (val score per round)."""

import argparse
import json
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


def rollout(env, expert, student, beta: float, seed: int, rng, keep: bool):
    """One episode; the expert acts with probability beta per step (student otherwise), every visited state is
    labelled with the expert's action. Returns (obs list, label list, transitions or None, success)."""
    obs, _ = env.reset(seed=seed)
    expert.path = None  # new episode: plan from scratch
    xs, ys, tr, done = [], [], [], False
    while not done:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--quick", action="store_true", help="tiny smoke run (tests)")
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
    run = ROOT / "runs" / ("_dry" if a.quick else "") / f"{date.today()}_{cfg['name']}_imitate"
    run.mkdir(parents=True, exist_ok=a.quick)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run / "git_hash.txt").write_text(git_hash() + "\n")
    t0, rng = time.time(), np.random.default_rng(ic["seed_base"])
    env = make_env(cfg)
    expert = PlannerExpert(env, **cfg.get("expert", {}))
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

    xs, Y, eps, seed, ok = [], [], [], ic["seed_base"], 0
    for i in range(ic["expert_episodes"]):
        x, y, tr, s = rollout(env, expert, None, 1.0, seed, rng, i < ic["prefill_episodes"])
        xs += x
        Y += y
        ok += s
        seed += 1
        if tr:
            eps.append(tr)
    (run / "expert_eps.pkl").write_bytes(pickle.dumps(eps))
    Y_arr = np.asarray(Y, np.float32)
    loss = fit(model, stack(xs), Y_arr, ic["epochs"], ic["batch_size"], ic["lr"])
    score("bc", loss)
    for it, beta in enumerate(ic["betas"][: ic["dagger_iters"]]):
        for _ in range(ic["dagger_episodes"]):
            x, y, _, _ = rollout(env, expert, model, beta, seed, rng, False)
            xs += x
            Y += y
            seed += 1
        Y_arr = np.asarray(Y, np.float32)
        loss = fit(model, stack(xs), Y_arr, ic["dagger_epochs"], ic["batch_size"], ic["lr"])
        score(f"dagger{it + 1}", loss)
    model.save(run / "bc.zip")
    X = stack(xs)
    np.savez_compressed(run / "demos.npz", actions=Y_arr, **{f"obs_{k}": v for k, v in X.items()})
    print(f"run={run.name} expert_success={ok / ic['expert_episodes']:.0%} samples={len(Y)}")
    print(
        "val per round: "
        + ", ".join(f"{h['round']} {h['success']:.0%}/{h['collision']:.0%}" for h in hist)
    )


if __name__ == "__main__":
    main()
