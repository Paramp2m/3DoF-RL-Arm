"""Phase gate. Reads artifacts (docs/best_runs.yaml + run folders); never trains.

Usage: check_phase.py <N> [--summary]. One line per check ID with its measured value.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RETURN_TARGET, STEP_CAP, DRY_LIMIT_S = -7.0, 300_000, 120.0


def best_run(key: str) -> Path | None:
    """Run dir for phase/task from docs/best_runs.yaml."""
    d = yaml.safe_load((ROOT / "docs/best_runs.yaml").read_text()) or {}
    node = d
    for k in key.split("."):
        node = node.get(k) if isinstance(node, dict) else None
    return ROOT / node if node else None


def sh(*cmd) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, cwd=ROOT, capture_output=True, text=True)


def c01():
    import gymnasium, mujoco, stable_baselines3, torch  # noqa: I001

    v = f"mujoco {mujoco.__version__}, gymnasium {gymnasium.__version__}, sb3 {stable_baselines3.__version__}, torch {torch.__version__}"
    return True, v


def c02():

    from rlarm.common import apply_threads, load_config

    cfg = load_config(ROOT / "configs/reacher_sac.yaml")
    n = apply_threads(cfg)
    return n == cfg["torch_threads"] and n != 16, f"threads={n} (config {cfg['torch_threads']})"


def c03():
    import imageio.v2 as imageio

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "snap.png"
        r = sh(
            "uv",
            "run",
            "python",
            "scripts/render_snapshot.py",
            "--config",
            "configs/reacher_sac.yaml",
            "--out",
            str(out),
        )
        if r.returncode:
            return False, "render failed: " + r.stderr.strip().splitlines()[-1]
        std = float(imageio.imread(out).std())
    return std > 5, f"pixel std={std:.1f}"


def c04():
    t = time.time()
    r = sh(
        "uv",
        "run",
        "python",
        "scripts/train.py",
        "--config",
        "configs/reacher_sac.yaml",
        "--dry-run",
    )
    dt = time.time() - t
    return r.returncode == 0 and dt < DRY_LIMIT_S, f"{dt:.0f}s rc={r.returncode}"


def c05():
    run = best_run("phase0.reacher")
    if not run or not (run / "eval.json").exists():
        return False, "no phase0.reacher run in best_runs.yaml / eval.json missing"
    ev = json.loads((run / "eval.json").read_text())
    sm = json.loads((run / "summary.json").read_text())
    steps = sm["steps"]
    ok = (
        sm["status"] == "done"
        and ev["return_mean"] >= RETURN_TARGET
        and ev["episodes"] >= 100
        and steps <= STEP_CAP
    )
    return ok, f"return={ev['return_mean']:.2f} over {ev['episodes']} eps, steps={steps}"


def c06():
    run = best_run("phase0.reacher")
    if not run:
        return False, "no run"
    need = ["config.yaml", "git_hash.txt", "model.zip", "eval.json", "summary.json", "video.gif"]
    missing = [f for f in need if not (run / f).exists()]
    if not list((run / "tb").glob("**/events.out.tfevents*")):
        missing.append("tb events")
    if (run / "video.gif").exists() and (run / "video.gif").stat().st_size == 0:
        missing.append("empty video.gif")
    threads = json.loads((run / "summary.json").read_text()).get("torch_threads")
    cfg_threads = yaml.safe_load((run / "config.yaml").read_text())["torch_threads"]
    if threads != cfg_threads or threads == 16:
        missing.append(f"train threads {threads} != config {cfg_threads}")
    return not missing, f"missing={missing or 'none'}"


def c07():
    t = sh("uv", "run", "pytest", "-q")
    r = sh("uv", "run", "ruff", "check", ".")
    last = t.stdout.strip().splitlines()[-1] if t.stdout.strip() else "no output"
    return t.returncode == 0 and r.returncode == 0, f"pytest: {last}; ruff rc={r.returncode}"


# ---- Phase 1: arm model and kinematics ----
ARM_CFG = "configs/arm.yaml"


def _arm():
    import mujoco

    from rlarm.arm import load_model
    from rlarm.common import load_config

    m = load_model(load_config(ROOT / ARM_CFG))
    return m, mujoco.MjData(m), m.jnt_range


def c11():
    import mujoco

    m, _, _ = _arm()
    hinges = int((m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE).sum())
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip") >= 0
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target")
    mocap = tid >= 0 and m.body_mocapid[tid] >= 0
    return m.njnt == 3 and hinges == 3 and site and mocap, (
        f"joints={m.njnt} hinges={hinges} tip={site} target_mocap={mocap}"
    )


def c12():
    import mujoco

    from rlarm.kinematics import fk

    m, d, lim = _arm()
    q = np.random.default_rng(0).uniform(lim[:, 0], lim[:, 1], (1000, 3))
    ref = np.empty((1000, 3))
    for i, qi in enumerate(q):
        d.qpos[:] = qi
        mujoco.mj_kinematics(m, d)
        ref[i] = d.site("tip").xpos
    err = float(np.abs(fk(q) - ref).max())
    return err < 1e-6, f"max err={err:.2e} m"


def c13():
    from rlarm.kinematics import fk, ik

    _, _, lim = _arm()
    q = np.random.default_rng(1).uniform(lim[:, 0], lim[:, 1], (1000, 3))
    fk_err, found, nsol = 0.0, 0, []
    for qi, p in zip(q, fk(q)):
        sols = ik(p)
        nsol.append(len(sols))
        if len(sols):
            fk_err = max(fk_err, float(np.abs(fk(sols) - p).max()))
        dq = np.abs((sols - qi + np.pi) % (2 * np.pi) - np.pi)
        found += bool(len(sols) and (dq.max(axis=1) < 1e-6).any())
    ok = fk_err < 1e-6 and found == 1000
    return (
        ok,
        f"max FK(IK) err={fk_err:.2e} m, sample config recovered {found}/1000, solutions {min(nsol)}-{max(nsol)}",
    )


def c14():
    import itertools

    import mujoco

    m, d, lim = _arm()
    rng = np.random.default_rng(2)
    corners = np.array(
        list(itertools.product(*lim.tolist()))
    )  # the 8 box corners, extra to the 2000
    qs = np.vstack([rng.uniform(lim[:, 0], lim[:, 1], (2000, 3)), corners])
    bad = 0
    for qi in qs:
        d.qpos[:] = qi
        mujoco.mj_forward(m, d)
        bad += d.ncon
    return bad == 0, f"contacts={bad} over {len(qs)} configs (2000 random + 8 corners)"


def c15():
    import mujoco

    from rlarm.arm import MOTOR

    m, d, lim = _arm()
    assert m.opt.disableactuator == 1 << MOTOR, "position variant not active"
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(5):
        goal = rng.uniform(lim[:, 0], lim[:, 1])
        mujoco.mj_resetData(m, d)
        d.qpos[:] = lim.mean(axis=1)
        d.ctrl[:3] = goal
        for _ in range(int(2.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        worst = max(worst, float(np.degrees(np.abs(d.qpos - goal).max())))
    return worst < 2.0, f"max steady-state err={worst:.2f} deg"


def c16():
    import imageio.v2 as imageio

    f = ROOT / "docs/media/phase1_arm.png"
    with tempfile.TemporaryDirectory() as d:  # re-render so a stale or arm-less image can't pass
        r = sh(
            "uv",
            "run",
            "python",
            "scripts/render_snapshot.py",
            "--config",
            ARM_CFG,
            "--out",
            f"{d}/a.png",
        )
        if r.returncode or not f.exists():
            return False, "render failed or docs/media/phase1_arm.png missing"
        fresh = imageio.imread(f"{d}/a.png")
    saved = imageio.imread(f)
    std = float(saved.std())
    same = saved.shape == fresh.shape and float(np.abs(saved.astype(int) - fresh).mean()) < 1.0
    return std > 5 and same and max(
        saved.shape[:2]
    ) <= 640, f"pixel std={std:.1f}, matches fresh render={same}"


# ---- Phase 2: reach environment ----
REACH_CFG = "configs/reach.yaml"


def _reach(mode: str = "delta_pos", **arm):
    from rlarm.common import load_config, make_env

    cfg = load_config(ROOT / REACH_CFG)
    cfg["env"]["action_mode"] = mode
    cfg["arm"].update(arm)
    return make_env(cfg)


def c21():
    import warnings

    from stable_baselines3.common.env_checker import check_env

    from rlarm.envs.base import MODES

    out = []
    for mode in MODES:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env(_reach(mode))  # raises on failure
        out.append(f"{mode} ok ({len(w)} warn)")
    return True, ", ".join(out)


def c22():
    from rlarm.envs.base import MODES

    bad, vmax = [], 0.0
    for mode in MODES:
        env = _reach(mode)
        env.action_space.seed(0)
        obs, _ = env.reset(seed=0)
        for _ in range(5000):
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            vmax = max(vmax, float(np.abs(env.data.qvel).max()))
            arrs = [*obs.values(), np.array([r])]
            if not all(np.isfinite(x).all() for x in arrs) or not env.observation_space.contains(
                obs
            ):
                bad.append(mode)
                break
            if term or trunc:
                obs, _ = env.reset()
    return not bad, f"modes failing={bad or 'none'}; max raw |qvel|={vmax:.1f} rad/s"


def c23():
    from rlarm.envs.base import MODES

    worst = 0.0
    for mode in MODES:
        env = _reach(mode)
        env.action_space.seed(1)
        obs, _ = env.reset(seed=1)
        ag, dg, infos, rs = [], [], [], []
        for _ in range(1000):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            (
                ag.append(obs["achieved_goal"]),
                dg.append(obs["desired_goal"]),
                infos.append(info),
                rs.append(r),
            )
            if term or trunc:
                obs, _ = env.reset()
        batched = env.compute_reward(np.array(ag), np.array(dg), np.array(infos))
        worst = max(worst, float(np.abs(batched - np.array(rs)).max()))
    return worst <= 1e-6, f"max |batched - per-step|={worst:.1e} over 4x1000 steps"


def c24():
    from rlarm.envs.base import MODES

    same = []
    for mode in MODES:
        trajs, env = [], _reach(mode)
        for _ in range(2):  # same env reset twice: also catches state leaking across resets
            rng = np.random.default_rng(7)
            obs, _ = env.reset(seed=123)
            t = [obs["observation"].tobytes()]
            for _ in range(200):
                obs, *_ = env.step(rng.uniform(-1, 1, 3).astype(np.float32))
                t.append(b"".join(v.tobytes() for v in obs.values()))
            trajs.append(t)
        same.append(trajs[0] == trajs[1])
    fresh = _reach(MODES[-1]).reset(seed=123)[0]["observation"].tobytes() == trajs[0][0]
    return all(same) and fresh, (
        f"bitwise identical per mode={dict(zip(MODES, same))}, fresh env same start={fresh}"
    )


def _agent_eval(kind: str) -> dict:
    from rlarm.agents import IKAgent, RandomAgent
    from rlarm.common import evaluate_goal

    env = _reach("delta_pos")
    return evaluate_goal((IKAgent if kind == "ik" else RandomAgent)(env), env, 100)


def c25():
    ev = _agent_eval("ik")
    return ev["success_rate"] >= 0.98, (
        f"success={ev['success_rate']:.0%}, final dist median={ev['final_dist_mm_median']:.1f} mm, "
        f"steps to success={ev['steps_to_success_mean']}"
    )


def c26():
    ev = _agent_eval("random")
    return ev["success_rate"] <= 0.05, (
        f"success={ev['success_rate']:.0%} (ever within 2 cm: {ev['ever_success_rate']:.0%}), "
        f"final dist median={ev['final_dist_mm_median']:.0f} mm"
    )


def c27():
    base = _reach().model
    m = _reach(
        body_mass={"shoulder": 0.8, "upper_arm": 2.0, "forearm": 0.6}, joint_damping=1.5
    ).model
    b = base.body("upper_arm").id
    ok = m.body_mass[b] == 2.0 and base.body_mass[b] == 1.0 and (m.dof_damping == 1.5).all()
    ok &= bool(np.allclose(m.body_inertia[b], 2 * base.body_inertia[b]))
    return ok, (
        f"upper_arm mass {base.body_mass[b]:.1f}->{m.body_mass[b]:.1f} kg, "
        f"damping {base.dof_damping[0]:.1f}->{m.dof_damping[0]:.1f}"
    )


def c28():
    from rlarm import rewards

    env = _reach()
    env.reset(seed=0)
    _, r, _, _, info = env.step(env.action_space.sample())
    names = list(rewards.terms(np.zeros(3), np.zeros(3), env.ecfg["reward"], 0.02))
    missing = [n for n in names if f"r_{n}" not in info]
    total = sum(info[f"r_{n}"] for n in names if f"r_{n}" in info)
    ok = not missing and abs(total - r) < 1e-9
    return ok, f"terms={names}, missing={missing or 'none'}, sum==reward {abs(total - r) < 1e-9}"


# ---- Phase 3: learned reach and hold ----
STEP_LIMIT, MEDIA_MAX = 1_000_000, 3_000_000
# ROADMAP task definitions each run's config copy must match (delta_pos, 2 cm, 200-step reach, 50-of-250 hold;
# phase 5: circle r 8 cm / T 8 s (guide D-3, locked before training), push goal within 3 cm, obstacle reach 2 cm)
TASK_DEF = {
    "track": {"action_mode": "delta_pos", "radius": 0.08, "period": 8.0, "episode_steps": 200},
    # test-time helpers (planner waypoint, residual, rollback filter) allowed again (user decision 2026-10-21):
    # the network-only variant (configs/obstacle_h*) reached 89.7% / 11.3%, see docs/reports/phase-5.md
    "obstacle": {"action_mode": "delta_pos", "success_dist": 0.02},
    "push": {"action_mode": "delta_pos", "success_dist": 0.03},
    "reach": {"action_mode": "delta_pos", "success_dist": 0.02, "episode_steps": 200},
    "hold": {
        "action_mode": "delta_pos",
        "success_dist": 0.02,
        "episode_steps": 250,
        "hold_steps": 50,
    },
}


PHASE5_STEPS = {"track": 1_000_000, "obstacle": 1_000_000, "push": 3_000_000}  # guide D-8


def _evals(task: str, phase: int = 3) -> list[dict]:
    """eval.json of the 3 best-run seeds for a task (docs/best_runs.yaml phase<N>.<task>), validated."""
    d = yaml.safe_load((ROOT / "docs/best_runs.yaml").read_text()) or {}
    runs = (d.get(f"phase{phase}") or {}).get(task) or []
    if len(runs) != 3:
        raise ValueError(f"phase{phase}.{task} needs 3 runs in best_runs.yaml, found {len(runs)}")
    limit = STEP_LIMIT if phase == 3 else PHASE5_STEPS[task]
    out, seeds = [], set()
    for r in runs:
        run = ROOT / r
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        steps = json.loads((run / "summary.json").read_text())["steps"]
        ev = json.loads((run / "eval.json").read_text())
        if cfg["task"] != task or steps > limit or ev["episodes"] != 100:
            raise ValueError(f"{r}: task={cfg['task']} steps={steps} episodes={ev['episodes']}")
        bad = {
            k: cfg["env"].get(k)
            for k, v in TASK_DEF.get(task, {}).items()
            if cfg["env"].get(k) != v
        }
        if bad:
            raise ValueError(f"{r}: env differs from ROADMAP task definition: {bad}")
        if (run / "eval.json").stat().st_mtime < (run / "model.zip").stat().st_mtime:
            raise ValueError(f"{r}: eval.json is older than model.zip (stale)")
        seeds.add(cfg["seed"])
        out.append(ev)
    if len(seeds) != 3:
        raise ValueError(f"{task}: seeds not distinct: {sorted(seeds)}")
    return out


def _mean(task: str, key: str, phase: int = 3) -> tuple[float, list[float]]:
    vals = [e[key] for e in _evals(task, phase)]
    return float(np.mean(vals)), vals


def _fmt(vals, scale=1.0, f=".1f") -> str:
    return "[" + ", ".join(format(v * scale, f) for v in vals) + "]"


def c31():
    m, v = _mean("reach", "success_rate")
    return m >= 0.95, f"reach success mean={m:.1%} per seed {_fmt(v, 100)}%"


def c32():
    m, v = _mean("reach", "final_dist_mm_median")
    return m <= 10, f"reach median final distance mean={m:.1f} mm per seed {_fmt(v)}"


def c33():
    m, v = _mean("hold", "success_rate")
    return m >= 0.90, f"hold success mean={m:.1%} per seed {_fmt(v, 100)}%"


def c34():
    m, v = _mean("hold", "final_joint_speed_mean")
    return (
        m < 0.05,
        f"hold mean joint speed (final 50 steps) mean={m:.4f} rad/s per seed {_fmt(v, f='.4f')}",
    )


def c35():
    gifs = {t: ROOT / f"docs/media/phase3_{t}.gif" for t in ("reach", "hold")}
    sizes = {t: p.stat().st_size for t, p in gifs.items()}
    real = all(p.read_bytes()[:6] in (b"GIF87a", b"GIF89a") for p in gifs.values())
    return real and all(0 < v < MEDIA_MAX for v in sizes.values()), (
        "gif sizes " + ", ".join(f"{t}={v / 1e6:.2f} MB" for t, v in sizes.items())
    )


def c41():
    from rlarm.common import EVAL_SCHEMA, arm_runs, experiments

    n, bad = 0, []
    for stem, exp in experiments().items():
        for label, arm in exp["arms"].items():
            for run in arm_runs(arm):
                n += 1
                f = run / "eval.json"
                ev = json.loads(f.read_text()) if f.exists() else {}
                ok = tuple(ev) == EVAL_SCHEMA and ev["episodes"] == 100
                ok = ok and all(v is None or isinstance(v, int | float) for v in ev.values())
                if not ok:
                    bad.append(f"{stem}/{label}/{run.name}")
    return not bad and n > 0, f"{n} runs, {len(bad)} without a valid eval.json {bad[:4]}"


def c42():
    r = sh(sys.executable, "scripts/make_results.py", "--check")
    committed = sh("git", "diff", "--quiet", "HEAD", "--", "docs/results.md").returncode == 0
    tracked = sh("git", "ls-files", "--error-unmatch", "docs/results.md").returncode == 0
    ok = r.returncode == 0 and committed and tracked
    return ok, f"{(r.stdout or r.stderr).strip()[-120:]}; committed={committed and tracked}"


def c43():
    from rlarm.common import arm_runs, experiments

    stems = [s for s in experiments() if s[:2] in ("e1", "e2", "e3")]
    pngs = {s: ROOT / f"docs/media/{s}_curves.png" for s in stems}
    exps = experiments()

    def fresh(s, p):  # a PNG older than any val.json it plots is stale
        vals = [r / "val.json" for a in exps[s]["arms"].values() for r in arm_runs(a)]
        return all(v.stat().st_mtime <= p.stat().st_mtime for v in vals)

    ok = {
        s: p.exists() and p.read_bytes()[:4] == b"\x89PNG" and fresh(s, p) for s, p in pngs.items()
    }
    return len(stems) == 3 and all(ok.values()), "curves " + ", ".join(
        f"{s}={'ok' if v else 'missing'}" for s, v in ok.items()
    )


def c44():
    text = (ROOT / "docs/results.md").read_text()
    secs = [s for s in text.split("\n## ")[1:] if "\n|" in s]  # sections holding a table
    need = ("Seeds:", "Step budget:", "Configs:")
    miss = [s.splitlines()[0] for s in secs if not all(k in s for k in need)]
    return len(secs) >= 4 and not miss, f"{len(secs)} tables, missing seeds/budget/configs: {miss}"


def c51():
    m, v = _mean("track", "track_err_mm_mean", 5)
    return m <= 25, f"track mean error after step 50 = {m:.1f} mm per seed {_fmt(v)}"


def c52():
    s, sv = _mean("obstacle", "success_rate", 5)
    c, cv = _mean("obstacle", "collision_rate", 5)
    return s >= 0.85 and c <= 0.05, (
        f"obstacle success={s:.1%} per seed {_fmt(sv, 100)}%; collisions={c:.1%} per seed {_fmt(cv, 100)}%"
    )


def c53():
    m, v = _mean("push", "success_rate", 5)
    return m >= 0.60, f"push success mean={m:.1%} per seed {_fmt(v, 100)}%"


def c54():
    gifs = {t: ROOT / f"docs/media/phase5_{t}.gif" for t in ("track", "obstacle", "push")}
    sizes = {t: p.stat().st_size for t, p in gifs.items()}
    real = all(p.read_bytes()[:6] in (b"GIF87a", b"GIF89a") for p in gifs.values())
    return real and all(0 < v < MEDIA_MAX for v in sizes.values()), (
        "gif sizes " + ", ".join(f"{t}={v / 1e6:.2f} MB" for t, v in sizes.items())
    )


def c61():
    r = sh("uv", "run", "python", "scripts/demo.py", "--headless", "--steps", "2000")
    if r.returncode:
        return False, f"demo.py exit {r.returncode}: {r.stderr.strip()[-200:]}"
    out = json.loads(r.stdout.strip().splitlines()[-1])
    m = out["mean_dist_mm_second_half"]
    return out["steps"] == 2000 and m <= 30, (
        f"{out['steps']} steps, mean tip-target distance in the second half = {m:.1f} mm ({out['run']})"
    )


README_SECTIONS = ("Overview", "Demo", "Results", "How to run", "Project structure")


def c62():
    heads = {
        ln.lstrip("#").strip()
        for ln in (ROOT / "README.md").read_text().splitlines()
        if ln.startswith("#")
    }
    miss = [h for h in README_SECTIONS if h not in heads]
    return not miss, f"missing sections: {miss}" if miss else "all 5 sections present"


def c63():
    t = sh("uv", "run", "pytest", "-q")
    rf = sh("uv", "run", "ruff", "check", ".")
    tail = (t.stdout.strip().splitlines() or ["no output"])[-1]
    tail += "".join(f" | {ln}" for ln in t.stdout.splitlines() if ln.startswith("FAILED"))
    return t.returncode == 0 and rf.returncode == 0, (
        f"pytest: {tail}; ruff: {(rf.stdout.strip().splitlines() or ['?'])[-1]}"
    )


def c64():
    env = {
        k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"
    }  # don't reuse this checkout's venv
    head = sh("git", "rev-parse", "--short", "HEAD").stdout.strip()
    with tempfile.TemporaryDirectory() as d:
        for cmd in (
            ["git", "clone", "-q", str(ROOT), d],
            ["uv", "sync", "-q"],
            ["uv", "run", "pytest", "-q", "-m", "not slow"],
        ):
            r = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, check=False)
            if r.returncode:
                return (
                    False,
                    f"clone of {head}: `{' '.join(cmd)}` exit {r.returncode}: {(r.stdout + r.stderr).strip()[-200:]}",
                )
        tail = r.stdout.strip().splitlines()[-1]
    return True, f"clone of committed HEAD {head}: uv sync ok, fast tests: {tail}"


CHECKS = {
    0: {"0.1": c01, "0.2": c02, "0.3": c03, "0.4": c04, "0.5": c05, "0.6": c06, "0.7": c07},
    1: {"1.1": c11, "1.2": c12, "1.3": c13, "1.4": c14, "1.5": c15, "1.6": c16},
    2: {
        "2.1": c21,
        "2.2": c22,
        "2.3": c23,
        "2.4": c24,
        "2.5": c25,
        "2.6": c26,
        "2.7": c27,
        "2.8": c28,
    },
    3: {"3.1": c31, "3.2": c32, "3.3": c33, "3.4": c34, "3.5": c35},
    4: {"4.1": c41, "4.2": c42, "4.3": c43, "4.4": c44},
    5: {"5.1": c51, "5.2": c52, "5.3": c53, "5.4": c54},
    6: {"6.1": c61, "6.2": c62, "6.3": c63, "6.4": c64},
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", type=int)
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    results = {}
    for cid, fn in CHECKS[a.phase].items():
        try:
            ok, val = fn()
        except Exception as e:  # noqa: BLE001 - a crashing check is a failing check
            ok, val = False, f"error: {e!r}"
        results[cid] = ok
        print(f"{cid} {'PASS' if ok else 'FAIL'} | {val}")
    allok = all(results.values())
    print("PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
