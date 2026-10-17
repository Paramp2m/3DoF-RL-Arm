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
MODES = ("delta_pos", "abs_pos", "torque", "torque_gravcomp")


def _reach(mode: str = "delta_pos", **arm):
    from rlarm.common import load_config, make_env

    cfg = load_config(ROOT / REACH_CFG)
    cfg["env"]["action_mode"] = mode
    cfg["arm"].update(arm)
    return make_env(cfg)


def c21():
    import warnings

    from stable_baselines3.common.env_checker import check_env

    out = []
    for mode in MODES:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_env(_reach(mode))  # raises on failure
        out.append(f"{mode} ok ({len(w)} warn)")
    return True, ", ".join(out)


def c22():
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
        f"success={ev['success_rate']:.0%}, final dist median={ev['final_dist_mm_median']:.0f} mm"
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
