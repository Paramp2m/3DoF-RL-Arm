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
    steps = json.loads((run / "summary.json").read_text())["steps"]
    ok = ev["return_mean"] >= RETURN_TARGET and ev["episodes"] >= 100 and steps <= STEP_CAP
    return ok, f"return={ev['return_mean']:.2f} over {ev['episodes']} eps, steps={steps}"


def c06():
    run = best_run("phase0.reacher")
    if not run:
        return False, "no run"
    need = ["config.yaml", "git_hash.txt", "model.zip", "eval.json", "summary.json", "video.gif"]
    missing = [f for f in need if not (run / f).exists()]
    if not list((run / "tb").glob("**/events.out.tfevents*")):
        missing.append("tb events")
    return not missing, f"missing={missing or 'none'}"


def c07():
    t = sh("uv", "run", "pytest", "-q")
    r = sh("uv", "run", "ruff", "check", ".")
    last = t.stdout.strip().splitlines()[-1] if t.stdout.strip() else "no output"
    return t.returncode == 0 and r.returncode == 0, f"pytest: {last}; ruff rc={r.returncode}"


CHECKS = {0: {"0.1": c01, "0.2": c02, "0.3": c03, "0.4": c04, "0.5": c05, "0.6": c06, "0.7": c07}}


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
