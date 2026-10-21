import json
import pickle
import shutil
import subprocess
import sys
import time
from datetime import date

import numpy as np
import pytest
import torch as th
import yaml

from rlarm.common import ROOT, algo_class, algo_kwargs, load_config, make_env
from rlarm.expert import PlannerExpert

sys.path.insert(0, str(ROOT / "scripts"))
from imitate import rollout, stack
from train import init_policy


def test_real_run_never_overwrites_existing_folder(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/reacher_sac.yaml").read_text())
    cfg["name"] = "test_no_overwrite"
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    run = ROOT / "runs" / f"{date.today()}_test_no_overwrite_s{cfg['seed']}"
    run.mkdir(parents=True)
    (run / "keep.txt").write_text("x")
    try:
        r = subprocess.run(
            ["uv", "run", "python", "scripts/train.py", "--config", str(tmp_path / "c.yaml")],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0 and "FileExistsError" in r.stderr
        assert (run / "keep.txt").exists()
    finally:
        shutil.rmtree(run)


@pytest.mark.slow
def test_dry_run_writes_summary_and_checkpoint():
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/train.py",
            "--config",
            "configs/reacher_sac.yaml",
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    run = max((ROOT / "runs/_dry").glob("*_reacher_sac_s0_dry"), key=lambda p: p.stat().st_mtime)
    assert json.loads((run / "summary.json").read_text())["status"] == "done"
    assert (run / "checkpoint.zip").exists() and not (run / "checkpoint_tmp.zip").exists()


@pytest.mark.slow
def test_dry_run_val_best_checkpoint():
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/train.py",
        "--config",
        "configs/pilot/base.yaml",
        "--dry-run",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    run = max((ROOT / "runs/_dry").glob("*_pilot_base_s0_dry"), key=lambda p: p.stat().st_mtime)
    val = json.loads((run / "val.json").read_text())
    best = max(val["history"], key=lambda h: (h["success"], -h["median_mm"]))
    assert val["best_step"] == best["step"]
    assert (run / "model.zip").exists() and (run / "final.zip").exists()
    assert not (run / "model_best.zip").exists()
    assert (
        f"@ {best['step']} steps"
        in json.loads((run / "summary.json").read_text())["best_checkpoint"]
    )


@pytest.mark.slow
def test_sigterm_marks_run_crashed(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/reacher_sac.yaml").read_text())
    cfg["name"] = "test_sigterm"
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    run = ROOT / "runs/_dry" / f"{date.today()}_test_sigterm_s{cfg['seed']}_dry"
    cmd = [sys.executable, "scripts/train.py", "--config", str(tmp_path / "c.yaml"), "--dry-run"]
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(600):  # summary.json is written after the SIGTERM handler is installed
            if (run / "summary.json").exists():
                break
            time.sleep(0.1)
        p.terminate()
        assert p.wait(60) != 0
        assert json.loads((run / "summary.json").read_text())["status"] == "crashed"
    finally:
        p.kill()
        shutil.rmtree(run, ignore_errors=True)


@pytest.mark.slow
def test_init_policy_loads_imitate_output(tmp_path):
    """train.init_policy reads what imitate.py writes: cloned weights, demos and expert episodes."""
    cfg = load_config(ROOT / "configs/obstacle_h2.yaml")
    cfg["algo"]["kwargs"]["buffer_size"] = 2000
    env = make_env(cfg)

    def new(kw):
        return algo_class(cfg)(cfg["algo"]["policy"], env, seed=0, device="cpu", **kw)

    bc_keys = ("bc_weight", "bc_decay", "actor_freeze")  # imitate.py builds its model without them
    teacher = new({k: v for k, v in algo_kwargs(cfg).items() if k not in bc_keys})
    expert = PlannerExpert(env, **cfg["expert"])
    xs, ys, tr, _ = rollout(env, expert, None, 1.0, 0, np.random.default_rng(0), keep=True)
    teacher.save(tmp_path / "bc.zip")
    (tmp_path / "expert_eps.pkl").write_bytes(pickle.dumps([tr]))
    obs = {f"obs_{k}": v for k, v in stack(xs).items()}
    np.savez_compressed(tmp_path / "demos.npz", actions=np.asarray(ys, np.float32), **obs)

    student = new(algo_kwargs(cfg))
    init_policy(student, {**cfg, "init_from": str(tmp_path / "bc.zip")})
    pairs = zip(teacher.policy.state_dict().values(), student.policy.state_dict().values())
    assert all(th.equal(a, b) for a, b in pairs)
    assert student.replay_buffer.size() == len(tr) and len(student._demo_act) == len(ys)
