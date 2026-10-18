import json
import shutil
import subprocess
from datetime import date

import pytest
import yaml

from rlarm.common import ROOT


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
