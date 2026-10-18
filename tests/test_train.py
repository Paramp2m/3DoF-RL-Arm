import json
import subprocess

import pytest

from rlarm.common import ROOT


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
