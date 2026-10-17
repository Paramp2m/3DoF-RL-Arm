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
    run = ROOT / "runs/_dry" / next(p.name for p in (ROOT / "runs/_dry").iterdir())
    assert json.loads((run / "summary.json").read_text())["status"] == "done"
    assert (run / "checkpoint.zip").exists() and not (run / "checkpoint_tmp.zip").exists()
