"""Re-eval every arm-task run with eval.json; check the old metrics are unchanged."""

import json
import subprocess
from pathlib import Path

import yaml

root = Path(".")
bad = 0
for ev in sorted(root.glob("runs/*/eval.json")):
    run = ev.parent
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    if "task" not in cfg or not (run / "model.zip").exists() and cfg.get("agent") != "ik":
        continue
    old = json.loads(ev.read_text())
    if "track_err_mm_mean" in old:
        continue
    subprocess.run(
        ["uv", "run", "python", "scripts/eval.py", "--run", str(run), "--no-video"], check=True
    )
    new = json.loads(ev.read_text())
    diff = {k: (v, new.get(k)) for k, v in old.items() if new.get(k) != v}
    print(run.name, "DIFF" if diff else "same", diff, flush=True)
    bad += bool(diff)
print("DONE, runs with changed old metrics:", bad)
