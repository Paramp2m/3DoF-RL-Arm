"""Run babysitter: every INTERVAL s, apply each running job's val.early_stop rule (rlarm.common.early_stop_reason)
to its val.json and `touch STOP` when it fires (the run then finalises: val-best saved, eval written). Logs each
decision to runs/_logs/babysit.log and exits when no run is training. Usage: babysit.py [interval_s=300] [glob]."""

import json
import sys
import time
from datetime import datetime

import yaml

from rlarm.common import ROOT, early_stop_reason

interval = int(sys.argv[1]) if len(sys.argv) > 1 else 300
pattern = sys.argv[2] if len(sys.argv) > 2 else "20*"
log = ROOT / "runs/_logs/babysit.log"
while True:
    live = 0
    for run in sorted((ROOT / "runs").glob(pattern)):
        try:
            sm = json.loads((run / "summary.json").read_text())
            if sm["status"] != "running":
                continue
            live += 1
            es = (yaml.safe_load((run / "config.yaml").read_text()).get("val") or {}).get(
                "early_stop"
            )
            hist = (
                json.loads((run / "val.json").read_text())["history"]
                if (run / "val.json").exists()
                else []
            )
            total = yaml.safe_load((run / "config.yaml").read_text())["total_steps"]
        except (OSError, ValueError, KeyError):
            continue
        scores = [h["success"] for h in hist]
        why = early_stop_reason(scores, sm["steps"], total, es) if es and hist else None
        line = f"{datetime.now():%H:%M} {run.name} {sm['steps'] // 1000}k last5={[round(x, 2) for x in scores[-5:]]}"
        if why and not (run / "STOP").exists():
            (run / "STOP").touch()
            line += f" -> STOP ({why})"
        with log.open("a") as f:
            f.write(line + "\n")
    if not live:
        break
    time.sleep(interval)
