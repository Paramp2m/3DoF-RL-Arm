"""Run babysitter: every INTERVAL s, apply each running job's val.early_stop rule (rlarm.common.early_stop_reason)
to its val.json, plus a batch rule: past min_frac of its budget, a run whose best val score trails the best
run of another config in the batch (same glob) by more than BEHIND is stopped ("behind"). `touch STOP` when a rule fires (the run then finalises: val-best saved, eval written). Logs each
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
BEHIND = 0.10  # val-score gap to the batch leader that marks a run as not worth its threads
while True:
    live, rows = 0, []
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
        name = yaml.safe_load((run / "config.yaml").read_text())["name"]
        rows.append((run, sm, es, total, scores, name))
    for run, sm, es, total, scores, name in rows:
        # leader among *other* configs: seeds of one config are never culled against each other (a 3-seed
        # final needs every seed, and dropping the weak one would cherry-pick)
        lead = max((max(r[4]) for r in rows if r[4] and r[5] != name), default=float("inf"))
        why = early_stop_reason(scores, sm["steps"], total, es) if es and scores else None
        frac = (es or {}).get("min_frac", 0.3)
        if (
            not why
            and scores
            and sm["steps"] >= frac * total
            and lead != float("inf")
            and max(scores) < lead - BEHIND
        ):
            why = "behind"
        line = f"{datetime.now():%H:%M} {run.name} {sm['steps'] // 1000}k last5={[round(x, 2) for x in scores[-5:]]}"
        if why and not (run / "STOP").exists():
            (run / "STOP").touch()
            line += f" -> STOP ({why})"
        with log.open("a") as f:
            f.write(line + "\n")
    if not live:
        break
    time.sleep(interval)
