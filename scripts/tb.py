"""Open TensorBoard on a tidy view of runs/: friendly run names, no dry runs, grouped charts.

Builds runs/_tb/ (symlinks only, safe to delete), writes a "Custom Scalars" chart layout into it,
then starts TensorBoard. Run names become `<config>/s<seed>`.
    uv run python scripts/tb.py                    # every real run
    uv run python scripts/tb.py --filter 'e1_|e2_'  # only runs whose name matches the regex
    uv run python scripts/tb.py --since 2026-10-19  # only runs started on/after a date
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS, VIEW = ROOT / "runs", ROOT / "runs" / "_tb"
NAME = re.compile(r"^(\d{4}-\d\d-\d\d)_(.+)_s(\d+)$")

# Custom Scalars tab: {section: {chart title: ["Multiline", [tag regexes]]}}
LAYOUT = {
    "1 Headline": {
        "Success rate (training)": ["Multiline", [r"rollout/success_rate"]],
        "Success rate (validation goals)": ["Multiline", [r"val/success"]],
        "Median final distance, mm (validation)": ["Multiline", [r"val/median_mm"]],
        "Episode reward": ["Multiline", [r"rollout/ep_rew_mean"]],
    },
    "2 Reward terms": {"All reward terms": ["Multiline", [r"reward/.*"]]},
    "3 Learning internals": {
        "Critic loss (SAC) / value loss (PPO)": [
            "Multiline",
            [r"train/critic_loss", r"train/value_loss"],
        ],
        "Actor loss (SAC)": ["Multiline", [r"train/actor_loss"]],
        "Entropy coefficient (SAC)": ["Multiline", [r"train/ent_coef"]],
        "PPO health": [
            "Multiline",
            [r"train/approx_kl", r"train/clip_fraction", r"train/explained_variance"],
        ],
    },
    "4 Speed": {"Steps per second": ["Multiline", [r"time/fps"]]},
}


def build_view(pattern: str, since: str) -> int:
    """Recreate runs/_tb/ with one symlink per run; return how many runs it holds."""
    shutil.rmtree(VIEW, ignore_errors=True)
    VIEW.mkdir()
    n = 0
    for run in sorted(RUNS.iterdir()):
        m = NAME.match(run.name)
        if not m or not (run / "tb").is_dir() or not re.search(pattern, run.name) or m[1] < since:
            continue
        link = VIEW / m[2] / f"s{m[3]}"
        if link.exists() or link.is_symlink():  # same config run on two dates: keep both
            link = VIEW / f"{m[2]}_{m[1]}" / f"s{m[3]}"
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(run / "tb")
        n += 1
    return n


def write_layout() -> None:
    from torch.utils.tensorboard import SummaryWriter

    w = SummaryWriter(VIEW / "_layout")
    w.add_custom_scalars(LAYOUT)
    w.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--filter", default="", help="regex on the run folder name")
    ap.add_argument("--since", default="0000-00-00", help="YYYY-MM-DD")
    ap.add_argument("--port", default="6006")
    a = ap.parse_args()
    n = build_view(a.filter, a.since)
    if not n:
        sys.exit("no runs match")
    write_layout()
    print(f"{n} runs -> http://localhost:{a.port}  (Custom Scalars tab = grouped charts)")
    subprocess.run(
        ["tensorboard", "--logdir", str(VIEW), "--port", a.port, "--reload_interval", "15"],
        check=False,
    )


if __name__ == "__main__":
    main()
