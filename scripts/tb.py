"""Open TensorBoard as separate browser tabs, one per group of runs.

Tabs (a tab only appears if it has runs):
    Best runs        the runs listed in docs/best_runs.yaml (the ones that count)
    E1 / E2 / E3     Phase 4 experiments: configs named e1_*, e2_*, e3_*
    Archive          everything else (old sweeps, pilots), only with --archive
Dry runs are never shown. Each tab is its own TensorBoard, so its sidebar holds only that group.
Views live in runs/_tb/ (symlinks only, safe to delete).
    uv run python scripts/tb.py [--archive] [--light] [--smoothing 0.6] [--no-open]
"""

import argparse
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS, VIEW = ROOT / "runs", ROOT / "runs" / "_tb"
NAME = re.compile(r"^(\d{4}-\d\d-\d\d)_(.+)_s(\d+)$")
EXPERIMENTS = {"1": "E1 Action modes", "2": "E2 SAC vs PPO", "3": "E3 HER vs shaped"}

# Custom Scalars tab: {section: {chart title: ["Multiline", [tag regexes]]}}
LAYOUT = {
    "1 Headline": {
        "Success rate, training (higher is better)": ["Multiline", [r"rollout/success_rate"]],
        "Success rate, validation goals (higher is better)": ["Multiline", [r"val/success"]],
        "Median final distance, mm (lower is better)": ["Multiline", [r"val/median_mm"]],
        "Episode reward (higher is better)": ["Multiline", [r"rollout/ep_rew_mean"]],
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


def best_runs() -> dict[str, str]:
    """Run folder name -> '<phase>/<task>' for every run in docs/best_runs.yaml."""
    out = {}
    for phase, tasks in yaml.safe_load((ROOT / "docs" / "best_runs.yaml").read_text()).items():
        for task, paths in tasks.items():
            for path in [paths] if isinstance(paths, str) else paths:
                out[Path(path).name] = f"{phase}/{task}"
    return out


def groups(archive: bool) -> dict[str, dict[str, Path]]:
    """Tab title -> {display name: tb folder}. Dry runs and runs without logs are skipped."""
    best, out = best_runs(), {}
    for run in sorted(RUNS.iterdir()):
        m = NAME.match(run.name)
        if not m or not (run / "tb").is_dir():
            continue
        _, config, seed = m.groups()
        exp = re.match(r"e([1-9])_(.+)", config)
        if run.name in best:
            tab, name = "Best runs", f"{best[run.name]}/{config}/s{seed}"
        elif exp and exp[1] in EXPERIMENTS:
            tab, name = EXPERIMENTS[exp[1]], f"{exp[2]}/s{seed}"
        elif archive:
            tab, name = "Archive", f"{config}/s{seed}"
        else:
            continue
        out.setdefault(tab, {})[name] = run / "tb"
    return out


def build_view(tab: str, runs: dict[str, Path]) -> Path:
    """Recreate runs/_tb/<tab>/ with one symlink per run and the chart layout; return its path."""
    view = VIEW / tab
    shutil.rmtree(view, ignore_errors=True)
    for name, tb in runs.items():
        link = view / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(tb)
    write_layout(view)
    return view


def write_layout(view: Path) -> None:
    from torch.utils.tensorboard import SummaryWriter

    w = SummaryWriter(view / "_layout")
    w.add_custom_scalars(LAYOUT)
    w.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--archive", action="store_true", help="add an Archive tab with all other runs")
    ap.add_argument(
        "--port", type=int, default=6006, help="first port; later tabs use the next ones"
    )
    ap.add_argument("--light", action="store_true", help="light theme (default: dark)")
    ap.add_argument("--smoothing", default="0.85", help="line smoothing 0-0.99 (default 0.85)")
    ap.add_argument("--no-open", action="store_true", help="don't open the browser")
    a = ap.parse_args()
    found = groups(a.archive)
    if not found:
        sys.exit("no runs to show (use --archive for old runs)")
    shutil.rmtree(VIEW, ignore_errors=True)
    procs = []
    for i, tab in enumerate(sorted(found, key=lambda t: (t == "Archive", t))):
        runs = found[tab]
        port = a.port + i
        view = build_view(tab, runs)
        print(f"{tab}: {len(runs)} runs -> http://localhost:{port}")
        procs.append(
            subprocess.Popen(
                ["tensorboard", "--logdir", str(view), "--port", str(port)]
                + ["--window_title", tab, "--reload_interval", "15"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
        if not a.no_open:  # dark theme + smoothing + Custom Scalars tab preselected via URL
            url = f"http://localhost:{port}/?darkMode={str(not a.light).lower()}"
            url += f"#custom_scalars&_smoothingWeight={a.smoothing}"
            threading.Timer(4 + i, webbrowser.open_new_tab, [url]).start()
    print("Ctrl+C to stop all tabs")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
