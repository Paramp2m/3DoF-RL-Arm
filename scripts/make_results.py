"""Generate docs/results.md (+ learning-curve PNGs) from the eval.json files of the Phase 4 experiments.

Every number comes from run folders listed by configs/experiments/*.yaml; nothing is typed by hand.
    uv run python scripts/make_results.py            # write docs/results.md and docs/media/<exp>_curves.png
    uv run python scripts/make_results.py --check    # exit 1 if docs/results.md would change (writes nothing)
"""

import argparse
import json
import sys

import numpy as np
import yaml

from rlarm.common import ROOT, arm_runs, experiments

OUT = ROOT / "docs/results.md"


def _ms(vals, scale=1.0, f=".1f") -> str:
    """mean ± sample std over seeds (just the value for one seed); n/a if any is None."""
    if any(v is None for v in vals):
        return "n/a"
    v = np.asarray(vals, float) * scale
    return format(v.mean(), f) + (f" ± {v.std(ddof=1):{f}}" if len(v) > 1 else "")


def _join(per_arm: dict) -> str:
    """One value if all arms agree, else 'arm: value' per arm."""
    vals = set(per_arm.values())
    return vals.pop() if len(vals) == 1 else "; ".join(f"{k}: {v}" for k, v in per_arm.items())


def section(stem: str, exp: dict) -> tuple[str, dict]:
    """Markdown for one experiment plus its learning curves {arm: [(steps, success per seed)]}."""
    rows, seeds, budget, curves = [], {}, {}, {}
    for label, arm in exp["arms"].items():
        runs = arm_runs(arm)
        ev = [json.loads((r / "eval.json").read_text()) for r in runs]
        cfgs = [yaml.safe_load((r / "config.yaml").read_text()) for r in runs]
        steps = [json.loads((r / "summary.json").read_text())["steps"] for r in runs]
        seeds[label] = ", ".join(str(c["seed"]) for c in cfgs)
        budget[label] = ", ".join(sorted({f"{s:,}" for s in steps}))
        g = lambda k, ev=ev: [e[k] for e in ev]
        rows.append(
            f"| {label} | {_ms(g('success_rate'), 100)} | "
            + ", ".join(f"{v:.0%}" for v in g("success_rate"))
            + f" | {_ms(g('final_dist_mm_median'))} | {_ms(g('steps_to_success_mean'))} | "
            f"{_ms(g('smoothness_mean'), f='.3f')} | {_ms(g('energy_mean'), f='.2f')} |"
        )
        if all((r / "val.json").exists() for r in runs):
            hist = [json.loads((r / "val.json").read_text())["history"] for r in runs]
            n = min(len(h) for h in hist)
            curves[label] = (
                [h["step"] for h in hist[0][:n]],
                np.array([[h["success"] for h in hs[:n]] for hs in hist]),
            )
    md = [
        f"## {exp['title']}",
        "",
        exp["question"],
        "",
        f"Seeds: {_join(seeds)}  ",
        f"Step budget: {_join(budget)} env steps per run  ",
        "Configs: " + ", ".join(f"{k} `{a['config']}`" for k, a in exp["arms"].items()),
        "",
        (
            "| arm | success % (mean ± std over seeds) | success per seed | median final mm | "
            "steps to success | smoothness ‖Δa‖ | energy J |"
        ),
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]
    if curves and len(curves) == len(exp["arms"]):
        md += [f"![learning curves](media/{stem}_curves.png)", ""]
    return "\n".join(md), curves


def tasks_section() -> str:
    """Markdown for the final policy of each task (docs/best_runs.yaml phase3 + phase5)."""
    best = yaml.safe_load((ROOT / "docs/best_runs.yaml").read_text())
    rows, seeds, budget, configs = [], {}, {}, {}
    for task, dirs in {**best["phase3"], **best["phase5"]}.items():
        runs = [ROOT / d for d in dirs]
        ev = [json.loads((r / "eval.json").read_text()) for r in runs]
        cfgs = [yaml.safe_load((r / "config.yaml").read_text()) for r in runs]
        steps = [json.loads((r / "summary.json").read_text())["steps"] for r in runs]
        seeds[task] = ", ".join(str(c["seed"]) for c in cfgs)
        budget[task] = ", ".join(sorted({f"{s:,}" for s in steps}))
        configs[task] = f"`configs/{cfgs[0]['name']}.yaml`"
        g = lambda k, ev=ev: [e[k] for e in ev]
        extra = {
            "track": f"tracking error {_ms(g('track_err_mm_mean'))} mm",
            "obstacle": f"collisions {_ms(g('collision_rate'), 100)} %",
        }.get(task, "")
        rows.append(
            f"| {task} | {_ms(g('success_rate'), 100)} | "
            + ", ".join(f"{v:.0%}" for v in g("success_rate"))
            + f" | {_ms(g('final_dist_mm_median'))} | {extra} |"
        )
    return "\n".join(
        [
            "## Final task policies (Phases 3 and 5)",
            "",
            "The policy kept for each task (docs/best_runs.yaml); the phase reports explain each.",
            "",
            f"Seeds: {_join(seeds)}  ",
            f"Step budget: {_join(budget)} env steps per run  ",
            "Configs: " + ", ".join(f"{k} {v}" for k, v in configs.items()),
            "",
            "| task | success % (mean ± std over seeds) | success per seed | median final mm | task metric |",
            "|---|---|---|---|---|",
            *rows,
            "",
        ]
    )


def plot(stem: str, title: str, curves: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=100)
    for label, (x, y) in curves.items():
        m, s = y.mean(0) * 100, y.std(0, ddof=1) * 100
        ax.plot(x, m, label=label)
        ax.fill_between(x, m - s, m + s, alpha=0.2)
    ax.set(
        title=title,
        xlabel="env steps",
        ylabel="validation success % (mean ± std, seeds)",
        ylim=(0, 100),
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / f"docs/media/{stem}_curves.png", metadata={"Software": None})
    plt.close(fig)


def build() -> tuple[str, dict]:
    parts = [
        "# Results",
        "",
        "Generated by `scripts/make_results.py` from each run's eval.json; do not edit by hand.",
        (
            "Eval: 100 episodes on fixed goals (seeds 10000+), deterministic policy, model.zip = "
            "best checkpoint on separate validation goals (seeds 50000+). Success = tip within 2 cm "
            "at the last step. Smoothness = mean ‖a_t − a_(t−1)‖. Energy (torque modes) = Σ|τ·q̇|·dt "
            "per episode. Learning curves: validation success every 20k steps."
        ),
        "",
    ]
    figs = {}
    for stem, exp in experiments().items():
        md, curves = section(stem, exp)
        parts.append(md)
        if curves and len(curves) == len(exp["arms"]):
            figs[stem] = (exp["title"], curves)
    parts.append(tasks_section())
    return "\n".join(parts), figs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    text, figs = build()
    if a.check:
        same = OUT.exists() and OUT.read_text() == text
        print("results.md up to date" if same else "results.md differs from generated output")
        return 0 if same else 1
    OUT.write_text(text)
    for stem, (title, curves) in figs.items():
        plot(stem, title, curves)
    print(f"wrote {OUT.relative_to(ROOT)} + {len(figs)} curve plots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
