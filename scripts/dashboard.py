"""Local, read-only dashboard: live training, experiments, phases, every run, and all docs in one page.

    uv run python scripts/dashboard.py [--port 8765] [--no-open]

Serves http://127.0.0.1:<port> until Ctrl+C or the terminal closes. Runs at the lowest CPU priority
and only re-reads files whose size/mtime changed, so it doesn't slow training. It never trains,
evaluates or writes project files; its buttons only open viewers and run the read-only phase gate.
Imports no torch/mujoco (rlarm.common pulls both in), so the run-folder rules are restated here.
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS, STATIC = ROOT / "runs", Path(__file__).resolve().parent / "dashboard"
NAME = re.compile(r"^(\d{4}-\d\d-\d\d)_(.+)_s(\d+)$")
MEDIA = {
    ".gif": "image/gif",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}
HZ = 25  # control rate (SPEC §4: frame_skip 20 x 2 ms), turns steps into seconds

# Plain-English text the project files don't hold. Phase summary: (what it builds, what you do, typical time).
PHASES = {
    0: (
        "SAC learns Gymnasium's 2-joint Reacher, to prove the train → eval → GIF → gate pipeline before our arm exists",
        "Watch the Reacher GIF: the fingertip should go straight to the dot and stop",
        "~1 h",
    ),
    1: (
        "The 3-joint arm in MuJoCo, plus the maths from joint angles to tip position (FK) and back (IK). No training",
        "Open the arm viewer, move the sliders, check nothing clips",
        "–",
    ),
    2: (
        "The reach task, 4 ways to drive the motors, a scripted IK agent and a random agent. No training",
        "Compare IK vs random: proves the task is solvable and not trivial",
        "–",
    ),
    3: (
        "SAC policies for reach (go to a target) and hold (go there and stay still), 3 seeds each",
        "Judge the GIFs: smooth? does the arm settle or jitter?",
        "~5 h",
    ),
    4: (
        "Experiments E1–E4: which choices matter? Results go to docs/results.md",
        "Read the Experiments page once it's done",
        "~6 h",
    ),
    5: (
        "Harder tasks: track a moving target, avoid an obstacle, push a cube",
        "Judge 3 GIFs; decide what to change if push stalls after 3 tries",
        "10 h+",
    ),
    6: (
        "A live demo (drag the target, the policy follows) and the README",
        "Run the demo and say whether it feels right",
        "~1 h",
    ),
    7: (
        "Optional: randomised physics and a second robot (UR5e)",
        "Decide whether to do it at all",
        "–",
    ),
}
STAGES = {
    "build": "Writing this phase's checks and code. Short messages in the terminal.",
    "train": "Training in the background. Claude is idle and wakes up by itself when the runs finish. Keep the laptop awake and plugged in.",
    "gate": "Checking results against the phase gate, retraining if a check fails, then a review.",
    "report": "Writing the report, copying media, committing.",
}
EXPERIMENTS = {  # manifest stem -> (plain question, why it matters, plain name per arm)
    "e1_action_modes": (
        "How should the network talk to the motors?",
        "The same learner, reward and budget, only the meaning of the network's output changes.",
        {
            "delta_pos": "Nudge each joint a bit",
            "abs_pos": "Jump to a joint angle",
            "torque": "Push with raw force",
            "torque_gravcomp": "Raw force, gravity cancelled",
        },
    ),
    "e2_algorithms": (
        "Which learning algorithm learns faster?",
        "SAC reuses old experience from a big memory; PPO learns only from fresh experience. Same task and budget.",
        {"SAC": "SAC (reuses experience)", "PPO": "PPO (fresh experience only)"},
    ),
    "e3_her_sparse": (
        "Can it learn from just 'hit / missed'?",
        "Shaped reward says 'warmer / colder' every step. Sparse reward only says 'success'; HER replays misses as if the arm had aimed where it ended up.",
        {"SAC + HER, sparse": "Only 'success' + HER", "SAC, shaped": "'Warmer / colder' reward"},
    ),
    "e4_rl_vs_ik": (
        "Does the trained network beat plain maths?",
        "The IK controller computes exact joint angles, so it is the ceiling. How close does the best learned policy get?",
        {
            "RL (reach_final, SAC + HER)": "Best trained network (Phase 3)",
            "IK controller": "Exact maths (IK)",
        },
    ),
}
SETTINGS = {  # flattened config key -> plain name; unknown keys show as the raw key
    "env.action_mode": "How the network drives the motors",
    "env.max_delta": "Largest joint nudge per step (rad)",
    "env.episode_steps": "Steps per attempt",
    "env.obs_err": "Sees the error vector directly",
    "env.reward.w_distance": "Weight: distance penalty",
    "env.reward.w_precision": "Weight: precision bonus",
    "env.reward.w_success": "Weight: success bonus",
    "env.reward.w_limit": "Weight: joint-limit penalty",
    "env.reward.w_settle": "Weight: stay-still penalty",
    "env.reward.limit_margin": "Joint-limit margin (rad)",
    "env.reward.sigma": "Precision bonus width (m)",
    "algo.name": "Learning algorithm",
    "algo.n_envs": "Parallel simulators",
    "algo.kwargs.learning_rate": "Learning rate",
    "algo.kwargs.batch_size": "Batch size",
    "algo.kwargs.buffer_size": "Replay memory size",
    "algo.kwargs.gamma": "Discount γ (how far ahead it plans)",
    "algo.kwargs.replay_buffer_class": "Replay memory type (HER = relabel misses)",
    "algo.kwargs.ent_coef": "Exploration bonus",
    "total_steps": "Training steps",
    "torch_threads": "CPU threads",
    "agent": "Agent",
    "task": "Task",
}
SKIP_DIFF = ("seed", "name", "val.", "dry_", "gif_episodes", "eval_episodes")
DOCS = [  # (group, title, path); a missing file is left out
    ("Start here", "Dashboard guide", "User Ref/GUIDE.md"),
    ("Start here", "First-time setup", "User Ref/SETUP.md"),
    ("Results", "Experiment results", "docs/results.md"),
    ("Design (yours, locked)", "Spec", "docs/SPEC.md"),
    ("Design (yours, locked)", "Roadmap", "docs/ROADMAP.md"),
    ("Project rules", "CLAUDE.md (instructions Claude follows)", "CLAUDE.md"),
]


# ---------- cached file reads ----------
_cache: dict[Path, tuple] = {}


def cached(path: Path, parse):
    """parse(path), re-read only when size/mtime change; keeps the last good value if a write is half-done."""
    try:
        st = path.stat()
    except OSError:
        return None
    key, hit = (st.st_mtime_ns, st.st_size), _cache.get(path)
    if hit and hit[0] == key:
        return hit[1]
    try:
        val = parse(path)
    except (OSError, ValueError, yaml.YAMLError):
        return hit[1] if hit else None
    _cache[path] = (key, val)
    return val


def jread(p: Path):
    return cached(p, lambda p: json.loads(p.read_text()))


def yread(p: Path):
    return cached(p, lambda p: yaml.safe_load(p.read_text()))


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, check=False, cwd=ROOT, capture_output=True, text=True).stdout.strip()


_git: dict = {"t": 0.0}


def git() -> dict:
    """Tags, uncommitted files and recent commits; refreshed every 10 s (git is the slowest call here)."""
    if time.time() - _git["t"] > 10:
        dirty = [
            ln[3:]
            for ln in sh("git", "status", "--porcelain").splitlines()
            if "User Ref/" not in ln
        ]
        log = [
            ln.split("\x1f")
            for ln in sh("git", "log", "-30", "--format=%h\x1f%cs\x1f%s").splitlines()
        ]
        tags = set(sh("git", "tag").split())
        starts = {
            t: int(sh("git", "log", "-1", "--format=%ct", t) or 0)
            for t in tags
            if t.endswith("-start")
        }
        if tags != _git.get("tags"):  # commits per finished phase only change when a tag is added
            _git["phase_log"] = {
                int(t[6:-5]): [
                    ln.split("\x1f")
                    for ln in sh(
                        "git", "log", "--format=%h\x1f%cs\x1f%s", f"phase-{t[6:-5]}-start..{t}"
                    ).splitlines()
                ]
                for t in tags
                if re.fullmatch(r"phase-\d+-done", t)
            }
        _git.update(t=time.time(), tags=tags, starts=starts, dirty=dirty, log=log)
    return _git


# ---------- training processes and machine ----------
def training() -> set[tuple[str, int | None]]:
    """(config stem, seed) of every live, non-dry train.py, from /proc (no subprocess)."""
    out = set()
    for c in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            a = [x.decode(errors="replace") for x in c.read_bytes().split(b"\0")]
        except OSError:
            continue
        if (
            any(x.endswith("scripts/train.py") for x in a)
            and "--config" in a
            and "--dry-run" not in a
        ):
            i = a.index("--config") + 1
            seed = int(a[a.index("--seed") + 1]) if "--seed" in a else None
            out.add((Path(a[i]).stem if i < len(a) else "?", seed))
    return out


_cpu = {"prev": None, "pct": 0.0}


def machine(n_train: int) -> dict:
    """CPU % since the last call, memory, battery: what decides whether training is being slowed down."""
    f = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    idle, total = f[3] + f[4], sum(f)
    if _cpu["prev"] and total > _cpu["prev"][1]:
        _cpu["pct"] = 100 * (1 - (idle - _cpu["prev"][0]) / (total - _cpu["prev"][1]))
    _cpu["prev"] = (idle, total)
    mem = dict(
        ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln
    )
    kb = lambda k: int(mem[k].split()[0])
    bat = None
    for b in Path("/sys/class/power_supply").glob("BAT*"):
        try:
            bat = {
                "pct": int((b / "capacity").read_text()),
                "status": (b / "status").read_text().strip(),
            }
        except OSError:
            pass
    return {
        "cpu": round(_cpu["pct"]),
        "threads": os.cpu_count(),
        "mem_used": round((kb("MemTotal") - kb("MemAvailable")) / 2**20, 1),
        "mem_total": round(kb("MemTotal") / 2**20, 1),
        "battery": bat,
        "train_procs": n_train,
    }


# ---------- TensorBoard scalars (incremental) ----------
_tb: dict[str, list] = {}  # run name -> [accumulator, last reload time, scalars at that reload]
_tb_lock = threading.Lock()


def scalars(run: Path, live: bool) -> dict:
    """{tag: [[step, value], ...]} from the run's TensorBoard log. Live runs reload at most every 5 s."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb = run / "tb"
    if not tb.exists():
        return {}
    with _tb_lock:
        ent = _tb.get(run.name)
        if ent is None:
            dirs = sorted({p.parent for p in tb.rglob("events.out.tfevents.*")})
            if not dirs:
                return {}
            ent = _tb[run.name] = [
                EventAccumulator(str(dirs[-1]), size_guidance={"scalars": 0}),
                0.0,
                {},
            ]
        if ent[1] == 0 or (live and time.time() - ent[1] > 5):
            acc = ent[0]
            acc.Reload()  # reads only events appended since the last call
            ent[1] = time.time()
            ent[2] = {t: [[e.step, e.value] for e in acc.Scalars(t)] for t in acc.Tags()["scalars"]}
        return ent[2]


def thin(pts: list, n: int = 300) -> list:
    """At most ~n points, always keeping the last one."""
    k = max(1, len(pts) // n)
    return pts[::k] + ([pts[-1]] if pts and (len(pts) - 1) % k else [])


# ---------- configs, runs ----------
def config_index() -> dict[str, Path]:
    """Config name -> its YAML under configs/ (the file Claude wrote, with its explaining comment)."""
    out = {}
    for p in sorted((ROOT / "configs").rglob("*.yaml")):
        c = yread(p)
        if isinstance(c, dict) and "name" in c:
            out.setdefault(c["name"], p)
    return out


def comment(p: Path | None) -> str:
    """The config's leading `# ...` lines: Claude's one-line reason for the experiment."""
    if p is None:
        return ""
    lines = []
    for ln in p.read_text().splitlines():
        if not ln.startswith("#"):
            break
        lines.append(ln.lstrip("# ").strip())
    return " ".join(lines)


def flat(d, pre: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{pre}{k}"
        if isinstance(v, dict) and k not in ("body_mass",):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def diff(cfg: dict, base: dict) -> list[dict]:
    """Settings that differ from the reference config, with plain names."""
    a, b = flat(base), flat(cfg)
    rows = []
    her = "algo.kwargs.replay_buffer_class"
    for k in sorted(set(a) | set(b)):
        if k.startswith(SKIP_DIFF) or a.get(k) == b.get(k):
            continue
        if k.startswith("algo.kwargs.replay_buffer_kwargs") and a.get(her) != b.get(her):
            continue  # HER's own settings: already covered by the replay-memory-type row
        rows.append(
            {"key": k, "name": SETTINGS.get(k, k), "base": a.get(k, "—"), "this": b.get(k, "—")}
        )
    return rows


def all_runs() -> list[Path]:
    return sorted(
        (p for p in RUNS.iterdir() if p.is_dir() and NAME.match(p.name)), key=lambda p: p.name
    )


def newest(name: str, seed: int) -> Path | None:
    found = sorted(RUNS.glob(f"????-??-??_{name}_s{seed}"))
    return found[-1] if found else None


def roles() -> dict[str, dict]:
    """Run folder name -> where it counts (best_runs.yaml phase/task, experiment arm). Others are archive."""
    out: dict[str, dict] = {}
    best = yread(ROOT / "docs/best_runs.yaml") or {}
    for phase, tasks in best.items():
        for task, dirs in (tasks or {}).items():
            for d in dirs if isinstance(dirs, list) else [dirs]:
                out.setdefault(Path(d).name, {})["best"] = (
                    f"Phase {phase.removeprefix('phase')} · {task}"
                )
    for stem, exp in experiments().items():
        for arm, spec in exp.get("arms", {}).items():
            c = yread(ROOT / spec["config"]) or {}
            for s in spec.get("seeds", []):
                r = newest(c.get("name", "?"), s)
                if r:
                    plain = EXPERIMENTS.get(stem, ("", "", {}))[2].get(arm, arm)
                    out.setdefault(r.name, {}).setdefault("exp", []).append(
                        f"{stem[:2].upper()} · {plain}"
                    )
    return out


def experiments() -> dict[str, dict]:
    return {p.stem: yread(p) or {} for p in sorted((ROOT / "configs/experiments").glob("*.yaml"))}


def run_meta(run: Path, live: set, role: dict) -> dict:
    """Everything the run lists and cards need. Steps of a live run come from its TensorBoard log."""
    date, name, seed = NAME.match(run.name).groups()
    seed = int(seed)
    sm = jread(run / "summary.json") or {}
    cfg = yread(run / "config.yaml") or {}
    total = cfg.get("total_steps") or 0
    alive = (name, seed) in live or ((name, None) in live and cfg.get("seed") == seed)
    status = sm.get("status", "?")
    steps, sps = sm.get("steps") or 0, sm.get("steps_per_s")
    val = jread(run / "val.json") or {}
    hist = val.get("history", [])
    if alive:
        status = "training"
        sc = scalars(run, live=True)
        if sc:
            steps = max(steps, max((v[-1][0] for v in sc.values() if v), default=0))
            fps = sc.get("time/fps")
            sps = fps[-1][1] if fps else sps
    elif status == "running":
        status = (
            "stopped"  # summary says running but no process: crashed, killed or the laptop slept
        )
    ev = jread(run / "eval.json") or {}
    agent = cfg.get("agent")
    return {
        "id": run.name,
        "date": date,
        "config": name,
        "seed": seed,
        "task": cfg.get("task", "reacher" if "reacher" in name else "?"),
        "algo": "IK (no learning)"
        if agent == "ik"
        else (cfg.get("algo") or {}).get("name", "?")
        + (" + HER" if "HerReplayBuffer" in str(cfg.get("algo")) else ""),
        "mode": (cfg.get("env") or {}).get("action_mode"),
        "status": status,
        "learns": agent != "ik",
        "steps": steps,
        "total": total,
        "sps": sps,
        "eta_s": (total - steps) / sps if alive and sps and total > steps else None,
        "wall_s": sm.get("wall_s"),
        "success": ev.get("success_rate", (sm.get("final_eval") or {}).get("success")),
        "miss_mm": ev.get("final_dist_mm_median", (sm.get("final_eval") or {}).get("median_mm")),
        "return": ev.get("mean_return", ev.get("return_mean")),
        "val_last": hist[-1]["success"] if hist else None,
        "val": [[h["step"], h["success"]] for h in hist],
        "best": role.get("best"),
        "exp": role.get("exp", []),
        "gif": (run / "video.gif").exists(),
        "mtime": max(
            (p.stat().st_mtime for p in (run / "summary.json", run / "config.yaml") if p.exists()),
            default=0,
        ),
    }


# ---------- phase / stage ----------
def phases(g: dict, runs: list[dict]) -> dict:
    text = (ROOT / "docs/ROADMAP.md").read_text()
    names = {
        int(n): t.strip() for n, t in re.findall(r"^## Phase (\d+)[^:]*: (.+)$", text, re.MULTILINE)
    }
    tags = g["tags"]
    started = sorted(int(t[6:-6]) for t in tags if re.fullmatch(r"phase-\d+-start", t))
    done = {int(t[6:-5]) for t in tags if re.fullmatch(r"phase-\d+-done", t)}
    cur = started[-1] if started else 0
    in_phase = bool(started) and cur not in done
    report = ROOT / f"docs/reports/phase-{cur}.md"
    t0 = g["starts"].get(f"phase-{cur}-start", 0)
    mine = [r for r in runs if r["mtime"] >= t0] if in_phase else []
    stage = (
        "train"
        if any(r["status"] == "training" for r in runs)
        else "report"
        if report.exists()
        else "gate"
        if mine
        else "build"
    )
    rows = []
    for n, title in sorted(names.items()):
        rep = ROOT / f"docs/reports/phase-{n}.md"
        builds, you, dur = PHASES.get(n, ("", "", ""))
        rows.append(
            {
                "n": n,
                "title": title,
                "state": "done"
                if n in done
                else "current"
                if (in_phase and n == cur)
                else "upcoming",
                "builds": builds,
                "you": you,
                "time": dur,
                "report": rep.relative_to(ROOT).as_posix() if rep.exists() else None,
                "media": sorted(p.name for p in (ROOT / "docs/media").glob(f"phase{n}_*")),
                "commits": [
                    {"hash": h, "date": d, "msg": m} for h, d, m in g["phase_log"].get(n, [])
                ],
            }
        )
    return {
        "current": cur,
        "in_phase": in_phase,
        "stage": stage if in_phase else "idle",
        "stage_text": STAGES.get(stage, "")
        if in_phase
        else "Idle between phases. The next phase starts when you paste its /goal in Claude's terminal.",
        "list": rows,
        "runs": [r["id"] for r in mine],
    }


def decisions() -> list[dict]:
    out = []
    for ln in (ROOT / "docs/decisions.md").read_text().splitlines():
        p = [x.strip() for x in ln.split(" | ")]
        if len(p) >= 3 and re.fullmatch(r"\d{4}-\d\d-\d\d", p[0]):
            out.append({"date": p[0], "phase": p[1], "what": p[2], "why": " | ".join(p[3:])})
    return out


# ---------- API payloads ----------
def state() -> dict:
    live = training()
    role = roles()
    runs = [run_meta(r, live, role.get(r.name, {})) for r in all_runs()]
    g = git()
    ph = phases(g, runs)
    docs = [{"group": grp, "title": t, "path": p} for grp, t, p in DOCS if (ROOT / p).exists()]
    docs += [
        {"group": "Phase reports", "title": f"Phase {r['n']}: {r['title']}", "path": r["report"]}
        for r in ph["list"]
        if r["report"]
    ]
    return {
        "phase": ph,
        "runs": runs,
        "machine": machine(len(live)),
        "dirty": g["dirty"],
        "commits": [{"hash": h, "date": d, "msg": m} for h, d, m in g["log"]],
        "decisions": decisions(),
        "docs": docs,
        "experiments": [
            {
                "id": k,
                "title": v.get("title", k),
                "plain": EXPERIMENTS.get(k, ("", "", {}))[0],
                "arms": [EXPERIMENTS.get(k, ("", "", {}))[2].get(a, a) for a in v.get("arms", {})],
            }
            for k, v in experiments().items()
        ],
        "jobs": {k: {kk: vv for kk, vv in j.items() if kk != "proc"} for k, j in JOBS.items()},
    }


def run_detail(rid: str) -> dict | None:
    run = RUNS / rid
    if not NAME.match(rid) or not run.is_dir():
        return None
    live = training()
    meta = run_meta(run, live, roles().get(rid, {}))
    cfg = yread(run / "config.yaml") or {}
    cfg_src = config_index().get(meta["config"])
    base_name = (
        "e1_delta_pos"
        if meta["exp"] and meta["config"] != "e1_delta_pos"
        else f"{meta['task']}_final"
    )
    base_path = config_index().get(base_name)
    base = yread(base_path) if base_path and base_name != meta["config"] else None
    sc = scalars(run, live=meta["status"] == "training")
    log = RUNS / "_logs" / f"{meta['config']}-s{meta['seed']}.log"
    return {
        **meta,
        "why": comment(cfg_src),
        "eval": jread(run / "eval.json"),
        "summary": jread(run / "summary.json"),
        "base": base_name if base else None,
        "diff": diff(cfg, base) if base else [],
        "config_text": (run / "config.yaml").read_text() if (run / "config.yaml").exists() else "",
        "git_hash": (run / "git_hash.txt").read_text().strip()
        if (run / "git_hash.txt").exists()
        else None,
        "scalars": {k: thin(v) for k, v in sc.items()},
        "log": log.read_text(errors="replace").splitlines()[-40:] if log.exists() else [],
        "log_path": log.relative_to(ROOT).as_posix() if log.exists() else None,
    }


def mean_std(vals: list) -> list | None:
    v = [x for x in vals if x is not None]
    if not v or len(v) < len(vals):
        return None
    m = sum(v) / len(v)
    return [m, (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5]


def experiment(stem: str) -> dict | None:
    exp = experiments().get(stem)
    if not exp:
        return None
    live = training()
    q, why, plain = EXPERIMENTS.get(stem, (exp.get("question", ""), "", {}))
    arms = []
    for label, spec in exp.get("arms", {}).items():
        c = yread(ROOT / spec["config"]) or {}
        seeds = []
        for s in spec.get("seeds", []):
            r = newest(c.get("name", "?"), s)
            if r is None:
                seeds.append({"seed": s, "id": None, "status": "not started"})
                continue
            m = run_meta(r, live, {})
            ev = jread(r / "eval.json") or {}
            seeds.append(
                {
                    **{k: m[k] for k in ("id", "seed", "status", "steps", "total", "val")},
                    "eval": ev if m["status"] == "done" else None,
                }
            )
        done = [s["eval"] for s in seeds if s.get("eval")]
        metric = lambda k, done=done, seeds=seeds: (
            mean_std([e.get(k) for e in done]) if len(done) == len(seeds) else None
        )
        arms.append(
            {
                "label": label,
                "plain": plain.get(label, label),
                "config": spec["config"],
                "steps": c.get("total_steps"),
                "learns": c.get("agent") != "ik",
                "seeds": seeds,
                "success": metric("success_rate"),
                "miss_mm": metric("final_dist_mm_median"),
                "steps_to_success": metric("steps_to_success_mean"),
                "smoothness": metric("smoothness_mean"),
                "energy": metric("energy_mean"),
            }
        )
    return {
        "id": stem,
        "title": exp.get("title", stem),
        "question": q,
        "why": why,
        "tech_question": exp.get("question", ""),
        "arms": arms,
        "hz": HZ,
    }


def search(q: str) -> list[dict]:
    """Case-insensitive search over every doc, decision, run and config name."""
    q = q.strip().lower()
    if len(q) < 2:
        return []
    hits = []
    s = state_cached()
    for d in s["docs"]:
        text = (ROOT / d["path"]).read_text(errors="replace")
        for i, ln in enumerate(text.splitlines()):
            if q in ln.lower():
                hits.append(
                    {
                        "kind": "doc",
                        "title": d["title"],
                        "path": d["path"],
                        "line": i,
                        "text": ln.strip()[:200],
                    }
                )
    for d in s["decisions"]:
        if q in (d["what"] + d["why"]).lower():
            hits.append(
                {
                    "kind": "decision",
                    "title": f"Decision · {d['date']} · phase {d['phase']}",
                    "text": d["what"][:200],
                }
            )
    for r in s["runs"]:
        if q in " ".join([r["id"], *r["exp"], r["best"] or "", r["algo"], r["mode"] or ""]).lower():
            hits.append(
                {
                    "kind": "run",
                    "title": r["id"],
                    "id": r["id"],
                    "text": " · ".join(r["exp"] + ([r["best"]] if r["best"] else [])),
                }
            )
    return hits[:80]


_state = {"t": 0.0, "v": None, "body": None}
_state_lock = threading.Lock()


def state_cached() -> dict:
    """state() at most once per second, however many tabs poll."""
    with _state_lock:
        if time.time() - _state["t"] >= 1 or _state["body"] is None:
            body = state()
            _state.update(
                t=time.time(), body=body, v=str(hash(json.dumps(body, sort_keys=True, default=str)))
            )
        return _state["body"]


# ---------- actions (viewers + read-only gate) ----------
JOBS: dict[str, dict] = {}


def low_prio_env(gui: bool) -> dict:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if gui:
        env.pop("MUJOCO_GL", None)  # GUI windows need the display backend, not EGL
    return env


def action(body: dict) -> dict:
    """Start one whitelisted command. GUIs open as their own window; the gate's output is kept as a job."""
    kind = body.get("kind")
    uv = ["nice", "-n", "19", "uv", "run", "python"]
    if kind in ("watch", "watch_random", "view"):
        if kind == "watch":
            rid = str(body.get("run", ""))
            if not NAME.match(rid) or not (RUNS / rid).is_dir():
                return {"error": "unknown run"}
            cmd = [*uv, "scripts/watch.py", "--run", f"runs/{rid}", "--speed", "2"]
        elif kind == "watch_random":
            cmd = [*uv, "scripts/watch.py", "--config", "configs/reach_final.yaml"]
        else:
            cmd = [*uv, "scripts/view.py"]
        (RUNS / "_logs").mkdir(exist_ok=True)
        with open(RUNS / "_logs" / "dashboard-gui.log", "a") as log:
            subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=low_prio_env(True),
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        return {"ok": "A window opens in a few seconds. Close it or press Ctrl+C in it to stop."}
    if kind == "gate":
        n = body.get("phase")
        if not isinstance(n, int) or not 0 <= n <= 7:
            return {"error": "bad phase"}
        jid = f"gate-{n}"
        if JOBS.get(jid, {}).get("state") == "running":
            return {"job": jid}
        p = subprocess.Popen(
            [*uv, "scripts/check_phase.py", str(n)],
            cwd=ROOT,
            env=low_prio_env(False),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        JOBS[jid] = {
            "state": "running",
            "title": f"Phase {n} gate",
            "out": "",
            "started": time.time(),
            "proc": p,
        }

        def collect(j=JOBS[jid]):
            out, _ = j["proc"].communicate()
            j.update(
                out=out[-20000:],
                state="PASS" if j["proc"].returncode == 0 else "FAIL",
                ended=time.time(),
            )

        threading.Thread(target=collect, daemon=True).start()
        return {"job": jid}
    if kind == "folder":
        rid = str(body.get("run", ""))
        target = RUNS / rid if NAME.match(rid) and (RUNS / rid).is_dir() else ROOT
        subprocess.Popen(
            ["xdg-open", str(target)],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"ok": f"Opened {target.relative_to(ROOT.parent)} in the file manager."}
    return {"error": "unknown action"}


# ---------- HTTP ----------
def safe_path(rel: str, allowed: tuple[str, ...]) -> Path | None:
    """Resolve rel inside the project, only under the allowed prefixes; None for anything else (e.g. ../)."""
    p = (ROOT / unquote(rel)).resolve()
    try:
        r = p.relative_to(ROOT).as_posix()
    except ValueError:
        return None
    return p if p.is_file() and r.startswith(allowed) else None


class Handler(BaseHTTPRequestHandler):
    server_version = "rlarm-dashboard"

    def log_message(self, *a):  # quiet: the terminal only shows the URL
        pass

    def send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code: int = 200) -> None:
        self.send(code, json.dumps(obj, default=str).encode(), "application/json")

    def local(self) -> bool:
        """Reject requests from other web pages (DNS rebinding / cross-site POST)."""
        host = self.headers.get("Host", "").split(":")[0]
        origin = self.headers.get("Origin")
        return host in ("127.0.0.1", "localhost") and (
            origin is None or urlparse(origin).hostname in ("127.0.0.1", "localhost")
        )

    def do_GET(self):
        if not self.local():
            return self.send(403, b"forbidden", "text/plain")
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self.send(
                    200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8"
                )
            if u.path == "/marked.min.js":
                return self.send(
                    200, (STATIC / "marked.min.js").read_bytes(), "text/javascript", cache=True
                )
            if u.path == "/api/state":
                body = state_cached()
                if q.get("v") == _state["v"]:
                    return self.json({"v": _state["v"], "same": True})
                return self.json({"v": _state["v"], **body})
            if u.path.startswith("/api/run/"):
                d = run_detail(u.path[len("/api/run/") :])
                return self.json(d) if d else self.json({"error": "no such run"}, 404)
            if u.path.startswith("/api/exp/"):
                d = experiment(u.path[len("/api/exp/") :])
                return self.json(d) if d else self.json({"error": "no such experiment"}, 404)
            if u.path == "/api/doc":
                p = safe_path(q.get("path", ""), ("docs/", "User Ref/", "CLAUDE.md", "README.md"))
                if p is None or p.suffix != ".md":
                    return self.json({"error": "not a project doc"}, 404)
                return self.json(
                    {
                        "path": p.relative_to(ROOT).as_posix(),
                        "text": p.read_text(errors="replace"),
                        "mtime": p.stat().st_mtime,
                    }
                )
            if u.path == "/api/search":
                return self.json(search(q.get("q", "")))
            if u.path.startswith("/file/"):
                p = safe_path(u.path[len("/file/") :], ("docs/media/", "runs/"))
                if p is None or p.suffix not in MEDIA:
                    return self.send(404, b"not found", "text/plain")
                return self.send(200, p.read_bytes(), MEDIA[p.suffix])
        except (BrokenPipeError, ConnectionResetError):
            return
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if (
            not self.local()
            or self.headers.get("Content-Type", "").split(";")[0] != "application/json"
        ):
            return self.send(403, b"forbidden", "text/plain")
        n = min(int(self.headers.get("Content-Length") or 0), 10_000)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self.json({"error": "bad json"}, 400)
        if self.path == "/api/action" and isinstance(body, dict):
            return self.json(action(body))
        self.send(404, b"not found", "text/plain")


def serve(port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    return srv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true", help="don't open the browser")
    a = p.parse_args()
    os.setpriority(os.PRIO_PROCESS, 0, 19)  # lowest priority: training always wins the CPU
    try:
        srv = serve(a.port)
    except OSError:
        raise SystemExit(
            f"port {a.port} is busy: is the dashboard already open? Try --port {a.port + 1}"
        )
    url = f"http://127.0.0.1:{a.port}"
    print(f"rl-arm dashboard: {url}   (Ctrl+C to stop)")
    if not a.no_open:
        threading.Timer(0.5, webbrowser.open, (url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        for j in JOBS.values():
            if j.get("state") == "running":
                j["proc"].terminate()


if __name__ == "__main__":
    main()
