"""Local, read-only dashboard: live training, experiments, phases, every run, and all docs in one page.

    uv run python scripts/dashboard.py [--port 8765] [--no-open]

Serves http://127.0.0.1:<port> until Ctrl+C or the terminal closes; `tailscale serve --bg <port>` also
puts it on the user's phone (only buttons that open laptop windows are refused there). Lowest CPU priority
and only re-reads files whose size/mtime changed, so it doesn't slow training. It never trains,
evaluates or writes project files; its buttons only open viewers and run the read-only phase gate.
Imports no torch/mujoco (rlarm.common pulls both in), so the run-folder rules are restated here.
"""

import argparse
import functools
import gzip
import json
import os
import re
import shutil
import subprocess
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS, STATIC = ROOT / "runs", Path(__file__).resolve().parent / "dashboard"
DISMISSED = (
    RUNS / "dismissed.json"
)  # run ids whose "stopped" warning was dismissed (by the user or Claude)
NAME = re.compile(r"^(\d{4}-\d\d-\d\d)_([^/]+)_s(\d+)$")
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
    ("Start here", "Phase 5 guide", "docs/PHASE5_GUIDE.md"),
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
        _git.update(t=time.time(), tags=tags, starts=starts, dirty=dirty)
    return _git


# ---------- training processes ----------
def train_procs() -> dict[int, tuple[str, int | None]]:
    """pid -> (config name, seed) of every live, non-dry train.py, from /proc (no subprocess)."""
    out = {}
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
            # runs are named after the config's `name:`, which can differ from the file name (configs/pilot/)
            cfg = yread(ROOT / a[i]) if i < len(a) else None
            name = (
                cfg.get("name") if isinstance(cfg, dict) else Path(a[i]).stem if i < len(a) else "?"
            )
            out[int(c.parent.name)] = (name, seed)
    return out


def training() -> set[tuple[str, int | None]]:
    return set(train_procs().values())


# ---------- machine vitals (sampled every 2 s by a background thread, ~1 ms each) ----------
def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def _num(p: Path, div: float = 1) -> float | None:
    try:
        return float(p.read_text()) / div
    except (OSError, ValueError):
        return None


_sensors: dict[str, Path] = {}


def sensor(*keys: str, div: float = 1000) -> float | None:
    """First readable hwmon value among keys '<chip>/<label>/<input|average>', e.g. 'k10temp/Tctl/input'."""
    if not _sensors:
        for h in Path("/sys/class/hwmon").glob("hwmon*"):
            chip = _read(h / "name") or h.name
            for f in h.glob("*_*"):
                m = re.fullmatch(r"(temp|power|fan)(\d+)_(input|average)", f.name)
                if m:
                    label = _read(h / f"{m[1]}{m[2]}_label") or f"{m[1]}{m[2]}"
                    _sensors.setdefault(f"{chip}/{label}/{m[3]}", f)
    for k in keys:
        if k in _sensors:
            v = _num(_sensors[k], div)
            if v is None:
                _sensors.clear()  # hwmon numbers can change after a resume: look them up again next time
            return None if v is None else round(v, 1)
    return None


def battery() -> dict | None:
    """Charge, plugged in or not (from the charger, not the battery: at a charge limit it reads 'Discharging'), health."""
    ps = Path("/sys/class/power_supply")
    if not ps.exists():
        return None
    plugged = any(_read(p / "online") == "1" for p in ps.iterdir() if _read(p / "type") == "Mains")
    for b in sorted(ps.glob("BAT*")):
        pct = _num(b / "capacity")
        if pct is None:
            continue
        full = _num(b / "energy_full") or _num(b / "charge_full")
        design = _num(b / "energy_full_design") or _num(b / "charge_full_design")
        watts, wh = _num(b / "power_now", 1e6), _num(b / "energy_now", 1e6)
        return {
            "pct": int(pct),
            "status": _read(b / "status") or "?",
            "plugged": plugged,
            "watts": round(watts, 1) if watts else None,
            "health": round(100 * full / design) if full and design else None,
            "cycles": int(_num(b / "cycle_count") or 0) or None,
            "hours_left": round(wh / watts, 1)
            if not plugged and watts and watts > 0.5 and wh
            else None,
        }
    return None


_vit: dict = {"prev": None, "latest": {}, "hist": deque(maxlen=600), "n": 0}
_vit_lock = threading.Lock()


def sample() -> dict:
    """One reading of every vital; every 3rd one also goes into the last hour's history."""
    rows = [
        ln.split() for ln in Path("/proc/stat").read_text().splitlines() if ln.startswith("cpu")
    ]
    now = {r[0]: (int(r[4]) + int(r[5]), sum(int(x) for x in r[1:9])) for r in rows}
    with _vit_lock:
        prev, _vit["prev"] = _vit["prev"] or {}, now

    def busy(k: str) -> int:
        p = prev.get(k)
        if not p or now[k][1] <= p[1]:
            return 0
        return round(100 * (1 - (now[k][0] - p[0]) / (now[k][1] - p[1])))

    mem = dict(
        ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln
    )
    gb = lambda k: round(int(mem[k].split()[0]) / 2**20, 1)
    cpus = Path("/sys/devices/system/cpu")
    ghz = [g for f in cpus.glob("cpu[0-9]*/cpufreq/scaling_cur_freq") if (g := _num(f, 1e6))]
    busy_gpu = [_read(p) for p in Path("/sys/class/drm").glob("card*/device/gpu_busy_percent")]
    v = {
        "t": time.time(),
        "cpu": busy("cpu"),
        "cores": [
            busy(k) for k in sorted((k for k in now if k != "cpu"), key=lambda k: int(k[3:]))
        ],
        "cpu_temp": sensor(
            "k10temp/Tctl/input",
            "zenpower/Tdie/input",
            "coretemp/Package id 0/input",
            "acpitz/temp1/input",
        ),
        "gpu_temp": sensor("amdgpu/edge/input"),
        "ssd_temp": sensor("nvme/Composite/input"),
        "chip_w": sensor("amdgpu/PPT/average", "amdgpu/PPT/input", div=1e6),
        "ghz": round(sum(ghz) / len(ghz), 2) if ghz else None,
        "ghz_max": _num(cpus / "cpu0/cpufreq/cpuinfo_max_freq", 1e6),
        "gpu_busy": next((int(b) for b in busy_gpu if b and b.isdigit()), None),
        "mem_used": round(gb("MemTotal") - gb("MemAvailable"), 1),
        "mem_total": gb("MemTotal"),
        "swap_used": round(gb("SwapTotal") - gb("SwapFree"), 1),
        "swap_total": gb("SwapTotal"),
        "load": [float(x) for x in Path("/proc/loadavg").read_text().split()[:3]],
        "battery": battery(),
    }
    with _vit_lock:
        _vit["latest"] = v
        _vit["n"] += 1
        if _vit["n"] % 3 == 1:
            _vit["hist"].append(
                [round(v["t"]), v["cpu"], v["cpu_temp"], v["chip_w"], v["mem_used"]]
            )
    return v


def sampler() -> None:
    while True:
        try:
            sample()
        except (OSError, ValueError, KeyError, IndexError):
            pass  # one odd reading; the next one comes in 2 s
        time.sleep(2)


def machine() -> dict:
    """The vitals every poll carries (the Laptop page asks /api/system for the rest)."""
    v = _vit["latest"] or sample()
    return {k: v[k] for k in ("cpu", "cpu_temp", "chip_w", "mem_used", "mem_total", "battery")}


_procs: dict = {"prev": None, "t": 0.0}
_procs_lock = threading.Lock()


def proc_times() -> dict[int, tuple[str, int, int, int]]:
    """pid -> (name, CPU ticks used so far, resident memory in pages, parent pid)."""
    out = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            s = (d / "stat").read_text()
        except OSError:
            continue
        f = s[s.rindex(")") + 2 :].split()  # f[k-3] = field k of proc(5)
        out[int(d.name)] = (
            s[s.index("(") + 1 : s.rindex(")")],
            int(f[11]) + int(f[12]),
            int(f[21]),
            int(f[1]),
        )
    return out


def procs() -> list[dict]:
    """Who uses the CPU since the last call: training runs (with their worker processes), apps, this dashboard."""
    with _procs_lock:
        now, t = proc_times(), time.time()
        prev, t0 = _procs["prev"], _procs["t"]
        if prev is None or t - t0 > 30:  # first look, or a stale one: measure over half a second
            time.sleep(0.5)
            prev, t0 = now, t
            now, t = proc_times(), time.time()
        _procs.update(prev=now, t=t)
    tick, page, ncpu = os.sysconf("SC_CLK_TCK"), os.sysconf("SC_PAGE_SIZE"), os.cpu_count() or 1
    runs = train_procs()

    def run_of(pid: int) -> tuple[str, int | None] | None:
        for _ in range(4):  # vec-env workers are children of the train.py process
            if pid in runs:
                return runs[pid]
            pid = now[pid][3] if pid in now else 0
        return None

    groups: dict[tuple, dict] = {}
    for pid, (name, ticks, rss, _) in now.items():
        r = run_of(pid)
        if r:
            rd = newest(r[0], r[1]) if r[1] is not None else None
            key = ("run", rd.name if rd else f"{r[0]} s{r[1]}")
        else:
            key = ("self", "this dashboard") if pid == os.getpid() else ("app", name)
        g = groups.setdefault(
            key, {"kind": key[0], "name": key[1], "cpu": 0.0, "mem_gb": 0.0, "n": 0}
        )
        if pid in prev:
            g["cpu"] += 100 * (ticks - prev[pid][1]) / tick / max(t - t0, 1e-3) / ncpu
        g["mem_gb"] += rss * page / 2**30
        g["n"] += 1
    rows = sorted(groups.values(), key=lambda g: -g["cpu"])
    apps = [g for g in rows if g["kind"] == "app" and g["cpu"] >= 0.2][:8]
    keep = [g for g in rows if g["kind"] != "app"] + apps
    for g in keep:
        g["cpu"], g["mem_gb"] = round(g["cpu"], 1), round(g["mem_gb"], 2)
    return keep


_inh: dict = {"t": 0.0, "v": None}


def sleep_blockers() -> list[dict] | None:
    """Apps holding a 'block' inhibitor on sleep/idle/lid (systemd-logind); None if logind can't be asked."""
    if time.time() - _inh["t"] > 15:
        try:
            out = subprocess.run(
                [
                    "busctl",
                    "--json=short",
                    "call",
                    "org.freedesktop.login1",
                    "/org/freedesktop/login1",
                    "org.freedesktop.login1.Manager",
                    "ListInhibitors",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout
            rows = json.loads(out)["data"][0]
            v = [
                {"what": what, "who": who, "why": why}
                for what, who, why, mode, *_ in rows
                if mode == "block" and {"sleep", "idle", "handle-lid-switch"} & set(what.split(":"))
            ]
        except (OSError, ValueError, KeyError, IndexError, TypeError, subprocess.TimeoutExpired):
            v = None
        _inh.update(t=time.time(), v=v)
    return _inh["v"]


def system() -> dict:
    """Everything the Laptop page shows."""
    du = shutil.disk_usage(ROOT)
    return {
        **(_vit["latest"] or sample()),
        "hist": list(_vit["hist"]),
        "procs": procs(),
        "sleep_blockers": sleep_blockers(),
        "disk_free": round(du.free / 2**30, 1),
        "disk_total": round(du.total / 2**30),
        "uptime_s": float(Path("/proc/uptime").read_text().split()[0]),
        "profile": _read(Path("/sys/firmware/acpi/platform_profile")),
        "kernel": os.uname().release,
        "host": os.uname().nodename,
        "threads": os.cpu_count(),
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
                dirs[-1],
            ]
        # live runs: at most every 5 s; finished ones: once more if the log grew after the last read
        grown = max((f.stat().st_mtime for f in ent[3].iterdir()), default=0) > ent[1]
        if ent[1] == 0 or (grown and (not live or time.time() - ent[1] > 5)):
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
    alive = ((name, seed) in live or ((name, None) in live and cfg.get("seed") == seed)) and newest(
        name, seed
    ) == run  # an older dated folder of the same config + seed is not the one training
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
        sc = scalars(run, live=False)  # the log goes further than the last checkpoint's summary
        steps = max(steps, max((v[-1][0] for v in sc.values() if v), default=0))
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
        "val_last": hist[-1]["success"] if hist else None,
        "val": [[h["step"], h["success"]] for h in hist],
        # val.clean configs score validation as success rate minus collision rate, so it can go below 0
        "val_kind": "clean" if (cfg.get("val") or {}).get("clean") else "success",
        "stopped_early": sm.get(
            "stopped_early"
        ),  # plateau / stuck_low / manual (val.early_stop, STOP file)
        "usable": sm.get("usable", True),
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
        else "Idle between phases. The next phase starts when you give Claude its go-ahead in the terminal (Guide §2).",
        "list": rows,
        "runs": [r["id"] for r in mine],
    }


@functools.cache
def me() -> str:
    """The user's first name (git user.name), for the greeting."""
    return (sh("git", "config", "user.name").split() or [""])[0]


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
        "machine": machine(),
        "me": me(),
        "dismissed": jread(DISMISSED) or [],
        "dirty": g["dirty"],
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
    idx = config_index()
    cfg_src = idx.get(meta["config"])
    base_name = (
        "e1_delta_pos"
        if meta["exp"] and meta["config"] != "e1_delta_pos"
        else f"{meta['task']}_final"
    )
    base_path = idx.get(base_name)
    base = yread(base_path) if base_path and base_name != meta["config"] else None
    sc = scalars(run, live=meta["status"] == "training")
    log = RUNS / "_logs" / f"{meta['config']}-s{meta['seed']}.log"
    return {
        **meta,
        "why": comment(cfg_src),
        "eval": jread(run / "eval.json"),
        "summary": jread(run / "summary.json"),
        "base": base_name if base else None,
        "base_missing": base_path is None,
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
            # version without the machine stats (they change every second), so unchanged polls stay tiny
            stable = {k: v for k, v in body.items() if k != "machine"}
            v = str(hash(json.dumps(stable, sort_keys=True, default=str)))
            _state.update(t=time.time(), body=body, v=v)
        return _state["body"]


# ---------- actions (viewers + read-only gate) ----------
JOBS: dict[str, dict] = {}
HOSTS = {"127.0.0.1", "localhost"}  # + this laptop's Tailscale name, added in main()
LAPTOP_ONLY = {"watch", "watch_random", "view", "folder"}  # they open a window on the laptop screen


def tailnet_name() -> str | None:
    """This laptop's Tailscale name (`tailscale serve` proxies the phone's requests to us under it)."""
    try:
        out = subprocess.run(
            ["tailscale", "status", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout
        return json.loads(out)["Self"]["DNSName"].rstrip(".") or None
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
        return None


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
        speed = body.get("speed", 1)
        if not isinstance(speed, (int, float)) or not 0 <= speed <= 16:
            return {"error": "speed must be 0 (as fast as possible) to 16"}
        if kind == "watch":
            rid = str(body.get("run", ""))
            if not NAME.match(rid) or not (RUNS / rid).is_dir():
                return {"error": "unknown run"}
            cmd = [*uv, "scripts/watch.py", "--run", f"runs/{rid}", "--speed", str(speed)]
        elif kind == "watch_random":
            cmd = [
                *uv,
                "scripts/watch.py",
                "--config",
                "configs/reach_final.yaml",
                "--speed",
                str(speed),
            ]
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
        if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= 7:
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
    if kind == "dismiss":
        rids = body.get("runs")
        if (
            not isinstance(rids, list)
            or not rids
            or not all(isinstance(r, str) and NAME.match(r) and (RUNS / r).is_dir() for r in rids)
        ):
            return {"error": "unknown run"}
        tmp = DISMISSED.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted({*(jread(DISMISSED) or []), *rids}), indent=1))
        tmp.replace(DISMISSED)  # atomic, so a poll never reads half a file
        return {"ok": "Dismissed. The run still shows as stopped on the Runs page."}
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
    try:
        p = (ROOT / unquote(rel)).resolve()
        r = p.relative_to(ROOT).as_posix()
    except (ValueError, OSError):  # outside the project, or a NUL byte in the path
        return None
    return p if p.is_file() and r.startswith(allowed) else None


class Handler(BaseHTTPRequestHandler):
    server_version = "rlarm-dashboard"
    timeout = 10  # a client that sends less body than it announced can't hold a thread forever

    def log_message(self, *a):  # quiet: the terminal only shows the URL
        pass

    def send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
        # gzip text: the full state is ~100 kB and changes every few seconds while training (phone data)
        gz = (
            len(body) > 1400
            and ctype.startswith(("application/json", "text/"))
            and "gzip" in self.headers.get("Accept-Encoding", "")
        )
        if gz:
            body = gzip.compress(body, 5)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if gz:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code: int = 200) -> None:
        self.send(code, json.dumps(obj, default=str).encode(), "application/json")

    def host(self) -> str:
        return self.headers.get("Host", "").split(":")[0]

    def allowed(self) -> bool:
        """Reject requests from other web pages (DNS rebinding / cross-site POST)."""
        origin = self.headers.get("Origin")
        return self.host() in HOSTS and (origin is None or urlparse(origin).hostname in HOSTS)

    def do_GET(self):
        if not self.allowed():
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
                    return self.json({"v": _state["v"], "same": True, "machine": body["machine"]})
                return self.json({"v": _state["v"], **body})
            if u.path == "/api/system":
                return self.json(system())
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
        except Exception as e:  # noqa: BLE001 - one odd run file must not make the page look dead
            return self.json({"error": f"{type(e).__name__}: {e}"}, 500)
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if (
            not self.allowed()
            or self.headers.get("Content-Type", "").split(";")[0] != "application/json"
        ):
            return self.send(403, b"forbidden", "text/plain")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 <= n <= 10_000:
                return self.json({"error": "bad length"}, 400)
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self.json({"error": "bad json"}, 400)
        if self.path == "/api/action" and isinstance(body, dict):
            if self.host() not in ("127.0.0.1", "localhost") and body.get("kind") in LAPTOP_ONLY:
                return self.json(
                    {"error": "That opens a window on the laptop, so it only works there."}
                )
            return self.json(action(body))
        self.send(404, b"not found", "text/plain")


def serve(port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    if not any(th.name == "vitals" for th in threading.enumerate()):
        threading.Thread(target=sampler, name="vitals", daemon=True).start()
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
    if ts := tailnet_name():
        HOSTS.add(ts)
        print(
            f"on your phone (Tailscale): https://{ts}   (needs `tailscale serve --bg {a.port}` once)"
        )
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
