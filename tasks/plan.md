# rl-arm: plan from kit to finished project

## Context
You dropped the kit (SPEC, ROADMAP, phase/train skills, hooks, settings) into the repo and asked for a plan that, for every phase, says **what you put in, what I hand back, what's needed, and how to move on**. The ROADMAP already defines the phases and gates, so this plan doesn't add new tasks. It covers the sequencing, the hand-offs, and the risks around them.

**Blocker found:** the kit sits in `rl-arm-kit/`, but it was meant to be unzipped at the repo root. Right now:
- `CLAUDE.md` points to `docs/SPEC.md` and `docs/ROADMAP.md`, which don't exist at the root.
- `.claude/settings.json` (MUJOCO_GL=egl, permissions, Sonnet default) isn't loaded.
- The hooks call `$CLAUDE_PROJECT_DIR/.claude/hooks/...`, which doesn't exist, so the SPEC/ROADMAP lock, auto-format and stop-checks are all inactive.
- The `phase`/`train` skills only trigger for files under `rl-arm-kit/`.

## Step 0: fix the layout (me, ~1 min, before Phase 0)
- `git mv` everything in `rl-arm-kit/` into the repo root: `docs/`, `.gitignore`, `.claude/hooks`, `.claude/settings.json`, `.claude/skills/{phase,train}`. Merge it beside the existing `.claude/skills/stable-baselines3`. No conflicts, since the root has no `.claude/settings.json` yet.
- Remove the empty `rl-arm-kit/`. Hooks are already mode 755 in git. Commit: `chore: move kit to repo root`.
- Line in `docs/decisions.md`.
- **You:** restart Claude so the settings and hooks load. Use `systemd-inhibit --what=idle:sleep claude --permission-mode auto`, as in `User Ref/SETUP.md`.
- Already checked: uv 0.12.9 and notify-send are installed. Python 3.12 isn't, but uv downloads it into user space. No sudo needed.

## Loop for every phase
**You type:**
```
/goal Phase N is finished: `uv run python scripts/check_phase.py N` printed PASS in this session, docs/reports/phase-N.md exists, and the work is committed. Use the phase skill for phase N. Stop after 40 turns, or earlier if you need a decision from me.
```
**I do (phase skill):**
1. Write the checks first and confirm they fail.
2. Build with tests.
3. Train in the background.
4. Run the gate, then one Sonnet review subagent.
5. Write the report, then commit and tag `phase-N-done`.

**You get back (≤8 lines):**
- PASS/FAIL per check ID, each with its value
- `docs/reports/phase-N.md`
- 1–3 media files to look at

**Then you:**
1. Read the report and open its GIF/PNG.
2. Ask for fixes in plain words if needed.
3. Run `/clear` and start the next phase.

**I stop and ask only when:**
- a SPEC/ROADMAP change is needed,
- something is destructive or outside this folder, or
- the same check has failed 3 times. You then get the evidence, never a lowered threshold.

## Per-phase hand-offs

| Phase | Your input | My output | Needs / risk |
|---|---|---|---|
| **0 Setup + Reacher** | Start the goal, keep the laptop awake (~1 h incl. training) | uv project (Py 3.12, CPU torch via the pytorch-cpu index), `src/rlarm/` skeleton, `check_phase.py`, `train.py`/`eval.py`/`render_snapshot.py`, Reacher SAC config + run. **Report:** versions, Reacher return (target ≥ −7), GIF of Reacher | ~1–2 GB download. EGL on the 780M should work through Mesa; osmesa is the fallback and gets logged. 300k SAC steps ≈ 30–60 min on CPU |
| **1 Arm + kinematics** | Judge the arm PNG. Optionally run `uv run python scripts/view.py` and drag things around | MJCF in `models/arm/`, DH FK, closed-form IK (both elbows), joint limits in decisions.md. **Report:** FK/IK error (<1e-6 m), contact count (0), hold error (<2°), arm PNG | Hardest part: limits that make **every** in-limit pose floor-free. Shoulder/elbow ranges have to be narrowed, which also shrinks the workspace. You'll see the final limits in the report |
| **2 Environment** | Judge the report (no training) | Goal-conditioned env, `reach` task, 4 action modes, physics from YAML, reward terms in `info`, IK scripted agent + random agent. **Report:** IK agent success (≥98%), random agent (≤5%), determinism, check_env | Low risk. Confirms the env can be solved before any RL time is spent on it |
| **3 Reach + hold (RL)** | Keep the laptop awake for **~3–5 h** of background training. Judge the reach/hold GIFs (smooth? settles?) | SAC configs, `hold` task, 3 seeds each, `docs/best_runs.yaml`. **Report:** reach success (≥95%), median mm (≤10), hold success (≥90%), joint speed (<0.05 rad/s), 2 GIFs | First real RL risk. 10 mm median and calm holding may need precision/settle reward terms. They're added one at a time and each one is logged. ≤1M steps/run, 6 runs, ~6 in parallel at 2 threads each |
| **4 Experiments** | Biggest compute block: **~10–15 h** total, best run overnight. Read `docs/results.md` | E1 (4 action modes), E2 (SAC vs PPO), E3 (HER sparse vs shaped), E4 (RL vs IK), ≥3 seeds each. `make_results.py` → `results.md`, learning-curve plots | No thresholds, the numbers just get reported. About 18+ runs, queued within the 14-thread budget. PPO uses 8 workers = 9 threads, so it runs mostly alone |
| **5 Track / obstacle / push** | Judge 3 GIFs. Expect pauses here: if a target looks out of reach, I'll ask instead of grinding | 3 env subclasses + configs + curricula. **Report:** tracking error (≤25 mm), obstacle success/collisions (≥85% / ≤5%), push (≥60%) | **Push ≥60% is the riskiest gate in the project** (3-DOF, flat pad, no gripper). It will likely need a curriculum. If 3 honest attempts fail, you decide whether to change the threshold |
| **6 Demo + README** | **Hands-on:** run `scripts/demo.py`, drag the target, say whether it feels right | `demo.py` (+ `--headless`), README with the required sections, final results.md, fresh-clone test | The viewer needs a real display, so the demo unsets `MUJOCO_GL=egl` for interactive mode. Otherwise low risk |
| **7 (optional) Robustness + UR5e** | Decide whether to do it at all | Domain-randomisation wrapper, UR5e from Menagerie (cloned into `third_party/`, git-ignored), robustness eval. **Report:** perturbed success (≥85%) vs nominal, UR5e reach (≥90%) | Link-length randomisation means rebuilding the model per episode. UR5e needs the env to be robot-agnostic, which was planned from Phase 1 |

## Usage notes
- `.claude/settings.json` defaults to **Sonnet**, and this session is Opus. Once settings load, phases run on Sonnet unless you `/model` switch, which the kit recommends to save usage.
- Do one phase per conversation and `/clear` between them. Training uses your CPU, not Claude usage.
- Total wall time is roughly: P0 1 h, P1–2 a few hours of my turns, P3 ~5 h, P4 ~15 h, P5 ~10 h+, P6 1 h.

## Files
- I write `tasks/plan.md` with this content, so it lasts beyond the session. `tasks/todo.md` is skipped: the ROADMAP checks plus the phase skill's per-phase task list already do that job.
- Step 0 touches: `rl-arm-kit/**` → root, `docs/decisions.md`.

## Verification (Step 0)
- `git status` is clean after the commit, and `ls docs .claude/hooks .claude/skills` shows the kit at the root.
- After the restart, a test Edit of `docs/ROADMAP.md` is blocked by the hook. That proves the hooks are live.
- Then Phase 0 starts with the `/goal` above, and its gate is `uv run python scripts/check_phase.py 0`.
