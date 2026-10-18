# rl-arm guide

How to run the project from start to finish: what you type, what Claude does, what you get back, and what to do when something goes wrong.

- All commands are **fish** and run from the project root: `cd ~/Projects/Claude/rl-arm`.
- Use a **second kitty window** for anything you run yourself, so Claude's session stays untouched.
- First time here? Read `User Ref/SETUP.md`. Charts confusing? Read `User Ref/TENSORBOARD.md`.

---

<!-- STATUS:BEGIN -->
## 0. Where you are now (auto-updated)

_Updated 2026-10-18 09:07 by `User Ref/guide-status` after each Claude turn. Run `python3 'User Ref/guide-status'` for a fresh copy. Don't edit inside this block._

### Now: between phases (Phase 3 done)
- **Claude is** idle. Phase 4 starts when you paste its `/goal` (§1).
- **Uncommitted changes:** 1 file(s) (normal mid-phase; they're committed at the end).

### What you can do right now
- Read `docs/reports/phase-3.md` and open its media (table below).
- Play with it (§3). Then `/clear` and paste the `/goal` for Phase 4.
- The arm by itself: `uv run python scripts/view.py` (rotate, drag, joint sliders).
- An env with a random policy: `uv run python scripts/watch.py --config configs/reach_final.yaml`.

### Done (past)
| Phase | Report | Media | Key numbers |
|---|---|---|---|
| 0 Setup and warm-up | `docs/reports/phase-0.md` | `phase0_reacher.gif`, `phase0_reacher_snapshot.png` | Run `2026-10-17_reacher_sac_s0`, seed 0, 300k steps, 32 min wall (155 steps/s, 2 threads). Random policy ≈ −43. |
| 1 Arm model and kinematics | `docs/reports/phase-1.md` | `phase1_arm.png` | Limits: yaw ±3.1, shoulder [0.4, 1.7], elbow [-1.1, 1.4] rad. Gains kp=400, kv=20, 12 N·m. No training this phase. |
| 2 Environment | `docs/reports/phase-2.md` | `phase2_ik_reach.gif`, `phase2_reach.png` | No training this phase. Control runs at 25 Hz (frame_skip 20 × 2 ms), and episodes last 200 steps (8 s). The IK agent reaches the target in ≈34 steps (1.4 s) and then holds it. A random agent ends ≈0.4 m away. |
| 3 Learned reach (task A) and hold (task B) | `docs/reports/phase-3.md` | `phase3_hold.gif`, `phase3_reach.gif` | **Runs:** `runs/2026-10-18_{reach,hold}_final_s{0,1,2}`. Each run is 1M steps and took about 4.8 h, with 6 runs in parallel. Val-best checkpoints came from 650k–950k steps. **How reach success changed** (seed 0): |

Look back at any phase N: its report; `git log --oneline phase-N-start..phase-N-done` (what changed); `git diff phase-N-start phase-N-done --stat`; `uv run python scripts/check_phase.py N` (re-run its gate); media in `docs/media/`.

### Next (future)
- **Phase 4: Experiments (results, not gates)**: builds experiments E1–E4 (action modes, SAC vs PPO, HER, RL vs IK) and `docs/results.md`. You: leave it running overnight, then read `docs/results.md`.
- **Phase 5: Harder tasks (C → D → E)**: builds track, obstacle and push tasks with curricula. You: judge 3 GIFs; decide if push stalls after 3 attempts.
- Read ahead: `grep -n -A 20 '## Phase <N>' docs/ROADMAP.md`. Full journey: §4.
<!-- STATUS:END -->


---

## 1. The loop (same for every phase)

### Start a phase
```fish
cd ~/Projects/Claude/rl-arm
systemd-inhibit --what=idle:sleep claude --permission-mode auto
```
Then paste this, with N set to the phase number:
```
/goal Phase N is finished: `uv run python scripts/check_phase.py N` printed PASS in this session,
docs/reports/phase-N.md exists, and the work is committed.
Use the phase skill for phase N. Stop after 40 turns, or earlier if you need a decision from me.
```

### What happens
| Step | What Claude does | What you see |
|---|---|---|
| 1. Orient | Reads this phase's section of the roadmap and tags `phase-N-start` | A short task list |
| 2. Checks first | Writes the phase's checks and runs them once, expecting FAIL | A few `FAIL` lines. Normal |
| 3. Build | Writes code and a test for each piece. Logs choices in `docs/decisions.md` | Short progress messages |
| 4. Train (phases 0, 3, 4, 5, 7) | Does a quick dry run, launches the real runs in the background, **and ends its turn** | One line with run ids and an ETA, then silence. **This is expected.** Claude wakes up when a run finishes |
| 5. Gate | Runs the checks. If one fails, it diagnoses, changes one thing and retrains | A PASS/FAIL table |
| 6. Review | One reviewer agent checks the work against the roadmap and spec | A short note |
| 7. Report and commit | Writes `docs/reports/phase-N.md`, copies media, commits, tags `phase-N-done` | A final message of at most 8 lines: PASS/FAIL per check, the report path, and 1–3 things to look at |

You get a desktop notification whenever Claude needs you.

### After the phase
1. Read `docs/reports/phase-N.md`: "Checks", "Look at this" and "How it works".
2. Open the GIF or PNG it points to, and play with the arm (§3).
3. Ask for fixes in plain words, e.g. "hold GIF looks jittery, fix it".
4. Run `/clear`, then start the next phase.

---

## 2. The phase gate

`scripts/check_phase.py` holds the pass/fail checks for every phase. It only reads existing results and never trains, so it's safe to run any time.

```fish
uv run python scripts/check_phase.py 3            # full output
uv run python scripts/check_phase.py 3 --summary  # compact table
```
Each line is `check-id PASS|FAIL | measured value`, followed by a final verdict. A check that crashes counts as FAIL and prints `error: ...`.

A gate is a minimum, not the definition of good. If the result looks wrong even though it passed, say so.

---

## 3. Playing with the arm

| Available from | What | Command |
|---|---|---|
| Phase 0 | Watch training charts. See `TENSORBOARD.md` | `uv run python scripts/tb.py`, then open http://localhost:6006 (Custom Scalars tab) |
| Phase 0 | Run status (steps, success, ETA) | `cat runs/*/summary.json` |
| Phase 0 | **Watch a policy live** in a MuJoCo window. During training it shows the latest saved checkpoint, so you see the arm improve | `uv run python scripts/watch.py` (latest run) or `--run runs/<id>` |
| Phase 0 | Still image of an env | `env MUJOCO_GL=egl uv run python scripts/render_snapshot.py --config configs/reach_final.yaml --out /tmp/snap.png`, then `xdg-open /tmp/snap.png` |
| Phase 0 | GIF of a finished run | open `runs/<run_id>/video.gif` |
| Phase 1 | **Interactive viewer:** rotate, zoom, move joints with the sliders. Drag the target: double-click it, then Ctrl + right-drag | `uv run python scripts/view.py` |
| Phase 6 | **Live demo:** drag the target and the trained policy follows | `uv run python scripts/demo.py` |

Handy `watch.py` options: `--config configs/<name>.yaml` (random policy, before any training), `--episodes 5`, `--stochastic`, `--speed 4` (4× faster; `0` = as fast as possible). Close the window or press Ctrl+C to stop.

**Windows Claude opens.** When Claude opens a GUI for you, the window floats in the bottom-right corner and focus returns to Claude's terminal. Move or close it like any window. Its output goes to `runs/_logs/gui.log`.

**While a phase is running:**
- Safe: TensorBoard, `summary.json`, GIFs, `watch.py`, `view.py`, `check_phase.py`.
- Don't run `scripts/eval.py` on a run that is still training: it overwrites that run's results.
- Don't run `git commit`, `git checkout` or `git reset`.
- The viewer and `watch.py` use a little CPU, which only slightly slows training.

---

## 4. The phases

| # | Phase | Time | You do | Gate (summary) |
|---|---|---|---|---|
| 0 | **Setup and warm-up.** SAC learns Gymnasium's built-in 2-joint Reacher. It's a stand-in that proves the pipeline works before our robot exists | ~1 h | Watch the Reacher GIF: the fingertip should go straight to the dot and stop | Reacher score ≥ −7 (random ≈ −43); tests and lint pass |
| 1 | **Arm model and kinematics** (no training). The 3-joint arm in MuJoCo, plus the maths to turn joint angles into a tip position (FK) and back (IK) | – | Run `view.py`, move the sliders, check the arm looks right and nothing clips | FK and IK agree with MuJoCo to under 1 µm; no collisions in 2000 random poses |
| 2 | **Environment** (no training). The reach task, 4 ways to control the arm, a scripted IK agent and a random agent | – | Compare IK agent vs random agent numbers: proves the task is solvable and not trivial | IK agent ≥ 98% success; random ≤ 5%; seeds reproduce exactly |
| 3 | **Learned reach and hold.** SAC trains on both tasks, 3 seeds each | ~5 h | Do the GIFs move smoothly? Does the arm settle or jitter? | Reach ≥ 95% and ≤ 10 mm; hold ≥ 90% and stays still |
| 4 | **Experiments.** Action modes, SAC vs PPO, with vs without HER, RL vs the IK agent | 10–15 h | Read `docs/results.md` and the plots. Good material for a CV or interview | No performance targets; only checks that results are complete and generated from data |
| 5 | **Harder tasks.** Track a moving target, avoid an obstacle, push a cube | 10 h+ | Judge 3 GIFs. Push is the hardest gate: if Claude can't reach it after 3 honest tries, it stops and you decide what to change | Tracking error ≤ 25 mm; obstacle ≥ 85% with ≤ 5% collisions; push ≥ 60% |
| 6 | **Demo and polish.** Drag-the-target demo, README, fresh-clone test | ~1 h | Run `demo.py`, drag the target around (including fast moves and workspace edges) | Headless demo mean distance ≤ 30 mm; all tests pass |
| 7 | **Robustness (optional).** Randomised physics; a UR5e arm | – | Decide whether to do it at all | ≥ 85% on perturbed arms; UR5e reach ≥ 90% |

Exact numbers and wording live in `docs/ROADMAP.md`. Read one phase ahead: `grep -n -A 20 '## Phase <N>' docs/ROADMAP.md`.

**Look back at a phase N:** read its report; `git log --oneline phase-N-start..phase-N-done` shows what changed; `uv run python scripts/check_phase.py N` re-runs its gate.

**Terms used above**
- **SAC / PPO:** two learning algorithms. SAC reuses old experience; PPO learns only from fresh experience.
- **HER:** replays failed attempts as if the point reached had been the goal, so sparse rewards still teach something.
- **Seed:** the random starting point of a run. Three seeds that agree mean the result is reliable.
- **Success:** the arm tip ends within 2 cm of the target.

---

## 5. Things to do while waiting

- **Watch:** TensorBoard's `success_rate` and reward charts; `tail docs/decisions.md` to see what Claude chose and why.
- **Understand:** read the previous report's "How it works" aloud. If you can't explain it, ask Claude in a fresh session. Read `docs/SPEC.md` §4–§5 (observation, action, reward) before Phase 3: it's the heart of the project.
- **Plan:** if you want to change `docs/SPEC.md` or `docs/ROADMAP.md`, do it between phases and commit. Only you can edit them.
- Avoid heavy CPU use (gaming, compiling) during long training. It slows runs and pushes back ETAs.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| Section 0 looks stale | It refreshes after each Claude turn, so it can lag during long training. Run `python3 'User Ref/guide-status'` |
| Claude went quiet after launching training | Expected. It wakes up when a run finishes. Check TensorBoard or `cat runs/*/summary.json`. Don't type into the session unless you want to interrupt |
| Laptop slept or rebooted mid-training | The runs died, but `summary.json` still says `running`. Start Claude again (with `systemd-inhibit`), re-paste the `/goal`, and say "runs were killed by sleep, relaunch them" |
| `/goal` stopped at the 40-turn cap | Re-paste the same `/goal`. Work and commits are kept |
| Hit the usage limit mid-phase | The goal pauses. After the reset, re-paste it. Background training keeps running |
| Claude asks for a decision | Read the evidence in the report. Give more steps or time, change the target (edit `docs/ROADMAP.md` yourself and commit), or accept the result and move on |
| Claude wants to change SPEC or ROADMAP | Only you can. Edit the file and commit, or run `touch .claude/unlock`, let Claude edit, then `rm .claude/unlock` |
| "Blocked: ... is user-owned and locked" | That's the lock above working as intended |
| Render error mentioning EGL | Claude falls back to osmesa. For your own renders: `env MUJOCO_GL=osmesa uv run python scripts/render_snapshot.py ...` |
| `view.py` / `demo.py` / `watch.py` window won't open | These need a display, not EGL. Run `set -e MUJOCO_GL` in that terminal first. On Wayland, if it still fails: `env GLFW_PLATFORM=x11 uv run python scripts/view.py` |
| TensorBoard "port in use" | `uv run python scripts/tb.py --port 6007` |
| Machine sluggish during training | Expected: up to 14 of 16 threads are busy. Close heavy apps and don't start extra training |
| `uv sync` is slow or fails | Retry. If torch starts downloading a CUDA build (several GB), tell Claude: "torch is pulling CUDA, pin the CPU index" |
| Disk filling up | `du -sh runs`. Ask Claude to clean up, so it doesn't delete a run listed in `docs/best_runs.yaml` |
| A phase went badly and you want to redo it | `git log --oneline phase-N-start..HEAD` shows its commits. To roll back (Claude isn't allowed to): `git reset --hard phase-N-start`, then restart the phase |
| Results look wrong even though the gate passed | Say so plainly ("reach GIF overshoots"). Gates are minimums |
| New session doesn't remember the last one | Normal after `/clear`. Claude re-reads the last report's "Known issues" and `docs/decisions.md` |
| Want to stop a training run | Tell Claude "stop run <id>", or `pkill -f "train.py --config configs/<name>"` |

---

## 7. Where things live

| Path | What |
|---|---|
| `docs/SPEC.md`, `docs/ROADMAP.md` | Your locked design and targets |
| `docs/reports/phase-N.md` | One report per phase. Read this first |
| `docs/decisions.md` | Every choice Claude made, one line each |
| `docs/best_runs.yaml` | Which run counts for each phase and task |
| `docs/media/` | GIFs and PNGs used in reports and the README |
| `docs/results.md` | Experiment tables (Phase 4+), generated by script |
| `scripts/` | `train`, `eval`, `tb`, `check_phase`, `render_snapshot`, `watch`, `view`, later `demo`, `make_results` |
| `configs/` | One YAML file per experiment |
| `runs/<run_id>/` | Everything about one run: config, logs, model, results, video (git-ignored) |
| `runs/_logs/` | Full training logs |
| `User Ref/guide-status` | Script that rewrites section 0; runs after every Claude turn |
| `src/rlarm/` | The code: kinematics, environments, rewards, wrappers |
