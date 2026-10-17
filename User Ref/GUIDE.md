# rl-arm: the full journey, phase by phase

How to drive the project from an empty folder to a trained arm you can drag around: what you type, what Claude does each turn, what you get back, how to poke at the arm along the way, and what to do when something goes wrong.

All commands are **fish** and run from the project root (`cd ~/Projects/Claude/rl-arm`). Use a **second kitty window** for anything you run yourself, so Claude's session stays untouched.

---

## 1. The loop (same for every phase)

### Start a phase
```fish
cd ~/Projects/Claude/rl-arm
systemd-inhibit --what=idle:sleep claude --permission-mode auto
```
Then paste the following, with N set to the phase number:
```
/goal Phase N is finished: `uv run python scripts/check_phase.py N` printed PASS in this session, docs/reports/phase-N.md exists, and the work is committed. Use the phase skill for phase N. Stop after 40 turns, or earlier if you need a decision from me.
```

### What happens turn by turn
| Step | What Claude does | What you see |
|---|---|---|
| 1. Orient | Reads only this phase's ROADMAP section plus the SPEC parts it needs; tags `phase-N-start` | A short task list |
| 2. Checks first | Adds the phase's checks to `scripts/check_phase.py` and runs them once, expecting FAIL | A few `FAIL` lines. This is normal |
| 3. Build | Writes code plus a unit test per piece. Logs non-obvious choices in `docs/decisions.md` | Short progress messages. The stop hook may make Claude fix failing tests before it stops |
| 4. Train (phases 0, 3, 4, 5, 7) | Runs a dry run (prints an ETA), then launches the real runs in the background, **and ends its turn** | One line: run ids + ETA. Then silence. **This is expected.** Claude wakes up by itself when a run finishes |
| 5. Gate | `check_phase.py N --summary`. Diagnoses any failure, changes one thing, retrains | PASS/FAIL table |
| 6. Review | One Sonnet subagent checks the diff against ROADMAP/SPEC; Claude fixes what it finds | A short note |
| 7. Report + commit | Writes `docs/reports/phase-N.md`, copies media to `docs/media/`, commits, tags `phase-N-done` | **Final message, at most 8 lines:** PASS/FAIL per check ID with values, the report path, and 1–3 things to look at |

A desktop notification (`notify-send`) fires whenever Claude needs you.

### After the phase
1. Read `docs/reports/phase-N.md`, especially "Checks", "Look at this" and "How it works".
2. Open the GIF/PNG it points to.
3. Play with the arm (see §3).
4. Ask for fixes in plain words, e.g. "hold GIF looks jittery, fix it".
5. Run `/clear`, then start the next phase.

---

## 2. The phase gate: `scripts/check_phase.py`

This file is at `scripts/check_phase.py` in the project root. Claude creates it in Phase 0, and each phase adds its own checks. It only reads existing results (`docs/best_runs.yaml` plus run folders). It may run a quick eval, but it never trains.

```fish
uv run python scripts/check_phase.py 0            # full gate
uv run python scripts/check_phase.py 0 --summary  # compact table
```
Output is one line per check ID (`ID PASS|FAIL | measured value`), then a final verdict:
```
0.1 PASS | mujoco 3.x, gymnasium 1.x, sb3 2.x, torch 2.x
0.2 PASS | torch threads = 4
0.3 PASS | pixel std = 41.2
0.4 PASS | dry run 38 s
0.5 PASS | mean return -4.1 (100 eps)
0.6 PASS | all artifacts present
0.7 PASS | pytest ok, ruff clean
PASS
```
The exit code is 0 on PASS and 1 on FAIL. A check that crashes counts as FAIL and shows `error: ...`. You can run it yourself at any time; it's safe.

---

## 3. Interacting with the arm

| Available from | What | Command |
|---|---|---|
| Phase 0 | Monitor learning live (curves, reward terms). Every chart is explained in `User Ref/TENSORBOARD.md` | `uv run tensorboard --logdir runs`, then open http://localhost:6006 |
| Phase 0 | Run status (steps, success, ETA) | `cat runs/*/summary.json` (each run updates it every ~10%) |
| Phase 0 | Render a still image of the env | `env MUJOCO_GL=egl uv run python scripts/render_snapshot.py --config configs/<name>.yaml --out /tmp/snap.png` |
| Phase 0 | GIF of a finished run | open `runs/<run_id>/video.gif` |
| **Phase 1** | **Interactive viewer:** rotate, zoom, drag the target (double-click it, then Ctrl + right-drag), move joints with the sliders in the right panel | `uv run python scripts/view.py` |
| Phase 2 | Watch the IK scripted agent drive the arm | Ask Claude: "add a way for me to watch the IK agent in the viewer" |
| **Phase 6** | **Live demo:** drag the target, and the trained policy follows | `uv run python scripts/demo.py` |

**Watching a policy before Phase 6.** `demo.py` only arrives in Phase 6. If you want to watch a trained or half-trained policy in the viewer from Phase 3 onward, start Phase 3 with this request: *"Also add `scripts/watch.py --run runs/<id> [--checkpoint latest]` that opens the viewer and runs that policy with a draggable target."* It's a small script, and Phase 6's demo can reuse it.

**Mid-phase rules (keep Claude's work clean):**
- Read-only things are always safe: TensorBoard, `summary.json`, the GIFs, `view.py`, `check_phase.py`.
- Don't run `scripts/eval.py` on a run that is still training. It rewrites that run's `eval.json` and GIF.
- Don't run `git commit`, `git checkout` or `git reset` while a phase is running.
- The viewer uses a little CPU, which only slightly slows training.

---

## 4. Phase by phase

### Phase 0: Setup and warm-up (~1 h)
- **Built:** uv project (Python 3.12, CPU-only torch), `src/rlarm/`, `scripts/{train,eval,render_snapshot,check_phase}.py`, `configs/reacher_sac.yaml`.
- **Training:** SAC on Gymnasium's built-in 2-joint Reacher, ≤300k steps (~30–60 min). This is a stand-in benchmark, not our robot: our 3-DOF arm is built in Phase 1.
- **Gate:** imports work; torch threads come from config; non-blank render; dry run under 120 s; Reacher return ≥ −7 (random ≈ −43); run folder complete; tests + lint pass.
- **You look at:** the Reacher GIF. The fingertip should go straight to the dot and stop.
- **This proves:** the whole train → eval → GIF → gate pipeline works before our own robot exists.

### Phase 1: Arm model and kinematics (no training)
- **Built:** the MJCF arm in `models/arm/` (base yaw → shoulder pitch → elbow pitch, capsule links, `tip` site, mocap `target`, floor, camera); FK and closed-form IK (elbow-up and elbow-down) in `src/rlarm/kinematics.py`; joint limits logged in decisions.md; `scripts/view.py`.
- **Gate:** FK matches MuJoCo to <1e-6 m; IK round-trips to <1e-6 m; zero self- or floor-collisions over 2000 random poses; position actuators hold poses to <2°; arm PNG.
- **You do:** run `view.py`. Move the joint sliders and check that nothing clips into the floor or itself, and that the arm looks like an arm.
- **Expect:** the joint limits are narrower than you might think, because every in-limit pose must stay clear of the floor.

### Phase 2: Environment (no training)
- **Built:** goal-conditioned env (`reach` task), 4 action modes (`delta_pos`, `abs_pos`, `torque`, `torque_gravcomp`), physics parameters from YAML, each reward term in `info`, IK scripted agent, random agent.
- **Gate:** SB3 `check_env` passes for all 4 modes; no NaN/Inf; batched reward matches per-step reward; same seed gives identical trajectories; IK agent ≥98% success; random agent ≤5%; a config change reaches the MuJoCo model; each reward term logged separately.
- **You look at:** the IK agent vs random agent numbers. This proves the task can be solved and isn't trivial, before any RL time is spent.

### Phase 3: Learned reach and hold (~3–5 h of training)
- **Built:** SAC configs for `reach` and `hold`, the `hold` task (tip within 2 cm for the final 50 of 250 steps), 3 seeds each, `docs/best_runs.yaml`.
- **Gate (mean of 3 seeds × 100 episodes):** reach success ≥95%, median final distance ≤10 mm; hold success ≥90%, joint speed in the final 50 steps <0.05 rad/s; both GIFs under 3 MB.
- **While waiting:** TensorBoard. Watch `success` rise and the reward terms. Each run is ≤1M steps; up to ~6 run in parallel.
- **You look at:** do the reach and hold GIFs move smoothly? Does the arm settle, or does it jitter?
- **Expect:** precision (≤10 mm) and stillness may take an extra reward term or two (precision, settle). Claude adds them one at a time and logs each one.

### Phase 4: Experiments (~10–15 h; best run overnight)
- **Runs (≥3 seeds each):** E1 action modes (4 modes, SAC); E2 SAC vs PPO; E3 SAC+HER sparse vs SAC shaped; E4 best RL vs IK controller (accuracy, steps, smoothness, energy).
- **Built:** `configs/experiments/`, `scripts/make_results.py` → `docs/results.md`, learning-curve plots (mean ± std).
- **No performance thresholds.** The gate only checks that every run has an eval, that results.md is regenerated from data (no hand-typed numbers), and that the plots and tables are complete.
- **You read:** `docs/results.md` and the plots. This is the "what I learned" section for your CV or an interview.

### Phase 5: Harder tasks: track → obstacle → push (~10 h+)
- **Built:** three env subclasses with configs and curricula. **Track:** the target moves on a circle. **Obstacle:** a box between start and target; touching it counts as a collision. **Push:** a flat pad on the tip pushes a cube to a goal marker.
- **Gate:** tracking error ≤25 mm after step 50; obstacle ≥85% success with ≤5% collisions; push ≥60% success; one GIF per task.
- **You look at:** 3 GIFs.
- **Expect pauses:** push is the hardest gate in the project (3 joints, no gripper). If Claude can't reach it after 3 honest attempts, it stops and shows you the evidence. You then decide whether to change the threshold, the task or the time budget.

### Phase 6: Demo and polish (~1 h)
- **Built:** `scripts/demo.py` (drag the target, the policy follows; `--headless` for tests), README (Overview, Demo, Results, How to run, Project structure), final results.md, fresh-clone test.
- **Gate:** headless demo with mean distance ≤30 mm; README sections present; all tests and lint pass; fresh clone + fast tests work in a temp folder.
- **You do:** run `uv run python scripts/demo.py`, drag the target around (including fast moves and edges of the workspace), and say if it feels right.

### Phase 7 (optional): Robustness and transfer
- **Built:** domain-randomisation wrapper (mass ±30%, damping ±50%, actuator gains ±20%, link length ±10%); UR5e from MuJoCo Menagerie with its wrist locked, selected by config only.
- **Gate:** randomisation-trained policy ≥85% on 50 perturbed arms × 20 episodes (nominal policy reported alongside); UR5e reach ≥90%.
- **You decide** whether to do it at all.

---

## 5. Things to do while waiting

**Watch:**
- TensorBoard: `success`, `ep_rew_mean` and every reward term. Learn what a healthy curve looks like.
- `docs/decisions.md` (`tail docs/decisions.md`): what Claude chose and why.

**Understand:**
- Read the previous phase's "How it works (say this in an interview)" section aloud. If you can't explain it, ask Claude in a fresh session.
- Read `docs/SPEC.md` §4–§5 (observation, action, reward) before Phase 3. That's the heart of the project.
- Background topics, one per phase: forward/inverse kinematics (Phase 1); goal-conditioned envs and HER (Phase 2); SAC and entropy (Phase 3); PPO vs SAC, on- vs off-policy (Phase 4); curriculum learning and reward shaping (Phase 5); domain randomisation and sim-to-real (Phase 7).

**Plan ahead:**
- Think about whether you want to change any SPEC/ROADMAP item for the next phase. Only you can edit those files; do it between phases and commit.
- Start a "lessons learned" note for the README or your CV.

Avoid anything heavy on the CPU (gaming, compiling) during long training: it slows runs and can push ETAs past what Claude planned for.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| Claude went quiet after launching training | Expected. It wakes up when a run finishes. Check progress with TensorBoard or `cat runs/*/summary.json`. Don't type into the session unless you want to interrupt |
| Laptop slept or rebooted mid-training | Runs died, and `summary.json` will still say `running`. Start Claude again (with `systemd-inhibit`), re-paste the `/goal`, and say "runs were killed by sleep, relaunch them". Always start with `systemd-inhibit` |
| `/goal` stopped at the 40-turn cap | Re-paste the same `/goal`. Work and commits are kept |
| Hit the usage limit mid-phase | The goal pauses. After the reset, re-paste the `/goal`. Background training keeps running in the meantime |
| Claude asks for a decision (3 failed attempts) | Read the evidence in the report. Choose: give more steps/time, change the threshold (edit ROADMAP yourself and commit), or accept the result and move on |
| Claude wants to change SPEC/ROADMAP | Only you can. Either edit the file yourself and commit, or `touch .claude/unlock`, let Claude edit, then `rm .claude/unlock` |
| "Blocked: ... is user-owned and locked" | The lock hook is working. See the row above |
| Render error / "EGL" in an error message | Claude should fall back to osmesa and log it. If you're rendering yourself: `env MUJOCO_GL=osmesa uv run python scripts/render_snapshot.py ...` |
| `view.py` / `demo.py` window won't open | The interactive viewer needs a display, not EGL: run `set -e MUJOCO_GL` in that terminal first. On Wayland, if it still fails: `env GLFW_PLATFORM=x11 uv run python scripts/view.py` |
| TensorBoard: "port in use" | `uv run tensorboard --logdir runs --port 6007` |
| Machine sluggish during training | Expected (up to 14 of 16 threads busy). Close heavy apps; don't start extra training yourself |
| `uv sync` fails or is very slow | Network issue: retry. If torch tries to download a CUDA build (several GB), tell Claude: "torch is pulling CUDA, pin the CPU index" |
| Disk filling up | `du -sh runs`. Old failed runs can go, but ask Claude to clean up so it doesn't remove a run listed in `docs/best_runs.yaml` |
| A phase went badly and you want to redo it | Every phase is tagged: `git log --oneline phase-N-start..HEAD` shows its commits. To roll back (Claude isn't allowed to): `git reset --hard phase-N-start`, then restart the phase |
| Results look wrong even though the gate passed | Say so plainly ("reach GIF overshoots", "hold looks shaky"). Gates are minimums, not the definition of good |
| New session doesn't remember the last one | Normal after `/clear`. Claude re-reads the previous report's "Known issues" and `docs/decisions.md` at the start of each phase |
| Want to stop a training run | Tell Claude "stop run <id>", or kill it yourself: `pkill -f "train.py --config configs/<name>"` |

---

## 7. Where things live

| Path | What |
|---|---|
| `docs/SPEC.md`, `docs/ROADMAP.md` | Your locked design and targets |
| `docs/reports/phase-N.md` | One report per phase: read this first |
| `docs/decisions.md` | Every choice Claude made, one line each |
| `docs/best_runs.yaml` | Which run counts for each phase/task |
| `docs/media/` | GIFs and PNGs for reports and README |
| `docs/results.md` | Experiment tables (Phase 4+), generated by script |
| `scripts/` | `train`, `eval`, `check_phase`, `render_snapshot`, later `view`, `demo`, `make_results` |
| `configs/` | One YAML per experiment |
| `runs/<run_id>/` | Config copy, git hash, TensorBoard logs, `model.zip`, `eval.json`, `summary.json`, `video.gif` (git-ignored) |
| `runs/_logs/` | Full training logs |
| `src/rlarm/` | The package: kinematics, envs, rewards, wrappers |
