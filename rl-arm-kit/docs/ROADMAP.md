# ROADMAP: phases and acceptance checks

These thresholds are owned by the user, and this file is locked against Claude edits.
Each check has an ID. `scripts/check_phase.py <N>` must implement exactly these checks, print one line per ID with its measured value, and end with `PASS` or `FAIL`.
A phase is done when:
- its checks PASS
- `docs/reports/phase-<N>.md` exists
- the work is committed

Phase 4 contains no performance thresholds on purpose: its numbers are results to report, not gates to clear.

---

## Phase 0: Setup and warm-up

**Build:**
- uv project with Python 3.12, CPU-only torch, and all SPEC §3 dependencies
- .gitignore
- package skeleton `src/rlarm/`
- the `scripts/check_phase.py` framework
- generic `scripts/train.py` and `scripts/eval.py`:
  - config-driven
  - `train.py` supports `--dry-run`
  - `train.py` writes `runs/<id>/summary.json`
- `scripts/render_snapshot.py`
- a config that trains SAC on Gymnasium `Reacher-v5`, to prove the pipeline end to end

**Checks:**

| ID | Check |
|---|---|
| 0.1 | `import mujoco, gymnasium, stable_baselines3, torch` succeeds; versions are printed |
| 0.2 | `torch.get_num_threads()` is configured from config (not the default 16) |
| 0.3 | Headless render works: `render_snapshot.py` writes a non-blank PNG of Reacher (pixel std > 5) |
| 0.4 | `train.py --dry-run` on the Reacher config completes in < 120 s |
| 0.5 | SAC on Reacher-v5, ≤ 300k env steps: 100-episode mean return **≥ −7** (a random policy scores ≈ −43) |
| 0.6 | The run folder contains config copy, git hash, TensorBoard logs, model.zip, eval.json, summary.json, and a GIF |
| 0.7 | `pytest -q` and `ruff check .` pass |

---

## Phase 1: Arm model and kinematics

**Build:**
- MJCF for the RRR arm per SPEC §2:
  - both actuator variants
  - `tip` site
  - mocap `target`
  - floor
  - camera
- `kinematics.py`:
  - forward kinematics (DH-based)
  - closed-form inverse kinematics returning all solutions (elbow-up and elbow-down), or a clear "unreachable"
- joint limits, chosen and logged in decisions.md
- `scripts/view.py`, which opens the interactive viewer (for the user)

**Checks:**

| ID | Check |
|---|---|
| 1.1 | Model loads; exactly 3 hinge joints; site `tip` and mocap body `target` exist |
| 1.2 | FK vs MuJoCo `site_xpos['tip']` over 1000 random in-limit configs: max error **< 1e-6 m** |
| 1.3 | IK round trip over 1000 targets sampled via FK: every returned solution satisfies FK(IK) error **< 1e-6 m**, and the sampling configuration is among the solutions (within 1e-6 rad, modulo 2π) |
| 1.4 | Over 2000 random in-limit configs: zero arm–arm and arm–floor contacts |
| 1.5 | Position actuators hold 5 random poses under gravity: steady-state error **< 2°** after 2 s |
| 1.6 | Rendered PNG of the arm (non-blank), saved to docs/media/ |

---

## Phase 2: Environment

**Build:**
- goal-conditioned base env plus the `reach` task, per SPEC §4–§5
- all 4 action modes
- physics parameters from config
- every reward term logged
- IK scripted agent
- random agent

**Checks:**

| ID | Check |
|---|---|
| 2.1 | SB3 `check_env` passes for all 4 action modes |
| 2.2 | 5000 random steps per action mode: no NaN/Inf in obs or reward; obs stays within declared bounds |
| 2.3 | `compute_reward` (batched) equals per-step rewards to within 1e-6 |
| 2.4 | Same seed gives identical 200-step trajectories (bitwise obs equality) |
| 2.5 | IK scripted agent (`delta_pos`), 100 eval episodes: success **≥ 98%** |
| 2.6 | Random agent, 100 eval episodes: success **≤ 5%** (the task is not trivial) |
| 2.7 | Changing a link mass or damping in config changes the loaded `MjModel` value (test proves params flow from config) |
| 2.8 | Each reward term appears as its own key in `info` |

---

## Phase 3: Learned reach (task A) and hold (task B)

**Build:**
- SAC configs for reach and hold (`delta_pos`)
- the `hold` task: success requires the tip within 2 cm for the final 50 consecutive steps of a 250-step episode
- training via the `train` skill, using 3 seeds
- best runs recorded in docs/best_runs.yaml

**Checks** (each is the mean over 3 seeds, 100 eval episodes per seed, ≤ 1M env steps per run):

| ID | Check |
|---|---|
| 3.1 | Reach success **≥ 95%** |
| 3.2 | Reach median final distance **≤ 10 mm** |
| 3.3 | Hold success **≥ 90%** |
| 3.4 | Hold mean joint speed over the final 50 steps **< 0.05 rad/s** |
| 3.5 | Reach and hold GIFs in docs/media/ (each < 3 MB) |

---

## Phase 4: Experiments (results, not gates)

**Run** each at ≥ 3 seeds, with the same step budget within each comparison:
- **E1, action modes:** `delta_pos` vs `abs_pos` vs `torque` vs `torque_gravcomp`, using SAC on reach
- **E2, algorithms:** SAC vs PPO on reach with `delta_pos`
- **E3, sparse + HER:** SAC + HER with sparse reward vs SAC with shaped reward
- **E4, RL vs classical:** best RL policy vs IK controller on accuracy, steps to success, smoothness, and energy where applicable

**Checks:**

| ID | Check |
|---|---|
| 4.1 | Every run listed in `configs/experiments/` has an eval.json matching the fixed schema |
| 4.2 | `scripts/make_results.py` regenerates `docs/results.md` with no diff from the committed file |
| 4.3 | Learning-curve plots (mean ± std across seeds) for E1–E3 exist in docs/media/ |
| 4.4 | Each table in results.md states seeds, step budget, and the configs used |

---

## Phase 5: Harder tasks (C → D → E)

**Build** these as env subclasses, each with its own config and curriculum where needed:
- **track (C):** target moves on a circle (radius and speed *(tunable)*; documented)
- **obstacle (D):** a box between start and target; any arm–box contact counts as a collision
- **push (E):** a flat pusher pad on the tip, plus a free cube on the floor; goal = cube centre within 3 cm of a goal marker

**Checks** (each is the mean over 3 seeds, 100 eval episodes per seed):

| ID | Check |
|---|---|
| 5.1 | Track: mean tracking error after step 50 **≤ 25 mm** |
| 5.2 | Obstacle: success **≥ 85%** and collision rate **≤ 5%** |
| 5.3 | Push: success **≥ 60%** |
| 5.4 | One GIF per task in docs/media/ |

---

## Phase 6: Demo and polish

**Build:**
- `scripts/demo.py`, the live viewer: the user drags the mocap target and the trained policy follows (reach/hold policy)
  - `--headless` mode moves the target on a script, for testing
- README:
  - what it is
  - GIFs
  - results summary
  - how to run
- final results.md

**Checks:**

| ID | Check |
|---|---|
| 6.1 | `demo.py --headless --steps 2000` runs without error; mean tip–target distance in the second half **≤ 30 mm** |
| 6.2 | README contains these sections: Overview, Demo, Results, How to run, Project structure |
| 6.3 | Full `pytest -q` passes; `ruff check .` is clean |
| 6.4 | A fresh clone setup (`uv sync`, then the fast tests) succeeds in a temp directory |

---

## Phase 7 (optional): Robustness and transfer

**Build:**
- domain randomisation wrapper with these ranges *(tunable, logged)*:
  - mass ±30%
  - damping ±50%
  - actuator gains ±20%
  - link length ±10%
- UR5e from Menagerie with the wrist locked, selected by config

**Checks:**

| ID | Check |
|---|---|
| 7.1 | On a fixed perturbed eval set (50 sampled arms × 20 episodes), the randomisation-trained policy reaches **≥ 85%** success; the nominal policy's score on the same set is reported alongside |
| 7.2 | UR5e (locked wrist) reach success **≥ 90%**, using the same env code with only the config changed |
