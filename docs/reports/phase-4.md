# Phase 4: Experiments

**Status:** PASS · **Date:** 2026-10-18 · **Commit:** see `git log -1 phase-4-done`

## What was built
Four experiment manifests (`configs/experiments/`), each listing a config and seeds for every arm. A queue (`scripts/phase4_runs.sh`) ran 18 training runs. `scripts/make_results.py` generates `docs/results.md` and the learning-curve plots from the eval.json and val.json files, so no number is typed by hand. `train.py` now also handles PPO with 8 workers (`algo.n_envs`) and the IK baseline (`agent: ik`).

## Checks
| ID | Check | Target | Measured | Result |
|---|---|---|---|---|
| 4.1 | experiment runs have eval.json in the fixed schema | all | 28/28 valid | PASS |
| 4.2 | make_results.py regenerates results.md with no diff | no diff, committed | up to date | PASS |
| 4.3 | E1–E3 learning curves (mean ± std over seeds) in docs/media | 3 | 3, newer than their val.json | PASS |
| 4.4 | each table states seeds, step budget, configs | 4 tables | 4/4 | PASS |

## Key numbers
Reach task, 100 fixed eval goals, success = tip within 2 cm. E1–E3: 3 seeds, 300k steps each, success as mean ± sample std.
- **E1 action modes:** delta_pos 70.0 ± 11.5% > torque_gravcomp 66.0 ± 9.6% > torque 45.7 ± 14.0% > abs_pos 31.7 ± 1.5%. The torque modes reach the target fastest (8–10 steps, against 36 for delta_pos), but they are jerkier (‖Δa‖ 0.43–0.50, against 0.18 for delta_pos). Gravity compensation adds 20 points over plain torque.
- **E2 algorithms:** SAC 70.0 ± 11.5% vs PPO 49.3 ± 2.3%. PPO trains about 14× faster in wall time (8.6 min vs about 2 h per run), so at equal wall time PPO could see far more steps. That was not tested.
- **E3:** SAC + HER with the sparse reward reached 85.0 ± 5.3% (median 8.4 mm), beating SAC with the shaped reward (70.0%, median 13.5 mm).
- **E4:** RL (reach_final, 1M steps) 96%, median 5.7 mm, 32.5 steps to success; IK 100%, 0.0 mm, 34 steps. IK is about 28× smoother (‖Δa‖ 0.017 vs 0.48).
- **Wall time:** 12 SAC jobs in parallel took about 2 h; the 3 HER runs took 1.2 h; the 3 PPO runs ran one after another, 8.6 min each.

## Look at this
- ![](../media/e1_action_modes_curves.png) Is the action-mode ranking stable? Note that every curve is still rising at 300k.
- ![](../media/e3_her_sparse_curves.png) Check how early HER pulls ahead of the shaped reward.
- ![](../media/e2_algorithms_curves.png) Check whether PPO has flattened out or is just slow per step.

## How it works (say this in an interview)
Every experiment changes one factor against a shared baseline. That baseline is plain SAC on reach with delta_pos actions and the Phase 3 shaped reward, at 300k steps and 3 seeds. Each run keeps its best checkpoint, chosen on separate validation goals, and is scored on the same 100 fixed eval goals. A manifest lists which runs belong to each experiment. One script turns their eval.json files into the tables and plots, and the gate re-runs that script to prove results.md matches the data. The IK controller solves the arm geometry in closed form and walks the joints there, which is why it is near-perfect: reach has no dynamics surprises for it.

## Decisions made
The four `2026-10-18 | 4 |` lines at the end of docs/decisions.md cover: the experiment design and budget, the manifests and make_results, the results summary, and the review fixes.

## Known issues / next
- **300k steps ranks learning speed, not final performance.** Every E1–E3 curve is still rising. With a 1M budget, plain SAC on delta_pos reached 88% in Phase 3.
- **Energy is reported for the torque modes only** (the SPEC §8 schema). That leaves E4 without an energy comparison.
- **The E4 comparison is uneven by design.** The IK arm has one run (it is deterministic) and a 0-step budget. The RL arm uses HER and 1M steps.
- **The E1 runs were launched from a dirty tree.** Commit 9f2eddc holds the identical code, made seconds after launch.
- No GIF was made this phase. The three plots use the three-PNG media limit.
