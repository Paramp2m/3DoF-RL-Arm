# Phase 3: Learned reach (task A) and hold (task B)

**Status:** FAIL (3.1 short by 0.7 pt; 3.2–3.5 PASS) · **Date:** 2026-10-18 · **Commit:** 7036dea

## What was built
`hold` task (success = tip within 2 cm for the final 50 of 250 steps), SAC + HER configs for both tasks (`configs/reach_her.yaml`, `configs/hold_her.yaml`), a goal-difference feature extractor, and the SPEC §5 settle term. There were 3 seeds × 1M steps per task, and checks 3.1–3.5.

## Checks
| ID | Check | Target | Measured (mean of 3 seeds) | Result |
|---|---|---|---|---|
| 3.1 | Reach success | ≥ 95% | 94.3% (96 / 91 / 96) | FAIL |
| 3.2 | Reach median final distance | ≤ 10 mm | 6.0 mm (6.1 / 5.5 / 6.5) | PASS |
| 3.3 | Hold success | ≥ 90% | 92.3% (94 / 92 / 91) | PASS |
| 3.4 | Hold joint speed, final 50 steps | < 0.05 rad/s | 0.0156 rad/s | PASS |
| 3.5 | GIFs in docs/media | < 3 MB each | reach 1.49 MB, hold 1.75 MB | PASS |

## Key numbers
- Runs: `runs/2026-10-17_{reach,hold}_her_s{0,1,2}`. Each has 1M steps and took about 4.6 h of wall time with 7 jobs in parallel (≈60 steps/s each).
- How reach success changed (seed 0, deterministic eval):

  | Setup | Steps | Success | Median distance |
  |---|---|---|---|
  | Baseline dense reward | 300k | 43% | 25 mm |
  | Tip→target vector scaled ×30 | 300k | 76% | 9.8 mm |
  | Same, dense reward | 1M | 88% | |
  | HER, no distance term | 1M | 96% | |

- 8 hyperparameter variants all landed at 73–76%: entropy target, gradient steps, learning rate, discount, batch size, Tanh activation, wider network, initial entropy.

## Look at this
- ![](../media/phase3_reach.gif) Reach, seed 0: does the tip settle on the red target without drifting?
- ![](../media/phase3_hold.gif) Hold, seed 0: does the arm arrive and then stay still?

## How it works (say this in an interview)
The agent sees the joint angles (sin/cos), velocities, tip position and goal. Inside the network, the tip→goal difference is multiplied by 30 so that millimetre errors are visible. It outputs 3 joint-angle increments (≤ 0.05 rad/step). Its reward is a precision bump exp(−(d/3 cm)²) plus +1 inside 2 cm, and hold also has a speed penalty inside the zone.

There is deliberately no "−distance" term. Greedy descent on distance gets stuck at a joint limit in 27% of episodes, and that was the 75% ceiling. HER replays every episode as if the place it ended up had been the goal, so the critic learns the real way around the limits.

## Decisions made
See docs/decisions.md, lines dated 2026-10-17 / 2026-10-18 with phase 3: the hold definition, the 3.4 metric (‖q̇‖ L2), err_scale and the dead-end diagnosis, the HER plan (user-approved), and the final results.

## Known issues / next
- **3.1 is 0.7 pt short.** 17/300 reach episodes fail. 12 of them end pinned at a joint limit (yaw 3, shoulder 7, elbow 7) and 8 end 2–4 cm out. Training success has been flat since about 400k, so more steps under the 1M cap won't fix it. The candidate fix is SPEC §5's joint-limit term for reach, which needs a ~3 h retrain of the 3 reach seeds.
- Reach without the settle term moves at 0.10 rad/s at the end. This is not gated, but it matters for Phase 4 smoothness comparisons.
- Dry-run ETAs underestimate parallel wall time by about 4× because CPU is shared, and the ETA is measured before learning starts.
- The exploratory configs `configs/reach_sac_*.yaml` are kept because decisions.md references their results.
