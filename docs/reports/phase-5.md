# Phase 5: Harder tasks (track, obstacle, push)

**Status:** DRAFT (gate pending: g2 seed-2 eval, regression gates, review) · **Date:** 2026-10-20 · **Commit:** <hash>

## What was built
Three env subclasses on a generalised base env (arm DOF sliced, task hooks): **track** (target circling r = 8 cm, T = 8 s), **obstacle** (box across the start→target line, contact-safe action filter, planner waypoint + learned residual), **push** (pusher pad + free 5 cm cube, push-only arm variant with elbow limit −2.2 rad).
Reward terms added: `w_smooth` (no chatter once settled), `w_reach` (push), `w_collision`, `w_clear`, `w_block`, `w_wp` (obstacle). Eval schema gained `track_err_mm_mean`, `collision_rate`.
Training tooling: `val.early_stop` (plateau → finalise and keep, stuck → unusable), `STOP` file, `scripts/babysit.py`, honest dry-run ETA.

## Checks
| ID | Check | Target | Measured (3 seeds) | Result |
|---|---|---|---|---|
| 5.1 | Track error after step 50 | ≤ 25 mm | 3.7 mm [3.5, 3.9, 3.9] | PASS |
| 5.2 | Obstacle success / collisions | ≥ 85% / ≤ 5% | s0 86% / 4%, s1 88% / 1%, s2 pending | pending |
| 5.3 | Push success | ≥ 60% | 92.7% [90, 94, 94] | PASS |
| 5.4 | 3 GIFs < 3 MB | 3 files | track 2.34, push 2.88, obstacle (g2) pending | pending |

## Key numbers
- track: `track_smooth` s0–2, 300k steps, ~1.4 h each; IK baseline 5.6 mm, branch-aware scripted oracle 4–5 mm.
- push: `push_smooth` s0–2, 1M steps, 3.4–4.6 h each; scripted IK push lands the cube within 1.6 cm in 8/10 seeds; the 2M runs (98%) plateaued by ~1M.
- obstacle: `obstacle_g2` s0–2, 400–450k steps (plateau stop). **Planner follower alone (zero residual, same filter): 89% success, 3% collisions.** Pure RL best (f1, safe filter): 72% success, 2% collisions (detour episodes 61%, direct 85%).
- jitter (mean action change once settled): track 0.26 → 0.02, push 0.49 → 0.13 with `w_smooth`.

## Look at this
- ![](../media/phase5_obstacle.gif) the arm goes round the box without touching it (red box, green target)
- ![](../media/phase5_push.gif) the round pad pushes the yellow cube onto the green floor marker
- ![](../media/phase5_track.gif) the tip follows the moving target without chatter

## How it works (say this in an interview)
The agent sees joint angles/speeds, the tip, the goal (and per task: target velocity, cube state, box pose and arm-to-box vectors, the next planner waypoint) and outputs joint-angle steps.
Track and push use SAC (push with HER) and a shaped reward plus an action-smoothness penalty.
Obstacle: pure RL learned to avoid touching the box only once a contact-safe filter took collisions off the table (2%), but plateaued at ~72% success because it could not find detours; a joint-space planner gives a collision-free waypoint path, a follower tracks it, and SAC learns a residual correction on top.

## Decisions made
30 lines tagged `| 5 |` in [docs/decisions.md](../decisions.md): D-1 push arm variant (elbow −2.2), schema, track design, each obstacle round (a1–a3, b, c, d, e, f, g) with its diagnosis, `w_smooth`, run-triage rules.

## Known issues / next
- **5.2 depends on privileged information.** The planner and the safe filter read the box's true pose from the simulator; the planner alone already passes 5.2 (89% / 3%). The learned residual adds a little on top. Pure RL without the planner tops out at ~72% success.
- Obstacle starts require ≥ 5 cm arm-box clearance and a plannable path (~1/60 episodes rejected); stricter than the ROADMAP wording.
- The collision metric was made stricter mid-phase (end-of-step contact refresh); earlier obstacle runs were not re-scored.
- Push uses a model variant whose forearm can touch the floor (SPEC §2 invariant waived for push only).
- Pure-RL obstacle with a bigger network was stopped (~27 h at 15 steps/s); a GPU or a longer budget is the next step for it.
