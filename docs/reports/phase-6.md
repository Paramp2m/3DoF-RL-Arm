# Phase 6: Demo and polish

**Status:** PASS · **Date:** 2026-10-21 · **Commit:** tag `phase-6-done`

## What was built
- `scripts/demo.py`: live window where you drag the green target and the trained hold policy follows; `--headless` moves the target on a script and prints the mean distance. Settings in `configs/demo.yaml`.
- `README.md` (overview, demo, results, how to run, project structure) and a generated "Final task policies" section in `docs/results.md`.

## Checks
| ID | Check | Target | Measured | Result |
|---|---|---|---|---|
| 6.1 | headless demo, 2000 steps, mean tip–target distance in the second half | ≤ 30 mm | 14.7 mm | PASS |
| 6.2 | README sections Overview, Demo, Results, How to run, Project structure | all 5 | all 5 | PASS |
| 6.3 | full `pytest -q` + `ruff check .` | clean | 86 passed, ruff clean | PASS |
| 6.4 | fresh clone (`uv sync` + fast tests) in a temp dir | succeeds | 83 passed, 3 slow deselected | PASS |

## Key numbers
- Demo policy: `runs/2026-10-18_hold_final_s2` (hold, SAC + HER, 1M steps, 98% eval success). No new training this phase.
- Headless script: target glides between random reachable points at ≤ 0.5 rad/s joint speed, pausing 2 s at each. Mean distance 11.5 mm over all 2000 steps, 14.7 mm over the second half; the demo takes ~8 s to run.
- Full test suite: 107 s. Fresh-clone check: ~50 s for sync + tests (uv cache warm).

## Look at this
- ![](../media/phase6_demo.gif) The arm following the scripted target for 16 s: does it look like it "follows" rather than lags or jitters?
- Try the live demo yourself: `uv run python scripts/demo.py`, double-click the ball, then Ctrl+right-drag it.
- [README.md](../../README.md): is it the summary you'd show someone?

## How it works (say this in an interview)
The demo reuses the hold policy unchanged. Every control step (25 Hz) it reads where the target ball is, writes that into the policy's goal input, and applies the policy's action (a small change of each joint setpoint). The policy was trained on still targets, but since it re-reads the goal every step, a slowly moving target is just a sequence of nearby still targets. It lags a moving target by about a centimetre and closes the gap when the target stops.

## Decisions made
See docs/decisions.md, lines dated 2026-10-21 with phase 6: demo policy choice, headless target script, how 6.4 clones, the results.md section. 

## Known issues / next
- `runs/` is not in git, so a fresh clone cannot run the demo, eval or the run-reading checks until it trains (the README says so). Shipping one small `model.zip` would fix that.
- The demo uses the reach/hold policy only; track, obstacle and push have no interactive mode.
- The live window was opened once for 250 steps (10 s) to confirm it starts, runs and closes; the arm reached the still target (3.8 mm in the second half). Dragging by mouse was not tested by Claude: please try it.
- The headless script is gentler than a fast human drag (smooth, ≤ 0.5 rad/s); a quick yank will show more lag.
- The README's numbers table is copied from results.md; results.md is the source of truth.
