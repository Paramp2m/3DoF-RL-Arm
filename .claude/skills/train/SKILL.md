---
name: train
description: Launch and summarise rl-arm training jobs (anything over ~5 min) without spending tokens while they run. Use whenever starting SB3 training or checking on runs.
argument-hint: "[launch <config> [seeds] | status]"
---

## Contract the scripts must keep (build or fix them if they don't)
- `scripts/train.py --config C [--seed S] [--dry-run]`:
  - prints at most 5 lines
  - writes `runs/<run_id>/summary.json` containing:
    - `status`: running / done / crashed
    - `steps`, `wall_s`, `steps_per_s`
    - `final_eval`: success, median_mm, return
    - `best_checkpoint`
  - updates `summary.json` every ~10% of training, so `status` works mid-run
  - `--dry-run` builds the env, runs ~2k steps and prints `steps_per_s` plus an ETA for the full budget
- Full stdout/stderr goes to `runs/_logs/<run_id>.log`, never to the conversation.

## launch
1. Run `uv run python scripts/train.py --config <C> --dry-run`. If it fails, fix it now. This is the cheap place to catch crashes.
2. Thread budget: 16 hardware threads, keep at most 14 busy.
   - SAC run with 1 env: 2 threads
   - PPO with 8 SubprocVecEnv workers: 9 threads
   - Run seeds in parallel when they fit; otherwise queue them in one background shell script.
3. Start each job with the Bash tool using `run_in_background: true`:
   `uv run python scripts/train.py --config <C> --seed <S> > runs/_logs/<name>-s<S>.log 2>&1`
4. Tell the user one line: run ids, plus the ETA from the dry run.
5. **End the turn.** Do not sleep, poll, tail or re-check. Claude Code delivers the job's completion as a new turn.
   Exception: the run triage below (automatic `early_stop` in every config, plus a cheap `val.json` peek whenever a turn happens anyway).

## Run triage: automatic first, then by hand
**1. Automatic (every new config).** Put `early_stop` in the config's `val:` block:
`val: {every: .., episodes: .., seed: .., early_stop: {min_frac: 0.3, min_points: 5, patience: 6, delta: 0.01, floor: <gate-ish score>}}`.
`train.py` then ends the run itself (rule in `rlarm.common.early_stop_reason`, tested):
- `plateau`: best val score has not improved by `delta` for `patience` val points, after >= `min_frac` of the budget and `min_points` val points. The run is **finalised, not discarded**: val-best becomes `model.zip`, final eval runs, `summary.json` gets `stopped_early: plateau`, the real step count and `usable: true`. A plateau at a good score is a finished run, so use it.
- `stuck_low`: best score still under `floor` and stalled for >= patience/2 val points. Also finalised (so the evidence exists) but `usable: false`: never put it in best_runs.yaml or results.
- `manual`: `touch runs/<id>/STOP` ends a run at its next 5000-step boundary with the same finalisation (val-best saved, eval run, `stopped_early: manual`). Use it instead of `kill` whenever the run is worth keeping (good plateau you spotted before the rule fired).
- Choose `floor` per task from the gate: ~half of it for success-type scores (obstacle uses success minus collision rate, floor 0.45; reach-type 0.5). `patience` 6 and `delta` 0.01 suit val points every 5% of the budget; with sparser val points use patience 4. Use `val.every` <= total_steps/20 so there are enough points.

**2. By hand (mid-run triage).** Whenever a turn happens anyway (user message, goal check-in, a sibling finishing) and a run over ~30 min is underway, spend one cheap call: read only the last 5 `val.json` history entries and the steps of each running job (never logs). Do not schedule extra wake-ups for it. Decide per run:
- Running, rising, or within ~3 val points of its best: leave it.
- Plateaued at a good score (>= gate or within 0.05 of the best sibling) but the auto rule has not fired yet: `touch STOP` (saves it). Do not `kill`: that leaves no `model.zip`.
- Clearly useless, i.e. score under the floor (e.g. < 0.45) or far below a sibling or its pilot at the same step, **and** flat or falling over the last 3 points, past ~30% of budget: `touch STOP` if you may want the evidence, else kill by PID from `ps -eo pid,args` (never `pkill -f` from the same shell: it matches itself and kills the tool call). Fix `touch STOP` first when in doubt: it finalises.
- Several seeds of one config where one is lagging but still rising: leave it. Where **all** seeds of a variant are below floor: stop the variant (all seeds), keep the siblings.
- Never judge before ~30% of budget or 5 val points. Never stop a run only because it is lower than a sibling while still rising.
- Always: log one line in docs/decisions.md (run, step, score trajectory, rule, action), commit, free the threads for the next queued job, and update any watcher waiting on the stopped runs. Stopped-useless runs are never used for results.

## On completion
- Read only `runs/<id>/summary.json`.
- If `status` is crashed: run `tail -n 40 runs/_logs/<id>.log`, fix the cause, and relaunch once.
- If finished and good: run `uv run python scripts/eval.py --run runs/<id>`, then update docs/best_runs.yaml if it beats the current entry.
- If finished but under target, diagnose from data, not by rereading the code:
  - grep the per-term reward scalars from the TensorBoard export
  - check success vs steps
- Change one thing, log it in decisions.md, relaunch.

## status
List `runs/*/summary.json` as a compact table (id, status, steps, success, median_mm). Nothing else.
