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
   Exception, the **mid-run triage**: whenever a turn happens anyway (a user message, a goal check-in, a sibling job finishing) and a run over ~30 min is underway, spend one cheap call on it: read only the last few `val.json` history entries and `summary.json` steps of every running job (never the logs). Do not schedule extra wake-ups just for this.

## Mid-run triage: kill useless runs
- Only judge a run past ~30% of its budget and at least 5 validation points.
- Kill it if it is clearly worse than a sibling (same config family, other seed/variant) or than its own pilot at the same step, **and** its score is flat or falling over the last 3 validation points. A seed that is merely lower than its siblings but still rising is not clearly useless.
- A run that already plateaued above its gate (e.g. >90% for 3 validation points while the gate is lower) may be stopped early too, but only if a `model.zip` can still be produced: otherwise let it finish.
- Kill by PID from `ps -eo pid,args` (never `pkill -f` from the same shell: it matches itself and kills the tool call). Then log one line in docs/decisions.md (run, step, scores, reason) and commit; stopped runs are never used for results.
- Freed threads go to the next queued job. Update any watcher that waits for the killed runs.

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
