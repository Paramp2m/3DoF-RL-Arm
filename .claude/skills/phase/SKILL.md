---
name: phase
description: Run one roadmap phase of rl-arm end to end. It builds the phase, gates it with check_phase.py, reviews, reports and commits. Use for any work on a phase in docs/ROADMAP.md.
argument-hint: "[phase-number]"
---

Run phase **$0** of docs/ROADMAP.md autonomously. The user judges the outputs, not the code.

## 1. Orient (cheap)
- Read only the `## Phase $0` section of docs/ROADMAP.md (grep for the heading).
- Read only the SPEC sections that phase relies on.
- If `docs/reports/phase-<$0 minus 1>.md` exists, read only its "Known issues" section.
- Run `git tag -f phase-$0-start`.
- Create a task list: one task per Build item, then "checks", "review", "report", "commit".

## 2. Checks first
- Add the phase's checks to `scripts/check_phase.py`, exactly as the ROADMAP lists them: same IDs, same thresholds, one printed line per ID with its measured value.
- Run the checks once and confirm they FAIL for the right reason.
- Checks only read artifacts: `docs/best_runs.yaml` plus run folders. They may run quick evals but never train.

## 3. Build
- Work in small steps. Write a unit test for each piece of logic as you go.
- Every new behaviour switch goes into config.
- For training or any job over ~5 min, follow the `train` skill. After launching, end your turn; the completion arrives as a new turn.
- Log every non-obvious choice in docs/decisions.md as one line.

## 4. Gate
- Run `uv run python scripts/check_phase.py $0`.
- If a check fails, diagnose it before changing anything.
- Three genuine attempts at the same failing check without progress means stop. Write the report with the evidence (numbers, curves, what you tried) and ask the user. Never lower a threshold.

## 5. Review (the only subagent of the phase)
- Spawn one subagent with model `sonnet` and this prompt: "Review `git diff phase-$0-start` against the `## Phase $0` section of docs/ROADMAP.md and docs/SPEC.md. Report only: (a) checks in scripts/check_phase.py that don't match the ROADMAP wording or thresholds, (b) bugs that would make results wrong, (c) SPEC violations. Max 15 lines."
- Fix (a)–(c), then re-run the gate.

## 6. Report
- Write `docs/reports/phase-$0.md` from docs/reports/TEMPLATE.md. Keep it to about one screen.
- Copy at most 1 GIF (< 3 MB) and 3 PNGs into docs/media/.
- If eval results changed, update docs/best_runs.yaml. From Phase 4 on, also regenerate docs/results.md.

## 7. Finish
- Run `uv run python scripts/check_phase.py $0` one final time. Its output must be in this turn.
- Commit with message `phase $0: <one-line summary>`, then run `git tag -f phase-$0-done`.
- Final message to the user, at most 8 lines:
  - PASS/FAIL per check ID, each with its value
  - the report path
  - the 1–3 things they should look at
