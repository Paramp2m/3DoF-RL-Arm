# rl-arm: 3-DOF RL robot arm (simulation only)

Train a neural-network policy (Stable-Baselines3) to control a 3-DOF arm in MuJoCo. No hardware.
- Design decisions (locked): docs/SPEC.md
- Phases + acceptance checks: docs/ROADMAP.md
- Decision log: docs/decisions.md

Read only the sections you need. Never read whole logs.

## Working mode: autonomous
- The user judges outputs (reports, GIFs, numbers), not code. Don't ask approval for implementation details. Decide, then add one line to docs/decisions.md: `YYYY-MM-DD | phase | decision | reason`.
- Ask the user only to:
  1. change anything in docs/SPEC.md or a threshold in docs/ROADMAP.md,
  2. do anything destructive or outside this folder,
  3. get unblocked after 3 genuine attempts at the same failing check.
- A phase is done only when `uv run python scripts/check_phase.py <N>` prints PASS **and** docs/reports/phase-<N>.md exists **and** the work is committed.
- Never weaken, skip or special-case a check to make it pass. If a target looks unreachable, say so in the report with evidence.
- Commit at the end of each phase and at stable checkpoints. Never push.
- Use the `phase` skill procedure for roadmap work, and the `train` skill for any job over ~5 min.

## Environment
- Machine: Ryzen 7 8845HS (8 cores/16 threads), Radeon 780M iGPU, CachyOS. No CUDA, so use CPU-only PyTorch.
- Python 3.12 via uv:
  - always run things with `uv run ...`
  - add deps with `uv add` (dev deps: `uv add --dev`)
  - never use global pip or sudo
- Headless MuJoCo rendering: MUJOCO_GL=egl (set in settings). Fall back to osmesa if EGL fails, and log it.

## Commands
- Fast tests: `uv run pytest -q -m "not slow"`. All tests: `uv run pytest -q`
- Lint/format: `uv run ruff check --fix . && uv run ruff format .` (a hook already formats edited files)
- Train: `uv run python scripts/train.py --config configs/<name>.yaml`, which writes runs/<run_id>/
- Eval: `uv run python scripts/eval.py --run runs/<run_id>`, which writes eval.json + video
- Visual check: `uv run python scripts/render_snapshot.py ...`, which writes a PNG (≤640 px)
- Phase gate: `uv run python scripts/check_phase.py <N>` (`--summary` gives a compact table)
- Dashboard (the user's live view of everything; read-only, nice 19): `uv run python scripts/dashboard.py`. The user starts it; its guide is `User Ref/GUIDE.md`. When adding a run field, doc or phase text the user should see, update `scripts/dashboard.py` too. When you stop a run on purpose, add its id to `runs/dismissed.json` so the dashboard doesn't warn about it
- Live view: `uv run python scripts/watch.py [--run runs/<id>]` (latest checkpoint during training). Claude never runs a GUI directly: always `scripts/claude_gui.sh <cmd...>`, which opens it as a floating, pinned window in the bottom-right corner and gives focus back to the terminal

## Credit discipline
- Scripts print a short summary. Full logs go to files; inspect them with `tail -n` / `grep`.
- Jobs over ~5 min: launch in the background (see the `train` skill). Do not sleep, poll or tail while they run. The one exception is the skill's mid-run triage: when a turn happens anyway, peek at `val.json` and kill clearly useless runs.
- Look at images only when a check needs eyes: one PNG at a time, ≤640 px.
- Prefer the project `stable-baselines3` skill for SB3 questions. Use context7 only for a specific mujoco/gymnasium API question.
- Don't re-read files you just wrote. No subagents except the single end-of-phase review.

## Code conventions
- Package src/rlarm/:
  - kinematics.py: FK/IK
  - envs/: base env + one subclass per task
  - rewards.py
  - wrappers.py: curriculum, randomisation
- Every behaviour switch comes from YAML in configs/, never hardcoded. This covers task, action mode, obs options, reward weights, physics params, robot model, algorithm and hyperparameters. A new experiment means a new config file.
- Each run folder holds: config copy, git hash, TensorBoard logs, model.zip, eval.json, video.
- Log every reward term separately to TensorBoard.
- Checks read existing artifacts (docs/best_runs.yaml maps phase/task → run dir). They may run quick evals but never train.
- Type hints and short docstrings. Tests live in tests/. Mark tests over 10 s with `@pytest.mark.slow`.
- runs/, videos and third_party/ are git-ignored. Each phase copies ≤1 GIF (<3 MB) and ≤3 PNGs to docs/media/.

# Compact instructions
When compacting, keep: the current phase, failing checks with their values, decisions made this session, and file paths touched.
