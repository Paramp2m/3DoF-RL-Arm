# rl-arm kit: setup and how to drive it

These files make Claude Code build the whole project with minimal input. Claude writes all the code; you start phases and judge the reports. Delete this file once you're set up.

## One-time setup
```bash
sudo pacman -S --needed uv git libnotify     # libnotify = desktop ping when Claude needs you
cd ~/Projects/Claude/rl-arm
unzip -n ~/Downloads/rl-arm-kit.zip           # -n: never overwrite (keeps your stable-baselines3 skill)
chmod +x .claude/hooks/*.sh
git init && git add -A && git commit -m "Project kit"
```
If you already had a `.claude/settings.json`, unzip skipped it, so ask Claude to merge the kit's version in.

## Running a phase (this is your whole job)
```bash
cd ~/Projects/Claude/rl-arm
systemd-inhibit --what=idle:sleep claude --permission-mode auto
```
- `systemd-inhibit` stops the laptop from sleeping. Sleep would kill training runs.
- `--permission-mode auto` removes per-command approval prompts; a safety classifier still screens each action. To make it the default everywhere instead, add `"permissions": {"defaultMode": "auto"}` to `~/.claude/settings.json` (project settings can't set it).

Then paste, changing the number for each phase:

```
/goal Phase 0 is finished: `uv run python scripts/check_phase.py 0` printed PASS in this session, docs/reports/phase-0.md exists, and the work is committed. Use the phase skill for phase 0. Stop after 40 turns, or earlier if you need a decision from me.
```

`/goal` keeps Claude working turn after turn until that condition is met. Training runs in the background, and Claude resumes when a run finishes. When the goal clears:
1. Read `docs/reports/phase-0.md` and open the GIF/PNG it points to.
2. Run `/clear`, then start the next phase.

## What you judge
- **The report:** check table, key numbers, "Look at this", and "How it works".
- **To ask for a fix,** say it plainly: "hold GIF looks jittery, fix it".
- **To change a target or design decision,** edit `docs/ROADMAP.md` or `docs/SPEC.md` yourself and commit. Claude is blocked from editing those two files. To let it edit them once, run `touch .claude/unlock`, then delete the file afterwards.

## Keeping usage low (Pro plan)
- The default model is Sonnet (set in `.claude/settings.json`). Only switch with `/model` for a bug it can't crack.
- Do one phase per conversation, and `/clear` between phases.
- Training runs on your CPU and doesn't use Claude usage; only Claude's turns do. Background check-ins on long runs are capped at one per 2 h.
- Check usage with `/usage`. If you hit the limit mid-goal, the goal pauses and can continue after the reset.
- Ask for report sections, not code walkthroughs.

## What's in the kit
| File | Purpose |
|---|---|
| `CLAUDE.md` | Standing instructions Claude reads every session: autonomy rules, commands, usage discipline |
| `docs/SPEC.md` | Your locked design decisions (arm, sim, actions, obs, reward, algorithms) |
| `docs/ROADMAP.md` | Phases 0–7 with numeric pass/fail checks |
| `docs/decisions.md` | Claude logs every choice it makes here, one line each |
| `docs/reports/TEMPLATE.md` | Shape of each phase report |
| `.claude/settings.json` | Model, permissions, env vars, hooks |
| `.claude/hooks/` | Auto-format Python; run fast tests before Claude stops; lock SPEC/ROADMAP |
| `.claude/skills/phase` | Step-by-step procedure for running a phase |
| `.claude/skills/train` | Launch training in the background without spending usage while it runs |

## Phases
| # | Outcome |
|---|---|
| 0 | Tooling works; SAC solves Gymnasium's 2-joint Reacher (proves the pipeline) |
| 1 | Your RRR arm exists in MuJoCo; FK/IK verified to 1e-6 m |
| 2 | The training environment works; a hand-coded IK agent solves it (proves the env) |
| 3 | RL learns reach (≥95%, ≤10 mm) and hold |
| 4 | Experiments: action modes, SAC vs PPO, HER, RL vs IK, written up in `results.md` |
| 5 | Moving target, obstacle, pushing a cube |
| 6 | Drag-the-target live demo + README |
| 7 | Optional: robustness to physics changes + UR5e transfer |
