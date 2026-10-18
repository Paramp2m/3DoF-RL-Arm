# rl-arm setup

You start each phase and judge the results. Claude writes all the code. Once you're set up and comfortable, you can delete this file: `GUIDE.md` covers day-to-day use.

## One-time setup
```fish
sudo pacman -S --needed uv git libnotify     # libnotify gives a desktop ping when Claude needs you
cd ~/Projects/Claude/rl-arm
unzip -n ~/Downloads/rl-arm-kit.zip          # -n never overwrites, so your stable-baselines3 skill is kept
chmod +x .claude/hooks/*.sh
git init; and git add -A; and git commit -m "Project kit"
```
If you already had a `.claude/settings.json`, unzip skipped it. Ask Claude to merge the kit's version in.

## Running a phase
```fish
cd ~/Projects/Claude/rl-arm
systemd-inhibit --what=idle:sleep claude --permission-mode auto
```
- `systemd-inhibit` stops the laptop sleeping, which would kill training.
- `--permission-mode auto` removes per-command prompts; a safety check still screens each action. To make it the default, add `"permissions": {"defaultMode": "auto"}` to `~/.claude/settings.json`.

Then paste this, changing the number for each phase:
```
/goal Phase 0 is finished: `uv run python scripts/check_phase.py 0` printed PASS in this session, docs/reports/phase-0.md exists, and the work is committed. Use the phase skill for phase 0. Stop after 40 turns, or earlier if you need a decision from me.
```
`/goal` keeps Claude working until that condition is met. Training runs in the background and Claude resumes when it finishes. When the goal clears:
1. Read `docs/reports/phase-0.md` and open the GIF or PNG it points to.
2. Run `/clear`, then start the next phase.

## What you judge
- **The report:** check table, key numbers, "Look at this", "How it works".
- **A fix:** ask in plain words, e.g. "hold GIF looks jittery, fix it".
- **A target or design change:** edit `docs/ROADMAP.md` or `docs/SPEC.md` yourself and commit. Claude is blocked from editing them. To allow it once: `touch .claude/unlock`, then delete that file afterwards.

## Keeping usage low
- Sonnet is the default model (set in `.claude/settings.json`). Switch with `/model` only for a bug it can't crack.
- One phase per conversation; `/clear` between phases.
- Training runs on your CPU and costs no Claude usage. Only Claude's turns do.
- Check usage with `/usage`. If you hit the limit mid-goal, it pauses and continues after the reset.
- Ask for report sections, not code walkthroughs.

## What's in the kit
| File | Purpose |
|---|---|
| `CLAUDE.md` | Standing instructions Claude reads every session |
| `docs/SPEC.md` | Your locked design decisions |
| `docs/ROADMAP.md` | Phases 0–7 with numeric pass/fail checks |
| `docs/decisions.md` | Every choice Claude makes, one line each |
| `docs/reports/TEMPLATE.md` | Shape of each phase report |
| `.claude/settings.json` | Model, permissions, hooks |
| `.claude/hooks/` | Auto-format Python, run fast tests before Claude stops, lock SPEC and ROADMAP |
| `.claude/skills/phase`, `train` | Step-by-step procedures for running a phase and launching training |

The phases themselves are listed in `GUIDE.md` §4.
