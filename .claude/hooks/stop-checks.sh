#!/usr/bin/env bash
# Stop: if Python files changed since the last commit, run fast tests + lint.
# On failure, exit 2 so Claude keeps working on the fix (once per stop; see stop_hook_active).
input=$(cat)
active=$(printf '%s' "$input" | python3 -c 'import sys,json; print(str(json.load(sys.stdin).get("stop_hook_active",False)).lower())' 2>/dev/null)
[ "$active" = "true" ] && exit 0

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0
[ -d tests ] || exit 0
[ -x .venv/bin/pytest ] && [ -x .venv/bin/ruff ] || exit 0   # deps not synced yet
git rev-parse --git-dir >/dev/null 2>&1 || exit 0
[ -n "$(git status --porcelain -- '*.py' | head -n 1)" ] || exit 0

out=$(timeout 300 .venv/bin/pytest -q -m "not slow" -x --no-header 2>&1); rc=$?
[ $rc -eq 5 ] && rc=0                     # pytest: no tests collected
lint=$(.venv/bin/ruff check --quiet . 2>&1); lrc=$?

if [ $rc -ne 0 ] || [ $lrc -ne 0 ]; then
  {
    echo "Fast checks failed. Fix them, or state clearly in your reply why they are expected to fail."
    [ $rc -ne 0 ] && printf '%s\n' "$out" | tail -n 25
    [ $lrc -ne 0 ] && printf '%s\n' "$lint" | head -n 15
  } >&2
  exit 2
fi
exit 0
