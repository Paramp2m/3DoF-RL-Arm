#!/usr/bin/env bash
# PostToolUse (Edit|Write): format + autofix the edited Python file. Silent, never blocks.
input=$(cat)
path=$(printf '%s' "$input" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)

[[ "$path" == *.py ]] || exit 0
cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0
[ -x .venv/bin/ruff ] || exit 0          # before Phase 0 installs ruff

.venv/bin/ruff format --quiet "$path" >/dev/null 2>&1
.venv/bin/ruff check --fix --quiet "$path" >/dev/null 2>&1
exit 0
