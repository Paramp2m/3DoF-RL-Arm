#!/usr/bin/env bash
# PreToolUse (Edit|Write): docs/SPEC.md and docs/ROADMAP.md belong to the user.
# To let Claude edit them temporarily: touch .claude/unlock  (delete it afterwards)
input=$(cat)
path=$(printf '%s' "$input" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)

case "$path" in
  *docs/SPEC.md|*docs/ROADMAP.md)
    [ -f "$CLAUDE_PROJECT_DIR/.claude/unlock" ] && exit 0
    echo "Blocked: $path is user-owned and locked. Propose the exact change to the user instead." >&2
    exit 2
    ;;
esac
exit 0
