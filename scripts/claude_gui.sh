#!/usr/bin/env bash
# For Claude only: open a GUI as its own floating, pinned window in the bottom-right corner
# and hand focus back to the window that was active (Claude's terminal), so nothing gets hidden.
# Usage: scripts/claude_gui.sh <command...>   (output goes to runs/_logs/gui.log)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/_logs
prev=$(hyprctl activewindow -j | python3 -c 'import json,sys; print(json.load(sys.stdin).get("address",""))')
count() { hyprctl clients -j | grep -c '"address"'; }
before=$(count)
cmd="cd $(printf '%q' "$PWD") && $(printf '%q ' "$@") > runs/_logs/gui.log 2>&1"
hyprctl eval "hl.exec_cmd([==[$cmd]==], { float = true, pin = true, no_initial_focus = true, size = \"monitor_w*0.4 monitor_h*0.55\", move = \"monitor_w*0.58 monitor_h*0.42\" })" >/dev/null
# the window may grab focus when it maps; once it has appeared, give focus back
for _ in $(seq 60); do
    sleep 0.25
    [ "$(count)" -gt "$before" ] && break
done
sleep 0.3
[ -n "$prev" ] && hyprctl dispatch "hl.dsp.focus({ window = \"address:$prev\" })" >/dev/null
echo "opened: $*"
