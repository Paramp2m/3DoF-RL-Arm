#!/usr/bin/env bash
# Serve this folder and open the site: the live demo needs http, not a double-clicked file.
cd "$(dirname "$0")" && (sleep 1; xdg-open http://localhost:8000/final/ >/dev/null 2>&1 &) ; python3 -m http.server 8000
