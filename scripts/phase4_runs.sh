#!/usr/bin/env bash
# Phase 4 experiment queue: train + eval every config x seed in configs/experiments/ (SPEC §6 thread budget <= 14).
# Wave 1: 12 SAC jobs x 1 torch thread. Wave 2: 3 SAC jobs alongside 3 sequential PPO jobs (8 workers + main).
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/_logs
job() {
  uv run python scripts/train.py --config "configs/$1.yaml" --seed "$2" > "runs/_logs/$1-s$2.log" 2>&1 &&
    uv run python scripts/eval.py --run "$(ls -d runs/*_"$1"_s"$2" | tail -1)" >> "runs/_logs/$1-s$2.log" 2>&1
  echo "$1 s$2 rc=$?"
}
export -f job
printf '%s\n' {e1_delta_pos,e1_abs_pos,e1_torque,e1_torque_gravcomp}' '{0,1,2} | xargs -P 12 -L 1 bash -c 'job $0 $1'
printf '%s\n' e3_her_sparse' '{0,1,2} | xargs -P 3 -L 1 bash -c 'job $0 $1' &
for s in 0 1 2; do job e2_ppo $s; done
wait
