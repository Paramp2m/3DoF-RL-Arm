# Decision log

One line per decision: `YYYY-MM-DD | phase | decision | reason`

2026-10-16 | - | Own spatial RRR arm first; UR5e (wrist locked) in Phase 7 | User choice D1-C
2026-10-16 | - | MuJoCo, MJCF only, no MCP server | User choice D2/D3
2026-10-16 | - | delta_pos main action mode; abs_pos/torque/torque_gravcomp as Phase 4 comparison | User choice D5
2026-10-16 | - | sin+cos joint encoding | User choice D6
2026-10-16 | - | Autonomous mode: user judges outputs, not code | User choice
2026-10-17 | - | Moved kit from rl-arm-kit/ to repo root | CLAUDE.md paths, hooks and skills expect root layout
2026-10-17 | 0 | CPU torch via [tool.uv.sources] pytorch-cpu index; ruff excludes .claude/ | PyPI linux torch is the CUDA build; skill scripts are vendored
2026-10-17 | 0 | Reacher SAC: default-ish hparams, 2 torch threads, 300k steps, eval 100 det. episodes | roadmap cap 300k; matches train skill thread budget
2026-10-17 | 0 | Policy class, dry-run steps/eval episodes moved into YAML; checks 0.5/0.6 also verify status=done, train thread count, non-empty GIF | Phase-0 review findings
2026-10-17 | 0 | eval.json schema deferred to Phase 1 (Reacher has no success/distance notion) | SPEC §8 fields need the real arm
