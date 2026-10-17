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
2026-10-17 | 0 | train.py saves checkpoint.zip every ~10% (atomic rename); watch.py reloads it between episodes; Claude-opened GUIs go through scripts/claude_gui.sh (hl.exec_cmd float+pin, then refocus terminal) | User asked for live viewing without hiding Claude's terminal; no_initial_focus alone did not keep focus
2026-10-17 | 1 | Joint limits: yaw ±3.1, shoulder pitch [0.4, 1.7], elbow [-1.1, 1.4] rad (positive pitch raises link; elbow<0 = elbow-up) | Box search maximising area s.t. tip/elbow ≥ 4 cm above floor (capsule r=2 cm) and forearm clear of column; 0 contacts in 2000 random + 8 corner configs
2026-10-17 | 1 | Column shortened to 0.06 m (joint stays at 0.10 m) | Column cylinder overlapped shoulder housing/upper arm (world-body geoms aren't parent-child filtered)
2026-10-17 | 1 | One MJCF with both actuator sets (group 0 position, group 1 motor); rlarm.arm.load_model disables one via opt.disableactuator and sets kp/kv/tau_max/damping/armature from config and position ctrlrange from the joint limits (XML holds no gains) | SPEC §2 allows either; single file keeps one source of geometry
2026-10-17 | 1 | Gains kp=400, kv=20, tau_max=12 N·m, damping 0.5, armature 0.01, integrator implicitfast | Worst-case gravity torque ≈4 N·m → ~0.6° sag; implicitfast keeps kv stable at dt=2 ms
2026-10-17 | 1 | ik() returns all 4 solutions {yaw, yaw+π}×{elbow up, down}, no limit filtering; empty array = unreachable | Check 1.3 needs every solution; limits are a separate concern
2026-10-17 | 1 | Check 1.4 also tests the 8 joint-box corners; 1.5 starts from mid-range pose and commands random poses | Stricter than roadmap wording, same threshold
2026-10-17 | 1 | render_snapshot.py renders the raw arm when the config has an `arm:` key | Reuse existing script for check 1.6
2026-10-17 | 1 | eval.json SPEC §8 schema moves to Phase 2 (first env with success/distance); masses/link lengths from config deferred to Phase 7 randomisation | Phase 1 has no env; ROADMAP Phase 1 lists neither
2026-10-17 | 1 | Post-phase review: removed unused in_limits()/joint_range(); added FK-vs-MuJoCo, config-override, ctrl-clipping and pose-hold unit tests | Mutation testing showed kv sign, damping override and ctrl limits were only guarded by the gate, not by tests
