# Phase 5 guide: harder tasks (track → obstacle → push)

Written 2026-10-18 as the input for `/plan`. It turns `## Phase 5` of docs/ROADMAP.md into ordered tasks that Claude can run on its own, following the `phase` skill (checks first → build → train → gate → review → report → commit). The ROADMAP and SPEC stay the authority. If this guide disagrees with them, they win; log the conflict in docs/decisions.md.

---

## 1. Definition of done (from ROADMAP + CLAUDE.md)

- `uv run python scripts/check_phase.py 5` prints PASS, with this output in the final turn
- `docs/reports/phase-5.md` exists (from docs/reports/TEMPLATE.md, about one screen)
- the work is committed (`phase 5: …`) and tagged `phase-5-done`
- `check_phase.py 3` and `check_phase.py 4` still PASS. Phase 5 touches the shared env and the eval schema, so this is a regression gate.

| ID | Check (each is the mean over 3 seeds × 100 eval episodes) |
|---|---|
| 5.1 | Track: mean tracking error after step 50 **≤ 25 mm** |
| 5.2 | Obstacle: success **≥ 85%** and collision rate **≤ 5%** |
| 5.3 | Push: success **≥ 60%** |
| 5.4 | One GIF per task in docs/media/ |

Thresholds are locked. Never weaken, skip or special-case a check.

---

## 2. Facts found in the code (read before planning)

| # | Fact | Consequence |
|---|---|---|
| F1 | `ArmGoalEnv` uses the whole `data.qpos`/`data.qvel`/`jnt_range` as the 3 arm joints (`_obs`, `_info`, `reset`) | A free-joint cube (push) breaks obs, joint speed and limit proximity. Slice to the arm's 3 DOF first (T2). |
| F2 | Check 4.1 needs `tuple(eval.json) == EVAL_SCHEMA` exactly | Adding eval keys means re-running eval.py on every phase-3 and experiment run, or 4.1 fails (T3) |
| F3 | `check_phase._evals()` hard-codes `phase3.<task>` | Generalise it to `phase<N>.<task>` for 5.1–5.3 |
| F4 | `src/rlarm/wrappers.py` is empty | The curriculum wrapper is new code (T4) |
| F5 | SAC+HER runs at ≈ 57 steps/s with 2 torch threads (reach_final: 1M steps in 4.8 h) | 7 parallel SAC jobs fit the 14-thread budget. Contacts (obstacle, push) may be slower, so take the ETA from `--dry-run`. |
| F6 | **Workspace: with the current limits (shoulder 0.4–1.7, elbow −1.1–1.4), the tip only gets below z = 0.15 m in a ring r ≈ 0.47–0.52 m** (measured with `fk` on a grid) | Push on the floor is nearly impossible with the current arm. See D-1. |
| F7 | With elbow ≥ −2.2 rad and a pad hanging ≈ 5 cm below the tip, the pad covers floor height over r ≈ 0.25–0.49 m (all yaw). The tip can then reach z ≈ −0.03, so the forearm can touch the floor. | This breaks the SPEC §2 invariant "no pose inside the limits … hits the floor", but only for the push variant |
| F8 | CLAUDE.md allows ≤ 1 GIF per phase; ROADMAP 5.4 needs 3 | ROADMAP wins (Phase 3 copied 2). Log it. |
| F9 | Every eval/val env is built with `make_env(cfg)` | The curriculum must wrap **only** the training env. Eval and val always run at full difficulty. |

---

## 3. Decisions

### D-1: needs the user (SPEC §2 change). Ask once, before T9.
Push needs a model variant whose arm can reach the floor (F6/F7).
- **Recommended:** `models/arm/arm_push.xml` = arm.xml + pusher pad + free cube + floor goal marker, with **elbow lower limit −2.2 rad** (shoulder unchanged), used only by push configs. Floor contact is physical (the floor stops the forearm/pad). All other tasks keep arm.xml and its invariant.
- Alternatives: lower the shoulder limit as well (wider band, but more floor contact); or keep the arm and use a raised platform (breaks "cube on the floor" in the ROADMAP, so not recommended).
- If the user hasn't answered when T9 starts: build T9a with the recommended variant behind config (`arm.xml: models/arm/arm_push.xml`), but **don't launch push training** until they say yes.

### Claude's defaults (decide, then log one line each in docs/decisions.md)
| ID | Default | Why |
|---|---|---|
| D-2 | Eval schema gains `track_err_mm_mean` (mean tip–goal distance over steps > 50, all tasks) and `collision_rate` (fraction of episodes with ≥ 1 arm–box contact; `null` when the env has no obstacle). Then re-eval every run listed in best_runs.yaml and configs/experiments/. | Keeps SPEC §8's one fixed schema. Eval is deterministic, so the old numbers must come out identical. |
| D-3 | Track: circle radius **0.08 m**, period **8 s** (≈ 0.063 m/s), horizontal plane, centre sampled through FK. Resample until every point (64 samples) has an in-limit `ik` solution. The target starts at a random phase. Fixed before any RL training and never retuned after seeing RL results (if they ever change, say so in the report). | The ROADMAP calls radius and speed tunable but documented. Locking them before training avoids tuning the task to pass. |
| D-4 | Track: plain SAC (**no HER**), shaped reward (precision + success), obs extras = target velocity (scaled), GoalDiffExtractor kept. 200-step episodes. | HER relabels with a static goal, which is inconsistent with a moving target and its velocity in the obs |
| D-5 | Obstacle: a box on a mocap body named `obstacle` (collides with the arm, not the floor) in `models/arm/arm_obstacle.xml`. Each reset puts it across the straight tip→target segment. Reject the sample if the arm touches it at reset, if it covers the target ± 4 cm, or if no in-limit IK pose at the target is contact-free. Obs extras = box centre + half-sizes. | Matches "a box between start and target" and keeps every episode solvable |
| D-6 | Obstacle: SAC+HER as in reach_final, plus reward term `collision` (−w_c per step with arm–box contact, read from `info`, so it works with HER `copy_info_dict`). **No termination** on contact. Contacts checked on every physics substep. 250-step episodes. | Keeps the never-terminate env contract. A contact between control steps still counts. |
| D-7 | Push: SAC+HER (FetchPush recipe). achieved_goal = cube centre, desired_goal = floor marker, `success_dist: 0.03`. Obs extras = cube pos, cube linear velocity, pad→cube vector. Reward = precision/success on cube→goal + `w_reach` pad→cube term (from info). Curriculum on goal distance from the cube (3 cm → full). 250-step episodes. | HER fits push naturally. The pad→cube shaping is goal-independent, so HER stays consistent. |
| D-8 | Step budgets (the ROADMAP sets no cap for Phase 5): track/obstacle ≤ 1M, push ≤ 3M per run. Pilots at 300k, seed 0. | Wall time ≈ 5 h per 1M at 57 steps/s |
| D-9 | GIF names: `docs/media/phase5_{track,obstacle,push}.gif`, each < 3 MB, from the seed-0 final run's `video.gif` | Matches the phase3_* naming |
| D-10 | Curriculum: `curriculum: {param, start, end, up_at, window}` in YAML. The wrapper raises `env.difficulty` by a step when the rolling success over `window` episodes is ≥ `up_at`. Logged to TensorBoard as `curriculum/difficulty`. | Config-driven, visible in curves, eval unaffected (F9) |

---

## 4. Dependency graph

```
T0 orient ─ T1 checks (FAIL) ─ T2 env generalisation ─┬─ T3 eval schema + re-eval (background)
                                                       ├─ T4 curriculum wrapper
                                                       │
            T5 TrackEnv ── T6 track pilot ─────────────┤        (train while building next)
            T7 ObstacleEnv ── T8 obstacle pilot ───────┤
  D-1 ──── T9a push model ── T9b PushEnv ── T10 push pilot(s)
                                                       │
                     T11 finals (3 seeds × 3 tasks) ── T12 eval + best_runs + GIFs
                                                       │
                     T13 gate + review ── T14 report + commit
```
Push (highest risk) gets a feasibility test in T9a before anything else is spent on it. Training overlaps with building: launch a pilot, end the turn, and build the next env when it completes.

---

## 5. Tasks

Every task: type hints + short docstrings, a unit test per piece of logic (`@pytest.mark.slow` if > 10 s), and every switch from YAML. Verify with `uv run pytest -q -m "not slow"` and `uv run ruff check .` unless more is stated.

### T0: Orient (XS)
- `git tag -f phase-5-start`. Read only the "Known issues" of docs/reports/phase-4.md.
- Create the task list (T1–T14).

### T1: Phase 5 checks first (S)
- Add `c51`–`c54` to scripts/check_phase.py with the exact IDs and thresholds above, one line each with the measured value (per-seed values like the 3.x lines). Generalise `_evals` to `(phase, task)` (F3); 3.x calls stay unchanged in behaviour.
- 5.1 reads `track_err_mm_mean`, 5.2 reads `success_rate` + `collision_rate`, 5.3 reads `success_rate` (best_runs `phase5.{track,obstacle,push}`, 3 runs each, `episodes == 100`), 5.4 checks the 3 GIFs exist and are < 3 MB.
- **Accept:** `check_phase.py 5` prints FAIL with "no phase5.track in best_runs.yaml"-style reasons. `check_phase.py 3` still PASS.
- Files: scripts/check_phase.py

### T2: Generalise the base env (M)
- Slice the arm DOF everywhere: `qpos[:3]`, `qvel[:3]`, `jnt_range[:3]` (F1); reset sets only the arm joints (plus task bodies via a hook).
- Hooks for subclasses: `_extras() -> np.ndarray` (appended to `observation`; `observation_space` sized from it), `_place_task()` at reset, and an `info` extension. Reach/hold return an empty extras vector.
- **Accept:** a regression test shows reach/hold obs, rewards and info are bit-identical to before for a fixed seed and action sequence (record the old values in the test before the refactor). All existing tests pass.
- Files: src/rlarm/envs/base.py, tests/test_env.py

### T3: Eval schema + re-eval (M, runs in the background)
- Extend `EVAL_SCHEMA` and `evaluate_goal` (D-2). The env reports `info["collision"]` only when it has an obstacle.
- Re-eval in one background shell script (`runs/_logs/phase5_reeval.log`): every run in docs/best_runs.yaml + every `arm_runs` of configs/experiments/. Compare each old eval.json value with the new one; they must match exactly except for the 2 new keys.
- **Accept:** `check_phase.py 3` and `4` PASS; `make_results.py --check` shows no diff; no old metric changed.
- Files: src/rlarm/common.py, scripts/check_phase.py (if needed), tests/test_results.py

### T4: Curriculum wrapper (S)
- `rlarm.wrappers.Curriculum` (D-10), applied in train.py **only** to the training env when `curriculum:` is in the config.
- **Accept:** test: difficulty rises after `window` successes, never passes `end`; a val/eval env built from the same config runs at full difficulty.
- Files: src/rlarm/wrappers.py, scripts/train.py, tests/test_train.py (or a new tests/test_wrappers.py)

### Checkpoint A (after T1–T4)
- [ ] fast tests + ruff clean · [ ] check 3 and 4 PASS · [ ] check 5 FAILs only for missing runs · commit `phase 5: checks, env hooks, eval schema, curriculum`

### T5: TrackEnv (S–M)
- `src/rlarm/envs/track.py` (registered in `TASKS`), configs/track_pilot.yaml (copy of reach_final minus HER, D-3/D-4). The target mocap moves each control step. `desired_goal` = its current position.
- Feasibility: run `IKAgent` on the track env (100 episodes, eval seeds) and record its `track_err_mm_mean`. If IK can't get ≤ 25 mm, RL likely can't either: say so in decisions.md before training.
- **Accept:** tests: the whole circle is reachable, speed matches the config (±1%), `compute_reward` is vectorised. `render_snapshot.py --config configs/track_pilot.yaml` gives a sane PNG (look once).
- Files: src/rlarm/envs/track.py, src/rlarm/envs/__init__.py, configs/track_pilot.yaml, tests/test_track.py

### T6: Track pilot (train skill)
- Dry run → `train.py --config configs/track_pilot.yaml --seed 0` in the background (300k). End the turn.
- **Accept (on completion):** eval.json has `track_err_mm_mean`. Next step: the diagnosis playbook (§7) or straight to the finals config `configs/track_final.yaml` (1M).

### T7: ObstacleEnv (M)
- `models/arm/arm_obstacle.xml`, `src/rlarm/envs/obstacle.py`, reward term `collision` in rewards.py (active only when `w_collision` is set), configs/obstacle_pilot.yaml (D-5/D-6).
- **Accept:** tests on 200 seeded resets: box intersects the start→target segment; no contact at reset; a contact-free IK goal pose exists; `IKAgent` (straight-line) has collision rate > 20% (the box really blocks). One PNG of the scene looked at once.
- Files: models/arm/arm_obstacle.xml, src/rlarm/envs/obstacle.py, src/rlarm/rewards.py, configs/obstacle_pilot.yaml, tests/test_obstacle.py

### T8: Obstacle pilot (train skill)
- Same as T6 with obstacle_pilot.yaml (300k, seed 0).

### Checkpoint B (after T5–T8)
- [ ] tests + ruff · [ ] both pilots launched or done · [ ] decisions.md has D-2–D-6 · commit

### T9a: Push model + feasibility (S) (needs D-1 for training)
- `models/arm/arm_push.xml`: flat pad on the tip (box, hangs ≈ 5 cm), free cube (≈ 5 cm, ≈ 0.1 kg, friction from config), floor goal marker = mocap `target` at cube height.
- **Accept (scripted, no RL):** the cube rests still for 250 steps with the arm parked; an IK-scripted move sweeps the pad through the cube and moves it ≥ 5 cm; cube and goal sampling stays inside the pad's floor band (F7). If the scripted push can't move the cube to within 3 cm of a nearby goal, fix the model before T9b.
- Files: models/arm/arm_push.xml, tests/test_push.py

### T9b: PushEnv (M)
- `src/rlarm/envs/push.py` (D-7) with a curriculum config configs/push_pilot.yaml.
- **Accept:** tests: achieved_goal = cube centre; success at < 3 cm; HER `compute_reward` matches the env reward for relabelled goals; obs extras have the right shape.
- Files: src/rlarm/envs/push.py, src/rlarm/envs/__init__.py, src/rlarm/rewards.py, configs/push_pilot.yaml, tests/test_push.py

### T10: Push pilots (train skill)
- With D-1 approved: 300k–1M pilots, up to 2 variants in parallel if slots are free (e.g. with/without `w_reach`), seed 0.

### T11: Finals (train skill)
- `configs/{track,obstacle,push}_final.yaml` (new files; never edit a config a finished run used). 3 seeds each, `model.zip` = val-best, as in Phase 3.
- Schedule (≤ 14 threads = 7 SAC jobs): track ×3 + obstacle ×3 (+1 push pilot) at once; push ×3 when its pilot is good. One background queue script if the slots are full.

### T12: Eval, best runs, media
- `eval.py --run …` for each final run → `docs/best_runs.yaml` `phase5: {track: [...], obstacle: [...], push: [...]}`.
- Copy the seed-0 `video.gif` of each task to `docs/media/phase5_<task>.gif` (< 3 MB; if bigger, re-render fewer frames rather than shrinking below a readable size). Log F8.

### T13: Gate + review
- `check_phase.py 5`. On a FAIL, use §7, and count attempts.
- One review subagent (model sonnet) with the phase skill's exact prompt for phase 5. Fix (a)–(c), then re-run gates 3, 4 and 5.

### T14: Report, commit
- `docs/reports/phase-5.md`: per-check values per seed; the track radius/speed (D-3); the D-1 outcome; IK baselines vs RL; known issues. Phase 5's media is the 3 GIFs (≤ 3 PNGs optional).
- Final `check_phase.py 5` in the same turn → commit `phase 5: track, obstacle, push` → `git tag -f phase-5-done`. Never push.

---

## 6. Wall-time plan (rough, ETA from `--dry-run` wins)

| Block | Jobs | ≈ Time |
|---|---|---|
| T3 re-eval | ~25 runs × 100 episodes | < 1 h, background |
| Pilots T6/T8/T10 | 3–4 SAC × 300k | ~1.5–2 h |
| Finals track + obstacle | 6 SAC × 1M | ~5 h |
| Finals push | 3 SAC × ≤ 3M | ≤ 15 h |

Total ≈ 1 day of wall time, mostly unattended. Never sleep, poll or tail while jobs run.

---

## 7. Diagnosis playbook (one change per attempt, log `attempt k/3` in decisions.md)

Read data, not code: `summary.json`, `val.json`, per-term TensorBoard scalars, `fresh_eval.py`.

| Symptom | First things to try (one at a time) |
|---|---|
| Track error high but stable | Compare with the IK baseline. If IK also lags: add target velocity scaling / a feed-forward hint (obs only). If only RL lags: progress term `w_g`, larger `features_extractor` scale, 1M steps. |
| Track error spikes at the start | Fine as long as step > 50 is good (the check ignores steps ≤ 50) |
| Obstacle collisions > 5% | Raise `w_collision`; curriculum on box size (small → full); longer episode (300) |
| Obstacle success < 85% but few collisions | The arm is stuck behind the box: longer episodes, `w_limit` check, HER `n_sampled_goal` 8 |
| Push: cube never moves | Raise `w_reach`, spawn the cube closer to the pad (curriculum start), check the pad really collides with the cube |
| Push: cube moves but misses | Curriculum on goal distance, more steps (up to 3M), `gamma` 0.98 → 0.99 |

**Stop rule:** 3 genuine attempts on the same failing check without progress → write the report with the evidence (numbers, curves, what was tried) and ask the user. Never lower a threshold.

---

## 8. Stop and ask the user only for

1. D-1 (push joint limits / floor contact: SPEC §2)
2. any other change to docs/SPEC.md or a ROADMAP threshold
3. anything destructive or outside this folder
4. the 3-attempt stop rule

---

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Push can't learn at 3-DOF + pad (no wrist) | High | T9a scripted feasibility first; curriculum; honest report with curves if < 60% |
| D-1 unanswered | Med | Build push behind config, train track/obstacle meanwhile |
| Base-env refactor shifts reach/hold numbers | Med | T2 bit-identical regression test; re-run gates 3 and 4 |
| Schema change breaks 4.1 / results.md | Med | T3 re-eval and compare before anything else depends on it |
| Contact-heavy envs train slower than 57 steps/s | Low | Dry-run ETA; queue push finals last |
| Moving target + HER inconsistent | Low | D-4: no HER on track |

---

## 10. Kick-off prompts for the user

```
/plan Completion of docs/PHASE5_GUIDE.md
```
then, once the plan is approved:
```
/goal Phase 5 is finished: `uv run python scripts/check_phase.py 5` printed PASS in this session, docs/reports/phase-5.md exists, and the work is committed. Follow docs/PHASE5_GUIDE.md with the phase skill. Stop after 40 turns, or earlier if you need a decision from me (D-1).
```
Answer D-1 in either prompt (e.g. "D-1: yes, elbow −2.2 push variant") to save a round trip.
