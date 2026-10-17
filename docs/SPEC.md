# SPEC: design decisions

Locked by the user. Changing anything here requires the user's approval; Claude cannot edit this file.
Items marked *(tunable)* may be adjusted by Claude, provided the change is logged in docs/decisions.md.

## 1. Goal
A 3-DOF arm in MuJoCo learns these tasks via reinforcement learning, in order:
reach → hold → track a moving target → avoid an obstacle → push an object.

Deliverables:
- an interactive "drag the target, the arm follows" demo
- an architecture ready for domain randomisation (Phase 7)

## 2. Robots
**Primary: our own spatial RRR arm**, MJCF in `models/arm/`.
- Joints: base yaw (z axis) → shoulder pitch → elbow pitch.
- Starting dimensions *(tunable)*:
  - base column 0.10 m
  - upper arm 0.30 m
  - forearm 0.25 m
  - capsule geoms
  - masses ≈ 1.0 kg (upper arm) and 0.6 kg (forearm)
- Joint limits *(tunable)*: chosen so that no pose inside the limits self-collides or hits the floor. Document the limits in decisions.md.
- Required elements:
  - a site named `tip` at the end-effector
  - the target is a **mocap body** named `target`: visual only, no collision, movable by code or by mouse in the viewer
  - a floor
  - one fixed camera for videos
- Actuators: one `position` actuator and one `motor` actuator set per joint. The active set is chosen from config (two model variants, or enabling via ctrlrange). All gains come from config.

**Secondary (Phase 7):** UR5e from MuJoCo Menagerie, cloned into `third_party/` (git-ignored), with its 3 wrist joints locked. Same env code, selected via config.

**Format:** MJCF only. No URDF.

## 3. Software
- MuJoCo (official python bindings)
- Gymnasium
- Stable-Baselines3
- PyTorch (CPU build)
- TensorBoard
- PyYAML
- imageio (videos/GIFs)
- matplotlib (plots)
- pytest, ruff
- uv for environment management
- No MCP servers.

## 4. Environment
**Interface.** Gymnasium, goal-conditioned (keeps HER possible):
- observation is a dict `{observation, achieved_goal, desired_goal}`
- `compute_reward(achieved_goal, desired_goal, info)` must be vectorised

**`observation` vector:**
- sin and cos of the 3 joint angles
- 3 joint velocities (scaled)
- tip position
- tip→target vector
- task-specific extras (target velocity, obstacle pose, object pose)

**Action:** 3 floats in [−1, 1]. Config key `action_mode`:

| mode | meaning |
|---|---|
| `delta_pos` (default) | position actuators; `target_q += a · max_delta` (max_delta 0.05 rad *(tunable)*), clipped to joint limits |
| `abs_pos` | position actuators; `a` mapped linearly into each joint's range |
| `torque` | motor actuators; `τ = a · τ_max` |
| `torque_gravcomp` | `torque` plus gravity compensation from `data.qfrc_bias` |

**Timing** *(tunable)*: control at ≈25 Hz (frame_skip so dt ≈ 0.04 s). Reach episodes are 200 steps.

**Success:** tip within 0.02 m of the target.

**Reset:**
- random joint configuration within limits
- target sampled in joint space and passed through FK, so it is always reachable
- optional curriculum on the target distance

**Physics parameters from config:** masses, damping, friction, actuator gains, link lengths where the model allows. This is what makes domain randomisation possible later.

## 5. Reward
Weights live in config. Add terms one at a time, only when a logged problem calls for them. Start with distance + precision + success.

| term | formula |
|---|---|
| distance | −w_d · d |
| precision | +w_p · exp(−(d/σ)²), σ ≈ 0.03 m |
| progress | +w_g · (d_prev − d) |
| action size | −w_a · ‖a‖² |
| action rate | −w_s · ‖a − a_prev‖² |
| settle | −w_v · ‖q̇‖² · [d < ε] |
| joint limits | −w_l · proximity to limit |
| energy (torque modes only) | −w_e · Σ\|τ · q̇\| |
| success | +R |

## 6. Algorithms
- **SAC** is the main algorithm, with `MultiInputPolicy`.
- **PPO** as comparison, with SubprocVecEnv of ~8 envs.
- **SAC + HER** with a sparse reward, as an experiment.
- Every reported comparison uses ≥3 seeds.
- CPU threads: each worker uses 1 torch thread; the main process uses at most 4. Never oversubscribe the 16 threads.

## 7. Classical baseline
- Closed-form inverse kinematics for the RRR arm, giving both elbow-up and elbow-down solutions.
- A joint-space controller, used two ways:
  1. a scripted agent that sanity-checks the env
  2. the "RL vs classical" comparison

## 8. Evaluation
- 100 episodes with fixed eval seeds.
- Metrics:
  - success rate
  - final distance in mm (mean and median)
  - steps to success
  - smoothness (mean ‖Δa‖)
  - energy (torque modes)
- `eval.json` follows one fixed schema across the whole project.

## 9. Deliverables
- README
- per-phase reports
- docs/results.md, generated from eval.json files by script (no hand-typed numbers)
- demo GIFs
- `scripts/demo.py`: live viewer with a draggable target followed by the trained policy
