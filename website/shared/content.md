# Copy deck (shared by v1-v4)

Every number comes from `docs/results.md` (generated from each run's eval.json) or a phase report. Eval: 100 fixed goals, deterministic policy, best checkpoint chosen on separate validation goals. Success = tip within 2 cm at the last step. 3 seeds per row, shown as mean ± std.

## Argument (claim -> proof)
1. A 3-joint arm learned five tasks with no hand-written controller -> clips + success table (88.7-100%).
2. The interface the policy acts through matters more than the algorithm tweak -> E1 (action modes) and E2 (SAC vs PPO).
3. A sparse "did it reach" reward with hindsight replay beat the hand-shaped reward -> E3 (85.0% vs 70.0%).
4. Learned policy is close to, but not better than, closed-form IK -> E4 (96.0%, 5.7 mm vs 100%, 0 mm). Said plainly.
5. Some things failed, and the log says so -> obstacle study: six network-only attempts, best 89.7% success but 11.3% collisions against a 5% limit.
6. Everything is reproducible and gated -> every behaviour switch in YAML, each run keeps config + git hash, each phase passes scripted checks.

## Hero
- Name: rl-arm
- Headline options (<= 32 chars): "Five tasks, learned from scratch" / "An arm that taught itself five tasks"
- Supporting sentence: A three-joint arm in MuJoCo learns to reach, hold, track, avoid a box and push a cube with reinforcement learning. 88.7 to 100% success, three seeds each.
- Action: "See the results" (to #results) ; second link: GitHub (TODO url)
- Media: phase6_demo (target glides, policy follows; 14.7 mm mean over the second half of a 2000-step headless run)

## Tasks (explorer)
Data in `data.json`. One sentence each:
- reach: move the tip to a random goal and stop within 2 cm.
- hold: reach, then stay within 2 cm for the last 50 of 250 steps.
- track: follow a target circling at 8 cm radius, one lap per 8 s.
- obstacle: reach around a box that sits across the straight line to the goal.
- push: push a free 5 cm cube to a goal with a pusher pad.

## Experiments (300k steps, reach, 3 seeds, unless stated)
- E1 action modes (SAC): delta_pos 70.0 ± 11.5 | torque_gravcomp 66.0 ± 9.6 | torque 45.7 ± 14.0 | abs_pos 31.7 ± 1.5. Torque modes reach in 8-10 steps vs 36 for delta_pos, but are jerkier.
- E2 algorithm (delta_pos): SAC 70.0 ± 11.5 | PPO 49.3 ± 2.3. PPO is smoother (0.050 vs 0.180 mean action change).
- E3 reward (SAC): SAC + HER with a sparse reward 85.0 ± 5.3 | shaped reward 70.0 ± 11.5.
- E4 versus classical (1M steps for RL): RL 96.0 ± 0.0, median 5.7 mm | IK + joint controller 100%, 0.0 mm, no training.
- Caveat to print: 300k steps ranks learning speed, not final performance; every curve was still rising. With 1M steps plain SAC on delta_pos reached 88%.

## What went wrong (about)
- Obstacle: a planner follower alone (no learning) gets 89% / 3% collisions; the kept policy (planner waypoint + learned residual + safety filter) gets 88.7% / 1.7%. Pure RL got 72% / 2%.
- Network-only study h1-h6: teacher-student imitation then SAC fine-tune. Best h6: 89.7% success, 11.3% collisions. Target was <= 5% collisions, so it is reported as a negative result; the kept policy is the planner-assisted one.
- Smaller notes: 300k-step comparisons are early; E1 runs were launched from a dirty working tree (identical code committed seconds later).

## Build (list)
- Robot: own RRR arm (base yaw, shoulder pitch, elbow pitch) as MJCF; target is a mocap ball.
- Observation: sin/cos of joint angles, joint speeds, tip position, target. Action: small change of each joint's position setpoint (delta_pos), 25 Hz.
- Learning: Stable-Baselines3 SAC (+HER), PPO for comparison; PyTorch on CPU only (Ryzen 7 8845HS, no GPU).
- Method: every switch (task, action mode, reward weights, physics, algorithm) in YAML; every reward term logged separately; each run stores config, git hash, TensorBoard logs.
- Process: seven phases, each closed by a scripted check and a written report.

## Links / footer
- GitHub: TODO url. Contact: TODO. Author: TODO name.
