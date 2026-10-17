# Reading TensorBoard for rl-arm

```fish
cd ~/Projects/Claude/rl-arm
uv run tensorboard --logdir runs        # then open http://localhost:6006
```
It's read-only and safe to leave open while Claude trains. It auto-refreshes about every 30 s.

> **Phase 0 uses a 2-joint arm on purpose.** Phase 0 trains on Gymnasium's built-in **Reacher**, a ready-made 2-DOF arm. It's a known benchmark, used only to prove the pipeline (train → eval → GIF → gate) works before our robot exists. **Our 3-DOF RRR arm** (base yaw → shoulder pitch → elbow pitch) is built in Phase 1 and is the robot used in every phase after that.

---

## 1. The screen layout

**Top bar**
- **SCALARS / TIME SERIES**: two views of the same line charts. Either is fine.
- **INACTIVE dropdown**: dashboards with no data yet (images, histograms). Ignore it.
- **⟳ Refresh**: reloads now.
- **⚙ Settings**: refresh interval and similar options.

**Left sidebar**
- **Runs list** (coloured checkboxes): one entry per run. A run's name looks like `<date>_<config>_s<seed>/tb/<ALGO>_1`, and the checkbox colour is that run's line colour.
  - Runs under `_dry/` are ~2k-step smoke tests that run before every real job. Untick them.
  - From Phase 3 there are 3 seeds per config. Tick all three to see whether they agree. Seeds that disagree mean training is fragile.
- **Smoothing slider** (default 0.6): the bold line is a moving average, and the faint line behind it is the raw data. Use about 0.9 to see the trend and 0 to see the noise.
- **Horizontal axis**:
  - **Step**: environment steps. Use this one.
  - **Relative**: hours since the start.
  - **Wall**: clock time.
- **Ignore outliers**: zooms the y-axis past spikes. Leave it on.
- **Filter tags (regex)**: e.g. `rew|success` shows only matching charts.

**Each chart card**
- **Hover** to see the exact step and value for each run.
- **Expand icon**: makes the chart bigger.
- **Log-scale icon**: good for losses.
- **Fit-domain icon**: resets the zoom. Drag on a chart to zoom in, and double-click to reset.
- **Pin** (Time Series view): keeps a chart at the top.

---

## 2. Background: what SAC is doing

SAC (Soft Actor-Critic, the main algorithm here) trains two kinds of network at once:
- **Critic:** guesses "how much future reward will I get if I take this action here?"
- **Actor (the policy):** picks actions the critic rates highly. It's also rewarded for keeping some randomness (*entropy*) so it keeps exploring.

Rewards in this project are mostly **negative** (a penalty for distance from the target), so "better" means **closer to zero**, or above zero once bonuses like success are added.

---

## 3. Every chart, explained

### `rollout/`: how the arm is doing (watch these)
| Chart | Meaning | Healthy shape |
|---|---|---|
| `ep_rew_mean` | Average total reward over the last 100 training episodes. **The main chart** | Rises fast, then flattens. Training includes deliberate randomness, so the final eval (no randomness) usually scores a bit better |
| `ep_len_mean` | Average episode length | Flat line when episodes have a fixed length (Reacher 50, reach 200, hold 250). Only moves if a task ends episodes early |
| `success_rate` *(our arm, Phase 2+)* | Fraction of recent episodes where the tip ended within 2 cm | Climbs toward 1.0. Phase 3 needs ≥ 0.95 at eval |

### `train/`: the learning machinery (don't judge progress from these)
| Chart | Meaning | Healthy shape |
|---|---|---|
| `critic_loss` | How wrong the critic's guesses are | Drops early and stays small. Spikes are normal; steady growth is bad |
| `actor_loss` | Roughly the negative of the critic's value estimate | **Often rises, and that's fine.** With negative rewards, a critic that has learned "some cost is still ahead" makes this positive |
| `ent_coef` (α) | How much exploring with random actions is rewarded; tuned automatically | Drifts down as the policy gets confident. Hitting ~0 very early can mean exploration stopped too soon |
| `ent_coef_loss` | The signal SAC uses to tune α | Hovers around 0. Ignore it |
| `learning_rate` | Step size of the network updates (e.g. 0.0003) | Flat line (fixed setting) |

### `time/`
| Chart | Meaning | Healthy shape |
|---|---|---|
| `fps` | Environment steps per second | High for the first few hundred steps (data collection only), then drops and holds steady once network updates start. A sudden dip means the CPU is busy elsewhere |

### `reward/`: one chart per reward term *(our arm, Phase 2+)*
Each term from SPEC §5 is logged separately, so you can see **why** the total moves:

| Term | What it shows | What you want |
|---|---|---|
| `distance` | −(distance to target) | Rises toward 0 |
| `precision` | Bonus that peaks within ~3 cm | Rises; this is what pushes accuracy to the mm level |
| `success` | Bonus for being within 2 cm | Rises as success rate rises |
| `progress` | Reward for getting closer each step | Positive early in each episode, then ~0 once there |
| `action_size`, `action_rate` | Penalties for large or jerky actions | Shrink toward 0, meaning smoother motion |
| `settle` | Penalty for joint velocity near the target (hold) | Shrinks toward 0, meaning the arm stops still |
| `joint_limits` | Penalty for hugging joint limits | Small |
| `energy` | Penalty for torque × speed (torque modes only) | Shrinks |

Only the terms switched on in that run's config appear. Claude starts with distance + precision + success and adds others only when a problem shows up.

### `eval/` *(if a run evaluates periodically)*
| Chart | Meaning |
|---|---|
| `mean_reward`, `success_rate` | Score with no randomness, on fixed eval episodes. More trustworthy than `rollout/` |
| `mean_ep_length` | As above |

### PPO-only charts *(Phase 4, experiment E2)*
PPO learns differently, so its `train/` charts differ:

| Chart | Meaning | Healthy |
|---|---|---|
| `value_loss` | Critic error | Falls, then stays moderate |
| `policy_gradient_loss` | Actor update signal | Small, noisy around 0 |
| `entropy_loss` | Negative randomness of the policy | Slowly rises toward 0 as it gets confident |
| `approx_kl` | How much each update changed the policy | Small (≈0.01–0.05). Big spikes mean unstable updates |
| `clip_fraction` | Share of updates PPO had to cap | ~0.05–0.3 |
| `explained_variance` | How well the critic predicts returns | Rises toward 1 |
| `std` | Action noise | Slowly falls |

---

## 4. What to expect in each phase

| Phase | Runs you'll see | What "good" looks like |
|---|---|---|
| 0 (Reacher, 2-DOF benchmark) | 1 SAC run, ≤300k steps, ~30 min at ~160 steps/s | `ep_rew_mean` climbs from about −43 (random) to about −4/−5 and flattens. Gate: eval ≥ −7 |
| 1–2 (our arm built, env tested) | No training | — |
| 3 (reach, hold) | 3 seeds × 2 tasks, ≤1M steps each, a few hours | `success_rate` → ~1.0. `distance` → near 0. For hold, `settle` → 0. The 3 seeds should look alike |
| 4 (experiments) | Many runs: action modes, SAC vs PPO, HER | Compare curves between configs, not absolute numbers. Expect `torque` modes to learn slower than `delta_pos` |
| 5 (track, obstacle, push) | Several tasks, often with a curriculum | Curves can step down when the curriculum makes the task harder, then recover. That's expected. Push learns slowest |
| 7 (robustness) | Randomised-physics runs | Noisier curves than nominal; slower to rise |

---

## 5. Warning signs

| You see | Likely meaning |
|---|---|
| `ep_rew_mean` / `success_rate` flat from the start | Not learning: reward or observation problem. Claude's gate will catch it |
| Rises, then collapses | Unstable training; often fixed by a lower learning rate or a reward-weight change |
| `critic_loss` growing steadily or exploding | Critic diverging. Results from that run are unreliable |
| `ent_coef` → ~0 within the first few % of training | Exploration died early; the policy may be stuck |
| One seed much worse than the others | Fragile setup. Claude reports the mean over seeds, so it shows up in the numbers |
| `fps` drops sharply mid-run | Something else is using the CPU |
| A run's line just stops | Run crashed or the laptop slept. See GUIDE.md §6 |

You don't need to act on any of these: Claude diagnoses from the same data at the gate. If you spot one mid-run and want to flag it, tell Claude in plain words.
