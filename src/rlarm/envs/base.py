"""Goal-conditioned RRR-arm env (SPEC §4): dict obs, 4 action modes, physics and reward from config."""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from rlarm import rewards
from rlarm.arm import load_model
from rlarm.kinematics import L1, L2, H

MODES = ("delta_pos", "abs_pos", "torque", "torque_gravcomp")


class ArmGoalEnv(gym.Env):
    """Base env. Subclasses implement `_sample_goal`.

    observation = [sin q (3), cos q (3), scaled q̇ (3), tip (3), (target - tip)·err_scale (3, unless obs_err false)].
    Episodes never terminate; they truncate after env.episode_steps. info["is_success"] = tip within
    env.success_dist of the target; info["r_<term>"] holds each reward term.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}  # noqa: RUF012 - gymnasium API

    def __init__(
        self, cfg: dict, render_mode: str | None = None, width: int = 320, height: int = 320
    ):
        e = self.ecfg = cfg["env"]
        self.mode = e["action_mode"]
        if self.mode not in MODES:
            raise ValueError(f"action_mode {self.mode!r} not in {MODES}")
        pos = self.mode.endswith("_pos")
        self.model = load_model(
            {**cfg, "arm": {**cfg["arm"], "actuators": "position" if pos else "motor"}}
        )
        self.data = mujoco.MjData(self.model)
        self.lim = self.model.jnt_range[:3].copy()  # the 3 arm joints; task bodies may add more
        self.act = slice(0, 3) if pos else slice(3, 6)  # actuator order in arm.xml
        self.tau_max = cfg["arm"]["tau_max"]
        self.mocap = self.model.body_mocapid[self.model.body("target").id]
        self.render_mode, self.size, self._renderer = render_mode, (height, width), None

        P, vc = H + L1 + L2, e["vel_clip"]  # P bounds any tip/target coordinate
        self.err_scale = e.get(
            "err_scale", 1.0
        )  # >1 magnifies the tip->target obs so the net can resolve mm
        # obs_err false: drop the goal-dependent tip->target entries (HER relabels goals, not observation);
        # rlarm.policies.GoalDiffExtractor then computes them inside the network.
        self.obs_err = e.get("obs_err", True)
        hi = np.r_[np.ones(6), np.full(3, vc), np.full(3, P)]
        if self.obs_err:
            hi = np.r_[hi, np.full(3, 2 * P * self.err_scale)]
        hi = np.r_[hi, np.full(self._extras_dim(), 10.0)]
        hi = hi.astype(np.float32)
        goal = spaces.Box(-P, P, (3,), np.float32)
        self.observation_space = spaces.Dict(
            {"observation": spaces.Box(-hi, hi), "achieved_goal": goal, "desired_goal": goal}
        )
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

    def _sample_goal(self) -> np.ndarray:
        raise NotImplementedError

    # --- hooks for task subclasses (reach/hold use none) ---
    def _extras_dim(self) -> int:
        """Length of the task-specific observation tail."""
        return 0

    def _extras(self) -> np.ndarray:
        """Task-specific observation tail (appended to `observation`)."""
        return np.zeros(0)

    def _place_task(self) -> None:
        """Called at reset after the goal is set: place task bodies (obstacle, cube...)."""

    def _info_extra(self) -> dict:
        """Extra info entries (e.g. collision)."""
        return {}

    def _substep(self) -> None:
        """Called after every physics substep (e.g. to catch contacts between control steps)."""

    def _update_goal(self) -> None:
        """Called every control step before the observation: move a moving target."""

    def compute_reward(self, achieved_goal, desired_goal, info) -> np.ndarray:
        """Sum of reward terms; batched over leading dims. info (a dict, or HER's array of dicts)
        supplies joint_speed / limit_prox for the settle / joint-limit terms."""
        w = self.ecfg["reward"]

        def get(key):
            return (
                info[key] if isinstance(info, dict) else np.array([i[key] for i in np.ravel(info)])
            )

        t = rewards.terms(
            achieved_goal,
            desired_goal,
            w,
            self.ecfg["success_dist"],
            get("joint_speed") if w.get("w_settle") else None,
            get("limit_prox") if w.get("w_limit") else None,
            get("collision") if w.get("w_collision") else None,
            get("reach_dist") if w.get("w_reach") else None,
            get("action_delta") if w.get("w_smooth") else None,
            get("clear_prox") if w.get("w_clear") else None,
            get("wp_dist") if w.get("w_wp") else None,
        )
        return sum(t.values())

    def _obs(self) -> dict:
        q, tip = self.data.qpos[:3], self.data.site("tip").xpos
        v = np.clip(
            self.data.qvel[:3] * self.ecfg["vel_scale"],
            -self.ecfg["vel_clip"],
            self.ecfg["vel_clip"],
        )
        o = np.r_[np.sin(q), np.cos(q), v, tip]
        if self.obs_err:
            o = np.r_[o, (self.goal - tip) * self.err_scale]
        o = np.r_[o, self._extras()]
        return {
            "observation": o.astype(np.float32),
            "achieved_goal": tip.astype(np.float32),
            "desired_goal": self.goal.astype(np.float32),
        }

    def _info(self, obs: dict) -> dict:
        speed = float(np.linalg.norm(self.data.qvel[:3]))
        q = self.data.qpos[:3]
        gap = np.minimum(q - self.lim[:, 0], self.lim[:, 1] - q)
        prox = float(np.clip(1 - gap / self.ecfg["reward"].get("limit_margin", 0.05), 0, 1).sum())
        extra = self._info_extra()
        t = rewards.terms(
            obs["achieved_goal"],
            obs["desired_goal"],
            self.ecfg["reward"],
            self.ecfg["success_dist"],
            speed,
            prox,
            extra.get("collision"),
            extra.get("reach_dist"),
            self.action_delta,
            extra.get("clear_prox"),
            extra.get("wp_dist"),
        )
        d = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"].astype(float)))
        return (
            {
                "is_success": d < self.ecfg["success_dist"],
                "distance": d,
                "joint_speed": speed,
                "limit_prox": prox,
                "action_delta": self.action_delta,
            }
            | {f"r_{k}": float(v) for k, v in t.items()}
            | extra
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        q = self.np_random.uniform(self.lim[:, 0], self.lim[:, 1])
        self.data.qpos[:3] = q
        self.target_q = q.copy()
        self.goal = self._sample_goal()
        self.data.mocap_pos[self.mocap] = self.goal
        self._place_task()
        mujoco.mj_forward(self.model, self.data)
        self.steps, self.action_delta, self._prev_a = 0, 0.0, np.zeros(3)
        obs = self._obs()
        return obs, self._info(obs)

    def step(self, action):
        a, e, (lo, hi) = np.clip(np.asarray(action, float), -1, 1), self.ecfg, self.lim.T
        if self.mode == "delta_pos":
            self.target_q = np.clip(self.target_q + a * e["max_delta"], lo, hi)
            ctrl = self.target_q
        elif self.mode == "abs_pos":
            ctrl = lo + (a + 1) / 2 * (hi - lo)
        else:
            ctrl = a * self.tau_max
        self.action_delta = float(np.sum((a - self._prev_a) ** 2))
        self._prev_a = a
        energy = 0.0
        for _ in range(e["frame_skip"]):
            # gravcomp: qfrc_bias is from the last mj_forward/mj_step; total torque still clipped to ±tau_max
            self.data.ctrl[self.act] = ctrl + (
                self.data.qfrc_bias if self.mode == "torque_gravcomp" else 0
            )
            mujoco.mj_step(self.model, self.data)
            self._substep()
            energy += (
                np.abs(self.data.actuator_force[self.act] * self.data.qvel[:3]).sum()
                * self.model.opt.timestep
            )
        self.steps += 1
        self._update_goal()
        obs = self._obs()
        info = self._info(obs) | {"energy": float(energy)}
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        return obs, reward, False, self.steps >= e["episode_steps"], info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.size)
        self._renderer.update_scene(self.data, camera="main")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
