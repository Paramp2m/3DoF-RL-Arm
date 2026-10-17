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

    observation = [sin q (3), cos q (3), scaled q̇ (3), tip (3), target - tip (3)].
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
        self.lim = self.model.jnt_range.copy()
        self.act = slice(0, 3) if pos else slice(3, 6)  # actuator order in arm.xml
        self.tau_max = cfg["arm"]["tau_max"]
        self.mocap = self.model.body_mocapid[self.model.body("target").id]
        self.render_mode, self.size, self._renderer = render_mode, (height, width), None

        P, vc = H + L1 + L2, e["vel_clip"]  # P bounds any tip/target coordinate
        self.err_scale = e.get("err_scale", 1.0)  # >1 magnifies the tip->target obs so the net can resolve mm
        hi = np.r_[np.ones(6), np.full(3, vc), np.full(3, P), np.full(3, 2 * P * self.err_scale)].astype(np.float32)
        goal = spaces.Box(-P, P, (3,), np.float32)
        self.observation_space = spaces.Dict(
            {"observation": spaces.Box(-hi, hi), "achieved_goal": goal, "desired_goal": goal}
        )
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

    def _sample_goal(self) -> np.ndarray:
        raise NotImplementedError

    def compute_reward(self, achieved_goal, desired_goal, info) -> np.ndarray:
        """Sum of reward terms; batched over leading dims (info unused by the current terms)."""
        t = rewards.terms(
            achieved_goal, desired_goal, self.ecfg["reward"], self.ecfg["success_dist"]
        )
        return sum(t.values())

    def _obs(self) -> dict:
        q, tip = self.data.qpos, self.data.site("tip").xpos
        v = np.clip(
            self.data.qvel * self.ecfg["vel_scale"], -self.ecfg["vel_clip"], self.ecfg["vel_clip"]
        )
        o = np.r_[np.sin(q), np.cos(q), v, tip, (self.goal - tip) * self.err_scale]
        return {
            "observation": o.astype(np.float32),
            "achieved_goal": tip.astype(np.float32),
            "desired_goal": self.goal.astype(np.float32),
        }

    def _info(self, obs: dict) -> dict:
        t = rewards.terms(
            obs["achieved_goal"],
            obs["desired_goal"],
            self.ecfg["reward"],
            self.ecfg["success_dist"],
        )
        d = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"].astype(float)))
        return {"is_success": d < self.ecfg["success_dist"], "distance": d, "joint_speed": float(np.linalg.norm(self.data.qvel))} | {
            f"r_{k}": float(v) for k, v in t.items()
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        q = self.np_random.uniform(self.lim[:, 0], self.lim[:, 1])
        self.data.qpos[:] = q
        self.target_q = q.copy()
        self.goal = self._sample_goal()
        self.data.mocap_pos[self.mocap] = self.goal
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
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
        energy = 0.0
        for _ in range(e["frame_skip"]):
            # gravcomp: qfrc_bias is from the last mj_forward/mj_step; total torque still clipped to ±tau_max
            self.data.ctrl[self.act] = ctrl + (
                self.data.qfrc_bias if self.mode == "torque_gravcomp" else 0
            )
            mujoco.mj_step(self.model, self.data)
            energy += (
                np.abs(self.data.actuator_force[self.act] * self.data.qvel).sum()
                * self.model.opt.timestep
            )
        self.steps += 1
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
