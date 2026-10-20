"""SAC fine-tuning from a behaviour-cloned actor (Stage 3 of the network-only obstacle plan).

SACBC = SB3 SAC (its `train` loop, copied from SB3 2.9) plus:
- a behaviour-cloning term on expert demonstrations: actor loss += λ(t) · |Q| · MSE(actor mean, expert action),
  λ(t) = bc_weight · max(0, 1 - num_timesteps / bc_decay); scaled by the mean |Q| so it stays comparable to the
  SAC objective (as in TD3+BC);
- `actor_freeze`: the first N gradient steps update only the critics (and not the entropy coefficient), so a
  freshly initialised critic cannot wreck the cloned actor before it has learned anything.
Demonstrations are given with `set_demos(obs_dict, actions)`; they are not saved with the model.
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


class SACBC(SAC):
    def __init__(
        self, *args, bc_weight: float = 0.0, bc_decay: int = 1, actor_freeze: int = 0, **kw
    ):
        self.bc_weight, self.bc_decay, self.actor_freeze = (
            bc_weight,
            max(int(bc_decay), 1),
            actor_freeze,
        )
        self._demo_obs, self._demo_act = None, None
        super().__init__(*args, **kw)

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "_demo_obs", "_demo_act"]

    def set_demos(self, obs: dict[str, np.ndarray], actions: np.ndarray) -> None:
        """Expert (observation dict, action in [-1, 1]) pairs for the BC term."""
        self._demo_obs = {
            k: th.as_tensor(v, dtype=th.float32, device=self.device) for k, v in obs.items()
        }
        self._demo_act = th.as_tensor(actions, dtype=th.float32, device=self.device)

    def bc_lambda(self) -> float:
        if self._demo_act is None:
            return 0.0
        return self.bc_weight * max(0.0, 1.0 - self.num_timesteps / self.bc_decay)

    def _bc_loss(self, batch_size: int) -> th.Tensor:
        idx = th.randint(0, len(self._demo_act), (batch_size,), device=self.device)
        obs = {k: v[idx] for k, v in self._demo_obs.items()}
        return F.mse_loss(self.actor(obs, deterministic=True), self._demo_act[idx])

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs, actor_losses, critic_losses, bc_losses = [], [], [], [], []
        lam = self.bc_lambda()
        for gradient_step in range(gradient_steps):
            frozen = self._n_updates + gradient_step < self.actor_freeze
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None and not frozen:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]
            self.critic.optimizer.zero_grad()
            critic_loss.backward()  # type: ignore[union-attr]
            self.critic.optimizer.step()

            if not frozen:
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
                if lam > 0:
                    bc = self._bc_loss(batch_size)
                    bc_losses.append(bc.item())
                    actor_loss = actor_loss + lam * min_qf_pi.abs().mean().detach() * bc
                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/bc_lambda", lam)
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        if bc_losses:
            self.logger.record("train/bc_loss", np.mean(bc_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
