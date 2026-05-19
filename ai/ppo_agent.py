"""PPO agent for hybrid continuous/discrete control."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from ai.action_space import HybridActionController
from ai.config import RLConfig
from ai.network import HybridActor, ValueNet
from ai.obs_extractor import ObsExtractor
from racing_ai.agent import Agent, Observation


@dataclass(slots=True)
class PPOActionSample:
    env_action: dict[str, float]
    cont_raw: np.ndarray
    gear: int
    log_prob: float
    value: float
    entropy: float
    mask_modified: bool
    mask_reasons: list[str]


class PPOAgent(Agent):
    def __init__(self, config: RLConfig, device: str = "auto") -> None:
        self.config = config
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.actor = HybridActor(config).to(self.device)
        self.value_net = ValueNet(config).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value_net.parameters()),
            lr=config.base_lr,
            eps=1e-5,
        )

        self.obs_extractor = ObsExtractor()
        self.action_controller = HybridActionController(config)
        self.training_mode = True

    def train_mode(self, mode: bool = True) -> None:
        self.training_mode = mode
        if mode:
            self.actor.train()
            self.value_net.train()
        else:
            self.actor.eval()
            self.value_net.eval()

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def sample_action(self, state: np.ndarray, obs: Observation) -> PPOActionSample:
        state_t = self._state_tensor(state)
        with torch.no_grad():
            if self.training_mode:
                cont_raw_t, gear_t, cont_lp_t, gear_lp_t, entropy_t = self.actor.sample(state_t)
            else:
                cont_raw_t, gear_t = self.actor.greedy(state_t)
                log_prob_t, entropy_t = self.actor.evaluate(state_t, cont_raw_t, gear_t)
                cont_lp_t = log_prob_t
                gear_lp_t = torch.zeros_like(log_prob_t)

            value_t = self.value_net(state_t)

        cont_raw = cont_raw_t.squeeze(0).cpu().numpy()
        gear = int(gear_t.item())
        mask_result = self.action_controller.from_policy(obs=obs, continuous_raw=cont_raw, gear_intent=gear)

        if self.training_mode:
            log_prob = float((cont_lp_t + gear_lp_t).item())
        else:
            log_prob = float(cont_lp_t.item())

        return PPOActionSample(
            env_action=mask_result.action,
            cont_raw=cont_raw,
            gear=gear,
            log_prob=log_prob,
            value=float(value_t.item()),
            entropy=float(entropy_t.mean().item()),
            mask_modified=mask_result.modified,
            mask_reasons=mask_result.reasons,
        )

    def act(self, observation: Observation) -> Mapping[str, float]:
        state = self.obs_extractor(observation)
        sample = self.sample_action(state=state, obs=observation)
        return sample.env_action

    def evaluate_batch(
        self,
        states: torch.Tensor,
        cont_raw: torch.Tensor,
        gear_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs, entropy = self.actor.evaluate(states, cont_raw, gear_actions)
        values = self.value_net(states)
        return log_probs, entropy, values

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        cont_raw = torch.tensor(batch["cont_raw"], dtype=torch.float32, device=self.device)
        gear = torch.tensor(batch["gear"], dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        num_items = states.size(0)
        idx = torch.arange(num_items, device=self.device)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "updates": 0.0,
        }

        for _ in range(self.config.ppo.update_epochs):
            perm = idx[torch.randperm(num_items, device=self.device)]
            for start in range(0, num_items, self.config.ppo.minibatch_size):
                sl = perm[start:start + self.config.ppo.minibatch_size]
                mb_states = states[sl]
                mb_cont = cont_raw[sl]
                mb_gear = gear[sl]
                mb_old_lp = old_log_probs[sl]
                mb_returns = returns[sl]
                mb_adv = advantages[sl]

                new_log_probs, entropy, value_preds = self.evaluate_batch(mb_states, mb_cont, mb_gear)
                ratio = (new_log_probs - mb_old_lp).exp()
                clipped_ratio = ratio.clamp(1.0 - self.config.ppo.clip_ratio, 1.0 + self.config.ppo.clip_ratio)
                policy_loss = -torch.min(ratio * mb_adv, clipped_ratio * mb_adv).mean()
                value_loss = F.mse_loss(value_preds, mb_returns)
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.ppo.value_coef * value_loss
                    - self.config.ppo.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.value_net.parameters()),
                    self.config.ppo.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_lp - new_log_probs).mean().item()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy_loss.item())
                metrics["approx_kl"] += float(approx_kl)
                metrics["updates"] += 1.0

                if approx_kl > self.config.ppo.target_kl:
                    break

        denom = max(metrics["updates"], 1.0)
        return {
            "policy_loss": metrics["policy_loss"] / denom,
            "value_loss": metrics["value_loss"] / denom,
            "entropy": metrics["entropy"] / denom,
            "approx_kl": metrics["approx_kl"] / denom,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "algorithm": "ppo",
                "config": {
                    "state_dim": self.config.state_dim,
                    "continuous_action_dim": self.config.continuous_action_dim,
                    "gear_action_dim": self.config.gear_action_dim,
                },
                "actor": self.actor.state_dict(),
                "value_net": self.value_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
