"""SAC agent for hybrid continuous/discrete control."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ai.action_space import HybridActionController
from ai.config import RLConfig
from ai.network import HybridActor, HybridQCritic
from ai.obs_extractor import ObsExtractor
from racing_ai.agent import Agent, Observation


@dataclass(slots=True)
class SACActionSample:
    env_action: dict[str, float]
    cont_raw: np.ndarray
    gear: int
    log_prob: float
    mask_modified: bool
    mask_reasons: list[str]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._idx = 0
        self._full = False
        self.states: list[np.ndarray | None] = [None] * capacity
        self.cont_actions: list[np.ndarray | None] = [None] * capacity
        self.gear_actions: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self.rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.next_states: list[np.ndarray | None] = [None] * capacity
        self.dones: np.ndarray = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.capacity if self._full else self._idx

    def push(
        self,
        state: np.ndarray,
        cont_action: np.ndarray,
        gear_action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self._idx] = state.astype(np.float32, copy=False)
        self.cont_actions[self._idx] = cont_action.astype(np.float32, copy=False)
        self.gear_actions[self._idx] = int(gear_action)
        self.rewards[self._idx] = float(reward)
        self.next_states[self._idx] = next_state.astype(np.float32, copy=False)
        self.dones[self._idx] = 1.0 if done else 0.0

        self._idx = (self._idx + 1) % self.capacity
        if self._idx == 0:
            self._full = True

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        n = len(self)
        ids = np.array(random.sample(range(n), batch_size), dtype=np.int64)

        states = np.stack([self.states[i] for i in ids if self.states[i] is not None], axis=0)
        cont_actions = np.stack([self.cont_actions[i] for i in ids if self.cont_actions[i] is not None], axis=0)
        next_states = np.stack([self.next_states[i] for i in ids if self.next_states[i] is not None], axis=0)

        return {
            "states": states,
            "cont_actions": cont_actions,
            "gear_actions": self.gear_actions[ids].copy(),
            "rewards": self.rewards[ids].copy(),
            "next_states": next_states,
            "dones": self.dones[ids].copy(),
        }


class SACAgent(Agent):
    def __init__(self, config: RLConfig, device: str = "auto") -> None:
        self.config = config
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.actor = HybridActor(config).to(self.device)
        self.critic1 = HybridQCritic(config).to(self.device)
        self.critic2 = HybridQCritic(config).to(self.device)
        self.target_critic1 = HybridQCritic(config).to(self.device)
        self.target_critic2 = HybridQCritic(config).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.sac.actor_lr)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=config.sac.critic_lr)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=config.sac.critic_lr)

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=config.sac.alpha_lr)
        self.target_entropy = config.sac.target_entropy_cont + config.sac.target_entropy_disc

        self.replay = ReplayBuffer(config.sac.replay_capacity)
        self.obs_extractor = ObsExtractor()
        self.action_controller = HybridActionController(config)

        self.training_mode = True
        self.total_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def train_mode(self, mode: bool = True) -> None:
        self.training_mode = mode
        if mode:
            self.actor.train()
            self.critic1.train()
            self.critic2.train()
        else:
            self.actor.eval()
            self.critic1.eval()
            self.critic2.eval()

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def sample_action(self, state: np.ndarray, obs: Observation) -> SACActionSample:
        state_t = self._state_tensor(state)
        with torch.no_grad():
            if self.training_mode:
                cont_raw_t, gear_t, cont_lp_t, gear_lp_t, _ = self.actor.sample(state_t)
                log_prob_t = cont_lp_t + gear_lp_t
            else:
                cont_raw_t, gear_t = self.actor.greedy(state_t)
                log_prob_t, _ = self.actor.evaluate(state_t, cont_raw_t, gear_t)

        cont_raw = cont_raw_t.squeeze(0).cpu().numpy()
        gear = int(gear_t.item())
        mask_result = self.action_controller.from_policy(obs=obs, continuous_raw=cont_raw, gear_intent=gear)

        return SACActionSample(
            env_action=mask_result.action,
            cont_raw=cont_raw,
            gear=gear,
            log_prob=float(log_prob_t.item()),
            mask_modified=mask_result.modified,
            mask_reasons=mask_result.reasons,
        )

    def act(self, observation: Observation) -> dict[str, float]:
        state = self.obs_extractor(observation)
        sample = self.sample_action(state=state, obs=observation)
        return sample.env_action

    def _gear_one_hot(self, gear_actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(gear_actions, num_classes=self.config.gear_action_dim).float()

    def maybe_update(self) -> dict[str, float] | None:
        self.total_steps += 1
        if len(self.replay) < self.config.sac.warmup_steps:
            return None

        metrics = {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "alpha_loss": 0.0,
            "alpha": float(self.alpha.item()),
            "q_value": 0.0,
            "updates": 0.0,
        }

        for _ in range(self.config.sac.updates_per_step):
            batch = self.replay.sample(self.config.sac.batch_size)
            states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
            cont_actions = torch.tensor(batch["cont_actions"], dtype=torch.float32, device=self.device)
            gear_actions = torch.tensor(batch["gear_actions"], dtype=torch.int64, device=self.device)
            rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device)
            next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
            dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device)

            gear_one_hot = self._gear_one_hot(gear_actions)

            with torch.no_grad():
                next_cont, next_gear, next_cont_lp, next_gear_lp, _ = self.actor.sample(next_states)
                next_log_prob = next_cont_lp + next_gear_lp
                next_gear_oh = self._gear_one_hot(next_gear)
                target_q1 = self.target_critic1(next_states, next_cont, next_gear_oh)
                target_q2 = self.target_critic2(next_states, next_cont, next_gear_oh)
                target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
                q_target = rewards + self.config.sac.gamma * (1.0 - dones) * target_q

            q1 = self.critic1(states, cont_actions, gear_one_hot)
            q2 = self.critic2(states, cont_actions, gear_one_hot)
            critic1_loss = F.mse_loss(q1, q_target)
            critic2_loss = F.mse_loss(q2, q_target)

            self.critic1_optim.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
            self.critic1_optim.step()

            self.critic2_optim.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
            self.critic2_optim.step()

            sampled_cont, sampled_gear, cont_lp, gear_lp, _ = self.actor.sample(states)
            sampled_log_prob = cont_lp + gear_lp
            sampled_gear_oh = self._gear_one_hot(sampled_gear)
            q1_pi = self.critic1(states, sampled_cont, sampled_gear_oh)
            q2_pi = self.critic2(states, sampled_cont, sampled_gear_oh)
            q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = (self.alpha.detach() * sampled_log_prob - q_pi).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optim.step()

            alpha_loss = -(self.log_alpha * (sampled_log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            tau = self.config.sac.tau
            with torch.no_grad():
                for target_param, src_param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(tau * src_param.data)
                for target_param, src_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                    target_param.data.mul_(1.0 - tau).add_(tau * src_param.data)

            metrics["critic_loss"] += float((critic1_loss + critic2_loss).item() * 0.5)
            metrics["actor_loss"] += float(actor_loss.item())
            metrics["alpha_loss"] += float(alpha_loss.item())
            metrics["q_value"] += float(q_pi.mean().item())
            metrics["updates"] += 1.0

        denom = max(metrics["updates"], 1.0)
        metrics["alpha"] = float(self.alpha.item())
        return {
            "critic_loss": metrics["critic_loss"] / denom,
            "actor_loss": metrics["actor_loss"] / denom,
            "alpha_loss": metrics["alpha_loss"] / denom,
            "alpha": metrics["alpha"],
            "q_value": metrics["q_value"] / denom,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "algorithm": "sac",
                "config": {
                    "state_dim": self.config.state_dim,
                    "continuous_action_dim": self.config.continuous_action_dim,
                    "gear_action_dim": self.config.gear_action_dim,
                },
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic1_optim": self.critic1_optim.state_dict(),
                "critic2_optim": self.critic2_optim.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu().item(),
                "alpha_optim": self.alpha_optim.state_dict(),
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2"])
        self.log_alpha.data.fill_(float(checkpoint.get("log_alpha", 0.0)))

        if "actor_optim" in checkpoint:
            self.actor_optim.load_state_dict(checkpoint["actor_optim"])
        if "critic1_optim" in checkpoint:
            self.critic1_optim.load_state_dict(checkpoint["critic1_optim"])
        if "critic2_optim" in checkpoint:
            self.critic2_optim.load_state_dict(checkpoint["critic2_optim"])
        if "alpha_optim" in checkpoint:
            self.alpha_optim.load_state_dict(checkpoint["alpha_optim"])

        self.total_steps = int(checkpoint.get("total_steps", 0))

