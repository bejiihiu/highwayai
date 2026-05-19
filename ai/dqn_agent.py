"""Main PyTorch DQN Agent implementation."""
from __future__ import annotations

import math
import random
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from ai.action_space import ActionSpace
from ai.config import DQNConfig
from ai.network import DuelingDQN
from ai.obs_extractor import ObsExtractor
from ai.replay_buffer import PrioritizedReplayBuffer
from ai.reward_shaper import RewardShaper
from racing_ai.agent import Agent, Observation


class DQNAgent(Agent):
    """Deep Q-Network Agent."""

    def __init__(self, config: DQNConfig, device: str = "auto") -> None:
        self.config = config
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy_net = DuelingDQN(config).to(self.device)
        self.target_net = DuelingDQN(config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)
        
        self.replay = PrioritizedReplayBuffer(config.replay_capacity)
        self.action_space = ActionSpace(config)
        self.obs_extractor = ObsExtractor()
        self.reward_shaper = RewardShaper(config.reward_clip)

        self.epsilon = config.eps_start
        self.steps_done = 0
        self.training_mode = False

    def train_mode(self, mode: bool = True) -> None:
        self.training_mode = mode
        if mode:
            self.policy_net.train()
        else:
            self.policy_net.eval()

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if self.training_mode and random.random() < self.epsilon:
            return random.randrange(self.config.num_actions)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def act(self, observation: Observation) -> Mapping[str, float]:
        """Implementation of the Agent protocol."""
        state = self.obs_extractor(observation)
        action_idx = self.select_action(state)
        return self.action_space.decode(action_idx)

    def train_step(self) -> float | None:
        """Sample a batch and perform a gradient descent step."""
        if len(self.replay) < self.config.batch_size or len(self.replay) < self.config.warmup_steps:
            return None

        states, actions, rewards, next_states, dones, weights, tree_indices = self.replay.sample(self.config.batch_size)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device).unsqueeze(1)
        weights_t = torch.tensor(weights, device=self.device).unsqueeze(1)

        # Q(s, a)
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # Double DQN: evaluate target using policy actions
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states_t).gather(1, next_actions)
            target_q_values = rewards_t + self.config.gamma * next_q_values * (1.0 - dones_t)

        # Compute TD error
        td_errors = (target_q_values - q_values).detach().cpu().numpy().flatten()
        self.replay.update_priorities(tree_indices, td_errors)

        # Compute loss
        loss = F.smooth_l1_loss(q_values, target_q_values, reduction="none")
        loss = (loss * weights_t).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.config.tau * policy_param.data + (1.0 - self.config.tau) * target_param.data)

        # Update epsilon
        self.epsilon = max(
            self.config.eps_end,
            self.config.eps_start - (self.config.eps_start - self.config.eps_end) * (self.steps_done / self.config.eps_decay_steps)
        )
        self.steps_done += 1

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.config.eps_end)
        self.steps_done = checkpoint.get("steps_done", self.config.eps_decay_steps)
