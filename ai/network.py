"""Dueling DQN network architecture."""
from __future__ import annotations

import torch
import torch.nn as nn

from ai.config import DQNConfig


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network.

    Architecture:
        shared → (value_stream, advantage_stream) → Q(s,a)
    """

    def __init__(self, config: DQNConfig) -> None:
        super().__init__()
        dims = config.hidden_dims  # e.g. (512, 384, 256)
        self.num_actions = config.num_actions

        # Shared feature extractor
        layers: list[nn.Module] = []
        in_dim = config.state_dim
        for h in dims[:-1]:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        self.features = nn.Sequential(*layers)

        last_hidden = dims[-1]

        # Value stream: single scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, last_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(last_hidden, 1),
        )

        # Advantage stream: one value per action A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, last_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(last_hidden, self.num_actions),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions. Shape: (batch, num_actions)."""
        features = self.features(state)
        value = self.value_stream(features)            # (batch, 1)
        advantage = self.advantage_stream(features)    # (batch, num_actions)
        # Dueling aggregation: Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
