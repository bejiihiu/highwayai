"""Neural networks for hybrid-action PPO/SAC."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from ai.config import RLConfig


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(last, hidden))
        layers.append(nn.ReLU(inplace=True))
        last = hidden
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class HybridActor(nn.Module):
    def __init__(self, config: RLConfig) -> None:
        super().__init__()
        self.config = config
        trunk_dims = config.hidden_dims
        self.trunk = _mlp(config.state_dim, trunk_dims, trunk_dims[-1])
        self.cont_mean = nn.Linear(trunk_dims[-1], config.continuous_action_dim)
        self.gear_logits = nn.Linear(trunk_dims[-1], config.gear_action_dim)
        self.log_std = nn.Parameter(torch.tensor([-0.65, -0.35, -1.25, -1.1, -1.35], dtype=torch.float32))
        self._init_weights()
        self._init_action_priors()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def _init_action_priors(self) -> None:
        # Bias initial behavior toward controllable launch:
        # throttle > 0, brake low, clutch mostly engaged, handbrake low.
        with torch.no_grad():
            self.cont_mean.bias[:] = torch.tensor([0.85, 0.0, -2.0, -1.6, -2.3], dtype=torch.float32)
            # gear logits order: down, hold, up -> prefer hold initially.
            self.gear_logits.bias[:] = torch.tensor([-1.4, 1.8, -1.1], dtype=torch.float32)

    def distributions(self, states: torch.Tensor) -> tuple[Normal, Categorical]:
        features = self.trunk(states)
        means = self.cont_mean(features)
        log_std = self.log_std.clamp(-2.8, 1.5).expand_as(means)
        std = log_std.exp()
        normal = Normal(means, std)
        categorical = Categorical(logits=self.gear_logits(features))
        return normal, categorical

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normal, categorical = self.distributions(states)
        cont_raw = normal.rsample()
        gear = categorical.sample()
        cont_log_prob = normal.log_prob(cont_raw).sum(dim=-1)
        gear_log_prob = categorical.log_prob(gear)
        entropy = normal.entropy().sum(dim=-1) + categorical.entropy()
        return cont_raw, gear, cont_log_prob, gear_log_prob, entropy

    def greedy(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normal, categorical = self.distributions(states)
        cont_raw = normal.mean
        gear = categorical.probs.argmax(dim=-1)
        return cont_raw, gear

    def evaluate(self, states: torch.Tensor, cont_raw: torch.Tensor, gear: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normal, categorical = self.distributions(states)
        cont_log_prob = normal.log_prob(cont_raw).sum(dim=-1)
        gear_log_prob = categorical.log_prob(gear)
        entropy = normal.entropy().sum(dim=-1) + categorical.entropy()
        return cont_log_prob + gear_log_prob, entropy


class ValueNet(nn.Module):
    def __init__(self, config: RLConfig) -> None:
        super().__init__()
        self.net = _mlp(config.state_dim, config.hidden_dims, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)


class HybridQCritic(nn.Module):
    def __init__(self, config: RLConfig) -> None:
        super().__init__()
        input_dim = config.state_dim + config.continuous_action_dim + config.gear_action_dim
        self.net = _mlp(input_dim, config.hidden_dims, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor, cont_actions: torch.Tensor, gear_one_hot: torch.Tensor) -> torch.Tensor:
        x = torch.cat((states, cont_actions, gear_one_hot), dim=-1)
        return self.net(x).squeeze(-1)
