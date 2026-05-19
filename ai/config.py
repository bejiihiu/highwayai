"""RL hyperparameters and runtime configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PPOHyperParams:
    rollout_steps: int = 1024
    update_epochs: int = 8
    minibatch_size: int = 256
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.015
    max_grad_norm: float = 1.0
    target_kl: float = 0.03


@dataclass(slots=True)
class SACHyperParams:
    replay_capacity: int = 250_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 2.0e-4
    critic_lr: float = 2.0e-4
    alpha_lr: float = 1.0e-4
    target_entropy_cont: float = -5.0
    target_entropy_disc: float = -1.0
    warmup_steps: int = 3500
    updates_per_step: int = 1


@dataclass(slots=True)
class RLConfig:
    # Observation / action dimensions
    state_dim: int = 60
    continuous_action_dim: int = 5  # throttle, steer, brake, clutch, handbrake
    gear_action_dim: int = 3        # down, hold, up

    hidden_dims: tuple[int, ...] = (384, 256, 192)

    # Generic training
    base_lr: float = 2.5e-4
    reward_clip: float = 18.0
    max_steps_per_episode: int = 3600
    save_every_episodes: int = 50
    checkpoint_dir: str = "ai/checkpoints"
    log_every_steps: int = 180

    ppo: PPOHyperParams = field(default_factory=PPOHyperParams)
    sac: SACHyperParams = field(default_factory=SACHyperParams)
