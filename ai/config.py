"""DQN hyperparameters and configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DQNConfig:
    """Central configuration for the DQN agent."""

    # --- Observation space ---
    state_dim: int = 47

    # --- Discretised action space ---
    # throttle: -1, -0.5, 0, 0.5, 1  (5)
    # steer:    -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1  (9)
    # brake:    0, 0.5, 1  (3)
    # handbrake: 0, 1  (2)
    # gear:     hold, up, down  (3)
    # total = 5 * 9 * 3 * 2 * 3 = 810
    throttle_bins: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    steer_bins: tuple[float, ...] = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
    brake_bins: tuple[float, ...] = (0.0, 0.5, 1.0)
    handbrake_bins: tuple[float, ...] = (0.0, 1.0)
    gear_actions: tuple[str, ...] = ("hold", "up", "down")

    @property
    def num_actions(self) -> int:
        return (
            len(self.throttle_bins)
            * len(self.steer_bins)
            * len(self.brake_bins)
            * len(self.handbrake_bins)
            * len(self.gear_actions)
        )

    # --- Network architecture ---
    hidden_dims: tuple[int, ...] = (512, 384, 256)

    # --- Training ---
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 200_000
    target_update_freq: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 80_000
    tau: float = 0.005  # soft target update weight

    # --- Reward shaping ---
    reward_clip: float = 15.0

    # --- Checkpointing ---
    save_every_episodes: int = 50
    checkpoint_dir: str = "ai/checkpoints"

    # --- Training loop ---
    max_steps_per_episode: int = 3600  # 60s at 60fps
    max_episodes: int = 5000
    warmup_steps: int = 1000  # fill replay before training
