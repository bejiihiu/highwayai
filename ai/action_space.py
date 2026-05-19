"""Discrete to continuous action mapping."""
from __future__ import annotations

from ai.config import DQNConfig


class ActionSpace:
    """Decodes a single integer into the multi-axis continuous action dict."""

    def __init__(self, config: DQNConfig) -> None:
        self.config = config

    def decode(self, action_idx: int) -> dict[str, float]:
        """Convert an integer [0, 809] to continuous actions."""
        # Unroll the flat index
        # order: gear -> handbrake -> brake -> steer -> throttle
        
        c = self.config
        
        # 1. Throttle
        n_thr = len(c.throttle_bins)
        thr_idx = action_idx % n_thr
        action_idx //= n_thr
        
        # 2. Steer
        n_str = len(c.steer_bins)
        str_idx = action_idx % n_str
        action_idx //= n_str
        
        # 3. Brake
        n_brk = len(c.brake_bins)
        brk_idx = action_idx % n_brk
        action_idx //= n_brk
        
        # 4. Handbrake
        n_hb = len(c.handbrake_bins)
        hb_idx = action_idx % n_hb
        action_idx //= n_hb
        
        # 5. Gear
        gear_idx = action_idx
        
        gear_action = c.gear_actions[gear_idx]
        gear_up = 1.0 if gear_action == "up" else 0.0
        gear_down = 1.0 if gear_action == "down" else 0.0
        
        # Auto-clutch logic: if we are shifting, fully press clutch. Otherwise 0.
        # This simplifies the action space for the agent while keeping the manual gearbox physics intact.
        clutch = 1.0 if (gear_up > 0.0 or gear_down > 0.0) else 0.0

        return {
            "throttle": c.throttle_bins[thr_idx],
            "steer": c.steer_bins[str_idx],
            "brake": c.brake_bins[brk_idx],
            "handbrake": c.handbrake_bins[hb_idx],
            "gear_up": gear_up,
            "gear_down": gear_down,
            "clutch": clutch,
        }
