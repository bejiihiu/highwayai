"""Advanced reward shaping."""
from __future__ import annotations

import math
from typing import Mapping

from racing_ai.agent import Observation


class RewardShaper:
    """Enhances the base frame reward with dense shaping for drifting, lines, and stunts."""

    def __init__(self, clip_val: float = 15.0) -> None:
        self.clip_val = clip_val
        self.drift_duration = 0.0
        self.prev_steer = 0.0
        self.prev_heading = 0.0

    def reset(self) -> None:
        self.drift_duration = 0.0
        self.prev_steer = 0.0
        self.prev_heading = 0.0

    def __call__(self, prev_obs: Observation, next_obs: Observation, action: Mapping[str, float]) -> float:
        # Start with base reward
        reward = float(next_obs.get("frame_reward", 0.0))
        
        speed = float(next_obs.get("speed", 0.0))
        drift_intensity = float(next_obs.get("drift_intensity", 0.0))
        off_track = bool(next_obs.get("off_track", False))
        heading_error = float(next_obs.get("heading_error", 0.0))
        stalled = bool(next_obs.get("stalled", False))
        
        steer = action.get("steer", 0.0)
        brake = action.get("brake", 0.0)
        throttle = action.get("throttle", 0.0)
        
        # 1. Speed bonus
        if not off_track and not stalled:
            reward += 0.3 * (speed / 400.0)
            
        # 2. Sustained drift bonus
        if drift_intensity > 0.5 and not off_track:
            self.drift_duration += 1.0 / 60.0  # approximate dt
            if self.drift_duration > 1.5:
                reward += 5.0  # huge bonus for holding it
                self.drift_duration = 0.0 # reset so we don't spam it every frame
        else:
            self.drift_duration = 0.0
            
        # 3. Trail-braking (steer + brake)
        if brake > 0.1 and abs(steer) > 0.1 and not off_track:
            reward += 1.5 * (1.0 / 60.0)
            
        # 4. Power-over (throttle + drift)
        if throttle > 0.7 and drift_intensity > 0.3 and not off_track:
            reward += 4.0 * (1.0 / 60.0)
            
        # 5. Smooth steering
        steer_delta = steer - self.prev_steer
        if abs(steer_delta) < 0.15:
            reward += 0.5 * (1.0 / 60.0)
            
        # 6. Idle penalty
        if speed < 15.0 and not stalled:
            reward -= 1.0 * (1.0 / 60.0)
            
        # 7. Reverse driving penalty
        forward_speed = float(next_obs.get("forward_speed", 0.0))
        if forward_speed < -10.0:
            reward -= 3.0 * (1.0 / 60.0)
            
        # 8. Heading deviation penalty
        reward -= 0.5 * abs(heading_error) * (1.0 / 60.0)
        
        # 9. Gear hunting penalty
        gear_shifts = int(next_obs.get("gear_shift_count", 0))
        if gear_shifts > 4:
            reward -= 2.0 * (1.0 / 60.0)

        # 10. Scandinavian flick (rapid steer reversal causing drift)
        # Check if steer changed sign rapidly and drift started
        if steer * self.prev_steer < -0.5 and drift_intensity > 0.2:
             reward += 8.0 * (1.0 / 60.0)

        # State updates
        self.prev_steer = float(steer)
        self.prev_heading = float(next_obs.get("heading", 0.0))

        # Clip and return
        return max(-self.clip_val, min(self.clip_val, reward))
