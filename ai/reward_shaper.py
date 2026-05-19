"""Reward shaping with explicit gearbox diagnostics."""
from __future__ import annotations

import math
from typing import Mapping

from racing_ai.agent import Observation


class RewardShaper:
    """Enhances frame reward and records component-level diagnostics."""

    def __init__(self, clip_val: float = 15.0) -> None:
        self.clip_val = clip_val
        self.drift_duration = 0.0
        self.prev_steer = 0.0
        self.prev_heading = 0.0
        self.last_components: dict[str, float] = {}

    def reset(self) -> None:
        self.drift_duration = 0.0
        self.prev_steer = 0.0
        self.prev_heading = 0.0
        self.last_components = {}

    def __call__(self, prev_obs: Observation, next_obs: Observation, action: Mapping[str, float]) -> float:
        components: dict[str, float] = {}

        def add(name: str, value: float) -> None:
            components[name] = components.get(name, 0.0) + value

        add("base", float(next_obs.get("frame_reward", 0.0)))

        speed = float(next_obs.get("speed", 0.0))
        forward_speed = float(next_obs.get("forward_speed", 0.0))
        prev_forward_speed = float(prev_obs.get("forward_speed", 0.0))
        prev_speed = float(prev_obs.get("speed", 0.0))
        drift_intensity = float(next_obs.get("drift_intensity", 0.0))
        prev_drift_intensity = float(prev_obs.get("drift_intensity", 0.0))
        off_track = bool(next_obs.get("off_track", False))
        heading_error = float(next_obs.get("heading_error", 0.0))
        stalled = bool(next_obs.get("stalled", False))
        lateral_g = abs(float(next_obs.get("lateral_g", 0.0)))
        traction = float(next_obs.get("traction", 0.0))
        wheel_spin_rear = float(next_obs.get("wheel_spin_rear", 0.0))
        time_prev = float(prev_obs.get("time", 0.0))
        time_next = float(next_obs.get("time", 0.0))
        dt = max(1e-3, time_next - time_prev)

        steer = float(action.get("steer", 0.0))
        brake = float(action.get("brake", 0.0))
        throttle = float(action.get("throttle", 0.0))
        clutch = float(next_obs.get("clutch", 0.0))

        # Driving quality bonuses
        if not off_track and not stalled:
            add("speed_bonus", 0.22 * (speed / 90.0))

        # Encourage active acceleration while preserving traction.
        forward_accel = (forward_speed - prev_forward_speed) / dt
        if not off_track and throttle > 0.18 and forward_accel > 0.0:
            accel_score = min(forward_accel / 145.0, 1.0)
            add("acceleration_bonus", 0.14 * accel_score)

        # Extra launch credit so the agent gets out of slow zones faster.
        if not off_track and speed < 18.0 and prev_speed < speed and throttle > 0.45:
            add("launch_accel_bonus", 0.05)

        # Reward clean, stable high-speed driving on the racing line.
        if not off_track and speed > 62.0 and abs(heading_error) < 0.28 and drift_intensity < 0.45:
            add("high_speed_stability_bonus", 0.09)

        # Reward carrying speed in cornering load when the car stays controlled.
        if not off_track and lateral_g > 0.58 and speed > 30.0 and abs(heading_error) < 0.58:
            add("cornering_speed_bonus", 0.075)

        # Reward entering a drift and holding it in a controllable range.
        if (
            not off_track
            and prev_drift_intensity < 0.34
            and drift_intensity >= 0.44
            and speed > 28.0
            and abs(heading_error) < 0.9
        ):
            add("drift_entry_bonus", 0.24)

        if not off_track and 0.35 <= drift_intensity <= 0.82 and speed > 24.0 and abs(heading_error) < 1.05:
            add("controlled_drift_bonus", 0.085 * drift_intensity)

        # Penalize unstable wheelspin/spinouts that do not contribute to forward pace.
        if throttle > 0.6 and traction < 0.24 and wheel_spin_rear > 0.52:
            add("traction_loss_penalty", -0.11)
        if drift_intensity > 0.93 and speed < 20.0:
            add("spinout_penalty", -0.14)

        prev_progress = float(prev_obs.get("progress", 0.0))
        progress = float(next_obs.get("progress", 0.0))
        delta_progress = progress - prev_progress
        if delta_progress < -0.5:
            delta_progress += 1.0
        if delta_progress > 0.0:
            add("progress_bonus", delta_progress * 6.5)

        if drift_intensity > 0.5 and not off_track:
            self.drift_duration += 1.0 / 60.0
            if self.drift_duration > 1.3:
                add("sustained_drift_bonus", 0.6)
                self.drift_duration = 0.0
        else:
            self.drift_duration = 0.0

        if brake > 0.1 and abs(steer) > 0.1 and not off_track:
            add("trail_brake_bonus", 1.0 / 60.0)

        steer_delta = steer - self.prev_steer
        if abs(steer_delta) < 0.16 and speed > 3.0:
            add("smooth_steer_bonus", 0.4 / 60.0)

        if speed < 1.2 and throttle < 0.2 and not stalled:
            add("idle_penalty", -1.1 / 60.0)

        if forward_speed < -1.0:
            add("reverse_penalty", -2.2 / 60.0)

        add("heading_penalty", -0.38 * abs(heading_error) * (1.0 / 60.0))

        # Gearbox-focused shaping
        shift_attempted = bool(next_obs.get("shift_attempted", False))
        shift_applied = bool(next_obs.get("shift_applied", False))
        shift_blocked = bool(next_obs.get("shift_blocked", False))
        overrev = bool(next_obs.get("overrev", False))
        money_shift_event = bool(next_obs.get("money_shift_event", False))
        stall_event = bool(next_obs.get("stall_event", False))
        clutch_slip = float(next_obs.get("clutch_slip", 0.0))
        rpm = float(next_obs.get("rpm", 0.0))

        if shift_applied and clutch > 0.55 and not overrev:
            add("valid_shift_bonus", 0.26)

        if shift_attempted and shift_blocked:
            add("invalid_shift_penalty", -0.23)

        if stall_event:
            add("stall_penalty", -1.2)

        if money_shift_event:
            add("money_shift_penalty", -1.4)

        if overrev and throttle > 0.25:
            add("overrev_penalty", -0.22)

        if speed > 2.0 and 2300.0 <= rpm <= 6500.0 and throttle > 0.1:
            add("rpm_band_bonus", 0.04)
        elif throttle > 0.2 and (rpm < 1200.0 or rpm > 7600.0):
            add("rpm_band_penalty", -0.05)

        if clutch_slip > 0.72 and throttle > 0.6:
            add("clutch_abuse_penalty", -0.06)

        gear_hunting = int(next_obs.get("gear_shift_count", 0))
        if gear_hunting > 4:
            add("gear_hunting_penalty", -0.03 * (gear_hunting - 4))

        if steer * self.prev_steer < -0.5 and drift_intensity > 0.2:
            add("scandi_flick_bonus", 2.8 / 60.0)

        reward = sum(components.values())
        clipped = max(-self.clip_val, min(self.clip_val, reward))
        components["clip_adjustment"] = clipped - reward
        components["total_unclipped"] = reward
        components["total_clipped"] = clipped
        self.last_components = components

        self.prev_steer = steer
        self.prev_heading = float(next_obs.get("heading", 0.0))
        return clipped
