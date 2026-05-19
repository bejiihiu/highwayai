"""Hybrid action mapping and hierarchical masking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ai.config import RLConfig
from racing_ai.agent import Observation
from racing_ai.math2d import clamp


@dataclass(slots=True)
class ActionMaskResult:
    action: dict[str, float]
    modified: bool
    reasons: list[str]


class HybridActionController:
    """Maps policy outputs into env actions and applies safety/realism masks."""
    UPSHIFT_ASSIST_MIN_RPM = 6100.0
    UPSHIFT_ASSIST_MIN_SPEED = 8.0
    DOWNSHIFT_ASSIST_MAX_RPM = 1600.0
    DOWNSHIFT_ASSIST_MIN_SPEED = 6.0
    SHIFT_ASSIST_MIN_CLUTCH = 0.72

    def __init__(self, config: RLConfig) -> None:
        self.config = config

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def decode_policy_output(self, continuous_raw: np.ndarray, gear_intent: int) -> dict[str, float]:
        throttle = clamp(float(np.tanh(continuous_raw[0])), -1.0, 1.0)
        steer = clamp(float(np.tanh(continuous_raw[1])), -1.0, 1.0)
        brake = clamp(self._sigmoid(float(continuous_raw[2])), 0.0, 1.0)
        clutch = clamp(self._sigmoid(float(continuous_raw[3])), 0.0, 1.0)
        handbrake = clamp(self._sigmoid(float(continuous_raw[4])), 0.0, 1.0)

        return {
            "throttle": throttle,
            "steer": steer,
            "brake": brake,
            "clutch": clutch,
            "handbrake": handbrake,
            "gear_intent": float(int(gear_intent)),
            "gear_up": 1.0 if int(gear_intent) == 2 else 0.0,
            "gear_down": 1.0 if int(gear_intent) == 0 else 0.0,
        }

    def apply_mask(self, obs: Observation, action: dict[str, float]) -> ActionMaskResult:
        masked = dict(action)
        reasons: list[str] = []

        speed = abs(float(obs.get("forward_speed", 0.0)))
        gear = int(obs.get("gear", 1))
        shift_lockout = float(obs.get("shift_lockout", 0.0))
        stalled = bool(obs.get("stalled", False))
        off_track = bool(obs.get("off_track", False))
        rpm = float(obs.get("rpm", 0.0))
        throttle = float(masked.get("throttle", 0.0))
        brake = float(masked.get("brake", 0.0))
        clutch = float(masked.get("clutch", 0.0))
        gear_intent = int(masked.get("gear_intent", 1.0))

        if throttle > 0.25 and brake > 0.25:
            if throttle >= brake:
                masked["brake"] = 0.0
                reasons.append("throttle_brake_conflict_drop_brake")
            else:
                masked["throttle"] = 0.0
                reasons.append("throttle_brake_conflict_drop_throttle")

        if stalled and clutch < 0.75:
            masked["clutch"] = 0.82
            reasons.append("stall_recovery_force_clutch")

        if shift_lockout > 0.0 and gear_intent != 1:
            gear_intent = 1
            reasons.append("shift_lockout_active")

        if gear_intent == 2 and gear >= 5:
            gear_intent = 1
            reasons.append("already_top_gear")

        if gear_intent == 0 and gear <= -1:
            gear_intent = 1
            reasons.append("already_reverse")

        if gear_intent == 0 and gear == 0 and speed > 4.0:
            gear_intent = 1
            reasons.append("reverse_speed_lockout")

        if gear_intent != 1 and clutch < 0.55:
            gear_intent = 1
            reasons.append("clutch_low_for_shift")

        if speed < 0.7 and abs(float(masked.get("steer", 0.0))) > 0.35 and throttle < 0.2:
            masked["steer"] = float(np.sign(float(masked["steer"]))) * 0.35
            reasons.append("low_speed_steer_damped")

        # Early-training launch helper:
        # prevent random brake/clutch/neutral behavior from pinning the car at spawn.
        if not stalled and speed < 1.2 and gear >= 1:
            if float(masked.get("throttle", 0.0)) < 0.35:
                masked["throttle"] = 0.35
                reasons.append("launch_min_throttle")
            if float(masked.get("brake", 0.0)) > 0.12:
                masked["brake"] = 0.0
                reasons.append("launch_drop_brake")
            if float(masked.get("clutch", 0.0)) > 0.32:
                masked["clutch"] = 0.2
                reasons.append("launch_clutch_engage")
            if float(masked.get("handbrake", 0.0)) > 0.05:
                masked["handbrake"] = 0.0
                reasons.append("launch_drop_handbrake")

        # Keep first gear during launch instead of bouncing into neutral/reverse.
        if speed < 2.0 and gear == 1 and gear_intent == 0:
            gear_intent = 1
            reasons.append("launch_hold_first_gear")

        # Recover from accidental neutral at near standstill.
        if speed < 0.8 and gear == 0:
            gear_intent = 2
            if float(masked.get("clutch", 0.0)) < self.SHIFT_ASSIST_MIN_CLUTCH:
                masked["clutch"] = self.SHIFT_ASSIST_MIN_CLUTCH
            reasons.append("launch_shift_from_neutral")

        # Shift assist for early training: helps policy escape first-gear lock.
        # Only engages while policy is "holding gear" and no lockout is active.
        if not stalled and shift_lockout <= 0.0 and gear_intent == 1 and not off_track:
            if (
                gear >= 1
                and gear < 5
                and speed >= self.UPSHIFT_ASSIST_MIN_SPEED
                and rpm >= self.UPSHIFT_ASSIST_MIN_RPM
                and throttle > 0.22
                and brake < 0.2
            ):
                gear_intent = 2
                if float(masked.get("clutch", 0.0)) < self.SHIFT_ASSIST_MIN_CLUTCH:
                    masked["clutch"] = self.SHIFT_ASSIST_MIN_CLUTCH
                reasons.append("assist_upshift")
            elif (
                gear > 1
                and speed >= self.DOWNSHIFT_ASSIST_MIN_SPEED
                and rpm <= self.DOWNSHIFT_ASSIST_MAX_RPM
                and throttle > 0.12
                and brake < 0.35
            ):
                gear_intent = 0
                if float(masked.get("clutch", 0.0)) < self.SHIFT_ASSIST_MIN_CLUTCH:
                    masked["clutch"] = self.SHIFT_ASSIST_MIN_CLUTCH
                reasons.append("assist_downshift")

        masked["gear_intent"] = float(gear_intent)
        masked["gear_up"] = 1.0 if gear_intent == 2 else 0.0
        masked["gear_down"] = 1.0 if gear_intent == 0 else 0.0

        return ActionMaskResult(action=masked, modified=len(reasons) > 0, reasons=reasons)

    def from_policy(self, obs: Observation, continuous_raw: np.ndarray, gear_intent: int) -> ActionMaskResult:
        decoded = self.decode_policy_output(continuous_raw=continuous_raw, gear_intent=gear_intent)
        return self.apply_mask(obs=obs, action=decoded)

    @staticmethod
    def sanitize_env_action(action: dict[str, Any]) -> dict[str, float]:
        """Normalize action dict for direct env stepping and logging."""
        return {
            "throttle": clamp(float(action.get("throttle", 0.0)), -1.0, 1.0),
            "steer": clamp(float(action.get("steer", 0.0)), -1.0, 1.0),
            "brake": clamp(float(action.get("brake", 0.0)), 0.0, 1.0),
            "clutch": clamp(float(action.get("clutch", 0.0)), 0.0, 1.0),
            "handbrake": clamp(float(action.get("handbrake", 0.0)), 0.0, 1.0),
            "gear_up": 1.0 if float(action.get("gear_up", 0.0)) > 0.5 else 0.0,
            "gear_down": 1.0 if float(action.get("gear_down", 0.0)) > 0.5 else 0.0,
        }
