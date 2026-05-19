from __future__ import annotations

import math
from typing import Mapping, Protocol


Action = dict[str, float]
Observation = dict[str, object]


class Agent(Protocol):
    def act(self, observation: Observation) -> Mapping[str, float]:
        """Return control values for throttle/steer/brake and optional drivetrain keys.

        Optional keys supported by the world are:
        clutch, handbrake, gear_up, gear_down.
        """


class ZeroAgent:
    """Default no-control agent. The car stays at the start until an AI is connected."""

    def act(self, observation: Observation) -> Mapping[str, float]:
        return {"throttle": 0.0, "steer": 0.0, "brake": 0.0}


class ScriptedDemoAgent:
    """Optional visual smoke-test agent. It is not used by default."""

    def act(self, observation: Observation) -> Mapping[str, float]:
        speed = float(observation.get("speed", 0.0))
        heading_error = float(observation.get("heading_error", 0.0))
        drift = float(observation.get("drift_intensity", 0.0))
        off_track = bool(observation.get("off_track", False))

        throttle = 0.65 if speed < 260.0 else 0.2
        brake = 0.35 if off_track else 0.0
        steer = max(-1.0, min(1.0, heading_error * 1.7))

        # Add a tiny oscillation so the drift sensors visibly change in demo mode.
        time_alive = float(observation.get("time", 0.0))
        steer += math.sin(time_alive * 1.8) * 0.08
        if drift > 0.55:
            throttle *= 0.6

        return {"throttle": throttle, "steer": steer, "brake": brake}
