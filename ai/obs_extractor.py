"""Observation extraction and processing."""
from __future__ import annotations

import math

import numpy as np

from racing_ai.agent import Observation


class ObsExtractor:
    """Converts world observation dictionary into a flat vector."""

    output_dim: int = 60

    def __call__(self, obs: Observation) -> np.ndarray:
        vec: list[float] = []

        rays = obs.get("rays", [])
        for i in range(11):
            if i < len(rays):
                vec.append(float(rays[i]["normalized_distance"]))
            else:
                vec.append(1.0)

        vec.append(float(obs.get("forward_speed", 0.0)) / 120.0)
        vec.append(float(obs.get("lateral_speed", 0.0)) / 80.0)
        vec.append(float(obs.get("speed", 0.0)) / 120.0)

        vec.append(float(obs.get("heading_error", 0.0)) / math.pi)
        vec.append(float(obs.get("slip_angle", 0.0)) / math.pi)

        vec.append(float(obs.get("drift_intensity", 0.0)))

        vec.append(float(obs.get("distance_to_center", 0.0)) / 59.0)
        vec.append(float(obs.get("edge_clearance", 0.0)) / 59.0)

        vec.append(float(obs.get("rpm", 0.0)) / 8200.0)

        gear = int(obs.get("gear", 1))
        gear_one_hot = [0.0] * 7
        idx = gear + 1
        if 0 <= idx < 7:
            gear_one_hot[idx] = 1.0
        vec.extend(gear_one_hot)

        vec.append(float(obs.get("clutch", 0.0)))
        vec.append(1.0 if bool(obs.get("stalled", False)) else 0.0)

        vec.append(float(obs.get("yaw_rate", 0.0)) / 8.0)
        vec.append(float(obs.get("longitudinal_g", 0.0)))
        vec.append(float(obs.get("lateral_g", 0.0)))
        vec.append(float(obs.get("angular_velocity", 0.0)) / 8.0)

        nearest = obs.get("nearest_markers", [])
        for i in range(3):
            if i < len(nearest):
                marker = nearest[i]
                vec.append(float(marker["dx"]) / 1000.0)
                vec.append(float(marker["dy"]) / 1000.0)
                vec.append(float(marker["reward"]) / 8.0)
            else:
                vec.extend([0.0, 0.0, 0.0])

        vec.append(float(obs.get("progress", 0.0)))

        vec.append(1.0 if bool(obs.get("off_track", False)) else 0.0)
        vec.append(1.0 if bool(obs.get("edge_collision", False)) else 0.0)

        vec.append(float(obs.get("front_load", 0.0)) / 12000.0)
        vec.append(float(obs.get("rear_load", 0.0)) / 12000.0)

        vec.append(float(obs.get("engine_load", 0.0)))
        vec.append(float(obs.get("clutch_slip", 0.0)))
        vec.append(float(obs.get("clutch_engagement", 0.0)))
        vec.append(float(obs.get("shift_lockout", 0.0)) / 0.2)
        vec.append(float(obs.get("traction", 0.0)))
        vec.append(float(obs.get("money_shift_damage", 0.0)))
        vec.append(float(obs.get("drive_force", 0.0)) / 12000.0)
        vec.append(float(obs.get("wheel_torque", 0.0)) / 4500.0)
        vec.append(1.0 if bool(obs.get("shift_attempted", False)) else 0.0)
        vec.append(1.0 if bool(obs.get("shift_applied", False)) else 0.0)
        vec.append(1.0 if bool(obs.get("shift_blocked", False)) else 0.0)
        vec.append(1.0 if bool(obs.get("overrev", False)) else 0.0)
        vec.append(1.0 if bool(obs.get("stall_event", False)) else 0.0)

        if len(vec) != self.output_dim:
            raise ValueError(f"ObsExtractor expected {self.output_dim} features, got {len(vec)}")

        return np.array(vec, dtype=np.float32)
