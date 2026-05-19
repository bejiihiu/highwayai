"""Observation extraction and processing."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from racing_ai.agent import Observation


class ObsExtractor:
    """Converts the rich Observation dictionary into a flat 47-dim vector."""

    def __call__(self, obs: Observation) -> np.ndarray:
        vec = []

        # 0-10: Rays
        rays = obs.get("rays", [])
        for i in range(11):
            if i < len(rays):
                vec.append(float(rays[i]["normalized_distance"]))
            else:
                vec.append(1.0)

        # 11-13: Speeds
        vec.append(float(obs.get("forward_speed", 0.0)) / 400.0)
        vec.append(float(obs.get("lateral_speed", 0.0)) / 200.0)
        vec.append(float(obs.get("speed", 0.0)) / 400.0)

        # 14-15: Angles
        vec.append(float(obs.get("heading_error", 0.0)) / math.pi)
        vec.append(float(obs.get("slip_angle", 0.0)) / math.pi)

        # 16: Drift
        vec.append(float(obs.get("drift_intensity", 0.0)))

        # 17-18: Track positioning
        vec.append(float(obs.get("distance_to_center", 0.0)) / 59.0)  # ~ half width
        vec.append(float(obs.get("edge_clearance", 0.0)) / 59.0)

        # 19: RPM
        vec.append(float(obs.get("rpm", 0.0)) / 8000.0)

        # 20-26: Gear (One-hot encoding for R, N, 1, 2, 3, 4, 5)
        gear = int(obs.get("gear", 1))
        gear_one_hot = [0.0] * 7
        idx = gear + 1  # -1 -> 0, 0 -> 1, 1 -> 2, etc.
        if 0 <= idx < 7:
            gear_one_hot[idx] = 1.0
        vec.extend(gear_one_hot)

        # 27: Clutch
        vec.append(float(obs.get("clutch", 0.0)))

        # 28: Stalled
        vec.append(1.0 if obs.get("stalled", False) else 0.0)

        # 29-32: Physics & G-forces
        vec.append(float(obs.get("yaw_rate", 0.0)) / 10.0)
        vec.append(float(obs.get("longitudinal_g", 0.0)))
        vec.append(float(obs.get("lateral_g", 0.0)))
        vec.append(float(obs.get("angular_velocity", 0.0)) / 10.0)

        # 33-41: Nearest Markers (up to 3)
        nearest = obs.get("nearest_markers", [])
        for i in range(3):
            if i < len(nearest):
                m = nearest[i]
                # Normalize dx, dy by ~1000 pixels (a reasonable local sight distance)
                vec.append(float(m["dx"]) / 1000.0)
                vec.append(float(m["dy"]) / 1000.0)
                # Normalize reward by ~7
                vec.append(float(m["reward"]) / 7.0)
            else:
                vec.extend([0.0, 0.0, 0.0])

        # 42: Progress
        vec.append(float(obs.get("progress", 0.0)))

        # 43-44: Flags
        vec.append(1.0 if obs.get("off_track", False) else 0.0)
        vec.append(1.0 if obs.get("edge_collision", False) else 0.0)

        # 45-46: Loads
        vec.append(float(obs.get("front_load", 0.0)) / 100000.0)
        vec.append(float(obs.get("rear_load", 0.0)) / 100000.0)

        # Total should be 47
        return np.array(vec, dtype=np.float32)
