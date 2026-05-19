from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from racing_ai.math2d import Point, angle_to_vector, clamp, dot


@dataclass(slots=True)
class CarPhysics:
    speed: float
    forward_speed: float
    lateral_speed: float
    slip_angle: float
    drift_intensity: float


class Car:
    def __init__(self, position: Point, heading: float) -> None:
        self.x = position[0]
        self.y = position[1]
        self.heading = heading
        self.vx = 0.0
        self.vy = 0.0
        self.length = 44.0
        self.width = 24.0

    @property
    def position(self) -> Point:
        return (self.x, self.y)

    @property
    def velocity(self) -> Point:
        return (self.vx, self.vy)

    def reset(self, position: Point, heading: float) -> None:
        self.x = position[0]
        self.y = position[1]
        self.heading = heading
        self.vx = 0.0
        self.vy = 0.0

    def update(self, action: Mapping[str, float], dt: float) -> CarPhysics:
        throttle = clamp(float(action.get("throttle", 0.0)), -1.0, 1.0)
        steer = clamp(float(action.get("steer", 0.0)), -1.0, 1.0)
        brake = clamp(float(action.get("brake", 0.0)), 0.0, 1.0)

        forward = angle_to_vector(self.heading)
        right = (-forward[1], forward[0])
        forward_speed = dot(self.velocity, forward)
        lateral_speed = dot(self.velocity, right)

        engine_accel = 460.0 * throttle
        brake_accel = -math.copysign(720.0 * brake, forward_speed) if abs(forward_speed) > 0.5 else 0.0
        rolling_drag = -forward_speed * 0.65
        aero_drag = -forward_speed * abs(forward_speed) * 0.0025
        forward_speed += (engine_accel + brake_accel + rolling_drag + aero_drag) * dt

        steering_speed = abs(forward_speed)
        turn_rate = steer * min(3.0, steering_speed / 105.0) * 1.65
        self.heading += turn_rate * dt

        # Lower lateral grip during hard steering so scripted/AI actions can create drift.
        grip = 5.2 - abs(steer) * 2.8
        grip = clamp(grip, 1.7, 5.2)
        lateral_speed *= max(0.0, 1.0 - grip * dt)

        if brake > 0.5 and abs(steer) > 0.25 and abs(forward_speed) > 80.0:
            lateral_speed += -steer * abs(forward_speed) * 0.55 * dt

        forward = angle_to_vector(self.heading)
        right = (-forward[1], forward[0])
        self.vx = forward[0] * forward_speed + right[0] * lateral_speed
        self.vy = forward[1] * forward_speed + right[1] * lateral_speed
        self.x += self.vx * dt
        self.y += self.vy * dt

        speed = math.hypot(self.vx, self.vy)
        slip_angle = math.atan2(lateral_speed, abs(forward_speed) + 1e-6)
        drift_intensity = 0.0
        if speed > 45.0:
            drift_intensity = clamp((abs(slip_angle) - 0.14) / 0.75, 0.0, 1.0)

        return CarPhysics(
            speed=speed,
            forward_speed=forward_speed,
            lateral_speed=lateral_speed,
            slip_angle=slip_angle,
            drift_intensity=drift_intensity,
        )
