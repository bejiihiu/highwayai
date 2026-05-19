from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from racing_ai.math2d import Point, angle_to_vector, clamp, dot, wrap_angle


@dataclass(slots=True)
class GearboxDiagnostics:
    shift_attempted: bool = False
    shift_applied: bool = False
    shift_blocked: bool = False
    shift_block_reason: str = ""
    shift_from: int = 1
    shift_to: int = 1
    money_shift_event: bool = False
    stall_event: bool = False
    overrev: bool = False


@dataclass(slots=True)
class CarPhysics:
    speed: float = 0.0
    forward_speed: float = 0.0
    lateral_speed: float = 0.0
    slip_angle: float = 0.0
    drift_intensity: float = 0.0
    rpm: float = 1000.0
    gear: int = 1
    wheel_spin_front: float = 0.0
    wheel_spin_rear: float = 0.0
    front_load: float = 0.0
    rear_load: float = 0.0
    yaw_rate: float = 0.0
    angular_velocity: float = 0.0
    longitudinal_g: float = 0.0
    lateral_g: float = 0.0
    clutch: float = 0.0
    stalled: bool = False
    engine_load: float = 0.0
    clutch_slip: float = 0.0
    clutch_engagement: float = 1.0
    shift_lockout: float = 0.0
    shift_attempted: bool = False
    shift_applied: bool = False
    shift_blocked: bool = False
    shift_block_reason: str = ""
    money_shift_event: bool = False
    stall_event: bool = False
    overrev: bool = False
    traction: float = 0.0
    drive_force: float = 0.0
    wheel_torque: float = 0.0


class Car:
    def __init__(self, position: Point, heading: float) -> None:
        self.x = position[0]
        self.y = position[1]
        self.heading = heading
        self.vx = 0.0
        self.vy = 0.0
        self.angular_velocity = 0.0
        self.length = 44.0
        self.width = 24.0

        # Vehicle constants (tuned for stable RWD behavior in world units)
        self.mass = 1450.0
        self.inertia = 2650.0
        self.cg_to_front = self.length * 0.48
        self.cg_to_rear = self.length * 0.52
        self.wheel_base = self.cg_to_front + self.cg_to_rear
        self.cg_height = 0.52
        self.gravity = 9.81

        self.speed_to_mps = 0.42
        self.long_force_scale = 0.28
        self.lat_force_scale = 0.1
        self.yaw_scale = 0.16

        # Engine and transmission
        self.rpm = 1000.0
        self.gear = 1
        self.max_rpm = 8200.0
        self.idle_rpm = 950.0
        self.gear_ratios = [0.0, 3.82, 2.26, 1.56, 1.21, 1.0]  # N, 1, 2, 3, 4, 5
        self.reverse_ratio = -3.42
        self.final_drive = 3.91
        self.wheel_radius = 0.34
        self.drivetrain_efficiency = 0.9
        self.max_steer_angle = 0.58

        # Manual gearbox state
        self.clutch = 0.0  # 0=engaged, 1=disengaged
        self.stalled = False
        self.stall_timer = 0.0
        self.shift_lockout = 0.0
        self.prev_gear = 1
        self.money_shift_damage = 0.0
        self.gear_shift_count = 0
        self.gear_shift_window = 0.0

        # State
        self.long_accel = 0.0
        self.lat_accel = 0.0

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
        self.angular_velocity = 0.0
        self.rpm = self.idle_rpm
        self.gear = 1
        self.clutch = 0.0
        self.stalled = False
        self.stall_timer = 0.0
        self.shift_lockout = 0.0
        self.prev_gear = 1
        self.money_shift_damage = 0.0
        self.gear_shift_count = 0
        self.gear_shift_window = 0.0
        self.long_accel = 0.0
        self.lat_accel = 0.0

    def get_torque(self, rpm: float) -> float:
        rpm = clamp(rpm, self.idle_rpm, self.max_rpm)
        if rpm < 2800.0:
            return 220.0 + (rpm - self.idle_rpm) * 0.042
        if rpm < 5200.0:
            return 300.0 + (rpm - 2800.0) * 0.03
        if rpm < 6800.0:
            return 372.0 - (rpm - 5200.0) * 0.03
        return 324.0 - (rpm - 6800.0) * 0.045

    def pacejka(self, slip: float, load: float, lateral: bool) -> float:
        b = 9.6 if lateral else 11.2
        c = 1.82 if lateral else 1.62
        d = load * (1.06 if lateral else 1.1)
        e = 0.95
        x = b * slip
        return d * math.sin(c * math.atan(x - e * (x - math.atan(x))))

    def _shift_target_gear(self, signal: int) -> int:
        if signal > 0:
            if self.gear == -1:
                return 0
            if self.gear < 5:
                return self.gear + 1
            return self.gear

        if signal < 0:
            if self.gear == 0:
                return -1
            if self.gear > -1:
                return self.gear - 1
            return self.gear

        return self.gear

    def _process_gearbox(self, action: Mapping[str, float], forward_speed: float, dt: float) -> GearboxDiagnostics:
        diag = GearboxDiagnostics()

        throttle = clamp(float(action.get("throttle", 0.0)), -1.0, 1.0)
        self.clutch = clamp(float(action.get("clutch", 0.0)), 0.0, 1.0)
        gear_up = float(action.get("gear_up", 0.0)) > 0.5
        gear_down = float(action.get("gear_down", 0.0)) > 0.5

        self.shift_lockout = max(0.0, self.shift_lockout - dt)

        self.gear_shift_window = max(0.0, self.gear_shift_window - dt)
        if self.gear_shift_window <= 0.0:
            self.gear_shift_count = 0

        if self.stalled:
            self.stall_timer -= dt
            if self.clutch > 0.75 and throttle > 0.2:
                self.stall_timer -= dt * 1.8
            if self.stall_timer <= 0.0:
                self.stalled = False
                self.rpm = self.idle_rpm + max(0.0, throttle) * 450.0
            return diag

        signal = 0
        if gear_up and not gear_down:
            signal = 1
        elif gear_down and not gear_up:
            signal = -1

        if signal != 0:
            diag.shift_attempted = True
            diag.shift_from = self.gear

            if self.shift_lockout > 0.0:
                diag.shift_blocked = True
                diag.shift_block_reason = "lockout"
            elif self.clutch < 0.55:
                diag.shift_blocked = True
                diag.shift_block_reason = "clutch_low"
            else:
                target_gear = self._shift_target_gear(signal)
                diag.shift_to = target_gear

                if target_gear == self.gear:
                    diag.shift_blocked = True
                    diag.shift_block_reason = "gear_limit"
                elif target_gear == -1 and abs(forward_speed) > 4.0:
                    diag.shift_blocked = True
                    diag.shift_block_reason = "reverse_lockout"
                else:
                    diag.shift_applied = True
                    self.prev_gear = self.gear
                    self.gear = target_gear
                    self.shift_lockout = 0.16
                    self.gear_shift_count += 1
                    self.gear_shift_window = 2.2

                    if target_gear > 0 and target_gear < self.prev_gear:
                        ratio = self.gear_ratios[target_gear]
                        projected_engine_rpm = abs(
                            forward_speed
                            * self.speed_to_mps
                            / (2.0 * math.pi * self.wheel_radius)
                            * 60.0
                            * ratio
                            * self.final_drive
                        )
                        if projected_engine_rpm > self.max_rpm * 1.05:
                            diag.money_shift_event = True
                            diag.overrev = True
                            self.money_shift_damage = min(1.0, self.money_shift_damage + 0.18)
                            self.vx *= 0.78
                            self.vy *= 0.78

        if self.clutch < 0.2 and self.gear != 0 and abs(forward_speed) < 1.5 and throttle < 0.08:
            if self.rpm < self.idle_rpm * 0.72:
                self.stalled = True
                self.stall_timer = 0.8
                self.rpm = 0.0
                diag.stall_event = True

        if diag.shift_applied:
            diag.shift_to = self.gear

        return diag

    def update(self, action: Mapping[str, float], dt: float) -> CarPhysics:
        if dt <= 0.0:
            return CarPhysics(clutch=self.clutch, stalled=self.stalled, gear=self.gear)

        throttle = clamp(float(action.get("throttle", 0.0)), -1.0, 1.0)
        steer = clamp(float(action.get("steer", 0.0)), -1.0, 1.0)
        brake = clamp(float(action.get("brake", 0.0)), 0.0, 1.0)
        handbrake = clamp(float(action.get("handbrake", 0.0)), 0.0, 1.0)

        forward = angle_to_vector(self.heading)
        right = (-forward[1], forward[0])
        forward_speed = dot(self.velocity, forward)
        lateral_speed = dot(self.velocity, right)

        if math.isnan(forward_speed) or math.isnan(lateral_speed):
            self.reset(self.position, self.heading)
            forward_speed = 0.0
            lateral_speed = 0.0

        gearbox = self._process_gearbox(action, forward_speed, dt)

        if throttle < 0.0:
            if self.gear == -1:
                throttle = abs(throttle)
            elif forward_speed > 1.0:
                brake = max(brake, abs(throttle))
                throttle = 0.0
            else:
                throttle = 0.0

        if self.gear > 0:
            gear_ratio = self.gear_ratios[self.gear]
        elif self.gear == -1:
            gear_ratio = self.reverse_ratio
        else:
            gear_ratio = 0.0

        clutch_engagement = 1.0 - self.clutch

        wheel_rpm = abs(
            forward_speed
            * self.speed_to_mps
            / (2.0 * math.pi * self.wheel_radius)
            * 60.0
            * abs(gear_ratio)
            * self.final_drive
        )
        free_rev_target = self.idle_rpm + max(0.0, throttle) * (self.max_rpm - self.idle_rpm)

        if self.stalled:
            self.rpm = 0.0
        elif self.clutch > 0.92 or gear_ratio == 0.0:
            rpm_target = free_rev_target
            rpm_response = 0.0
            if self.rpm < rpm_target:
                rpm_response = 9200.0 * dt
            else:
                rpm_response = -5200.0 * dt
            self.rpm = clamp(self.rpm + rpm_response, self.idle_rpm, self.max_rpm)
        else:
            engaged_target = clamp(max(self.idle_rpm, wheel_rpm), self.idle_rpm, self.max_rpm)
            blended_target = free_rev_target * self.clutch + engaged_target * clutch_engagement
            if max(0.0, throttle) > 0.2 and clutch_engagement > 0.75 and abs(forward_speed) < 2.0:
                blended_target = max(blended_target, self.idle_rpm + max(0.0, throttle) * 1500.0)
            rpm_error = blended_target - self.rpm
            self.rpm = clamp(self.rpm + rpm_error * min(1.0, 8.5 * dt), self.idle_rpm, self.max_rpm)

        overrev = self.rpm > self.max_rpm * 0.985 or gearbox.overrev
        if overrev and throttle > 0.45:
            self.money_shift_damage = min(1.0, self.money_shift_damage + 0.0025 * dt * 60.0)

        effective_throttle = max(0.0, throttle)
        engine_torque = self.get_torque(self.rpm) * effective_throttle
        clutch_transfer = clutch_engagement ** 1.2
        transmitted_torque = engine_torque * clutch_transfer
        damage_factor = 1.0 - self.money_shift_damage * 0.45
        wheel_torque = transmitted_torque * gear_ratio * self.final_drive * self.drivetrain_efficiency * damage_factor

        static_front_load = self.mass * self.gravity * (self.cg_to_rear / self.wheel_base)
        static_rear_load = self.mass * self.gravity * (self.cg_to_front / self.wheel_base)
        weight_transfer = (self.mass * self.long_accel * self.cg_height) / max(self.wheel_base, 1e-6)
        front_load = max(0.0, static_front_load - weight_transfer)
        rear_load = max(0.0, static_rear_load + weight_transfer)

        max_drive_force = rear_load * (1.15 - 0.65 * handbrake)
        drive_force_raw = wheel_torque / max(self.wheel_radius, 1e-6)
        drive_force = clamp(drive_force_raw, -max_drive_force, max_drive_force)

        rolling_drag = -forward_speed * (128.0 + 6.8 * abs(forward_speed))
        aero_drag = -forward_speed * abs(forward_speed) * 2.2
        long_force_env = rolling_drag + aero_drag

        brake_force = 9800.0 * brake
        handbrake_force = 6200.0 * handbrake
        if abs(forward_speed) > 0.12:
            direction = math.copysign(1.0, forward_speed)
            front_brake_force = -direction * brake_force * 0.7
            rear_brake_force = -direction * (brake_force * 0.3 + handbrake_force)
        else:
            front_brake_force = 0.0
            rear_brake_force = 0.0

        speed = math.hypot(forward_speed, lateral_speed)
        speed_ref = abs(forward_speed) + max(0.0, throttle) * 6.0
        steer_authority = clamp((speed_ref - 1.2) / 12.0, 0.0, 1.0)
        traction = clamp(speed / 22.0 + 0.12, 0.1, 1.0)

        steer_angle = steer * self.max_steer_angle
        front_lat_speed = lateral_speed + self.angular_velocity * self.cg_to_front
        rear_lat_speed = lateral_speed - self.angular_velocity * self.cg_to_rear

        slip_angle_front = math.atan2(front_lat_speed, abs(forward_speed) + 4.0) - steer_angle
        slip_angle_rear = math.atan2(rear_lat_speed, abs(forward_speed) + 4.0)

        front_lat_force = -self.pacejka(slip_angle_front, front_load, True) * steer_authority * traction
        rear_lat_force = -self.pacejka(slip_angle_rear, rear_load, True) * traction * (1.0 - handbrake * 0.85)

        total_long_force = drive_force + long_force_env + front_brake_force + rear_brake_force
        total_lat_force = front_lat_force * math.cos(steer_angle) + rear_lat_force

        self.long_accel = total_long_force / self.mass * self.long_force_scale
        self.lat_accel = total_lat_force / self.mass * self.lat_force_scale

        forward_speed += self.long_accel * dt
        lateral_speed += self.lat_accel * dt

        if abs(forward_speed) < 0.2 and (brake > 0.2 or handbrake > 0.2):
            forward_speed = 0.0
        if abs(lateral_speed) < 0.3 and abs(forward_speed) < 0.4:
            lateral_speed = 0.0

        torque = (front_lat_force * self.cg_to_front * math.cos(steer_angle) - rear_lat_force * self.cg_to_rear)
        torque *= 0.28 + 0.72 * steer_authority
        angular_accel = torque / max(self.inertia, 1e-6) * self.yaw_scale
        self.angular_velocity += angular_accel * dt

        yaw_damping = 1.35 + (1.0 - steer_authority) * 2.55
        self.angular_velocity *= max(0.0, 1.0 - yaw_damping * dt)
        if speed < 1.2 and abs(steer) < 0.22:
            self.angular_velocity *= max(0.0, 1.0 - 10.0 * dt)
        if speed < 0.4 and abs(self.angular_velocity) < 0.08:
            self.angular_velocity = 0.0

        self.heading = wrap_angle(self.heading + self.angular_velocity * dt)

        forward_vec = angle_to_vector(self.heading)
        right_vec = (-forward_vec[1], forward_vec[0])
        self.vx = forward_vec[0] * forward_speed + right_vec[0] * lateral_speed
        self.vy = forward_vec[1] * forward_speed + right_vec[1] * lateral_speed

        self.x += self.vx * dt
        self.y += self.vy * dt

        speed = math.hypot(self.vx, self.vy)
        slip_angle = math.atan2(lateral_speed, abs(forward_speed) + 1e-6)
        drift_intensity = 0.0
        if speed > 5.0:
            drift_intensity = clamp((abs(slip_angle) - 0.09) / 0.65, 0.0, 1.0)

        long_g = self.long_accel / self.gravity
        lat_g = self.lat_accel / self.gravity

        engine_load = clamp(
            abs((wheel_rpm if gear_ratio != 0 else self.idle_rpm) - self.rpm) / max(self.max_rpm - self.idle_rpm, 1.0)
            + effective_throttle * 0.4,
            0.0,
            1.0,
        )
        clutch_slip = clamp(
            abs(free_rev_target - wheel_rpm) / max(self.max_rpm, 1.0) * clutch_engagement,
            0.0,
            1.0,
        )

        return CarPhysics(
            speed=speed,
            forward_speed=forward_speed,
            lateral_speed=lateral_speed,
            slip_angle=slip_angle,
            drift_intensity=drift_intensity,
            rpm=self.rpm,
            gear=self.gear,
            wheel_spin_front=0.0,
            wheel_spin_rear=max(0.0, abs(drive_force_raw - drive_force) / max(rear_load, 1.0)),
            front_load=front_load,
            rear_load=rear_load,
            yaw_rate=self.angular_velocity,
            angular_velocity=self.angular_velocity,
            longitudinal_g=long_g,
            lateral_g=lat_g,
            clutch=self.clutch,
            stalled=self.stalled,
            engine_load=engine_load,
            clutch_slip=clutch_slip,
            clutch_engagement=clutch_engagement,
            shift_lockout=self.shift_lockout,
            shift_attempted=gearbox.shift_attempted,
            shift_applied=gearbox.shift_applied,
            shift_blocked=gearbox.shift_blocked,
            shift_block_reason=gearbox.shift_block_reason,
            money_shift_event=gearbox.money_shift_event,
            stall_event=gearbox.stall_event,
            overrev=overrev,
            traction=traction,
            drive_force=drive_force,
            wheel_torque=wheel_torque,
        )
