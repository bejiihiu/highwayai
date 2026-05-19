from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from racing_ai.math2d import Point, angle_to_vector, clamp, dot, wrap_angle


@dataclass(slots=True)
class CarPhysics:
    speed: float
    forward_speed: float
    lateral_speed: float
    slip_angle: float
    drift_intensity: float
    rpm: float
    gear: int
    wheel_spin_front: float
    wheel_spin_rear: float
    front_load: float
    rear_load: float
    yaw_rate: float
    angular_velocity: float
    longitudinal_g: float
    lateral_g: float
    clutch: float
    stalled: bool


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
        
        # Physics constants
        self.mass = 1200.0
        self.inertia = 1500.0
        self.cg_to_front = self.length * 0.5
        self.cg_to_rear = self.length * 0.5
        self.wheel_base = self.cg_to_front + self.cg_to_rear
        self.cg_height = 14.0
        self.gravity = 9.81 * 10.0  # Scale gravity for the game units
        
        # Engine & Transmission
        self.rpm = 1000.0
        self.gear = 1
        self.max_rpm = 8000.0
        self.idle_rpm = 1000.0
        self.gear_ratios = [0.0, 3.5, 2.3, 1.6, 1.2, 0.9] # Neutral, 1, 2, 3, 4, 5
        self.reverse_ratio = -3.5
        self.final_drive = 3.8
        self.wheel_radius = 12.0
        
        # Manual gearbox state
        self.clutch = 0.0           # 0=fully engaged, 1=fully disengaged
        self.stalled = False
        self.stall_timer = 0.0
        self.shift_lockout = 0.0    # countdown timer (seconds)
        self.prev_gear = 1
        self.money_shift_damage = 0.0  # accumulated mechanical damage 0..1
        self.gear_shift_count = 0
        self.gear_shift_window = 0.0   # timer for gear-hunting detection
        
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
        # Simple torque curve
        rpm = clamp(rpm, self.idle_rpm, self.max_rpm)
        peak_rpm = 5500.0
        # Peak torque 350 at 5500 rpm
        return 350.0 * (1.0 - 0.5 * ((rpm - peak_rpm) / peak_rpm)**2)

    def pacejka(self, slip: float, load: float, lateral: bool) -> float:
        b = 10.0 if lateral else 12.0
        c = 1.9 if lateral else 1.65
        d = load * (1.1 if lateral else 1.2)
        e = 0.97
        x = b * slip
        return d * math.sin(c * math.atan(x - e * (x - math.atan(x))))

    def _process_gearbox(self, action: Mapping[str, float], forward_speed: float, dt: float) -> tuple[bool, bool]:
        """Process manual gearbox. Returns (stall_event, money_shift_event)."""
        stall_event = False
        money_shift_event = False

        self.clutch = clamp(float(action.get("clutch", 0.0)), 0.0, 1.0)
        gear_up = float(action.get("gear_up", 0.0)) > 0.5
        gear_down = float(action.get("gear_down", 0.0)) > 0.5

        # Shift lockout countdown
        self.shift_lockout = max(0.0, self.shift_lockout - dt)

        # Gear-hunting tracker
        self.gear_shift_window = max(0.0, self.gear_shift_window - dt)
        if self.gear_shift_window <= 0.0:
            self.gear_shift_count = 0

        # Handle stall recovery
        if self.stalled:
            self.stall_timer -= dt
            if self.stall_timer <= 0.0:
                self.stalled = False
                self.rpm = self.idle_rpm
            return stall_event, money_shift_event

        # Process gear changes (only if clutch is mostly disengaged and no lockout)
        if self.shift_lockout <= 0.0 and self.clutch > 0.6:
            new_gear = self.gear
            if gear_up and not gear_down:
                if self.gear == -1:
                    new_gear = 0  # R -> N
                elif self.gear < 5:
                    new_gear = self.gear + 1
            elif gear_down and not gear_up:
                if self.gear == 0:
                    new_gear = -1  # N -> R
                elif self.gear > -1:
                    new_gear = self.gear - 1

            if new_gear != self.gear:
                # Check for money-shift on downshift
                if new_gear > 0 and new_gear < self.gear:
                    target_ratio = self.gear_ratios[new_gear]
                    projected_rpm = abs(forward_speed * target_ratio * self.final_drive * 60.0 / (2.0 * math.pi * self.wheel_radius))
                    if projected_rpm > self.max_rpm * 1.1:
                        # Money-shift! Mechanical damage
                        money_shift_event = True
                        self.money_shift_damage = min(1.0, self.money_shift_damage + 0.25)
                        self.vx *= 0.7
                        self.vy *= 0.7

                self.prev_gear = self.gear
                self.gear = new_gear
                self.shift_lockout = 0.15  # 150ms synchro time
                self.gear_shift_count += 1
                self.gear_shift_window = 2.0

        # Check for rev-match quality on clutch engagement
        # RPM matching happens naturally through the clutch engagement below

        # Stall check: RPM too low with clutch engaged and car nearly stopped
        if self.clutch < 0.3 and self.gear != 0 and abs(forward_speed) < 20.0:
            if self.rpm < 600.0:
                self.stalled = True
                self.stall_timer = 1.0
                stall_event = True
                self.rpm = 0.0

        return stall_event, money_shift_event

    def update(self, action: Mapping[str, float], dt: float) -> CarPhysics:
        if dt <= 0.0:
            return CarPhysics(0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0.0,False)
            
        throttle = clamp(float(action.get("throttle", 0.0)), -1.0, 1.0)
        steer = clamp(float(action.get("steer", 0.0)), -1.0, 1.0)
        brake = clamp(float(action.get("brake", 0.0)), 0.0, 1.0)
        handbrake = clamp(float(action.get("handbrake", 0.0)), 0.0, 1.0)

        forward = angle_to_vector(self.heading)
        right = (-forward[1], forward[0])
        forward_speed = dot(self.velocity, forward)
        lateral_speed = dot(self.velocity, right)

        # NaN guard
        if math.isnan(forward_speed) or math.isnan(lateral_speed):
            self.reset(self.position, self.heading)
            forward_speed = 0.0
            lateral_speed = 0.0

        # Manual gearbox processing
        stall_event, money_shift_event = self._process_gearbox(action, forward_speed, dt)

        # Handle reverse throttle mapping
        if throttle < 0.0:
            if self.gear == -1:
                throttle = abs(throttle)
            elif forward_speed > 10.0:
                brake = max(brake, abs(throttle))
                throttle = 0.0
            else:
                throttle = 0.0

        gear_ratio = self.gear_ratios[self.gear] if self.gear > 0 else (self.reverse_ratio if self.gear == -1 else 0.0)
        
        # RPM calculation with clutch
        wheel_rpm = abs(forward_speed * gear_ratio * self.final_drive * 60.0 / (2.0 * math.pi * self.wheel_radius))
        
        if self.stalled:
            self.rpm = 0.0
        elif self.clutch > 0.95:
            # Clutch fully disengaged - engine free-revs
            if throttle > 0.0:
                self.rpm = clamp(self.rpm + throttle * 12000.0 * dt, self.idle_rpm, self.max_rpm)
            else:
                self.rpm = clamp(self.rpm - 4000.0 * dt, self.idle_rpm, self.max_rpm)
        else:
            # Blend between engine RPM and wheel RPM based on clutch
            clutch_blend = self.clutch  # 0=fully engaged, 1=disengaged
            target_rpm = clamp(wheel_rpm, self.idle_rpm, self.max_rpm)
            free_rpm = self.rpm
            if throttle > 0.0:
                free_rpm = clamp(self.rpm + throttle * 12000.0 * dt, self.idle_rpm, self.max_rpm)
            self.rpm = clamp(
                free_rpm * clutch_blend + target_rpm * (1.0 - clutch_blend),
                0.0 if self.stalled else self.idle_rpm,
                self.max_rpm
            )
        
        # Engine Torque (reduced by clutch slip and damage)
        effective_throttle = throttle if not self.stalled else 0.0
        engine_torque = self.get_torque(self.rpm) * effective_throttle
        # Clutch transmits torque proportional to engagement
        clutch_transfer = 1.0 - self.clutch  # 0 when disengaged, 1 when engaged
        transmitted_torque = engine_torque * clutch_transfer
        # Apply mechanical damage from money-shifts
        damage_factor = 1.0 - self.money_shift_damage * 0.5
        drive_torque = transmitted_torque * gear_ratio * self.final_drive * damage_factor
        drive_force = drive_torque / self.wheel_radius

        # Aero and rolling resistance
        rolling_drag = -forward_speed * 0.65
        aero_drag = -forward_speed * abs(forward_speed) * 0.0025
        long_force_env = rolling_drag + aero_drag

        # Weight transfer
        static_front_load = self.mass * self.gravity * (self.cg_to_rear / self.wheel_base)
        static_rear_load = self.mass * self.gravity * (self.cg_to_front / self.wheel_base)
        weight_transfer = (self.mass * self.long_accel * self.cg_height) / self.wheel_base
        
        front_load = max(0.0, static_front_load - weight_transfer)
        rear_load = max(0.0, static_rear_load + weight_transfer)

        # Slip angles
        steer_angle = steer * 0.6  # max 34 degrees
        front_lat_speed = lateral_speed + self.angular_velocity * self.cg_to_front
        rear_lat_speed = lateral_speed - self.angular_velocity * self.cg_to_rear
        
        slip_angle_front = math.atan2(front_lat_speed, abs(forward_speed) + 1.0) - steer_angle * math.copysign(1.0, forward_speed + 1e-6)
        slip_angle_rear = math.atan2(rear_lat_speed, abs(forward_speed) + 1.0)

        # Pacejka lateral forces
        front_lat_force = -self.pacejka(slip_angle_front, front_load, True)
        rear_lat_force = -self.pacejka(slip_angle_rear, rear_load, True) * (1.0 - handbrake * 0.8) # Handbrake kills rear grip

        # Longitudinal forces (Traction/Braking)
        brake_force = 12000.0 * brake
        handbrake_force = 15000.0 * handbrake
        
        # Apply braking to axles (70% front, 30% rear typical)
        front_brake_force = -math.copysign(brake_force * 0.7, forward_speed) if abs(forward_speed) > 1.0 else 0.0
        rear_brake_force = -math.copysign(brake_force * 0.3 + handbrake_force, forward_speed) if abs(forward_speed) > 1.0 else 0.0
        
        # Sum forces
        total_long_force = drive_force + long_force_env + front_brake_force + rear_brake_force
        total_lat_force = front_lat_force * math.cos(steer_angle) + rear_lat_force
        
        self.long_accel = total_long_force / self.mass
        self.lat_accel = total_lat_force / self.mass

        # Update velocities (local)
        forward_speed += self.long_accel * dt
        lateral_speed += self.lat_accel * dt

        # Stop completely if very slow and braking
        if abs(forward_speed) < 5.0 and (brake > 0.1 or handbrake > 0.1):
            forward_speed = 0.0
        if abs(lateral_speed) < 5.0 and abs(forward_speed) < 1.0:
            lateral_speed = 0.0

        # Angular torque
        torque = front_lat_force * self.cg_to_front * math.cos(steer_angle) - rear_lat_force * self.cg_to_rear
        angular_accel = torque / self.inertia
        self.angular_velocity += angular_accel * dt

        # Update global velocities
        self.heading += self.angular_velocity * dt
        self.heading = wrap_angle(self.heading)

        # Convert back to global vx, vy
        forward_vec = angle_to_vector(self.heading)
        right_vec = (-forward_vec[1], forward_vec[0])
        
        self.vx = forward_vec[0] * forward_speed + right_vec[0] * lateral_speed
        self.vy = forward_vec[1] * forward_speed + right_vec[1] * lateral_speed
        
        self.x += self.vx * dt
        self.y += self.vy * dt

        speed = math.hypot(self.vx, self.vy)
        slip_angle = math.atan2(lateral_speed, abs(forward_speed) + 1e-6)
        drift_intensity = 0.0
        if speed > 45.0:
            drift_intensity = clamp((abs(slip_angle) - 0.14) / 0.75, 0.0, 1.0)
            
        long_g = self.long_accel / self.gravity
        lat_g = self.lat_accel / self.gravity

        return CarPhysics(
            speed=speed,
            forward_speed=forward_speed,
            lateral_speed=lateral_speed,
            slip_angle=slip_angle,
            drift_intensity=drift_intensity,
            rpm=self.rpm,
            gear=self.gear,
            wheel_spin_front=0.0,
            wheel_spin_rear=0.0,
            front_load=front_load,
            rear_load=rear_load,
            yaw_rate=self.angular_velocity,
            angular_velocity=self.angular_velocity,
            longitudinal_g=long_g,
            lateral_g=lat_g,
            clutch=self.clutch,
            stalled=self.stalled,
        )
