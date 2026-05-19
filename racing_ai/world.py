from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping

from racing_ai.agent import Agent, Observation, ZeroAgent
from racing_ai.car import Car, CarPhysics
from racing_ai.math2d import Point, angle_to_vector, clamp, distance, dot, wrap_angle
from racing_ai.track import Track, TrackSample


@dataclass(slots=True)
class WorldStats:
    total_reward: float = 0.0
    frame_reward: float = 0.0
    marker_reward: float = 0.0
    drift_score: float = 0.0
    lap_markers_collected: int = 0
    # Legacy alias preserved for existing tests/callers.
    markers_collected: int = 0
    total_markers_collected: int = 0
    off_track_count: int = 0
    edge_collision_count: int = 0
    last_edge_impact: float = 0.0
    lap: int = 0
    progress: float = 0.0
    time: float = 0.0
    smooth_steering_bonus: float = 0.0
    trail_brake_bonus: float = 0.0
    clean_lap_bonus: float = 0.0
    stall_count: int = 0
    money_shift_count: int = 0
    death_count: int = 0


class RacingWorld:
    DEATH_PENALTY: float = 50.0

    def __init__(self, agent: Agent | None = None) -> None:
        self.track = Track.build_default()
        spawn_position, spawn_heading = self.track.spawn_pose()
        self.car = Car(spawn_position, spawn_heading)
        self.markers = self.track.reward_markers()
        self.agent = agent or ZeroAgent()
        self.stats = WorldStats()
        self.max_ray_distance = 280.0
        self.ray_angles = [
            math.radians(angle)
            for angle in (-135, -95, -65, -35, -15, 0, 15, 35, 65, 95, 135)
        ]
        self.previous_off_track = False
        self.previous_edge_collision = False
        self.edge_collision = False
        self.death_event = False
        self.lap_cooldown = 0.0
        self.previous_progress = self.track.sample_at(self.car.position).progress
        self.last_physics = CarPhysics()
        self.last_action = self._sanitize_action({})
        self._prev_money_damage = 0.0
        self.observation = self.build_observation(0.0, self.track.sample_at(self.car.position), self.last_physics)

    def reset(self) -> Observation:
        spawn_position, spawn_heading = self.track.spawn_pose()
        self.car.reset(spawn_position, spawn_heading)
        for marker in self.markers:
            marker.collected = False
        self.stats = WorldStats()
        self.previous_off_track = False
        self.previous_edge_collision = False
        self.edge_collision = False
        self.death_event = False
        self.lap_cooldown = 0.0
        self.previous_progress = self.track.sample_at(self.car.position).progress
        self.last_physics = CarPhysics()
        self.last_action = self._sanitize_action({})
        self._prev_money_damage = 0.0
        self.observation = self.build_observation(0.0, self.track.sample_at(self.car.position), self.last_physics)
        return self.observation

    def update(self, dt: float) -> Observation:
        action = self.agent.act(self.observation)
        return self.step(action, dt)

    def step(self, action: Mapping[str, float], dt: float) -> Observation:
        dt = max(0.0, min(dt, 1.0 / 20.0))
        self.stats.time += dt
        self.death_event = False

        env_action = self._sanitize_action(action)
        self.last_action = env_action

        physics = self.car.update(env_action, dt)
        sample = self.track.sample_at(self.car.position)
        frame_reward = 0.0

        edge_collision, edge_impact = self._apply_edge_collision(sample)
        if edge_collision:
            frame_reward -= self._trigger_death()
            sample = self.track.sample_at(self.car.position)
            physics = self._measure_physics()

        self.edge_collision = edge_collision
        self.stats.last_edge_impact = edge_impact
        if edge_collision:
            self.stats.edge_collision_count += 1

        off_track = not sample.on_track
        if off_track:
            frame_reward -= 4.0 * dt
            if not self.previous_off_track:
                self.stats.off_track_count += 1
                frame_reward -= 10.0

        marker_reward = self._collect_markers()
        frame_reward += marker_reward
        self.stats.marker_reward += marker_reward

        if not off_track and physics.drift_intensity > 0.0:
            drift_points = physics.drift_intensity * min(1.0, physics.speed / 90.0) * dt * 18.0
            self.stats.drift_score += drift_points
            frame_reward += drift_points * 0.2

        brake_val = float(env_action.get("brake", 0.0))
        steer_val = float(env_action.get("steer", 0.0))
        if brake_val > 0.1 and abs(steer_val) > 0.1:
            self.stats.trail_brake_bonus += dt * 0.5
            frame_reward += dt * 0.35

        # Gearbox events are available directly from car physics.
        if physics.stall_event:
            self.stats.stall_count += 1
            frame_reward -= 10.0

        if physics.money_shift_event:
            self.stats.money_shift_count += 1
            frame_reward -= 12.0

        if physics.shift_attempted and physics.shift_blocked:
            frame_reward -= 0.07

        if physics.shift_applied and physics.clutch > 0.55 and not physics.overrev:
            frame_reward += 0.04

        if self.car.money_shift_damage > self._prev_money_damage:
            frame_reward -= (self.car.money_shift_damage - self._prev_money_damage) * 5.0
        self._prev_money_damage = self.car.money_shift_damage

        self._update_lap_and_progress(sample, dt)
        self.stats.frame_reward = frame_reward
        self.stats.total_reward += frame_reward
        if self.death_event:
            self.previous_off_track = False
            self.previous_edge_collision = False
        else:
            self.previous_off_track = off_track
            self.previous_edge_collision = edge_collision
        self.last_physics = physics
        self.observation = self.build_observation(frame_reward, sample, physics)
        return self.observation

    def build_observation(
        self,
        frame_reward: float,
        sample: TrackSample,
        physics: CarPhysics,
    ) -> Observation:
        rays = []
        for relative_angle in self.ray_angles:
            world_angle = self.car.heading + relative_angle
            ray_distance, hit_point = self.track.raycast(self.car.position, world_angle, self.max_ray_distance)
            rays.append(
                {
                    "relative_angle": relative_angle,
                    "distance": ray_distance,
                    "normalized_distance": ray_distance / self.max_ray_distance,
                    "hit": hit_point,
                }
            )

        nearest_markers = self._nearest_visible_markers(limit=6)
        heading_error = wrap_angle(sample.heading - self.car.heading)
        off_track = not sample.on_track
        speed_kmh = physics.speed * self.car.speed_to_mps * 3.6
        virtual_keys = self._virtual_keys_from_action(self.last_action)

        return {
            "time": self.stats.time,
            "speed": physics.speed,
            "speed_kmh": speed_kmh,
            "forward_speed": physics.forward_speed,
            "lateral_speed": physics.lateral_speed,
            "heading": self.car.heading,
            "track_heading": sample.heading,
            "heading_error": heading_error,
            "slip_angle": physics.slip_angle,
            "drift_intensity": physics.drift_intensity,
            "drift_score": self.stats.drift_score,
            "rpm": physics.rpm,
            "gear": physics.gear,
            "clutch": physics.clutch,
            "stalled": physics.stalled,
            "wheel_spin_front": physics.wheel_spin_front,
            "wheel_spin_rear": physics.wheel_spin_rear,
            "front_load": physics.front_load,
            "rear_load": physics.rear_load,
            "yaw_rate": physics.yaw_rate,
            "angular_velocity": physics.angular_velocity,
            "longitudinal_g": physics.longitudinal_g,
            "lateral_g": physics.lateral_g,
            "engine_load": physics.engine_load,
            "clutch_slip": physics.clutch_slip,
            "clutch_engagement": physics.clutch_engagement,
            "shift_lockout": physics.shift_lockout,
            "shift_attempted": physics.shift_attempted,
            "shift_applied": physics.shift_applied,
            "shift_blocked": physics.shift_blocked,
            "shift_block_reason": physics.shift_block_reason,
            "money_shift_event": physics.money_shift_event,
            "stall_event": physics.stall_event,
            "overrev": physics.overrev,
            "traction": physics.traction,
            "drive_force": physics.drive_force,
            "wheel_torque": physics.wheel_torque,
            "off_track": off_track,
            "off_track_count": self.stats.off_track_count,
            "distance_to_center": sample.distance_to_center,
            "signed_distance_to_center": sample.signed_distance_to_center,
            "edge_clearance": sample.edge_clearance,
            "edge_collision": self.edge_collision,
            "edge_collision_count": self.stats.edge_collision_count,
            "last_edge_impact": self.stats.last_edge_impact,
            "death_event": self.death_event,
            "death_count": self.stats.death_count,
            "progress": self.stats.progress,
            "lap": self.stats.lap,
            "frame_reward": frame_reward,
            "total_reward": self.stats.total_reward,
            "markers_collected": self.stats.lap_markers_collected,
            "total_markers_collected": self.stats.total_markers_collected,
            "markers_total": len(self.markers),
            "marker_reward": self.stats.marker_reward,
            "rays": rays,
            "nearest_markers": nearest_markers,
            "car_position": self.car.position,
            "car_velocity": self.car.velocity,
            "last_action": dict(self.last_action),
            "virtual_keys": virtual_keys,
            "stall_count": self.stats.stall_count,
            "money_shift_count": self.stats.money_shift_count,
            "money_shift_damage": self.car.money_shift_damage,
            "gear_shift_count": self.car.gear_shift_count,
        }

    def _sanitize_action(self, action: Mapping[str, float]) -> dict[str, float]:
        return {
            "throttle": clamp(float(action.get("throttle", 0.0)), -1.0, 1.0),
            "steer": clamp(float(action.get("steer", 0.0)), -1.0, 1.0),
            "brake": clamp(float(action.get("brake", 0.0)), 0.0, 1.0),
            "clutch": clamp(float(action.get("clutch", 0.0)), 0.0, 1.0),
            "handbrake": clamp(float(action.get("handbrake", 0.0)), 0.0, 1.0),
            "gear_up": 1.0 if float(action.get("gear_up", 0.0)) > 0.5 else 0.0,
            "gear_down": 1.0 if float(action.get("gear_down", 0.0)) > 0.5 else 0.0,
        }

    def _virtual_keys_from_action(self, action: Mapping[str, float]) -> dict[str, bool]:
        throttle = float(action.get("throttle", 0.0))
        steer = float(action.get("steer", 0.0))
        brake = float(action.get("brake", 0.0))
        clutch = float(action.get("clutch", 0.0))
        handbrake = float(action.get("handbrake", 0.0))
        gear_up = float(action.get("gear_up", 0.0))
        gear_down = float(action.get("gear_down", 0.0))
        w_pressed = throttle > 0.15
        a_pressed = steer < -0.15
        d_pressed = steer > 0.15
        s_pressed = brake > 0.15 or throttle < -0.15
        return {
            "w": w_pressed,
            "a": a_pressed,
            "s": s_pressed,
            "d": d_pressed,
            "ru_ts": w_pressed,
            "ru_ef": a_pressed,
            "ru_y": s_pressed,
            "ru_ve": d_pressed,
            "clutch_key": clutch > 0.5,
            "handbrake_key": handbrake > 0.1,
            "gear_up_key": gear_up > 0.5,
            "gear_down_key": gear_down > 0.5,
        }

    def _trigger_death(self) -> float:
        self.death_event = True
        self.stats.death_count += 1
        spawn_position, spawn_heading = self.track.spawn_pose()
        self.car.reset(spawn_position, spawn_heading)
        self._restore_markers_for_next_lap()
        sample = self.track.sample_at(self.car.position)
        self.previous_progress = sample.progress
        self.previous_off_track = False
        self.previous_edge_collision = False
        self.last_physics = CarPhysics(rpm=self.car.rpm, gear=self.car.gear, clutch=self.car.clutch, stalled=self.car.stalled)
        self._prev_money_damage = self.car.money_shift_damage
        return self.DEATH_PENALTY

    def _apply_edge_collision(self, sample: TrackSample) -> tuple[bool, float]:
        collision_radius = self.car.width * 0.55
        penetration = collision_radius - sample.edge_clearance
        if penetration <= 0.0:
            return False, 0.0

        collision_band = self.track.half_width + collision_radius + 24.0
        if sample.distance_to_center > collision_band:
            return False, 0.0

        side = 1.0 if sample.signed_distance_to_center >= 0.0 else -1.0
        outward = (sample.normal[0] * side, sample.normal[1] * side)
        inward = (-outward[0], -outward[1])

        self.car.x += inward[0] * (penetration + 0.01)
        self.car.y += inward[1] * (penetration + 0.01)

        outward_speed = max(0.0, dot(self.car.velocity, outward))
        if outward_speed > 0.0:
            self.car.vx -= outward[0] * outward_speed
            self.car.vy -= outward[1] * outward_speed

        self.car.vx *= 0.84
        self.car.vy *= 0.84

        impact_from_depth = clamp(penetration / max(collision_radius, 1.0), 0.0, 1.0)
        impact_from_speed = clamp(outward_speed / 240.0, 0.0, 1.0)
        return True, clamp(impact_from_depth + impact_from_speed, 0.0, 1.0)

    def _measure_physics(self) -> CarPhysics:
        forward = angle_to_vector(self.car.heading)
        right = (-forward[1], forward[0])
        forward_speed = dot(self.car.velocity, forward)
        lateral_speed = dot(self.car.velocity, right)
        speed = math.hypot(self.car.vx, self.car.vy)
        slip_angle = math.atan2(lateral_speed, abs(forward_speed) + 1e-6)
        drift_intensity = 0.0
        if speed > 5.0:
            drift_intensity = clamp((abs(slip_angle) - 0.09) / 0.65, 0.0, 1.0)

        return CarPhysics(
            speed=speed,
            forward_speed=forward_speed,
            lateral_speed=lateral_speed,
            slip_angle=slip_angle,
            drift_intensity=drift_intensity,
            rpm=self.last_physics.rpm,
            gear=self.last_physics.gear,
            wheel_spin_front=0.0,
            wheel_spin_rear=self.last_physics.wheel_spin_rear,
            front_load=self.last_physics.front_load,
            rear_load=self.last_physics.rear_load,
            yaw_rate=self.last_physics.yaw_rate,
            angular_velocity=self.last_physics.angular_velocity,
            longitudinal_g=self.last_physics.longitudinal_g,
            lateral_g=self.last_physics.lateral_g,
            clutch=self.last_physics.clutch,
            stalled=self.last_physics.stalled,
            engine_load=self.last_physics.engine_load,
            clutch_slip=self.last_physics.clutch_slip,
            clutch_engagement=self.last_physics.clutch_engagement,
            shift_lockout=self.last_physics.shift_lockout,
            shift_attempted=self.last_physics.shift_attempted,
            shift_applied=self.last_physics.shift_applied,
            shift_blocked=self.last_physics.shift_blocked,
            shift_block_reason=self.last_physics.shift_block_reason,
            money_shift_event=self.last_physics.money_shift_event,
            stall_event=self.last_physics.stall_event,
            overrev=self.last_physics.overrev,
            traction=self.last_physics.traction,
            drive_force=self.last_physics.drive_force,
            wheel_torque=self.last_physics.wheel_torque,
        )

    def _collect_markers(self) -> float:
        reward = 0.0
        for marker in self.markers:
            if marker.collected:
                continue
            if distance(self.car.position, marker.position) <= marker.radius + 12.0:
                marker.collected = True
                self.stats.lap_markers_collected += 1
                self.stats.markers_collected = self.stats.lap_markers_collected
                self.stats.total_markers_collected += 1
                reward += marker.reward
        return reward

    def _restore_markers_for_next_lap(self) -> None:
        for marker in self.markers:
            marker.collected = False
        self.stats.lap_markers_collected = 0
        self.stats.markers_collected = 0

    def _nearest_visible_markers(self, limit: int) -> list[dict[str, object]]:
        car_x, car_y = self.car.x, self.car.y
        active = [
            (math.hypot(m.position[0] - car_x, m.position[1] - car_y), m)
            for m in self.markers
            if not m.collected
        ]
        nearest = heapq.nsmallest(limit, active, key=lambda pair: pair[0])
        visible: list[dict[str, object]] = []
        for dist, marker in nearest:
            dx = marker.position[0] - car_x
            dy = marker.position[1] - car_y
            visible.append(
                {
                    "id": marker.marker_id,
                    "kind": marker.kind,
                    "dx": dx,
                    "dy": dy,
                    "distance": dist,
                    "reward": marker.reward,
                }
            )
        return visible

    def _update_lap_and_progress(self, sample: TrackSample, dt: float) -> None:
        progress = sample.progress
        self.lap_cooldown -= dt
        if self.previous_progress > 0.86 and progress < 0.14 and self.lap_cooldown <= 0.0:
            self.stats.lap += 1
            self.lap_cooldown = 5.0
            self._restore_markers_for_next_lap()
        self.previous_progress = progress
        self.stats.progress = progress
