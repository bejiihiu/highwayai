from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping

from racing_ai.agent import Agent, Observation, ZeroAgent
from racing_ai.car import Car, CarPhysics
from racing_ai.math2d import Point, angle_to_vector, clamp, distance, dot, wrap_angle
from racing_ai.track import RewardMarker, Track, TrackSample


@dataclass(slots=True)
class WorldStats:
    total_reward: float = 0.0
    frame_reward: float = 0.0
    marker_reward: float = 0.0
    drift_score: float = 0.0
    lap_markers_collected: int = 0
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


class RacingWorld:
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
        self.lap_cooldown = 0.0
        self.previous_progress = self.track.sample_at(self.car.position).progress
        self.last_physics = CarPhysics(0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
        self.lap_cooldown = 0.0
        self.previous_progress = self.track.sample_at(self.car.position).progress
        self.last_physics = CarPhysics(0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.observation = self.build_observation(0.0, self.track.sample_at(self.car.position), self.last_physics)
        return self.observation

    def update(self, dt: float) -> Observation:
        action = self.agent.act(self.observation)
        return self.step(action, dt)

    def step(self, action: Mapping[str, float], dt: float) -> Observation:
        dt = max(0.0, min(dt, 1.0 / 20.0))
        self.stats.time += dt

        physics = self.car.update(action, dt)
        sample = self.track.sample_at(self.car.position)
        frame_reward = 0.0
        edge_collision, edge_impact = self._apply_edge_collision(sample)
        if edge_collision:
            sample = self.track.sample_at(self.car.position)
            physics = self._measure_physics()

        self.edge_collision = edge_collision
        self.stats.last_edge_impact = edge_impact
        if edge_collision:
            if not self.previous_edge_collision:
                self.stats.edge_collision_count += 1
                frame_reward -= 8.0
            frame_reward -= 2.0 * dt
            frame_reward -= edge_impact * 4.0

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
            drift_points = physics.drift_intensity * min(1.0, physics.speed / 180.0) * dt * 24.0
            self.stats.drift_score += drift_points
            frame_reward += drift_points * 0.25

        # Example RL bonuses
        brake_val = action.get("brake", 0.0)
        steer_val = action.get("steer", 0.0)
        if brake_val > 0.1 and abs(steer_val) > 0.1:
            self.stats.trail_brake_bonus += dt * 0.5
            frame_reward += dt * 0.5

        self._update_lap_and_progress(sample, dt)
        self.stats.frame_reward = frame_reward
        self.stats.total_reward += frame_reward
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

        return {
            "time": self.stats.time,
            "speed": physics.speed,
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
            "wheel_spin_front": physics.wheel_spin_front,
            "wheel_spin_rear": physics.wheel_spin_rear,
            "front_load": physics.front_load,
            "rear_load": physics.rear_load,
            "yaw_rate": physics.yaw_rate,
            "angular_velocity": physics.angular_velocity,
            "longitudinal_g": physics.longitudinal_g,
            "lateral_g": physics.lateral_g,

            "off_track": off_track,
            "off_track_count": self.stats.off_track_count,
            "distance_to_center": sample.distance_to_center,
            "signed_distance_to_center": sample.signed_distance_to_center,
            "edge_clearance": sample.edge_clearance,
            "edge_collision": self.edge_collision,
            "edge_collision_count": self.stats.edge_collision_count,
            "last_edge_impact": self.stats.last_edge_impact,
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
        }

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
        if speed > 45.0:
            drift_intensity = clamp((abs(slip_angle) - 0.14) / 0.75, 0.0, 1.0)
            
        return CarPhysics(
            speed=speed,
            forward_speed=forward_speed,
            lateral_speed=lateral_speed,
            slip_angle=slip_angle,
            drift_intensity=drift_intensity,
            rpm=self.last_physics.rpm,
            gear=self.last_physics.gear,
            wheel_spin_front=0.0,
            wheel_spin_rear=0.0,
            front_load=self.last_physics.front_load,
            rear_load=self.last_physics.rear_load,
            yaw_rate=self.last_physics.yaw_rate,
            angular_velocity=self.last_physics.angular_velocity,
            longitudinal_g=self.last_physics.longitudinal_g,
            lateral_g=self.last_physics.lateral_g
        )

    def _collect_markers(self) -> float:
        reward = 0.0
        for marker in self.markers:
            if marker.collected:
                continue
            if distance(self.car.position, marker.position) <= marker.radius + 12.0:
                marker.collected = True
                self.stats.lap_markers_collected += 1
                self.stats.total_markers_collected += 1
                reward += marker.reward
        return reward

    def _restore_markers_for_next_lap(self) -> None:
        for marker in self.markers:
            marker.collected = False
        self.stats.lap_markers_collected = 0

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

