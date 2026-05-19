from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from racing_ai.agent import Agent, Observation, ZeroAgent
from racing_ai.car import Car, CarPhysics
from racing_ai.math2d import Point, distance, wrap_angle
from racing_ai.track import RewardMarker, Track, TrackSample


@dataclass(slots=True)
class WorldStats:
    total_reward: float = 0.0
    frame_reward: float = 0.0
    marker_reward: float = 0.0
    drift_score: float = 0.0
    markers_collected: int = 0
    off_track_count: int = 0
    lap: int = 0
    progress: float = 0.0
    time: float = 0.0


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
        self.previous_progress = 0.0
        self.last_physics = CarPhysics(0.0, 0.0, 0.0, 0.0, 0.0)
        self.observation = self.build_observation(0.0, self.track.sample_at(self.car.position), self.last_physics)

    def reset(self) -> Observation:
        spawn_position, spawn_heading = self.track.spawn_pose()
        self.car.reset(spawn_position, spawn_heading)
        for marker in self.markers:
            marker.collected = False
        self.stats = WorldStats()
        self.previous_off_track = False
        self.previous_progress = 0.0
        self.last_physics = CarPhysics(0.0, 0.0, 0.0, 0.0, 0.0)
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

        self._update_lap_and_progress(sample)
        self.stats.frame_reward = frame_reward
        self.stats.total_reward += frame_reward
        self.previous_off_track = off_track
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
            "off_track": off_track,
            "off_track_count": self.stats.off_track_count,
            "distance_to_center": sample.distance_to_center,
            "progress": self.stats.progress,
            "lap": self.stats.lap,
            "frame_reward": frame_reward,
            "total_reward": self.stats.total_reward,
            "markers_collected": self.stats.markers_collected,
            "markers_total": len(self.markers),
            "marker_reward": self.stats.marker_reward,
            "rays": rays,
            "nearest_markers": nearest_markers,
            "car_position": self.car.position,
            "car_velocity": self.car.velocity,
        }

    def _collect_markers(self) -> float:
        reward = 0.0
        for marker in self.markers:
            if marker.collected:
                continue
            if distance(self.car.position, marker.position) <= marker.radius + self.car.length * 0.35:
                marker.collected = True
                self.stats.markers_collected += 1
                reward += marker.reward
        return reward

    def _nearest_visible_markers(self, limit: int) -> list[dict[str, object]]:
        active_markers = [marker for marker in self.markers if not marker.collected]
        active_markers.sort(key=lambda marker: distance(self.car.position, marker.position))
        visible = []
        for marker in active_markers[:limit]:
            dx = marker.position[0] - self.car.x
            dy = marker.position[1] - self.car.y
            visible.append(
                {
                    "id": marker.marker_id,
                    "kind": marker.kind,
                    "dx": dx,
                    "dy": dy,
                    "distance": math.hypot(dx, dy),
                    "reward": marker.reward,
                }
            )
        return visible

    def _update_lap_and_progress(self, sample: TrackSample) -> None:
        progress = sample.progress
        if self.previous_progress > 0.86 and progress < 0.14:
            self.stats.lap += 1
        self.previous_progress = progress
        self.stats.progress = progress
