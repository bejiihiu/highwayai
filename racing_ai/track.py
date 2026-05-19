from __future__ import annotations

import math
from dataclasses import dataclass

from racing_ai.math2d import (
    Point,
    add,
    angle_to_vector,
    distance,
    mul,
    normalize,
    project_point_to_segment,
    ray_segment_intersection,
    sub,
    vector_angle,
)


@dataclass(slots=True)
class TrackSample:
    distance_to_center: float
    closest_point: Point
    heading: float
    progress: float
    segment_index: int
    on_track: bool


@dataclass(slots=True)
class RewardMarker:
    marker_id: int
    position: Point
    radius: float
    reward: float
    kind: str
    collected: bool = False


class Track:
    def __init__(
        self,
        centerline: list[Point],
        inner_boundary: list[Point],
        outer_boundary: list[Point],
        width: float,
    ) -> None:
        self.centerline = centerline
        self.inner_boundary = inner_boundary
        self.outer_boundary = outer_boundary
        self.width = width
        self.half_width = width * 0.5
        self.center_segments = self._segments(centerline)
        self.inner_segments = self._segments(inner_boundary)
        self.outer_segments = self._segments(outer_boundary)
        self.edge_segments = self.inner_segments + self.outer_segments

    @classmethod
    def build_default(cls) -> "Track":
        center = (550.0, 380.0)
        radius_x = 350.0
        radius_y = 215.0
        width = 96.0
        samples = 180

        centerline: list[Point] = []
        for index in range(samples):
            t = math.tau * index / samples
            x = center[0] + radius_x * math.cos(t) + 34.0 * math.cos(3.0 * t)
            y = center[1] + radius_y * math.sin(t) + 24.0 * math.sin(2.0 * t)
            centerline.append((x, y))

        inner: list[Point] = []
        outer: list[Point] = []
        for index, point in enumerate(centerline):
            previous_point = centerline[index - 1]
            next_point = centerline[(index + 1) % samples]
            tangent = normalize(sub(next_point, previous_point))
            normal = (-tangent[1], tangent[0])
            inner.append(add(point, mul(normal, width * 0.5)))
            outer.append(add(point, mul(normal, -width * 0.5)))

        return cls(centerline=centerline, inner_boundary=inner, outer_boundary=outer, width=width)

    def spawn_pose(self) -> tuple[Point, float]:
        sample = self.sample_at(self.centerline[0])
        return self.centerline[0], sample.heading

    def start_line(self) -> tuple[Point, Point]:
        return self.inner_boundary[0], self.outer_boundary[0]

    def reward_markers(self) -> list[RewardMarker]:
        markers: list[RewardMarker] = []
        marker_id = 0
        for index in range(10, len(self.centerline), 10):
            if index >= len(self.centerline) - 4:
                continue

            kind = ("apex", "speed", "drift")[marker_id % 3]
            reward = {"apex": 4.0, "speed": 5.0, "drift": 7.0}[kind]
            markers.append(
                RewardMarker(
                    marker_id=marker_id,
                    position=self.centerline[index],
                    radius=18.0,
                    reward=reward,
                    kind=kind,
                )
            )
            marker_id += 1
        return markers

    def sample_at(self, point: Point) -> TrackSample:
        best_distance = float("inf")
        best_point = self.centerline[0]
        best_index = 0
        best_t = 0.0

        for index, (start, end) in enumerate(self.center_segments):
            segment_distance, closest, t = project_point_to_segment(point, start, end)
            if segment_distance < best_distance:
                best_distance = segment_distance
                best_point = closest
                best_index = index
                best_t = t

        start, end = self.center_segments[best_index]
        heading = vector_angle(sub(end, start))
        progress = (best_index + best_t) / len(self.center_segments)
        on_track = best_distance <= self.half_width
        return TrackSample(
            distance_to_center=best_distance,
            closest_point=best_point,
            heading=heading,
            progress=progress,
            segment_index=best_index,
            on_track=on_track,
        )

    def raycast(self, origin: Point, angle: float, max_distance: float) -> tuple[float, Point]:
        direction = angle_to_vector(angle)
        best_distance = max_distance

        for start, end in self.edge_segments:
            hit_distance = ray_segment_intersection(origin, direction, start, end)
            if hit_distance is not None and hit_distance < best_distance:
                best_distance = hit_distance

        hit_point = add(origin, mul(direction, best_distance))
        return best_distance, hit_point

    def marker_progress(self, marker: RewardMarker) -> float:
        return self.sample_at(marker.position).progress

    @staticmethod
    def _segments(points: list[Point]) -> list[tuple[Point, Point]]:
        return list(zip(points, points[1:] + points[:1]))
