from __future__ import annotations

import math
from dataclasses import dataclass

from racing_ai.math2d import (
    Point,
    add,
    angle_to_vector,
    distance,
    dot,
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
    signed_distance_to_center: float
    edge_clearance: float
    closest_point: Point
    heading: float
    normal: Point
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
        self.length = self._polyline_length(centerline)
        self.bounds = self._bounds(inner_boundary + outer_boundary)

    @classmethod
    def build_default(cls) -> "Track":
        width = 118.0
        samples_per_segment = 32
        control_points: list[Point] = [
            (520.0, 500.0),
            (1600.0, 420.0),
            (3000.0, 470.0),
            (4500.0, 540.0),
            (5050.0, 820.0),
            (5100.0, 1180.0),
            (4200.0, 1320.0),
            (2650.0, 1220.0),
            (900.0, 1320.0),
            (420.0, 1600.0),
            (500.0, 2000.0),
            (1700.0, 2130.0),
            (3300.0, 2050.0),
            (4700.0, 2180.0),
            (5200.0, 2500.0),
            (4900.0, 2900.0),
            (3300.0, 3050.0),
            (1600.0, 2920.0),
            (420.0, 3180.0),
            (260.0, 3700.0),
            (1200.0, 4050.0),
            (3000.0, 4020.0),
            (4700.0, 3850.0),
            (5550.0, 3100.0),
            (5650.0, 2050.0),
            (5400.0, 900.0),
            (4700.0, 260.0),
            (3000.0, 180.0),
            (1500.0, 240.0),
        ]

        centerline: list[Point] = []
        point_count = len(control_points)
        for index in range(point_count):
            p0 = control_points[(index - 1) % point_count]
            p1 = control_points[index]
            p2 = control_points[(index + 1) % point_count]
            p3 = control_points[(index + 2) % point_count]
            for sample_index in range(samples_per_segment):
                t = sample_index / samples_per_segment
                centerline.append(cls._catmull_rom(p0, p1, p2, p3, t))

        centerline = cls._rotate_to_smooth_seam(centerline, width)
        inner, outer = cls._build_boundaries(centerline, width)

        return cls(centerline=centerline, inner_boundary=inner, outer_boundary=outer, width=width)

    def spawn_pose(self) -> tuple[Point, float]:
        sample = self.sample_at(self.centerline[0])
        return self.centerline[0], sample.heading

    def start_line(self) -> tuple[Point, Point]:
        return self.inner_boundary[0], self.outer_boundary[0]

    def reward_markers(self) -> list[RewardMarker]:
        markers: list[RewardMarker] = []
        marker_spacing = 300.0
        marker_count = max(1, int(self.length // marker_spacing))
        for marker_id in range(marker_count):
            marker_distance = (marker_id + 0.5) * self.length / marker_count
            position, segment_index = self._point_at_distance(marker_distance)
            kind = self._marker_kind(segment_index)
            reward = {"apex": 4.0, "speed": 5.0, "drift": 7.0}[kind]
            markers.append(
                RewardMarker(
                    marker_id=marker_id,
                    position=position,
                    radius=18.0,
                    reward=reward,
                    kind=kind,
                )
            )
        return markers

    def sample_at(self, point: Point) -> TrackSample:
        best_distance = float("inf")
        best_point = self.centerline[0]
        best_index = 0
        best_t = 0.0
        best_normal = (0.0, 1.0)
        best_signed_offset = 0.0

        for index, (start, end) in enumerate(self.center_segments):
            segment_distance, closest, t = project_point_to_segment(point, start, end)
            if segment_distance < best_distance:
                best_distance = segment_distance
                best_point = closest
                best_index = index
                best_t = t
                tangent = normalize(sub(end, start))
                best_normal = (-tangent[1], tangent[0])
                best_signed_offset = dot(sub(point, closest), best_normal)

        start, end = self.center_segments[best_index]
        heading = vector_angle(sub(end, start))
        progress = (best_index + best_t) / len(self.center_segments)
        signed_distance = 0.0
        if best_distance > 1e-9:
            side = 1.0 if best_signed_offset >= 0.0 else -1.0
            signed_distance = best_distance * side
        edge_clearance = self.half_width - best_distance
        on_track = best_distance <= self.half_width
        return TrackSample(
            distance_to_center=best_distance,
            signed_distance_to_center=signed_distance,
            edge_clearance=edge_clearance,
            closest_point=best_point,
            heading=heading,
            normal=best_normal,
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

    @classmethod
    def _build_boundaries(cls, centerline: list[Point], width: float) -> tuple[list[Point], list[Point]]:
        inner: list[Point] = []
        outer: list[Point] = []
        for index, point in enumerate(centerline):
            previous_point = centerline[index - 1]
            next_point = centerline[(index + 1) % len(centerline)]
            tangent = normalize(sub(next_point, previous_point))
            normal = (-tangent[1], tangent[0])
            inner.append(add(point, mul(normal, width * 0.5)))
            outer.append(add(point, mul(normal, -width * 0.5)))
        return inner, outer

    @classmethod
    def _rotate_to_smooth_seam(cls, centerline: list[Point], width: float) -> list[Point]:
        inner, outer = cls._build_boundaries(centerline, width)
        best_index = min(
            range(len(centerline)),
            key=lambda index: max(
                distance(inner[index - 1], inner[index]),
                distance(outer[index - 1], outer[index]),
            ),
        )
        return centerline[best_index:] + centerline[:best_index]

    @staticmethod
    def _catmull_rom(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
        t2 = t * t
        t3 = t2 * t
        x = 0.5 * (
            (2.0 * p1[0])
            + (-p0[0] + p2[0]) * t
            + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
            + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
        )
        y = 0.5 * (
            (2.0 * p1[1])
            + (-p0[1] + p2[1]) * t
            + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
            + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
        )
        return (x, y)

    @staticmethod
    def _polyline_length(points: list[Point]) -> float:
        return sum(distance(start, end) for start, end in Track._segments(points))

    @staticmethod
    def _bounds(points: list[Point]) -> tuple[float, float, float, float]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _point_at_distance(self, target_distance: float) -> tuple[Point, int]:
        wrapped_distance = target_distance % self.length
        traveled = 0.0
        for index, (start, end) in enumerate(self.center_segments):
            segment_length = distance(start, end)
            if traveled + segment_length >= wrapped_distance:
                t = (wrapped_distance - traveled) / max(segment_length, 1e-9)
                return add(start, mul(sub(end, start), t)), index
            traveled += segment_length
        return self.centerline[-1], len(self.center_segments) - 1

    def _marker_kind(self, segment_index: int) -> str:
        segment_count = len(self.center_segments)
        lookahead = 6
        previous_segment = self.center_segments[(segment_index - lookahead) % segment_count]
        next_segment = self.center_segments[(segment_index + lookahead) % segment_count]
        previous_heading = vector_angle(sub(previous_segment[1], previous_segment[0]))
        next_heading = vector_angle(sub(next_segment[1], next_segment[0]))
        turn_amount = abs(self._wrap_angle(next_heading - previous_heading))

        if turn_amount >= 0.20:
            return "drift"
        if turn_amount <= 0.065:
            return "speed"
        return "apex"

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle <= -math.pi:
            angle += math.tau
        while angle > math.pi:
            angle -= math.tau
        return angle
