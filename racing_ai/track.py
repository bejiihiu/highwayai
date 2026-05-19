from __future__ import annotations

import bisect
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
    wrap_angle,
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


class _SegmentGrid:
    """Uniform spatial grid for O(1)-amortised segment lookup."""

    __slots__ = ("_inv_cell", "_ox", "_oy", "_grid")

    def __init__(
        self,
        segments: list[tuple[Point, Point]],
        bounds: tuple[float, float, float, float],
        cell_size: float = 280.0,
    ) -> None:
        inv = 1.0 / cell_size
        self._inv_cell = inv
        self._ox = bounds[0] - cell_size
        self._oy = bounds[1] - cell_size

        grid: dict[tuple[int, int], list[int]] = {}
        for idx, (s, e) in enumerate(segments):
            sx, sy = s
            ex, ey = e
            c0x = int((min(sx, ex) - self._ox) * inv)
            c0y = int((min(sy, ey) - self._oy) * inv)
            c1x = int((max(sx, ex) - self._ox) * inv)
            c1y = int((max(sy, ey) - self._oy) * inv)
            for cx in range(c0x, c1x + 1):
                for cy in range(c0y, c1y + 1):
                    key = (cx, cy)
                    if key in grid:
                        grid[key].append(idx)
                    else:
                        grid[key] = [idx]
        self._grid = grid

    def query_nearby(self, x: float, y: float) -> list[int]:
        """Return segment indices in the cell containing (x,y) plus its 8 neighbours."""
        inv = self._inv_cell
        cx = int((x - self._ox) * inv)
        cy = int((y - self._oy) * inv)
        result: list[int] = []
        grid = self._grid
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cell = (cx + dx, cy + dy)
                if cell in grid:
                    result.extend(grid[cell])
        return result

    def query_rect(self, x0: float, y0: float, x1: float, y1: float) -> set[int]:
        """Return unique segment indices overlapping the given rectangle (with 1-cell margin)."""
        inv = self._inv_cell
        c0x = int((x0 - self._ox) * inv) - 1
        c0y = int((y0 - self._oy) * inv) - 1
        c1x = int((x1 - self._ox) * inv) + 1
        c1y = int((y1 - self._oy) * inv) + 1
        result: set[int] = set()
        grid = self._grid
        for cx in range(c0x, c1x + 1):
            for cy in range(c0y, c1y + 1):
                cell = (cx, cy)
                if cell in grid:
                    result.update(grid[cell])
        return result


class Track:
    DEFAULT_MARKER_SPACING = 100.0
    LEGACY_MARKER_SPACING = 150.0
    MIN_MARKER_SPACING = 45.0

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

        # Pre-compute segment lengths and cumulative distances for binary search.
        seg_lens: list[float] = []
        cumulative: list[float] = []
        total = 0.0
        for s, e in self.center_segments:
            d = distance(s, e)
            seg_lens.append(d)
            total += d
            cumulative.append(total)
        self._segment_lengths = seg_lens
        self._cumulative_distances = cumulative

        # Build spatial grids for fast lookup.
        padded_bounds = (
            self.bounds[0] - width,
            self.bounds[1] - width,
            self.bounds[2] + width,
            self.bounds[3] + width,
        )
        self._center_grid = _SegmentGrid(self.center_segments, padded_bounds, cell_size=280.0)
        self._edge_grid = _SegmentGrid(self.edge_segments, padded_bounds, cell_size=280.0)

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

    def reward_markers(
        self,
        marker_spacing: float = DEFAULT_MARKER_SPACING,
        normalize_total_reward: bool = True,
    ) -> list[RewardMarker]:
        markers: list[RewardMarker] = []
        spacing = max(self.MIN_MARKER_SPACING, float(marker_spacing))
        marker_count = max(1, int(self.length // spacing))
        reward_scale = spacing / self.LEGACY_MARKER_SPACING if normalize_total_reward else 1.0
        
        reward_map = {
            "apex": 4.0,
            "speed": 5.0,
            "drift": 7.0,
            "braking_zone": 3.5,
            "late_apex": 4.5,
            "chicane": 5.5,
            "overtake_zone": 3.0,
            "pit_entry": 2.0,
            "clean_line": 4.0,
            "acceleration_zone": 4.5,
            "elevation_crest": 3.0,
            "fuel_save": 2.5,
            "time_bonus": 6.0,
        }

        for marker_id in range(marker_count):
            marker_distance = (marker_id + 0.5) * self.length / marker_count
            position, segment_index = self._point_at_distance(marker_distance)
            kind = self._marker_kind(segment_index, marker_id)
            reward = reward_map.get(kind, 3.0) * reward_scale
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
        candidate_indices = self._center_grid.query_nearby(point[0], point[1])

        if not candidate_indices:
            # Fallback: full scan (should not happen for points near the track).
            candidate_indices = range(len(self.center_segments))

        best_distance = float("inf")
        best_point = self.centerline[0]
        best_index = 0
        best_t = 0.0
        best_normal = (0.0, 1.0)
        best_signed_offset = 0.0

        segments = self.center_segments
        for index in candidate_indices:
            start, end = segments[index]
            segment_distance, closest, t = project_point_to_segment(point, start, end)
            if segment_distance < best_distance:
                best_distance = segment_distance
                best_point = closest
                best_index = index
                best_t = t
                tangent = normalize(sub(end, start))
                best_normal = (-tangent[1], tangent[0])
                best_signed_offset = dot(sub(point, closest), best_normal)

        start, end = segments[best_index]
        heading = vector_angle(sub(end, start))
        progress = (best_index + best_t) / len(segments)
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

        # Query only edge segments overlapping the ray's bounding rectangle.
        end_x = origin[0] + direction[0] * max_distance
        end_y = origin[1] + direction[1] * max_distance
        candidate_indices = self._edge_grid.query_rect(
            min(origin[0], end_x),
            min(origin[1], end_y),
            max(origin[0], end_x),
            max(origin[1], end_y),
        )

        edge_segs = self.edge_segments
        for idx in candidate_indices:
            start, end = edge_segs[idx]
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
        # Binary search on cumulative distances instead of linear scan.
        index = bisect.bisect_left(self._cumulative_distances, wrapped_distance)
        if index >= len(self.center_segments):
            return self.centerline[-1], len(self.center_segments) - 1
        start, end = self.center_segments[index]
        seg_len = self._segment_lengths[index]
        prev_cum = self._cumulative_distances[index - 1] if index > 0 else 0.0
        t = (wrapped_distance - prev_cum) / max(seg_len, 1e-9)
        return add(start, mul(sub(end, start), t)), index

    def _marker_kind(self, segment_index: int, marker_id: int) -> str:
        segment_count = len(self.center_segments)
        
        def get_turn(offset: int, lookahead: int = 6) -> float:
            prev_seg = self.center_segments[(segment_index + offset - lookahead) % segment_count]
            next_seg = self.center_segments[(segment_index + offset + lookahead) % segment_count]
            prev_head = vector_angle(sub(prev_seg[1], prev_seg[0]))
            next_head = vector_angle(sub(next_seg[1], next_seg[0]))
            return wrap_angle(next_head - prev_head)

        current_turn = get_turn(0)
        abs_turn = abs(current_turn)
        next_turn = get_turn(15)
        prev_turn = get_turn(-15)

        if marker_id % 17 == 0:
            return "time_bonus"
        if marker_id % 23 == 0:
            return "pit_entry"
        if marker_id % 31 == 0:
            return "elevation_crest"
            
        if abs_turn <= 0.05:
            if abs(next_turn) > 0.15:
                return "braking_zone"
            if abs(prev_turn) > 0.15:
                return "acceleration_zone"
            if marker_id % 5 == 0:
                return "fuel_save"
            if marker_id % 7 == 0:
                return "overtake_zone"
            return "speed"
            
        if abs_turn >= 0.25:
            if current_turn * next_turn < -0.05:
                return "chicane"
            return "drift"
            
        if abs(next_turn) > abs_turn:
            return "late_apex"
        if abs(prev_turn) > abs_turn:
            return "clean_line"
            
        return "apex"
