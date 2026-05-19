from __future__ import annotations

import math

Point = tuple[float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Point, scalar: float) -> Point:
    return (a[0] * scalar, a[1] * scalar)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def length(a: Point) -> float:
    return math.hypot(a[0], a[1])


def distance(a: Point, b: Point) -> float:
    return length(sub(a, b))


def normalize(a: Point) -> Point:
    size = length(a)
    if size <= 1e-9:
        return (1.0, 0.0)
    return (a[0] / size, a[1] / size)


def angle_to_vector(angle: float) -> Point:
    return (math.cos(angle), math.sin(angle))


def vector_angle(a: Point) -> float:
    return math.atan2(a[1], a[0])


def wrap_angle(angle: float) -> float:
    return math.remainder(angle, math.tau)


def project_point_to_segment(point: Point, a: Point, b: Point) -> tuple[float, Point, float]:
    ab = sub(b, a)
    denom = dot(ab, ab)
    if denom <= 1e-9:
        return distance(point, a), a, 0.0

    t = clamp(dot(sub(point, a), ab) / denom, 0.0, 1.0)
    closest = add(a, mul(ab, t))
    return distance(point, closest), closest, t


def ray_segment_intersection(origin: Point, direction: Point, a: Point, b: Point) -> float | None:
    segment = sub(b, a)
    denom = cross(direction, segment)
    if abs(denom) <= 1e-9:
        return None

    offset = sub(a, origin)
    ray_t = cross(offset, segment) / denom
    segment_t = cross(offset, direction) / denom
    if ray_t >= 0.0 and 0.0 <= segment_t <= 1.0:
        return ray_t
    return None
