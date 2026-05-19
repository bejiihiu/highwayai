"""Quick benchmark for the simulation hot path (no rendering)."""
from __future__ import annotations

import time

from racing_ai.agent import ScriptedDemoAgent
from racing_ai.world import RacingWorld


def bench_steps(n: int = 2000) -> None:
    world = RacingWorld(agent=ScriptedDemoAgent())
    dt = 1.0 / 60.0

    # Warm up
    for _ in range(20):
        world.update(dt)

    start = time.perf_counter()
    for _ in range(n):
        world.update(dt)
    elapsed = time.perf_counter() - start

    steps_per_sec = n / elapsed
    us_per_step = elapsed / n * 1e6
    print(f"  {n} steps in {elapsed:.3f}s  ->  {steps_per_sec:,.0f} steps/s  ({us_per_step:.0f} us/step)")


def bench_sample_at(n: int = 5000) -> None:
    world = RacingWorld()
    track = world.track
    # Sample at many different centerline points.
    points = track.centerline[::4][:n]
    if len(points) < n:
        points = (points * (n // len(points) + 1))[:n]

    start = time.perf_counter()
    for p in points:
        track.sample_at(p)
    elapsed = time.perf_counter() - start

    calls_per_sec = n / elapsed
    us_per_call = elapsed / n * 1e6
    print(f"  {n} sample_at calls in {elapsed:.3f}s  ->  {calls_per_sec:,.0f} calls/s  ({us_per_call:.0f} us/call)")


def bench_raycast(n: int = 5000) -> None:
    import math
    world = RacingWorld()
    track = world.track
    origin = track.centerline[0]
    angles = [math.radians(a) for a in range(0, 360, 5)]

    start = time.perf_counter()
    count = 0
    while count < n:
        for angle in angles:
            track.raycast(origin, angle, 280.0)
            count += 1
            if count >= n:
                break
    elapsed = time.perf_counter() - start

    calls_per_sec = n / elapsed
    us_per_call = elapsed / n * 1e6
    print(f"  {n} raycast calls in {elapsed:.3f}s  ->  {calls_per_sec:,.0f} calls/s  ({us_per_call:.0f} us/call)")


if __name__ == "__main__":
    print("=== Simulation Benchmark ===\n")
    print("world.update (full step):")
    bench_steps()
    print("\ntrack.sample_at:")
    bench_sample_at()
    print("\ntrack.raycast:")
    bench_raycast()
