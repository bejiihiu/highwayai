from __future__ import annotations

import unittest

from racing_ai.math2d import distance
from racing_ai.track import Track
from racing_ai.world import RacingWorld


class TrackGenerationTest(unittest.TestCase):
    def test_default_track_is_extreme_drift_layout(self) -> None:
        track = Track.build_default()
        markers = track.reward_markers()
        min_x, min_y, max_x, max_y = track.bounds

        self.assertGreaterEqual(track.length, 18_000.0)
        self.assertGreater(max_x - min_x, 1100.0)
        self.assertGreater(max_y - min_y, 760.0)
        self.assertTrue(track.sample_at(track.spawn_pose()[0]).on_track)
        self.assertGreaterEqual(len(markers), 60)
        self.assertGreaterEqual(sum(marker.kind == "drift" for marker in markers), 25)
        self.assertLess(distance(track.inner_boundary[-1], track.inner_boundary[0]), 20.0)
        self.assertLess(distance(track.outer_boundary[-1], track.outer_boundary[0]), 20.0)


class RacingWorldTest(unittest.TestCase):
    def test_zero_action_keeps_car_at_start(self) -> None:
        world = RacingWorld()
        start = world.car.position

        for _ in range(20):
            observation = world.step({"throttle": 0.0, "steer": 0.0, "brake": 0.0}, 1.0 / 60.0)

        self.assertAlmostEqual(world.car.x, start[0], places=6)
        self.assertAlmostEqual(world.car.y, start[1], places=6)
        self.assertFalse(observation["off_track"])
        self.assertEqual(observation["markers_collected"], 0)

    def test_action_moves_car_and_updates_rays(self) -> None:
        world = RacingWorld()

        for _ in range(45):
            observation = world.step({"throttle": 1.0, "steer": 0.15, "brake": 0.0}, 1.0 / 60.0)

        self.assertGreater(observation["speed"], 1.0)
        self.assertEqual(len(observation["rays"]), len(world.ray_angles))
        self.assertTrue(all(0.0 <= ray["normalized_distance"] <= 1.0 for ray in observation["rays"]))
        for key in (
            "speed",
            "speed_kmh",
            "heading_error",
            "off_track",
            "distance_to_center",
            "progress",
            "rays",
            "nearest_markers",
            "car_position",
            "car_velocity",
            "death_event",
            "death_count",
            "last_action",
            "virtual_keys",
        ):
            self.assertIn(key, observation)
        for key in (
            "signed_distance_to_center",
            "edge_clearance",
            "edge_collision",
            "edge_collision_count",
            "last_edge_impact",
        ):
            self.assertIn(key, observation)

    def test_marker_collection_rewards_agent(self) -> None:
        world = RacingWorld()
        marker = world.markers[0]
        world.car.x, world.car.y = marker.position

        observation = world.step({"throttle": 0.0, "steer": 0.0, "brake": 0.0}, 1.0 / 60.0)

        self.assertTrue(marker.collected)
        self.assertEqual(observation["markers_collected"], 1)
        self.assertGreaterEqual(observation["frame_reward"], marker.reward)

    def test_markers_restore_after_lap_wrap(self) -> None:
        world = RacingWorld()
        for marker in world.markers:
            marker.collected = True
        world.stats.markers_collected = len(world.markers)
        world.previous_progress = 0.9
        world.car.x, world.car.y = world.track.centerline[0]

        observation = world.step({"throttle": 0.0, "steer": 0.0, "brake": 0.0}, 1.0 / 60.0)

        self.assertEqual(observation["lap"], 1)
        self.assertEqual(observation["markers_collected"], 0)
        self.assertTrue(all(not marker.collected for marker in world.markers))
        self.assertGreater(len(observation["nearest_markers"]), 0)

    def test_off_track_penalty_is_reported(self) -> None:
        world = RacingWorld()
        world.car.x = 5.0
        world.car.y = 5.0

        observation = world.step({"throttle": 0.0, "steer": 0.0, "brake": 0.0}, 1.0 / 60.0)

        self.assertTrue(observation["off_track"])
        self.assertEqual(observation["off_track_count"], 1)
        self.assertLess(observation["frame_reward"], 0.0)

    def test_edge_collision_triggers_death_and_respawn(self) -> None:
        world = RacingWorld()
        for marker in world.markers:
            marker.collected = True
        world.stats.lap_markers_collected = len(world.markers)
        world.stats.markers_collected = len(world.markers)

        center = world.track.centerline[0]
        sample = world.track.sample_at(center)
        collision_radius = world.car.width * 0.55
        offset = world.track.half_width - collision_radius * 0.35
        world.car.x = center[0] + sample.normal[0] * offset
        world.car.y = center[1] + sample.normal[1] * offset
        world.car.vx = sample.normal[0] * 90.0
        world.car.vy = sample.normal[1] * 90.0

        observation = world.step({"throttle": 0.0, "steer": 0.0, "brake": 0.0}, 1.0 / 60.0)
        spawn_position, _ = world.track.spawn_pose()

        self.assertTrue(observation["edge_collision"])
        self.assertEqual(observation["edge_collision_count"], 1)
        self.assertTrue(observation["death_event"])
        self.assertEqual(observation["death_count"], 1)
        self.assertGreater(observation["last_edge_impact"], 0.0)
        self.assertLessEqual(float(observation["frame_reward"]), -50.0)
        self.assertLess(distance(world.car.position, spawn_position), 1e-6)
        self.assertEqual(observation["markers_collected"], 0)
        self.assertTrue(all(not marker.collected for marker in world.markers))

    def test_drift_score_increases_when_velocity_slips(self) -> None:
        world = RacingWorld()
        world.car.vx = 120.0
        world.car.vy = 130.0

        observation = world.step({"throttle": 0.0, "steer": 0.0, "brake": 0.0}, 1.0 / 60.0)

        self.assertGreater(observation["drift_intensity"], 0.0)
        self.assertGreater(observation["drift_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
