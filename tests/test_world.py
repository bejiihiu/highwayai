from __future__ import annotations

import unittest

from racing_ai.world import RacingWorld


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

    def test_marker_collection_rewards_agent(self) -> None:
        world = RacingWorld()
        marker = world.markers[0]
        world.car.x, world.car.y = marker.position

        observation = world.step({"throttle": 0.0, "steer": 0.0, "brake": 0.0}, 1.0 / 60.0)

        self.assertTrue(marker.collected)
        self.assertEqual(observation["markers_collected"], 1)
        self.assertGreaterEqual(observation["frame_reward"], marker.reward)

    def test_off_track_penalty_is_reported(self) -> None:
        world = RacingWorld()
        world.car.x = 5.0
        world.car.y = 5.0

        observation = world.step({"throttle": 0.0, "steer": 0.0, "brake": 0.0}, 1.0 / 60.0)

        self.assertTrue(observation["off_track"])
        self.assertEqual(observation["off_track_count"], 1)
        self.assertLess(observation["frame_reward"], 0.0)

    def test_drift_score_increases_when_velocity_slips(self) -> None:
        world = RacingWorld()
        world.car.vx = 120.0
        world.car.vy = 130.0

        observation = world.step({"throttle": 0.0, "steer": 0.0, "brake": 0.0}, 1.0 / 60.0)

        self.assertGreater(observation["drift_intensity"], 0.0)
        self.assertGreater(observation["drift_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
