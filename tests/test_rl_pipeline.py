from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai.reward_shaper import RewardShaper
from racing_ai.math2d import angle_to_vector
from racing_ai.world import RacingWorld


class GearboxPhysicsTest(unittest.TestCase):
    def test_standstill_steer_does_not_spin_up(self) -> None:
        world = RacingWorld()
        obs = world.reset()

        for _ in range(60):
            obs = world.step(
                {
                    "throttle": 0.0,
                    "steer": 1.0,
                    "brake": 0.0,
                    "clutch": 0.0,
                    "handbrake": 0.0,
                    "gear_up": 0.0,
                    "gear_down": 0.0,
                },
                1.0 / 60.0,
            )

        self.assertLess(float(obs["speed"]), 0.5)
        self.assertLess(abs(float(obs["yaw_rate"])), 0.8)

    def test_launch_reaches_min_speed(self) -> None:
        world = RacingWorld()
        obs = world.reset()

        for _ in range(120):
            obs = world.step(
                {
                    "throttle": 1.0,
                    "steer": 0.0,
                    "brake": 0.0,
                    "clutch": 0.0,
                    "handbrake": 0.0,
                    "gear_up": 0.0,
                    "gear_down": 0.0,
                },
                1.0 / 60.0,
            )

        self.assertGreater(float(obs["speed"]), 2.0)

    def test_clutch_disengaged_revs_with_low_wheel_torque(self) -> None:
        world = RacingWorld()
        obs = world.reset()

        for _ in range(100):
            obs = world.step(
                {
                    "throttle": 1.0,
                    "steer": 0.0,
                    "brake": 0.0,
                    "clutch": 1.0,
                    "handbrake": 0.0,
                    "gear_up": 0.0,
                    "gear_down": 0.0,
                },
                1.0 / 60.0,
            )

        self.assertGreater(float(obs["rpm"]), 3000.0)
        self.assertLess(float(obs["speed"]), 1.0)

    def test_valid_shift_applies_with_clutch(self) -> None:
        world = RacingWorld()
        world.reset()

        for _ in range(30):
            world.step(
                {
                    "throttle": 0.7,
                    "steer": 0.0,
                    "brake": 0.0,
                    "clutch": 0.0,
                    "handbrake": 0.0,
                    "gear_up": 0.0,
                    "gear_down": 0.0,
                },
                1.0 / 60.0,
            )

        obs = world.step(
            {
                "throttle": 0.4,
                "steer": 0.0,
                "brake": 0.0,
                "clutch": 1.0,
                "handbrake": 0.0,
                "gear_up": 1.0,
                "gear_down": 0.0,
            },
            1.0 / 60.0,
        )

        self.assertTrue(bool(obs["shift_applied"]))
        self.assertEqual(int(obs["gear"]), 2)

    def test_bad_downshift_triggers_money_shift(self) -> None:
        world = RacingWorld()
        world.reset()
        world.car.gear = 5
        forward = angle_to_vector(world.car.heading)
        world.car.vx = forward[0] * 220.0
        world.car.vy = forward[1] * 220.0

        obs = world.step(
            {
                "throttle": 0.2,
                "steer": 0.0,
                "brake": 0.0,
                "clutch": 1.0,
                "handbrake": 0.0,
                "gear_up": 0.0,
                "gear_down": 1.0,
            },
            1.0 / 60.0,
        )

        self.assertTrue(bool(obs["money_shift_event"]))
        self.assertGreater(float(obs["money_shift_damage"]), 0.0)


class RewardDiagnosticsTest(unittest.TestCase):
    def test_reward_components_include_totals(self) -> None:
        shaper = RewardShaper(clip_val=18.0)

        prev_obs = {
            "progress": 0.2,
            "heading": 0.0,
        }
        next_obs = {
            "frame_reward": 0.0,
            "speed": 8.0,
            "forward_speed": 8.0,
            "drift_intensity": 0.0,
            "off_track": False,
            "heading_error": 0.0,
            "stalled": False,
            "progress": 0.24,
            "rpm": 4100.0,
            "clutch": 0.8,
            "shift_attempted": True,
            "shift_applied": True,
            "shift_blocked": False,
            "overrev": False,
            "money_shift_event": False,
            "stall_event": False,
            "clutch_slip": 0.15,
            "gear_shift_count": 1,
            "heading": 0.0,
        }
        action = {"throttle": 0.5, "steer": 0.0, "brake": 0.0}

        shaped = shaper(prev_obs, next_obs, action)

        self.assertIn("total_clipped", shaper.last_components)
        self.assertIn("total_unclipped", shaper.last_components)
        self.assertAlmostEqual(shaper.last_components["total_clipped"], shaped, places=5)
        self.assertGreater(shaper.last_components.get("valid_shift_bonus", 0.0), 0.0)


class TerminalReasonTest(unittest.TestCase):
    def test_death_event_terminates_episode_with_crash_reason(self) -> None:
        try:
            from ai.trainer import _terminal_reason
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise unittest.SkipTest("torch not installed in this environment") from exc
            raise

        done, reason = _terminal_reason({"death_event": True}, step=0, max_steps=120)
        self.assertTrue(done)
        self.assertEqual(reason, "crash")


class TrainingSmokeTest(unittest.TestCase):
    def _require_torch(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest("torch not installed in this environment") from exc

    def test_ppo_smoke_train_writes_checkpoint(self) -> None:
        self._require_torch()

        from ai.config import RLConfig
        from ai.trainer import train

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = RLConfig()
            cfg.checkpoint_dir = tmpdir
            cfg.max_steps_per_episode = 48
            cfg.log_every_steps = 9999
            cfg.save_every_episodes = 1000
            cfg.ppo.rollout_steps = 16
            cfg.ppo.update_epochs = 1
            cfg.ppo.minibatch_size = 8

            train(cfg, episodes=1, device="cpu", render=False, algo="ppo")

            self.assertTrue((Path(tmpdir) / "final.pt").exists())
            self.assertTrue((Path(tmpdir) / "best.pt").exists())

    def test_sac_smoke_train_writes_checkpoint(self) -> None:
        self._require_torch()

        from ai.config import RLConfig
        from ai.trainer import train

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = RLConfig()
            cfg.checkpoint_dir = tmpdir
            cfg.max_steps_per_episode = 64
            cfg.log_every_steps = 9999
            cfg.save_every_episodes = 1000
            cfg.sac.replay_capacity = 512
            cfg.sac.batch_size = 8
            cfg.sac.warmup_steps = 8
            cfg.sac.updates_per_step = 1

            train(cfg, episodes=1, device="cpu", render=False, algo="sac")

            self.assertTrue((Path(tmpdir) / "final.pt").exists())
            self.assertTrue((Path(tmpdir) / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
