"""Training loop for PPO/SAC hybrid-action agents."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from ai.config import RLConfig
from ai.ppo_agent import PPOAgent
from ai.reward_shaper import RewardShaper
from ai.sac_agent import SACAgent
from racing_ai.world import RacingWorld


@dataclass(slots=True)
class EpisodeSummary:
    done_reason: str
    reward: float
    shaped_reward: float
    progress: float
    duration_sec: float
    steps: int
    avg_speed: float
    max_speed: float
    masked_actions: int


def _terminal_reason(obs: dict[str, object], step: int, max_steps: int) -> tuple[bool, str]:
    if bool(obs.get("death_event", False)):
        return True, "crash"

    if int(obs.get("lap", 0)) > 0:
        return True, "lap_complete"

    if int(obs.get("off_track_count", 0)) > 10:
        return True, "off_track_limit"

    time_elapsed = float(obs.get("time", 0.0))
    speed = float(obs.get("speed", 0.0))
    stalled = bool(obs.get("stalled", False))

    if time_elapsed > 5.0 and speed < 0.75:
        return True, "failed_launch"

    if stalled and time_elapsed > 4.0:
        return True, "stalled"

    if bool(obs.get("money_shift_event", False)) and float(obs.get("money_shift_damage", 0.0)) > 0.9:
        return True, "gearbox_damage"

    if step >= max_steps - 1:
        return True, "timeout"

    return False, "running"


def _gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    size = len(rewards)
    advantages = np.zeros(size, dtype=np.float32)
    gae = 0.0
    for idx in reversed(range(size)):
        non_terminal = 0.0 if dones[idx] else 1.0
        next_value = last_value if idx == size - 1 else values[idx + 1]
        delta = rewards[idx] + gamma * next_value * non_terminal - values[idx]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[idx] = gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def _top_components(components: dict[str, float], n: int = 4) -> str:
    if not components:
        return "{}"
    filtered = {k: v for k, v in components.items() if k not in {"total_unclipped", "total_clipped", "clip_adjustment"}}
    if not filtered:
        return "{}"
    top = sorted(filtered.items(), key=lambda item: abs(item[1]), reverse=True)[:n]
    return ", ".join(f"{k}={v:+.3f}" for k, v in top)


def _physics_sanity_probe(world: RacingWorld) -> None:
    obs = world.reset()
    for _ in range(40):
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
            dt=1.0 / 60.0,
        )
    idle_spin = abs(float(obs.get("yaw_rate", 0.0)))

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
            dt=1.0 / 60.0,
        )
    launch_speed = float(obs.get("speed", 0.0))
    print(f"[sanity] idle steer yaw={idle_spin:.3f} | 2s full-throttle speed={launch_speed:.2f}")


def train(
    config: RLConfig,
    episodes: int,
    device: str = "auto",
    render: bool = False,
    algo: str = "ppo",
) -> None:
    algo = algo.lower().strip()
    if algo not in {"ppo", "sac"}:
        raise ValueError(f"Unsupported algorithm '{algo}'. Use 'ppo' or 'sac'.")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    world = RacingWorld(
        marker_spacing=config.marker_spacing,
        normalize_marker_reward=config.normalize_marker_reward,
    )
    reward_shaper = RewardShaper(config.reward_clip)

    if algo == "ppo":
        agent = PPOAgent(config, device=device)
    else:
        agent = SACAgent(config, device=device)
    agent.train_mode(True)

    renderer = None
    if render:
        from racing_ai.renderer import RacingWindow

        renderer = RacingWindow(world)

    print(f"Starting {algo.upper()} training on {device}...")
    _physics_sanity_probe(world)

    best_score = -float("inf")
    ppo_last_metrics: dict[str, float] = {}
    sac_last_metrics: dict[str, float] = {}

    # PPO rollout storage
    ppo_states: list[np.ndarray] = []
    ppo_cont_raw: list[np.ndarray] = []
    ppo_gear: list[int] = []
    ppo_log_probs: list[float] = []
    ppo_values: list[float] = []
    ppo_rewards: list[float] = []
    ppo_dones: list[bool] = []

    def flush_ppo_rollout(last_value: float) -> None:
        nonlocal ppo_last_metrics
        if not ppo_states:
            return

        advantages, returns = _gae(
            rewards=ppo_rewards,
            values=ppo_values,
            dones=ppo_dones,
            last_value=last_value,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
        )
        batch = {
            "states": np.asarray(ppo_states, dtype=np.float32),
            "cont_raw": np.asarray(ppo_cont_raw, dtype=np.float32),
            "gear": np.asarray(ppo_gear, dtype=np.int64),
            "log_probs": np.asarray(ppo_log_probs, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
        }
        ppo_last_metrics = agent.update(batch)  # type: ignore[arg-type]

        ppo_states.clear()
        ppo_cont_raw.clear()
        ppo_gear.clear()
        ppo_log_probs.clear()
        ppo_values.clear()
        ppo_rewards.clear()
        ppo_dones.clear()

    for episode in range(1, episodes + 1):
        obs = world.reset()
        reward_shaper.reset()
        state = agent.obs_extractor(obs)

        episode_reward = 0.0
        episode_shaped_reward = 0.0
        episode_speeds: list[float] = []
        masked_actions = 0

        start_time = time.time()
        done_reason = "timeout"

        for step in range(config.max_steps_per_episode):
            sample = agent.sample_action(state=state, obs=obs)  # type: ignore[attr-defined]
            action = sample.env_action

            next_obs = world.step(action, dt=1.0 / 60.0)
            shaped_reward = reward_shaper(obs, next_obs, action)
            done, done_reason = _terminal_reason(next_obs, step, config.max_steps_per_episode)

            next_state = agent.obs_extractor(next_obs)

            if algo == "ppo":
                ppo_states.append(state)
                ppo_cont_raw.append(sample.cont_raw)
                ppo_gear.append(sample.gear)
                ppo_log_probs.append(sample.log_prob)
                ppo_values.append(sample.value)
                ppo_rewards.append(shaped_reward)
                ppo_dones.append(done)

                if len(ppo_states) >= config.ppo.rollout_steps:
                    if done:
                        bootstrap_value = 0.0
                    else:
                        with torch.no_grad():
                            state_t = torch.tensor(next_state, dtype=torch.float32, device=agent.device).unsqueeze(0)  # type: ignore[attr-defined]
                            bootstrap_value = float(agent.value_net(state_t).item())  # type: ignore[attr-defined]
                    flush_ppo_rollout(last_value=bootstrap_value)
            else:
                agent.replay.push(state, sample.cont_raw, sample.gear, shaped_reward, next_state, done)  # type: ignore[attr-defined]
                update_metrics = agent.maybe_update()  # type: ignore[attr-defined]
                if update_metrics is not None:
                    sac_last_metrics = update_metrics

            if sample.mask_modified:
                masked_actions += 1

            state = next_state
            obs = next_obs
            episode_reward += float(obs.get("frame_reward", 0.0))
            episode_shaped_reward += shaped_reward
            episode_speeds.append(float(obs.get("speed", 0.0)))

            if renderer:
                renderer.dispatch_events()
                renderer.on_draw()
                renderer.flip()

            if step % config.log_every_steps == 0:
                components = _top_components(reward_shaper.last_components)
                print(
                    f"  step {step:4d} | spd={float(obs.get('speed', 0.0)):6.2f} | "
                    f"gear={int(obs.get('gear', 1)):2d} | rpm={float(obs.get('rpm', 0.0)):6.0f} | "
                    f"shaped={shaped_reward:+6.3f} | mask={sample.mask_modified} | comps: {components}"
                )
                if sample.mask_modified and sample.mask_reasons:
                    print(f"    mask reasons: {', '.join(sample.mask_reasons[:3])}")

            if done:
                if algo == "ppo" and ppo_states:
                    flush_ppo_rollout(last_value=0.0)
                break

        duration = time.time() - start_time
        avg_speed = float(np.mean(episode_speeds)) if episode_speeds else 0.0
        max_speed = float(np.max(episode_speeds)) if episode_speeds else 0.0
        progress = float(obs.get("progress", 0.0)) * 100.0
        lap = int(obs.get("lap", 0))

        summary = EpisodeSummary(
            done_reason=done_reason,
            reward=episode_reward,
            shaped_reward=episode_shaped_reward,
            progress=progress,
            duration_sec=duration,
            steps=step + 1,
            avg_speed=avg_speed,
            max_speed=max_speed,
            masked_actions=masked_actions,
        )

        if algo == "ppo":
            algo_metrics = (
                f"pl={ppo_last_metrics.get('policy_loss', 0.0):.4f} "
                f"vl={ppo_last_metrics.get('value_loss', 0.0):.4f} "
                f"ent={ppo_last_metrics.get('entropy', 0.0):.4f} "
                f"kl={ppo_last_metrics.get('approx_kl', 0.0):.4f}"
            )
        else:
            algo_metrics = (
                f"qloss={sac_last_metrics.get('critic_loss', 0.0):.4f} "
                f"aloss={sac_last_metrics.get('actor_loss', 0.0):.4f} "
                f"alpha={sac_last_metrics.get('alpha', 0.0):.4f} "
                f"q={sac_last_metrics.get('q_value', 0.0):.3f}"
            )

        print(
            f"Ep {episode:4d} [{algo.upper()}] | reason={summary.done_reason:<14} | "
            f"reward={summary.reward:8.2f} | shaped={summary.shaped_reward:8.2f} | "
            f"progress={summary.progress:5.1f}% lap={lap} | "
            f"spd(avg/max)={summary.avg_speed:5.2f}/{summary.max_speed:5.2f} | "
            f"masked={summary.masked_actions:4d} | "
            f"stall={int(obs.get('stall_count', 0)):2d} money={int(obs.get('money_shift_count', 0)):2d} | "
            f"time={summary.duration_sec:5.1f}s | {algo_metrics}"
        )

        score = progress + lap * 100.0 + episode_reward * 0.01
        if score > best_score:
            best_score = score
            agent.save(os.path.join(config.checkpoint_dir, "best.pt"))  # type: ignore[attr-defined]

        if episode % config.save_every_episodes == 0:
            agent.save(os.path.join(config.checkpoint_dir, f"checkpoint_{episode}.pt"))  # type: ignore[attr-defined]

    if algo == "ppo" and ppo_states:
        flush_ppo_rollout(last_value=0.0)

    agent.save(os.path.join(config.checkpoint_dir, "final.pt"))  # type: ignore[attr-defined]
    print("Training complete.")



