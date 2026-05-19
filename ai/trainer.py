"""Training loop for the DQN Agent."""
from __future__ import annotations

import os
import time

from ai.config import DQNConfig
from ai.dqn_agent import DQNAgent
from racing_ai.world import RacingWorld


def is_terminal(obs: dict[str, object]) -> bool:
    """Check if episode should end."""
    if int(obs.get("off_track_count", 0)) > 10:
        return True
    
    speed = float(obs.get("speed", 0.0))
    time_elapsed = float(obs.get("time", 0.0))
    if time_elapsed > 5.0 and speed < 5.0:
        return True
        
    stalled = bool(obs.get("stalled", False))
    if stalled and time_elapsed > 5.0:
        return True

    return False


def train(config: DQNConfig, episodes: int, device: str = "auto", render: bool = False) -> None:
    print(f"Starting training on {device}...")
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    world = RacingWorld()
    agent = DQNAgent(config, device=device)
    agent.train_mode(True)
    
    # Optional rendering setup
    renderer = None
    if render:
        from racing_ai.renderer import RacingWindow
        import pyglet
        renderer = RacingWindow(world)
        # Use a clock to prevent runaway rendering if needed, but we'll pump events manually

    best_reward = -float("inf")
    
    for episode in range(1, episodes + 1):
        obs = world.reset()
        agent.reward_shaper.reset()
        state = agent.obs_extractor(obs)
        
        episode_reward = 0.0
        episode_shaped_reward = 0.0
        episode_loss = 0.0
        loss_steps = 0
        
        start_time = time.time()
        
        for step in range(config.max_steps_per_episode):
            # 1. Select action
            action_idx = agent.select_action(state)
            action_dict = agent.action_space.decode(action_idx)
            
            # 2. Step environment
            next_obs = world.step(action_dict, dt=1.0/60.0)
            
            # 3. Shape reward
            shaped_reward = agent.reward_shaper(obs, next_obs, action_dict)
            
            # 4. Check done
            done = is_terminal(next_obs)
            if step >= config.max_steps_per_episode - 1:
                done = True
                
            # Bonus for completing lap
            lap = int(next_obs.get("lap", 0))
            if lap > 0:
                shaped_reward += 50.0
                done = True
                
            next_state = agent.obs_extractor(next_obs)
            
            # 5. Store transition
            agent.replay.push(state, action_idx, shaped_reward, next_state, done)
            
            # 6. Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss += loss
                loss_steps += 1
                
            # Update state
            state = next_state
            obs = next_obs
            episode_reward += float(obs.get("frame_reward", 0.0))
            episode_shaped_reward += shaped_reward
            
            # Render if requested
            if renderer:
                renderer.dispatch_events()
                renderer.update(1.0/60.0)
                renderer.on_draw()
                renderer.flip()
                
            if done:
                break
                
        # Logging
        duration = time.time() - start_time
        avg_loss = episode_loss / max(loss_steps, 1)
        progress = float(obs.get("progress", 0.0)) * 100.0
        
        print(f"Ep {episode:4d} | "
              f"Reward: {episode_reward:7.1f} | "
              f"Shaped: {episode_shaped_reward:7.1f} | "
              f"Progress: {progress:5.1f}% | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Loss: {avg_loss:.4f} | "
              f"Time: {duration:.1f}s")
              
        # Checkpointing
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(config.checkpoint_dir, "best.pt"))
            
        if episode % config.save_every_episodes == 0:
            agent.save(os.path.join(config.checkpoint_dir, f"checkpoint_{episode}.pt"))

    # Final save
    agent.save(os.path.join(config.checkpoint_dir, "final.pt"))
    print("Training complete.")
