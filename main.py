from __future__ import annotations

import argparse

from racing_ai.agent import ScriptedDemoAgent, ZeroAgent
from racing_ai.world import RacingWorld


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-down racing sandbox for AI experiments.")
    parser.add_argument(
        "--demo-agent",
        action="store_true",
        help="Use a tiny scripted agent so the car moves without keyboard controls.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Start headless DQN training.",
    )
    parser.add_argument(
        "--train-render",
        action="store_true",
        help="Start DQN training with live pyglet visualization.",
    )
    parser.add_argument(
        "--play",
        type=str,
        help="Path to a checkpoint file to run the trained agent.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for PyTorch (auto, cpu, cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Training modes bypass the normal run loop
    if args.train or args.train_render:
        try:
            from ai.trainer import train
            from ai.config import DQNConfig
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency for training. Run: python -m pip install -r requirements.txt") from exc
        
        config = DQNConfig()
        train(config, episodes=args.episodes, device=args.device, render=args.train_render)
        return

    # Play mode
    if args.play:
        try:
            from ai.dqn_agent import DQNAgent
            from ai.config import DQNConfig
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency for playing. Run: python -m pip install -r requirements.txt") from exc
        
        config = DQNConfig()
        agent = DQNAgent(config, device=args.device)
        try:
            agent.load(args.play)
            agent.train_mode(False)
            print(f"Loaded checkpoint from {args.play}")
        except FileNotFoundError:
            raise SystemExit(f"Checkpoint not found: {args.play}")
    else:
        # Default agents
        agent = ScriptedDemoAgent() if args.demo_agent else ZeroAgent()

    world = RacingWorld(agent=agent)

    try:
        from racing_ai.renderer import run_pyglet_app
    except ModuleNotFoundError as exc:
        if exc.name == "pyglet":
            raise SystemExit("pyglet is not installed. Run: python -m pip install -r requirements.txt") from exc
        raise

    run_pyglet_app(world)


if __name__ == "__main__":
    main()
