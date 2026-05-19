from __future__ import annotations

import argparse

from racing_ai.agent import ScriptedDemoAgent, ZeroAgent
from racing_ai.track import Track
from racing_ai.world import RacingWorld


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-down racing sandbox for actor-critic AI experiments.")
    parser.add_argument(
        "--demo-agent",
        action="store_true",
        help="Use a tiny scripted agent so the car moves without keyboard controls.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Start headless RL training.",
    )
    parser.add_argument(
        "--train-render",
        action="store_true",
        help="Start RL training with live pyglet visualization.",
    )
    parser.add_argument(
        "--play",
        type=str,
        help="Path to a checkpoint file to run the trained policy.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "sac"],
        help="RL algorithm to use (ppo or sac).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for PyTorch (auto, cpu, cuda).",
    )
    parser.add_argument(
        "--marker-spacing",
        type=float,
        default=Track.DEFAULT_MARKER_SPACING,
        help=(
            "Distance between reward markers in world units. "
            "Lower value means more markers (minimum enforced in world logic)."
        ),
    )
    return parser.parse_args()


def _infer_algorithm_from_checkpoint(path: str, fallback: str) -> str:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return fallback

    algo = str(checkpoint.get("algorithm", fallback)).lower()
    if algo in {"ppo", "sac"}:
        return algo
    return fallback


def main() -> None:
    args = parse_args()

    if args.train or args.train_render:
        try:
            from ai.config import RLConfig
            from ai.trainer import train
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency for training. Run: python -m pip install -r requirements.txt") from exc

        config = RLConfig()
        config.marker_spacing = float(args.marker_spacing)
        train(config, episodes=args.episodes, device=args.device, render=args.train_render, algo=args.algo)
        return

    if args.play:
        try:
            from ai.config import RLConfig
            from ai.ppo_agent import PPOAgent
            from ai.sac_agent import SACAgent
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency for playing. Run: python -m pip install -r requirements.txt") from exc

        config = RLConfig()
        algo = _infer_algorithm_from_checkpoint(args.play, args.algo)
        if algo != args.algo:
            print(f"Checkpoint metadata detected algorithm={algo}; overriding --algo {args.algo}.")

        if algo == "sac":
            agent = SACAgent(config, device=args.device)
        else:
            agent = PPOAgent(config, device=args.device)

        try:
            agent.load(args.play)
            agent.train_mode(False)
            print(f"Loaded {algo.upper()} checkpoint from {args.play}")
        except FileNotFoundError:
            raise SystemExit(f"Checkpoint not found: {args.play}")
    else:
        agent = ScriptedDemoAgent() if args.demo_agent else ZeroAgent()

    world = RacingWorld(
        agent=agent,
        marker_spacing=float(args.marker_spacing),
    )

    try:
        from racing_ai.renderer import run_pyglet_app
    except ModuleNotFoundError as exc:
        if exc.name == "pyglet":
            raise SystemExit("pyglet is not installed. Run: python -m pip install -r requirements.txt") from exc
        raise

    run_pyglet_app(world)


if __name__ == "__main__":
    main()
