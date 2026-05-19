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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
