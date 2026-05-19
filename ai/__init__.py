"""Actor-critic racing AI with hybrid actions."""

from ai.config import RLConfig

__all__ = ["RLConfig", "PPOAgent", "SACAgent"]

try:
    from ai.ppo_agent import PPOAgent
    from ai.sac_agent import SACAgent
except ModuleNotFoundError:
    # Allow importing light modules (e.g. reward shaping tests) without torch.
    PPOAgent = None  # type: ignore[assignment]
    SACAgent = None  # type: ignore[assignment]
