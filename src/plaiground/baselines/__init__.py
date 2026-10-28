"""Non-learning baseline policies."""

from .heuristic import agent_brain as heuristic_action
from .random import agent_brain as random_action

__all__ = ["heuristic_action", "random_action"]
