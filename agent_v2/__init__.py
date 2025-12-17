"""
Polydev SWE-bench Agent v2

A redesigned agent based on mini-swe-agent architecture with Polydev MCP integration.

Key differences from v1:
- Actually executes bash commands in repositories
- Iterative problem-solving with feedback
- Polydev MCP consultation when stuck
"""

__version__ = "2.0.0"

from .agent import PolydevAgent
from .environment import LocalEnvironment
from .model import ClaudeModel

__all__ = ["PolydevAgent", "LocalEnvironment", "ClaudeModel"]
