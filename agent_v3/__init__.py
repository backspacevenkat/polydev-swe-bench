"""
Polydev SWE-bench Agent v3

Improvements:
- REAL Polydev MCP consultation (Claude, GPT-5.2, Gemini)
- 15-worker parallel execution with staggered starts
- Increased limits (100 steps, 60 min timeout)
- Pre-cloning of repositories
"""

from .agent import PolydevAgent, AgentConfig
from .runner import SWEBenchRunner, RunConfig, TaskResult
from .model import ClaudeModel, ModelConfig
from .environment import LocalEnvironment, EnvironmentConfig

__all__ = [
    "PolydevAgent",
    "AgentConfig",
    "SWEBenchRunner",
    "RunConfig",
    "TaskResult",
    "ClaudeModel",
    "ModelConfig",
    "LocalEnvironment",
    "EnvironmentConfig",
]
