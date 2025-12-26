"""
Polydev SWE-bench Agent v3

Improvements:
- REAL Polydev MCP consultation (Claude, GPT-5.2, Gemini)
- 15-worker parallel execution with staggered starts
- Increased limits (100 steps, 60 min timeout)
- Pre-cloning of repositories

Chief Resident Architecture (v4):
- Two-stage gating for consultation triggers
- Structured consult_polydev tool interface
- Unstuck Rate metric tracking
- 4 experimental configurations (A, B, C, D)
"""

# Original v3 components
from .agent import PolydevAgent, AgentConfig
from .runner import SWEBenchRunner, RunConfig, TaskResult
from .model import ClaudeModel, ModelConfig
from .environment import LocalEnvironment, EnvironmentConfig

# Chief Resident Architecture (v4) components
from .agent_state import AgentState, ErrorSignature, TestResult as TestResultState, PatchInfo
from .consultation_trigger import (
    ConsultationTrigger,
    TriggerType,
    ConsultType,
    TriggerConfig
)
from .chief_resident_agent import (
    ChiefResidentAgent,
    ChiefResidentConfig,
    ExperimentConfig,
    run_experiment,
    run_all_configs
)
from .evaluation_harness import (
    EvaluationHarness,
    AggregateMetrics,
    ExperimentResult
)

__all__ = [
    # Original v3
    "PolydevAgent",
    "AgentConfig",
    "SWEBenchRunner",
    "RunConfig",
    "TaskResult",
    "ClaudeModel",
    "ModelConfig",
    "LocalEnvironment",
    "EnvironmentConfig",
    # Chief Resident Architecture (v4)
    "AgentState",
    "ErrorSignature",
    "TestResultState",
    "PatchInfo",
    "ConsultationTrigger",
    "TriggerType",
    "ConsultType",
    "TriggerConfig",
    "ChiefResidentAgent",
    "ChiefResidentConfig",
    "ExperimentConfig",
    "run_experiment",
    "run_all_configs",
    "EvaluationHarness",
    "AggregateMetrics",
    "ExperimentResult",
]
