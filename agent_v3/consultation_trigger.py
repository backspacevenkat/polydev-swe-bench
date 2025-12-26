"""
ConsultationTrigger: Two-Stage Gating for Polydev Consultation

Implements the "Chief Resident" architecture where the base model
only consults Polydev when specific trigger conditions are met.

Stage A: Deterministic Rule-Based Triggers (cheap, fast)
Stage B: Model-Based Self-Diagnostic (if Stage A passes)
"""

import re
import logging
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of consultation triggers."""
    STUCK_LOOP = auto()       # Same error after 2+ fix attempts
    STAGNATION = auto()       # No progress in 2+ iterations
    CONTRADICTION = auto()    # Model's assertions conflict
    LOW_CONFIDENCE = auto()   # Model expresses uncertainty
    SEARCH_EXPLOSION = auto() # Too many plausible causes
    API_UNCERTAINTY = auto()  # Unsure about library usage
    REPEATED_COMPILE_ERROR = auto()  # Same type/syntax error 2+ times
    SECURITY_SENSITIVE = auto()      # Auth, crypto, sandboxing changes


class ConsultType(Enum):
    """Types of consultation requests."""
    DEBUG = "debug"
    DESIGN_ALTERNATIVE = "design_alternative"
    EDGE_CASES = "edge_cases"
    API_VERIFICATION = "api_verification"
    ARCHITECTURE = "architecture"


@dataclass
class ErrorSignature:
    """Normalized error signature for comparison."""
    error_type: str
    error_message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None

    def __hash__(self):
        return hash((self.error_type, self.error_message[:100]))

    def __eq__(self, other):
        if not isinstance(other, ErrorSignature):
            return False
        return (self.error_type == other.error_type and
                self.error_message[:100] == other.error_message[:100])


@dataclass
class TestResult:
    """Test execution result."""
    passed: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_message: str = ""

    @property
    def failures(self) -> int:
        return self.failed_tests


@dataclass
class TriggerConfig:
    """Configuration for trigger thresholds."""
    stuck_loop_threshold: int = 2      # Same error N times
    stagnation_threshold: int = 2      # No progress for N iterations
    repeated_error_threshold: int = 2  # Same error type N times
    low_confidence_threshold: float = 0.7
    max_hypotheses_before_explosion: int = 3

    # Security-sensitive file patterns
    security_patterns: List[str] = field(default_factory=lambda: [
        r'auth', r'login', r'password', r'credential', r'secret',
        r'crypto', r'encrypt', r'decrypt', r'hash',
        r'sandbox', r'permission', r'privilege', r'security',
        r'token', r'api.?key', r'oauth',
    ])

    # Uncertainty keywords in model output
    uncertainty_keywords: List[str] = field(default_factory=lambda: [
        "i'm not sure", "i assume", "might work", "not certain",
        "unclear", "uncertain", "possibly", "maybe",
        "i think", "i believe", "don't know", "unsure",
        "could be", "possibly", "perhaps"
    ])


class ConsultationTrigger:
    """
    Determines when the base model should call consult_polydev.

    Implements two-stage gating:
    - Stage A: Deterministic checks (cheap, fast)
    - Stage B: Model-based self-assessment (if Stage A passes)
    """

    def __init__(self, config: Optional[TriggerConfig] = None):
        self.config = config or TriggerConfig()

        # Compiled regex patterns for security detection
        self._security_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.config.security_patterns
        ]

    def should_consult(
        self,
        state: 'AgentState',
        last_response: Optional[str] = None
    ) -> Tuple[bool, Optional[TriggerType]]:
        """
        Determine if consultation should be triggered.

        Returns:
            (should_consult, trigger_type) tuple
        """
        # Stage A: Deterministic checks
        stage_a_result = self._stage_a_deterministic(state)
        if stage_a_result[0]:
            logger.info(f"Trigger Stage A: {stage_a_result[1]}")
            return stage_a_result

        # Stage B: Model-based self-assessment
        stage_b_result = self._stage_b_model_based(state, last_response)
        if stage_b_result[0]:
            logger.info(f"Trigger Stage B: {stage_b_result[1]}")
            return stage_b_result

        return False, None

    def _stage_a_deterministic(
        self,
        state: 'AgentState'
    ) -> Tuple[bool, Optional[TriggerType]]:
        """
        Stage A: Fast, deterministic checks based on observable signals.
        """
        # Check stuck loop (same error repeated)
        if self._is_stuck_loop(state):
            return True, TriggerType.STUCK_LOOP

        # Check stagnation (no test progress)
        if self._is_stagnating(state):
            return True, TriggerType.STAGNATION

        # Check repeated compile/type errors
        if self._has_repeated_error(state):
            return True, TriggerType.REPEATED_COMPILE_ERROR

        # Check security-sensitive changes
        if self._is_security_sensitive(state):
            return True, TriggerType.SECURITY_SENSITIVE

        # Check for contradiction
        if self._has_contradiction(state):
            return True, TriggerType.CONTRADICTION

        return False, None

    def _stage_b_model_based(
        self,
        state: 'AgentState',
        last_response: Optional[str] = None
    ) -> Tuple[bool, Optional[TriggerType]]:
        """
        Stage B: Model-based self-assessment.
        Check if the model's output indicates uncertainty.
        """
        if not last_response:
            return False, None

        # Check for uncertainty keywords
        response_lower = last_response.lower()
        uncertainty_count = sum(
            1 for kw in self.config.uncertainty_keywords
            if kw in response_lower
        )

        if uncertainty_count >= 2:
            return True, TriggerType.LOW_CONFIDENCE

        # Check for search explosion (multiple hypotheses without ranking)
        hypothesis_indicators = [
            "could be", "might be", "another possibility",
            "alternatively", "option 1", "option 2", "option 3"
        ]
        hypothesis_count = sum(
            1 for ind in hypothesis_indicators
            if ind in response_lower
        )

        if hypothesis_count >= self.config.max_hypotheses_before_explosion:
            return True, TriggerType.SEARCH_EXPLOSION

        # Check for API uncertainty
        api_uncertainty_patterns = [
            r"not sure (how|if) .* (api|library|function|method)",
            r"(deprecated|breaking change|new version)",
            r"documentation (says|shows|unclear)",
        ]

        for pattern in api_uncertainty_patterns:
            if re.search(pattern, response_lower):
                return True, TriggerType.API_UNCERTAINTY

        return False, None

    def _is_stuck_loop(self, state: 'AgentState') -> bool:
        """Same error signature after N fix attempts."""
        if len(state.error_history) < self.config.stuck_loop_threshold:
            return False

        # Check if the last N errors are identical
        recent_errors = state.error_history[-self.config.stuck_loop_threshold:]
        return len(set(recent_errors)) == 1

    def _is_stagnating(self, state: 'AgentState') -> bool:
        """No reduction in failing tests for N iterations."""
        if len(state.test_results) < self.config.stagnation_threshold:
            return False

        recent_results = state.test_results[-self.config.stagnation_threshold:]

        # Check if failures haven't decreased
        failures = [r.failures for r in recent_results]
        return all(f >= failures[0] for f in failures)

    def _has_repeated_error(self, state: 'AgentState') -> bool:
        """Same error type N times (e.g., TypeError, SyntaxError)."""
        if len(state.error_history) < self.config.repeated_error_threshold:
            return False

        recent_errors = state.error_history[-self.config.repeated_error_threshold:]
        error_types = [e.error_type for e in recent_errors]

        return len(set(error_types)) == 1

    def _is_security_sensitive(self, state: 'AgentState') -> bool:
        """Check if current work touches security-sensitive files."""
        modified_files = state.get_modified_files()

        for file_path in modified_files:
            for pattern in self._security_patterns:
                if pattern.search(file_path):
                    return True

        return False

    def _has_contradiction(self, state: 'AgentState') -> bool:
        """Check if model's assertions contradict each other."""
        # This is hard to detect deterministically,
        # but we can check for reversing changes
        if len(state.patch_history) < 2:
            return False

        # Check if recent patches are reverting each other
        # (simplified heuristic)
        for i in range(len(state.patch_history) - 1):
            p1 = state.patch_history[i]
            p2 = state.patch_history[i + 1]

            # Check if p2 undoes p1 (simplified check)
            if p1.added_lines and p2.removed_lines:
                overlap = set(p1.added_lines) & set(p2.removed_lines)
                if len(overlap) > len(p1.added_lines) * 0.5:
                    return True

        return False

    def get_consult_type(self, trigger: TriggerType) -> ConsultType:
        """Map trigger type to appropriate consultation type."""
        mapping = {
            TriggerType.STUCK_LOOP: ConsultType.DEBUG,
            TriggerType.STAGNATION: ConsultType.DESIGN_ALTERNATIVE,
            TriggerType.CONTRADICTION: ConsultType.DEBUG,
            TriggerType.LOW_CONFIDENCE: ConsultType.EDGE_CASES,
            TriggerType.SEARCH_EXPLOSION: ConsultType.DESIGN_ALTERNATIVE,
            TriggerType.API_UNCERTAINTY: ConsultType.API_VERIFICATION,
            TriggerType.REPEATED_COMPILE_ERROR: ConsultType.DEBUG,
            TriggerType.SECURITY_SENSITIVE: ConsultType.ARCHITECTURE,
        }
        return mapping.get(trigger, ConsultType.DEBUG)
