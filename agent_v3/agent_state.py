"""
AgentState: State Tracking for Chief Resident Architecture

Tracks all history needed for:
- Trigger detection (error history, test results)
- Consultation context (what was tried, current hypothesis)
- Metrics collection (Unstuck Rate, consultation effectiveness)
"""

import time
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class ErrorSignature:
    """Normalized error signature for comparison."""
    error_type: str
    error_message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    raw_output: str = ""

    def __hash__(self):
        # Hash based on error type and first 100 chars of message
        return hash((self.error_type, self.error_message[:100] if self.error_message else ""))

    def __eq__(self, other):
        if not isinstance(other, ErrorSignature):
            return False
        return (self.error_type == other.error_type and
                (self.error_message[:100] if self.error_message else "") ==
                (other.error_message[:100] if other.error_message else ""))

    @classmethod
    def from_output(cls, output: str, returncode: int) -> 'ErrorSignature':
        """Parse error signature from command output."""
        # Default values
        error_type = "UnknownError"
        error_message = output[:500] if output else ""
        file_path = None
        line_number = None

        if returncode != 0:
            # Try to extract Python exception type
            exception_match = re.search(
                r'^(\w*Error|\w*Exception): (.+)$',
                output, re.MULTILINE
            )
            if exception_match:
                error_type = exception_match.group(1)
                error_message = exception_match.group(2)

            # Try to extract file and line
            file_line_match = re.search(
                r'File "([^"]+)", line (\d+)',
                output
            )
            if file_line_match:
                file_path = file_line_match.group(1)
                line_number = int(file_line_match.group(2))

            # Check for common error patterns
            if "SyntaxError" in output:
                error_type = "SyntaxError"
            elif "TypeError" in output:
                error_type = "TypeError"
            elif "ImportError" in output or "ModuleNotFoundError" in output:
                error_type = "ImportError"
            elif "AttributeError" in output:
                error_type = "AttributeError"
            elif "NameError" in output:
                error_type = "NameError"
            elif "FAILED" in output:
                error_type = "TestFailure"

        return cls(
            error_type=error_type,
            error_message=error_message,
            file_path=file_path,
            line_number=line_number,
            raw_output=output[:2000] if output else ""
        )


@dataclass
class TestResult:
    """Result from running tests."""
    passed: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_message: str = ""
    test_names: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def failures(self) -> int:
        return self.failed_tests

    @classmethod
    def from_pytest_output(cls, output: str, returncode: int) -> 'TestResult':
        """Parse test result from pytest output."""
        passed = returncode == 0
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        failed_names = []

        # Parse pytest summary line: "X passed, Y failed, Z error"
        summary_match = re.search(
            r'(\d+) passed',
            output
        )
        if summary_match:
            passed_tests = int(summary_match.group(1))

        failed_match = re.search(
            r'(\d+) failed',
            output
        )
        if failed_match:
            failed_tests = int(failed_match.group(1))

        error_match = re.search(
            r'(\d+) error',
            output
        )
        if error_match:
            failed_tests += int(error_match.group(1))

        total_tests = passed_tests + failed_tests

        # Extract failed test names
        failed_pattern = re.findall(
            r'FAILED (.+)::\w+',
            output
        )
        failed_names.extend(failed_pattern)

        return cls(
            passed=passed,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_message=output[:1000] if not passed else "",
            test_names=failed_names
        )


@dataclass
class PatchInfo:
    """Information about a generated patch."""
    diff: str
    files_modified: List[str] = field(default_factory=list)
    added_lines: List[str] = field(default_factory=list)
    removed_lines: List[str] = field(default_factory=list)
    rationale: str = ""
    iteration: int = 0
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_diff(cls, diff: str, rationale: str = "", iteration: int = 0) -> 'PatchInfo':
        """Parse patch info from git diff output."""
        files = []
        added = []
        removed = []

        # Parse file names
        file_pattern = re.findall(r'^diff --git a/(.+) b/', diff, re.MULTILINE)
        files.extend(file_pattern)

        # Parse added/removed lines
        for line in diff.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                added.append(line[1:].strip())
            elif line.startswith('-') and not line.startswith('---'):
                removed.append(line[1:].strip())

        return cls(
            diff=diff,
            files_modified=files,
            added_lines=added,
            removed_lines=removed,
            rationale=rationale,
            iteration=iteration
        )


@dataclass
class ConsultationRecord:
    """Record of a Polydev consultation."""
    trigger_type: str
    consult_type: str
    context_summary: str
    current_hypothesis: str
    specific_questions: List[str]
    response: Dict[str, str]  # provider -> response
    iteration: int
    timestamp: float = field(default_factory=time.time)
    helped: Optional[bool] = None  # Set after seeing if it helped

    @property
    def unstuck_after(self) -> bool:
        """Did we solve the task after this consultation?"""
        return self.helped is True


class AgentState:
    """
    Complete state of the agent during task solving.

    Tracks everything needed for:
    - Trigger detection
    - Consultation context
    - Metrics collection
    """

    def __init__(self, task: Dict[str, Any]):
        self.task = task
        self.instance_id = task.get("instance_id", "unknown")
        self.problem_statement = task.get("problem_statement", "")
        self.start_time = time.time()

        # History tracking
        self.error_history: List[ErrorSignature] = []
        self.test_results: List[TestResult] = []
        self.patch_history: List[PatchInfo] = []
        self.consultation_history: List[ConsultationRecord] = []

        # Current iteration info
        self.current_iteration = 0
        self.current_hypothesis = ""
        self.steps_since_last_consult = 0

        # Modified files tracking
        self._modified_files: Set[str] = set()

        # Metrics
        self.total_steps = 0
        self.total_consultations = 0
        self.solved = False
        self.solve_iteration: Optional[int] = None

    def add_error(self, output: str, returncode: int):
        """Record an error from command execution."""
        error = ErrorSignature.from_output(output, returncode)
        self.error_history.append(error)

    def add_test_result(self, result: TestResult):
        """Record a test execution result."""
        self.test_results.append(result)

    def add_patch(self, diff: str, rationale: str = ""):
        """Record a generated patch."""
        patch = PatchInfo.from_diff(
            diff=diff,
            rationale=rationale,
            iteration=self.current_iteration
        )
        self.patch_history.append(patch)
        self._modified_files.update(patch.files_modified)

    def add_consultation(
        self,
        trigger_type: str,
        consult_type: str,
        context_summary: str,
        current_hypothesis: str,
        specific_questions: List[str],
        response: Dict[str, str]
    ):
        """Record a Polydev consultation."""
        record = ConsultationRecord(
            trigger_type=trigger_type,
            consult_type=consult_type,
            context_summary=context_summary,
            current_hypothesis=current_hypothesis,
            specific_questions=specific_questions,
            response=response,
            iteration=self.current_iteration
        )
        self.consultation_history.append(record)
        self.total_consultations += 1
        self.steps_since_last_consult = 0

    def add_perspectives(self, perspectives: Dict[str, str]):
        """Convenience method to update last consultation with response."""
        if self.consultation_history:
            self.consultation_history[-1].response = perspectives

    def mark_solved(self):
        """Mark the task as solved."""
        self.solved = True
        self.solve_iteration = self.current_iteration

        # Mark if any consultation helped
        if self.consultation_history:
            self.consultation_history[-1].helped = True

    def get_modified_files(self) -> Set[str]:
        """Get all files modified during this session."""
        return self._modified_files

    def get_context(self) -> str:
        """Build context summary for consultation."""
        context_parts = [
            f"Task: {self.instance_id}",
            f"Problem: {self.problem_statement[:500]}",
            f"Iteration: {self.current_iteration}",
            f"Steps taken: {self.total_steps}",
        ]

        if self.patch_history:
            context_parts.append(
                f"Files modified: {', '.join(self._modified_files)}"
            )

        if self.error_history:
            last_error = self.error_history[-1]
            context_parts.append(
                f"Last error: {last_error.error_type}: {last_error.error_message[:200]}"
            )

        if self.test_results:
            last_test = self.test_results[-1]
            context_parts.append(
                f"Test status: {last_test.passed_tests}/{last_test.total_tests} passed"
            )

        return "\n".join(context_parts)

    def get_attempt_history(self) -> str:
        """Summarize what has been tried so far."""
        if not self.patch_history:
            return "No patches attempted yet"

        attempts = []
        for i, patch in enumerate(self.patch_history[-5:]):  # Last 5 patches
            attempts.append(
                f"Attempt {i+1}: Modified {', '.join(patch.files_modified[:3])} - {patch.rationale[:100]}"
            )

        return "\n".join(attempts)

    @property
    def best_patch(self) -> str:
        """Get the best patch (most test passes or last attempt)."""
        if not self.patch_history:
            return ""

        # If we have test results, find patch with most passes
        if self.test_results:
            # Zip patches with their corresponding test results
            paired = list(zip(self.patch_history, self.test_results))
            best = max(paired, key=lambda x: x[1].passed_tests)
            return best[0].diff

        return self.patch_history[-1].diff

    def increment_iteration(self):
        """Move to next iteration."""
        self.current_iteration += 1
        self.total_steps += 1
        self.steps_since_last_consult += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for analysis."""
        consultations_that_helped = sum(
            1 for c in self.consultation_history if c.helped
        )

        return {
            "instance_id": self.instance_id,
            "solved": self.solved,
            "solve_iteration": self.solve_iteration,
            "total_iterations": self.current_iteration,
            "total_steps": self.total_steps,
            "total_errors": len(self.error_history),
            "total_test_runs": len(self.test_results),
            "total_patches": len(self.patch_history),
            "total_consultations": self.total_consultations,
            "consultations_that_helped": consultations_that_helped,
            "unstuck_rate": consultations_that_helped / self.total_consultations if self.total_consultations > 0 else 0,
            "files_modified": list(self._modified_files),
            "duration_seconds": time.time() - self.start_time,
            "consultation_breakdown": {
                c.trigger_type: c.helped for c in self.consultation_history
            }
        }
