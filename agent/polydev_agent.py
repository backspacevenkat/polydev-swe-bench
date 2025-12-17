"""
Polydev SWE-bench Agent

Main agent class that orchestrates the evaluation pipeline:
1. Task ingestion
2. Code analysis
3. Solution generation with confidence
4. Consultation (if needed)
5. Synthesis
6. Patch generation
"""

import os
import json
import time
import re
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .confidence import ConfidenceAssessor
from .consultation import PolydevConsultation
from .patch_generator import PatchGenerator

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage tracking per provider."""
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens


@dataclass
class TaskContext:
    """Context for a SWE-bench task."""
    instance_id: str
    repository: str
    base_commit: str
    problem_statement: str
    hints_text: str = ""
    relevant_files: List[str] = field(default_factory=list)
    file_contents: Dict[str, str] = field(default_factory=dict)


@dataclass
class Analysis:
    """Result of code analysis."""
    root_cause: str
    affected_files: List[str]
    relevant_code: Dict[str, str]
    expected_behavior: str
    actual_behavior: str
    edge_cases: List[str] = field(default_factory=list)


@dataclass
class Solution:
    """Proposed solution with confidence."""
    approach: str
    confidence: int  # 1-10
    confidence_reasoning: str
    uncertainties: List[str]
    proposed_changes: List[Dict[str, Any]]


@dataclass
class ConsultationResult:
    """Result of multi-model consultation."""
    triggered: bool
    trigger_reason: str = ""
    gpt_response: str = ""
    gemini_response: str = ""
    synthesis: str = ""
    approach_changed: bool = False
    final_approach: str = ""
    final_confidence: int = 0
    duration_ms: int = 0
    cost_usd: float = 0.0
    # Token tracking
    gpt_input_tokens: int = 0
    gpt_output_tokens: int = 0
    gemini_input_tokens: int = 0
    gemini_output_tokens: int = 0


@dataclass
class TaskResult:
    """Complete result for a task."""
    instance_id: str
    configuration: str  # "baseline" or "polydev"

    # Phase timings
    analysis_duration_ms: int = 0
    solution_duration_ms: int = 0
    consultation_duration_ms: int = 0
    patch_duration_ms: int = 0
    total_duration_ms: int = 0

    # Results
    initial_confidence: int = 0
    consultation_triggered: bool = False
    final_confidence: int = 0
    patch_generated: bool = False
    patch: str = ""

    # Detailed logs
    analysis: Optional[Analysis] = None
    solution: Optional[Solution] = None
    consultation: Optional[ConsultationResult] = None

    # Evaluation (filled in by harness)
    tests_passed: Optional[bool] = None
    error: Optional[str] = None

    # Cost
    cost_usd: float = 0.0
    
    # Token tracking per provider
    token_usage: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "claude": {"input": 0, "output": 0},
        "gpt": {"input": 0, "output": 0},
        "gemini": {"input": 0, "output": 0}
    })


class PolydevAgent:
    """
    Lightweight agent for SWE-bench evaluation with Polydev MCP consultation.

    Usage:
        agent = PolydevAgent(consultation_enabled=True, threshold=8)
        result = agent.solve_task(task)
    """

    def __init__(
        self,
        consultation_enabled: bool = True,
        confidence_threshold: int = 8,
        max_retries: int = 3,
        timeout_seconds: int = 600,  # 10 minutes for complex SWE-bench tasks
        log_dir: Optional[Path] = None,
        mock_mode: bool = False  # For testing without actual API calls
    ):
        """
        Initialize the agent.

        Args:
            consultation_enabled: Whether to use Polydev MCP consultation
            confidence_threshold: Consult if confidence < threshold
            max_retries: Number of retries on model failures
            timeout_seconds: Timeout for model calls (default: 600s for complex tasks)
            log_dir: Directory for detailed logs
            mock_mode: If True, return simulated responses (for testing)
        """
        self.consultation_enabled = consultation_enabled
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.log_dir = log_dir
        self.mock_mode = mock_mode

        # Components
        self.confidence_assessor = ConfidenceAssessor()
        self.consultation = PolydevConsultation()
        self.patch_generator = PatchGenerator()

        # Load prompts
        self.prompts = self._load_prompts()
        
        # Token tracking for current task
        self._current_token_usage = {
            "claude": {"input": 0, "output": 0},
            "gpt": {"input": 0, "output": 0},
            "gemini": {"input": 0, "output": 0}
        }

        logger.info(
            f"PolydevAgent initialized: consultation={consultation_enabled}, "
            f"threshold={confidence_threshold}"
        )

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from files."""
        prompts = {}
        prompt_dir = Path(__file__).parent / "prompts"

        prompt_files = [
            "analysis.txt",
            "code_reading.txt",
            "solution.txt",
            "confidence.txt",
            "consultation.txt",
            "synthesis.txt",
            "patch.txt"
        ]

        for filename in prompt_files:
            filepath = prompt_dir / filename
            if filepath.exists():
                prompts[filename.replace(".txt", "")] = filepath.read_text()
            else:
                logger.warning(f"Prompt file not found: {filepath}")
                # Provide default empty template
                prompts[filename.replace(".txt", "")] = ""

        return prompts
    
    def _reset_token_tracking(self):
        """Reset token tracking for a new task."""
        self._current_token_usage = {
            "claude": {"input": 0, "output": 0},
            "gpt": {"input": 0, "output": 0},
            "gemini": {"input": 0, "output": 0}
        }

    def solve_task(self, task: Dict[str, Any]) -> TaskResult:
        """
        Solve a SWE-bench task.

        Args:
            task: SWE-bench task dictionary with keys:
                - instance_id
                - repo
                - base_commit
                - problem_statement
                - hints_text (optional)

        Returns:
            TaskResult with patch and metadata
        """
        start_time = time.time()
        self._reset_token_tracking()

        # Initialize result
        result = TaskResult(
            instance_id=task["instance_id"],
            configuration="polydev" if self.consultation_enabled else "baseline"
        )

        try:
            # Create task context
            context = self._create_context(task)

            # Phase 1: Analyze
            logger.info(f"[{context.instance_id}] Starting analysis")
            analysis_start = time.time()
            analysis = self._analyze(context)
            result.analysis = analysis
            result.analysis_duration_ms = int((time.time() - analysis_start) * 1000)

            # Phase 2: Generate solution with confidence
            logger.info(f"[{context.instance_id}] Generating solution")
            solution_start = time.time()
            solution = self._generate_solution(context, analysis)
            result.solution = solution
            result.initial_confidence = solution.confidence
            result.solution_duration_ms = int((time.time() - solution_start) * 1000)

            logger.info(
                f"[{context.instance_id}] Solution generated, "
                f"confidence: {solution.confidence}/10"
            )

            # Phase 3: Consultation (if needed and enabled)
            consultation_result = ConsultationResult(triggered=False)

            if (self.consultation_enabled and
                solution.confidence < self.confidence_threshold):

                logger.info(
                    f"[{context.instance_id}] Confidence {solution.confidence} < "
                    f"{self.confidence_threshold}, triggering consultation"
                )

                consultation_start = time.time()
                consultation_result = self._consult(context, analysis, solution)
                result.consultation_duration_ms = int(
                    (time.time() - consultation_start) * 1000
                )

                # Update solution if consultation changed approach
                if consultation_result.approach_changed:
                    solution.approach = consultation_result.final_approach
                    solution.confidence = consultation_result.final_confidence
                
                # Update token tracking from consultation
                self._current_token_usage["gpt"]["input"] += consultation_result.gpt_input_tokens
                self._current_token_usage["gpt"]["output"] += consultation_result.gpt_output_tokens
                self._current_token_usage["gemini"]["input"] += consultation_result.gemini_input_tokens
                self._current_token_usage["gemini"]["output"] += consultation_result.gemini_output_tokens

            result.consultation = consultation_result
            result.consultation_triggered = consultation_result.triggered
            result.final_confidence = solution.confidence
            result.cost_usd = consultation_result.cost_usd

            # Phase 4: Generate patch
            logger.info(f"[{context.instance_id}] Generating patch")
            patch_start = time.time()
            patch = self._generate_patch(context, analysis, solution)
            result.patch = patch
            result.patch_generated = bool(patch)
            result.patch_duration_ms = int((time.time() - patch_start) * 1000)

            result.total_duration_ms = int((time.time() - start_time) * 1000)
            
            # Copy token usage to result
            result.token_usage = self._current_token_usage.copy()

            logger.info(
                f"[{context.instance_id}] Complete: "
                f"{result.total_duration_ms}ms, "
                f"consulted={result.consultation_triggered}, "
                f"tokens={{claude: {self._current_token_usage['claude']}, "
                f"gpt: {self._current_token_usage['gpt']}, "
                f"gemini: {self._current_token_usage['gemini']}}}"
            )

        except Exception as e:
            logger.error(f"[{task['instance_id']}] Error: {e}")
            result.error = str(e)
            result.total_duration_ms = int((time.time() - start_time) * 1000)
            result.token_usage = self._current_token_usage.copy()

        # Save detailed log if log_dir specified
        if self.log_dir:
            self._save_task_log(result)

        return result

    def _create_context(self, task: Dict[str, Any]) -> TaskContext:
        """Create TaskContext from raw task dictionary."""
        return TaskContext(
            instance_id=task["instance_id"],
            repository=task["repo"],
            base_commit=task.get("base_commit", ""),
            problem_statement=task["problem_statement"],
            hints_text=task.get("hints_text", "")
        )

    def _analyze(self, context: TaskContext) -> Analysis:
        """Analyze the issue and codebase."""
        # Call Claude via CLI to analyze
        prompt = self.prompts["analysis"].format(
            repo=context.repository,
            instance_id=context.instance_id,
            problem_statement=context.problem_statement,
            repo_context=context.hints_text or "No additional context provided"
        )

        response = self._call_claude(prompt)

        # Parse analysis from response
        analysis = self._parse_analysis(response)
        return analysis

    def _generate_solution(
        self,
        context: TaskContext,
        analysis: Analysis
    ) -> Solution:
        """Generate solution with confidence assessment."""
        # Generate solution
        solution_prompt = self.prompts["solution"].format(
            repo=context.repository,
            instance_id=context.instance_id,
            problem_statement=context.problem_statement,
            analysis=analysis.root_cause,
            relevant_code=json.dumps(analysis.relevant_code, indent=2)
        )

        solution_response = self._call_claude(solution_prompt)

        # Assess confidence
        confidence_prompt = self.prompts["confidence"].format(
            proposed_solution=solution_response
        )

        confidence_response = self._call_claude(confidence_prompt)
        confidence_score, reasoning, uncertainties = self.confidence_assessor.assess(
            confidence_response
        )

        return Solution(
            approach=solution_response,
            confidence=confidence_score,
            confidence_reasoning=reasoning,
            uncertainties=uncertainties,
            proposed_changes=[]  # Parsed from solution_response
        )

    def _consult(
        self,
        context: TaskContext,
        analysis: Analysis,
        solution: Solution
    ) -> ConsultationResult:
        """Consult other models via Polydev MCP."""
        # Build consultation request
        consultation_prompt = self.prompts["consultation"].format(
            repo=context.repository,
            instance_id=context.instance_id,
            problem_statement=context.problem_statement,
            analysis=analysis.root_cause,
            proposed_solution=solution.approach,
            confidence=solution.confidence
        )

        # Call Polydev MCP
        start_time = time.time()
        perspectives = self.consultation.get_perspectives(consultation_prompt)
        consultation_time = int((time.time() - start_time) * 1000)

        # Synthesize responses
        synthesis_prompt = self.prompts["synthesis"].format(
            repo=context.repository,
            instance_id=context.instance_id,
            problem_statement=context.problem_statement,
            analysis=analysis.root_cause,
            original_solution=solution.approach,
            gpt_perspective=perspectives.get("gpt-5.2", "No response"),
            gemini_perspective=perspectives.get("gemini-3-pro", "No response")
        )

        synthesis = self._call_claude(synthesis_prompt)

        # Parse synthesis result
        final_approach, final_confidence, approach_changed = self._parse_synthesis(
            synthesis, solution.approach
        )

        return ConsultationResult(
            triggered=True,
            trigger_reason=f"confidence ({solution.confidence}) < threshold ({self.confidence_threshold})",
            gpt_response=perspectives.get("gpt-5.2", ""),
            gemini_response=perspectives.get("gemini-3-pro", ""),
            synthesis=synthesis,
            approach_changed=approach_changed,
            final_approach=final_approach,
            final_confidence=final_confidence,
            duration_ms=consultation_time,
            cost_usd=perspectives.get("cost_usd", 0.0),
            # Token tracking from consultation
            gpt_input_tokens=perspectives.get("gpt_input_tokens", 0),
            gpt_output_tokens=perspectives.get("gpt_output_tokens", 0),
            gemini_input_tokens=perspectives.get("gemini_input_tokens", 0),
            gemini_output_tokens=perspectives.get("gemini_output_tokens", 0)
        )

    def _generate_patch(
        self,
        context: TaskContext,
        analysis: Analysis,
        solution: Solution
    ) -> str:
        """Generate unified diff patch."""
        prompt = self.prompts["patch"].format(
            final_approach=solution.approach,
            files_to_modify=", ".join(analysis.affected_files),
            current_contents=json.dumps(analysis.relevant_code, indent=2)
        )

        response = self._call_claude(prompt)

        # Extract patch from response
        patch = self.patch_generator.extract_patch(response)
        return patch

    def _call_claude(self, prompt: str) -> str:
        """Call Claude via Claude Code CLI with token tracking."""
        # Mock mode for testing
        if self.mock_mode:
            return self._generate_mock_response(prompt)

        for attempt in range(self.max_retries):
            try:
                # Use JSON output to get token usage
                result = subprocess.run(
                    ["claude", "-p", prompt, "--output-format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )

                if result.returncode == 0:
                    # Parse JSON response for token usage
                    try:
                        response_data = json.loads(result.stdout)
                        
                        # Extract token usage from response
                        if isinstance(response_data, dict):
                            usage = response_data.get("usage", {})
                            input_tokens = usage.get("input_tokens", 0)
                            output_tokens = usage.get("output_tokens", 0)
                            
                            # Update tracking
                            self._current_token_usage["claude"]["input"] += input_tokens
                            self._current_token_usage["claude"]["output"] += output_tokens
                            
                            # Get the actual response text
                            if "result" in response_data:
                                return response_data["result"]
                            elif "content" in response_data:
                                return response_data["content"]
                            elif "text" in response_data:
                                return response_data["text"]
                            else:
                                # Fallback to full response
                                return json.dumps(response_data)
                        else:
                            return str(response_data)
                            
                    except json.JSONDecodeError:
                        # Fallback to text parsing if JSON fails
                        # Estimate tokens from response length
                        input_tokens = len(prompt) // 4  # Rough estimate
                        output_tokens = len(result.stdout) // 4
                        self._current_token_usage["claude"]["input"] += input_tokens
                        self._current_token_usage["claude"]["output"] += output_tokens
                        return result.stdout.strip()
                else:
                    logger.warning(
                        f"Claude CLI error (attempt {attempt + 1}): {result.stderr}"
                    )

            except subprocess.TimeoutExpired:
                logger.warning(f"Claude CLI timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Claude CLI exception (attempt {attempt + 1}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError(f"Claude CLI failed after {self.max_retries} attempts")

    def _generate_mock_response(self, prompt: str) -> str:
        """Generate a mock response for testing."""
        import random

        # Determine what type of response is needed based on prompt content
        if "analyze" in prompt.lower() or "analysis" in prompt.lower():
            return """
<root_cause>Mock root cause: The issue appears to be in the model handling logic.</root_cause>
<affected_files>
- src/model.py: Main model file
- src/utils.py: Utility functions
</affected_files>
<expected_behavior>The function should return correct results.</expected_behavior>
<actual_behavior>The function returns incorrect results in edge cases.</actual_behavior>
<edge_cases>
- Empty input
- Large numbers
</edge_cases>
"""
        elif "confidence" in prompt.lower():
            score = random.randint(5, 9)  # Random confidence to test consultation
            return f"""
<confidence_score>{score}</confidence_score>
<confidence_reasoning>Mock reasoning: The solution appears reasonable but has some edge cases.</confidence_reasoning>
<uncertainties>
- Edge case handling
- Performance implications
</uncertainties>
"""
        elif "solution" in prompt.lower():
            return """
### Solution Explanation
Mock solution: Fix the bug by updating the logic in the affected function.

### Patch
```diff
diff --git a/src/model.py b/src/model.py
--- a/src/model.py
+++ b/src/model.py
@@ -10,7 +10,7 @@
 def process_data(data):
-    return data
+    return data if data else []
```
"""
        elif "synthesis" in prompt.lower():
            return """
<approach>Mock synthesized approach: After considering multiple perspectives,
the best fix is to add proper validation.</approach>
<new_confidence>8</new_confidence>
"""
        elif "patch" in prompt.lower():
            return """
```diff
diff --git a/src/model.py b/src/model.py
--- a/src/model.py
+++ b/src/model.py
@@ -10,7 +10,7 @@
 def process_data(data):
-    return data
+    return data if data else []
```
"""
        else:
            return "Mock response for prompt."

    def _parse_analysis(self, response: str) -> Analysis:
        """Parse analysis from Claude response."""
        # Simple parsing - in production would use XML parsing
        return Analysis(
            root_cause=self._extract_tag(response, "root_cause") or "Unknown",
            affected_files=self._extract_list(response, "affected_files"),
            relevant_code={},
            expected_behavior=self._extract_tag(response, "expected_behavior") or "",
            actual_behavior=self._extract_tag(response, "actual_behavior") or "",
            edge_cases=self._extract_list(response, "edge_cases")
        )

    def _parse_synthesis(
        self,
        response: str,
        original_approach: str
    ) -> tuple[str, int, bool]:
        """Parse synthesis result."""
        final_approach = self._extract_tag(response, "approach") or original_approach

        confidence_str = self._extract_tag(response, "new_confidence") or "5"
        try:
            final_confidence = int(confidence_str)
        except ValueError:
            final_confidence = 5

        approach_changed = final_approach.strip() != original_approach.strip()

        return final_approach, final_confidence, approach_changed

    def _extract_tag(self, text: str, tag: str) -> Optional[str]:
        """Extract content from XML-like tag."""
        import re
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_list(self, text: str, tag: str) -> List[str]:
        """Extract list items from tag content."""
        content = self._extract_tag(text, tag)
        if not content:
            return []

        items = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                items.append(line[2:].split(":")[0].strip())
        return items

    def _save_task_log(self, result: TaskResult) -> None:
        """Save detailed task log to file."""
        if not self.log_dir:
            return

        log_file = self.log_dir / "tasks" / f"{result.instance_id}.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dict
        log_data = {
            "instance_id": result.instance_id,
            "configuration": result.configuration,
            "timings": {
                "analysis_ms": result.analysis_duration_ms,
                "solution_ms": result.solution_duration_ms,
                "consultation_ms": result.consultation_duration_ms,
                "patch_ms": result.patch_duration_ms,
                "total_ms": result.total_duration_ms
            },
            "confidence": {
                "initial": result.initial_confidence,
                "final": result.final_confidence,
                "consultation_triggered": result.consultation_triggered
            },
            "patch_generated": result.patch_generated,
            "cost_usd": result.cost_usd,
            "error": result.error
        }

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
