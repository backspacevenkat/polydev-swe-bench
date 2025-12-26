"""
Chief Resident Agent: Base Model + Selective Polydev Consultation

Implements the "Chief Resident" architecture where:
- Base model (Claude) is the lead engineer doing the work
- Polydev is the board of specialists consulted only when triggered

Core principle: Base Model + Polydev > Base Model Alone
"""

import re
import time
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .agent_state import AgentState, ErrorSignature, TestResult, PatchInfo
from .consultation_trigger import (
    ConsultationTrigger, TriggerType, ConsultType, TriggerConfig
)
from .environment import LocalEnvironment, EnvironmentConfig
from .model import ClaudeModel, ModelConfig

logger = logging.getLogger(__name__)


class ExperimentConfig(Enum):
    """Experimental configurations from the paper."""
    BASE_ALONE = "A"           # Claude only, no consultation
    BASE_PLUS_POLYDEV_GATED = "B"  # Claude + Polydev when triggered (main hypothesis)
    BASE_PLUS_POLYDEV_ALWAYS = "C"  # Claude + Polydev every iteration
    BASE_PLUS_SELF_REFLECT = "D"   # Claude + same tokens for self-critique


@dataclass
class ChiefResidentConfig:
    """Configuration for Chief Resident Agent."""
    # Iteration limits - INCREASED based on mini-SWE-agent (250 steps, 74% SOTA)
    max_iterations: int = 10  # Increased from 5
    max_steps_per_iteration: int = 50  # Increased from 20 (total: 500 max)

    # Consultation budget
    max_consults: int = 3  # Increased from 2
    consult_cooldown: int = 5  # Increased from 3 - more steps between consults

    # Behavior
    run_tests_before_submit: bool = True
    require_patch_changes: bool = True

    # Experiment mode
    experiment_config: ExperimentConfig = ExperimentConfig.BASE_PLUS_POLYDEV_GATED

    # Trigger configuration
    trigger_config: TriggerConfig = field(default_factory=TriggerConfig)


# === Prompt Templates ===
# Improved prompts based on mini-SWE-agent (74% SOTA) patterns

SYSTEM_TEMPLATE = """You are an expert software engineer solving a GitHub issue.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.

<format_example>
THOUGHT: Your reasoning and analysis here. Explain why you want to perform the action.

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.

## Important Rules

1. Every response must contain exactly one bash code block
2. The bash block must contain exactly one command (or commands connected with && or ||)
3. Directory or environment variable changes are not persistent - every action is executed in a new subshell
4. You can prefix any action with `cd /path/to/dir && ...` or write/load environment variables from files
5. Do NOT modify test files - only modify source code files

## Useful Command Examples

### Create a new file:
```bash
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:
```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:
```bash
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'

# View first N lines
head -n 50 filename.py

# View with context around a pattern
grep -n -B5 -A5 "pattern" filename.py
```

## Submission
When you've completed your work and verified your fix works, submit with:
```bash
echo POLYDEV_SUBMIT_PATCH && git add -A && git diff --cached
```

After this command, you cannot continue working on this task."""

TASK_TEMPLATE = """## Problem Statement
{problem_statement}

## Repository Location
{repo_path}

## Recommended Workflow

Follow this workflow step-by-step to solve the issue:

1. **Explore the codebase** - Find and read relevant files to understand the code structure
2. **Create a reproduction script** - Write a minimal script that reproduces the issue
3. **Run the reproduction script** - Verify you can reproduce the bug
4. **Edit the source code** - Make changes to fix the issue
5. **Verify your fix** - Run the reproduction script again to confirm the fix works
6. **Test edge cases** - Make sure your fix handles edge cases correctly
7. **Submit your changes** - Use `echo POLYDEV_SUBMIT_PATCH && git add -A && git diff --cached`

Begin by exploring the repository structure to understand the codebase."""

OBSERVATION_TEMPLATE = """<returncode>{returncode}</returncode>
{output_section}"""

# Template for truncated output (used when output > 10000 chars)
OBSERVATION_TRUNCATED_TEMPLATE = """<warning>
The output of your last command was too long ({total_chars} chars).
Please try a different command that produces less output.
If you're looking at a file, use head, tail, or sed to view a smaller number of lines.
If you're using grep or find, use a more selective pattern.
</warning>
<output_head>
{head}
</output_head>
<elided_chars>{elided} characters elided</elided_chars>
<output_tail>
{tail}
</output_tail>"""

# Template for format errors (when model doesn't provide valid bash command)
FORMAT_ERROR_TEMPLATE = """Please always provide EXACTLY ONE bash code block with your command.
Your last response had {num_actions} bash code blocks instead of 1.

If you want to end the task, please issue the following command:
```bash
echo POLYDEV_SUBMIT_PATCH && git add -A && git diff --cached
```

Otherwise, format your response exactly as follows:

<response_example>
THOUGHT: Here are some thoughts about why you want to perform the action.

```bash
<your_command_here>
```
</response_example>

Note: If you need to run multiple commands, combine them with && or || in a single bash block."""

# === Consultation Templates ===

CONSULT_POLYDEV_TEMPLATE = """## Polydev Consultation Request

### Context Summary
{context_summary}

### Current Hypothesis
{current_hypothesis}

### What I've Tried
{what_you_tried}

### Failure Signals
{failure_signals}

### Specific Questions
{specific_questions}

### Consultation Type
{consult_type}

Please provide expert perspectives on how to solve this issue."""

CONSULTATION_RESULT_TEMPLATE = """## Cross-Provider Consultation Results

I consulted external expert models for a second opinion:

{perspectives}

As the lead engineer, please:
1. Evaluate which advice is valid for this specific task
2. Reject any hallucinations or incorrect suggestions
3. Synthesize the final approach yourself

Continue with your solution."""

SELF_REFLECT_TEMPLATE = """## Self-Reflection

Please critically analyze your current approach:

### What you're trying to do
{current_hypothesis}

### What you've tried
{what_you_tried}

### Current errors
{failure_signals}

Please:
1. Identify any blind spots in your reasoning
2. Consider alternative approaches
3. Identify what you might be missing

Then continue with a refined approach."""


class ChiefResidentAgent:
    """
    Chief Resident Agent implementing selective Polydev consultation.

    The base model (Claude) is the "Chief Resident" who:
    - Does the primary work (analyze, code, test)
    - Decides when to consult specialists (Polydev)
    - Synthesizes final solutions from all inputs
    """

    def __init__(
        self,
        model: Optional[ClaudeModel] = None,
        env: Optional[LocalEnvironment] = None,
        config: Optional[ChiefResidentConfig] = None
    ):
        self.model = model or ClaudeModel()
        self.env = env or LocalEnvironment()
        self.config = config or ChiefResidentConfig()

        # Initialize trigger with config
        self.trigger = ConsultationTrigger(self.config.trigger_config)

        # State tracking
        self.state: Optional[AgentState] = None
        self.messages: List[Dict[str, Any]] = []
        self.consults_used = 0

    def solve(
        self,
        task: Dict[str, Any],
        repo_path: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Solve a SWE-bench task.

        Returns:
            (success, patch, metrics)
        """
        # Initialize state
        self.state = AgentState(task)
        self.messages = []
        self.consults_used = 0

        # Build initial conversation
        self._init_conversation(task, repo_path)

        logger.info(f"Starting task: {task.get('instance_id', 'unknown')}")
        logger.info(f"Config: {self.config.experiment_config.name}")

        # Main loop: SOLVE → TEST → REFLECT
        for iteration in range(self.config.max_iterations):
            self.state.increment_iteration()
            logger.info(f"Iteration {iteration + 1}/{self.config.max_iterations}")

            try:
                # Run one iteration of solve-test-reflect
                result = self._run_iteration(repo_path)

                if result == "SUBMITTED":
                    patch = self._extract_patch(repo_path)
                    if patch and len(patch.strip()) > 10:
                        self.state.mark_solved()
                        return True, patch, self.state.get_metrics()
                    else:
                        # Empty patch, continue
                        self._add_message("user",
                            "Warning: No code changes detected. Please make changes before submitting.")

            except StepLimitReached:
                logger.warning("Step limit reached for iteration")
                # Trigger consultation if we're stuck
                if self._can_consult():
                    self._do_consultation(TriggerType.STAGNATION)

            except Exception as e:
                logger.error(f"Error in iteration: {e}")
                self.state.add_error(str(e), 1)

        # Return best effort
        patch = self.state.best_patch
        return False, patch, self.state.get_metrics()

    def _init_conversation(self, task: Dict[str, Any], repo_path: str):
        """Initialize conversation with system prompt and task."""
        self._add_message("system", SYSTEM_TEMPLATE)
        self._add_message("user", TASK_TEMPLATE.format(
            problem_statement=task.get("problem_statement", ""),
            repo_path=repo_path
        ))

    def _run_iteration(self, repo_path: str) -> str:
        """
        Run one solve-test-reflect iteration.

        Returns:
            "SUBMITTED" if task completed, "CONTINUE" otherwise
        """
        steps_this_iteration = 0

        while steps_this_iteration < self.config.max_steps_per_iteration:
            steps_this_iteration += 1
            self.state.total_steps += 1

            # === SOLVE: Query model ===
            response = self._query_model()
            last_response = response.get("content", "")

            # === TEST: Parse and execute action ===
            action = self._parse_action(last_response)
            if not action:
                self._add_message("user", self._get_format_error_message())
                continue

            output = self._execute_action(action, repo_path)

            # Check for submission
            if self._check_submission(output):
                return "SUBMITTED"

            # Record error if command failed
            if output.get("returncode", 0) != 0:
                self.state.add_error(
                    output.get("output", ""),
                    output.get("returncode", 1)
                )

            # Check for test results
            if "pytest" in action.lower() or "test" in action.lower():
                test_result = TestResult.from_pytest_output(
                    output.get("output", ""),
                    output.get("returncode", 0)
                )
                self.state.add_test_result(test_result)

            # Add observation to conversation
            output_text = output.get("output", "")
            output_section = self._format_output(output_text)
            observation = OBSERVATION_TEMPLATE.format(
                returncode=output.get("returncode", 0),
                output_section=output_section
            )
            self._add_message("user", observation)

            # === REFLECT: Check if consultation needed ===
            self._maybe_consult(last_response)

        raise StepLimitReached()

    def _query_model(self) -> Dict[str, Any]:
        """Query the base model."""
        response = self.model.query(self.messages)
        self._add_message("assistant", response.get("content", ""))
        return response

    def _parse_action(self, response: str) -> Optional[str]:
        """Parse bash command from response."""
        matches = re.findall(r"```bash\s*\n(.*?)\n```", response, re.DOTALL)
        if len(matches) == 1:
            return matches[0].strip()
        
        # Store the number of matches for error feedback
        self._last_action_count = len(matches)
        return None

    def _get_format_error_message(self) -> str:
        """Get format error message based on last parse attempt."""
        num_actions = getattr(self, '_last_action_count', 0)
        return FORMAT_ERROR_TEMPLATE.format(num_actions=num_actions)

    def _execute_action(self, action: str, repo_path: str) -> Dict[str, Any]:
        """Execute bash command in repo."""
        return self.env.execute(action, cwd=repo_path)

    def _check_submission(self, output: Dict[str, Any]) -> bool:
        """Check if agent submitted."""
        text = output.get("output", "")
        markers = ["POLYDEV_SUBMIT_PATCH", "COMPLETE_TASK", "FINAL_OUTPUT"]
        return any(m in text for m in markers)

    def _extract_patch(self, repo_path: str) -> str:
        """Extract git diff patch."""
        result = self.env.execute("git diff", cwd=repo_path)
        patch = result.get("output", "")
        if patch:
            self.state.add_patch(patch, self.state.current_hypothesis)
        return patch

    def _maybe_consult(self, last_response: str):
        """Check triggers and maybe do consultation."""
        if not self._can_consult():
            return

        # Different behavior based on experiment config
        if self.config.experiment_config == ExperimentConfig.BASE_ALONE:
            # No consultation ever
            return

        elif self.config.experiment_config == ExperimentConfig.BASE_PLUS_POLYDEV_ALWAYS:
            # Always consult (every few steps)
            if self.state.steps_since_last_consult >= 3:
                self._do_consultation(TriggerType.LOW_CONFIDENCE)

        elif self.config.experiment_config == ExperimentConfig.BASE_PLUS_SELF_REFLECT:
            # Self-reflect instead of consulting
            should_consult, trigger = self.trigger.should_consult(
                self.state, last_response
            )
            if should_consult:
                self._do_self_reflect()

        else:  # BASE_PLUS_POLYDEV_GATED (main hypothesis)
            # Only consult when triggered
            should_consult, trigger = self.trigger.should_consult(
                self.state, last_response
            )
            if should_consult:
                self._do_consultation(trigger)

    def _can_consult(self) -> bool:
        """Check if we can do another consultation."""
        if self.consults_used >= self.config.max_consults:
            return False
        if self.state.steps_since_last_consult < self.config.consult_cooldown:
            return False
        return True

    def _do_consultation(self, trigger: TriggerType):
        """Perform Polydev consultation."""
        logger.info(f"Triggering Polydev consultation: {trigger.name}")
        self.consults_used += 1

        # Build consultation request
        consult_type = self.trigger.get_consult_type(trigger)
        context_summary = self.state.get_context()
        current_hypothesis = self.state.current_hypothesis or "Exploring the codebase"
        what_you_tried = self.state.get_attempt_history()
        failure_signals = ""
        if self.state.error_history:
            last_error = self.state.error_history[-1]
            failure_signals = f"{last_error.error_type}: {last_error.error_message[:300]}"

        specific_questions = self._generate_questions(trigger, consult_type)

        # Build the consultation prompt
        consult_prompt = CONSULT_POLYDEV_TEMPLATE.format(
            context_summary=context_summary,
            current_hypothesis=current_hypothesis,
            what_you_tried=what_you_tried,
            failure_signals=failure_signals,
            specific_questions="\n".join(f"- {q}" for q in specific_questions),
            consult_type=consult_type.value
        )

        # Call Polydev MCP
        perspectives = self.model.consult_polydev(consult_prompt)

        # Record consultation
        self.state.add_consultation(
            trigger_type=trigger.name,
            consult_type=consult_type.value,
            context_summary=context_summary,
            current_hypothesis=current_hypothesis,
            specific_questions=specific_questions,
            response=perspectives
        )

        # Format and add to conversation
        formatted_perspectives = self._format_perspectives(perspectives)
        self._add_message("user", CONSULTATION_RESULT_TEMPLATE.format(
            perspectives=formatted_perspectives
        ))

    def _do_self_reflect(self):
        """Self-reflect instead of consulting (control condition)."""
        logger.info("Triggering self-reflection (control)")

        current_hypothesis = self.state.current_hypothesis or "Exploring the codebase"
        what_you_tried = self.state.get_attempt_history()
        failure_signals = ""
        if self.state.error_history:
            last_error = self.state.error_history[-1]
            failure_signals = f"{last_error.error_type}: {last_error.error_message[:300]}"

        self._add_message("user", SELF_REFLECT_TEMPLATE.format(
            current_hypothesis=current_hypothesis,
            what_you_tried=what_you_tried,
            failure_signals=failure_signals
        ))

    def _generate_questions(
        self,
        trigger: TriggerType,
        consult_type: ConsultType
    ) -> List[str]:
        """Generate specific questions based on trigger type."""
        base_questions = {
            TriggerType.STUCK_LOOP: [
                "Why am I getting the same error repeatedly?",
                "What am I missing about this error?",
                "What alternative approach should I try?"
            ],
            TriggerType.STAGNATION: [
                "Why aren't the tests passing?",
                "What edge cases am I missing?",
                "Is my understanding of the problem correct?"
            ],
            TriggerType.LOW_CONFIDENCE: [
                "Is my current approach correct?",
                "What blind spots might I have?",
                "Are there better alternatives?"
            ],
            TriggerType.API_UNCERTAINTY: [
                "What is the correct API usage here?",
                "Has this API changed in recent versions?",
                "What are common pitfalls with this library?"
            ],
            TriggerType.SECURITY_SENSITIVE: [
                "Are there security implications I'm missing?",
                "Is this the safest way to implement this?",
                "What vulnerabilities should I watch for?"
            ],
        }

        return base_questions.get(trigger, [
            "What am I missing?",
            "What alternative approaches should I consider?",
            "Is my understanding correct?"
        ])

    def _format_perspectives(self, perspectives: Dict[str, str]) -> str:
        """Format Polydev perspectives for the conversation."""
        if not perspectives:
            return "No perspectives received."

        parts = []
        for provider, response in perspectives.items():
            if provider != "error" and response:
                parts.append(f"### {provider.upper()}\n{response[:2000]}")

        if "error" in perspectives:
            parts.append(f"\n(Note: {perspectives['error']})")

        return "\n\n".join(parts) if parts else "No perspectives available."

    def _add_message(self, role: str, content: str):
        """Add message to conversation."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

    def _truncate(self, text: str, max_len: int = 10000) -> str:
        """Truncate long text."""
        if len(text) <= max_len:
            return text
        return text[:4000] + f"\n\n... [{len(text) - 8000} chars elided] ...\n\n" + text[-4000:]

    def _format_output(self, output: str, max_len: int = 10000) -> str:
        """Format command output, truncating if too long (mini-SWE-agent style)."""
        if len(output) <= max_len:
            return f"<output>\n{output}</output>"
        
        # Use truncated template for long output
        head = output[:5000]
        tail = output[-5000:]
        elided = len(output) - 10000
        
        return OBSERVATION_TRUNCATED_TEMPLATE.format(
            total_chars=len(output),
            head=head,
            elided=elided,
            tail=tail
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "config": self.config.experiment_config.name,
            "consults_used": self.consults_used,
            "model_stats": self.model.get_stats(),
            **(self.state.get_metrics() if self.state else {})
        }


class StepLimitReached(Exception):
    """Step limit reached for current iteration."""
    pass


# === Convenience functions for running experiments ===

def run_experiment(
    task: Dict[str, Any],
    repo_path: str,
    config: ExperimentConfig
) -> Tuple[bool, str, Dict[str, Any]]:
    """Run a single experiment configuration on a task."""
    agent_config = ChiefResidentConfig(experiment_config=config)
    agent = ChiefResidentAgent(config=agent_config)
    return agent.solve(task, repo_path)


def run_all_configs(
    task: Dict[str, Any],
    repo_path: str
) -> Dict[str, Tuple[bool, str, Dict[str, Any]]]:
    """Run all experiment configurations on a task."""
    results = {}
    for config in ExperimentConfig:
        logger.info(f"Running config: {config.name}")
        results[config.name] = run_experiment(task, repo_path, config)
    return results
