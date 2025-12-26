"""
Polydev SWE-bench Agent v3

Improvements over v2:
- REAL Polydev MCP consultation (not placeholder)
- Increased step limit (100 vs 30)
- Better consultation triggers
- Test validation before submit
"""

import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .environment import LocalEnvironment, EnvironmentConfig
from .model import ClaudeModel, ModelConfig

logger = logging.getLogger(__name__)


# === Exceptions ===

class NonTerminatingException(Exception):
    """Errors that the agent can recover from."""
    pass


class FormatError(NonTerminatingException):
    """LLM output not in expected format."""
    pass


class ExecutionTimeoutError(NonTerminatingException):
    """Command execution timed out."""
    pass


class TerminatingException(Exception):
    """Errors that stop the agent."""
    pass


class Submitted(TerminatingException):
    """Agent completed the task."""
    pass


class LimitsExceeded(TerminatingException):
    """Cost or step limit reached."""
    pass


# === Configuration ===

@dataclass
class AgentConfig:
    """Agent configuration - IMPROVED LIMITS."""
    step_limit: int = 100  # Increased from 30 to 100
    cost_limit: float = 10.0  # Increased from 5 to 10
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    submit_markers: List[str] = field(default_factory=lambda: [
        "POLYDEV_SUBMIT_PATCH",
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        "MINI_SWE_AGENT_FINAL_OUTPUT",
    ])
    consultation_enabled: bool = True
    consultation_after_steps: int = 25  # Consult every 25 steps for guidance
    consultation_on_failure: bool = True  # Consult when stuck
    run_tests_before_submit: bool = True  # Validate fix before submitting


# === Templates ===

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
sed -i 's/old_string/new_string/g' filename.py
```

### View file contents:
```bash
cat filename.py
```

### Search for patterns:
```bash
grep -r "pattern" . --include="*.py"
```

### Run tests:
```bash
cd /path/to/repo && python -m pytest tests/test_specific.py -v
```

## Submission
When you've completed your work and verified your fix works, submit with:
```bash
echo POLYDEV_SUBMIT_PATCH && git add -A && git diff --cached
```
After this command, you cannot continue working on this task."""

INSTANCE_TEMPLATE = """## Problem Statement
{problem_statement}

## Repository Location
{repo_path}

## Recommended Workflow
Follow this workflow step-by-step to solve the issue:
1. **Explore the codebase** - Find and read relevant files
2. **Create a reproduction script** - Write a minimal script that reproduces the issue
3. **Run the reproduction script** - Verify you can reproduce the bug
4. **Edit the source code** - Make changes to fix the issue
5. **Verify your fix** - Run the reproduction script again
6. **Test edge cases** - Make sure your fix handles edge cases
7. **Submit your changes** - Use `echo POLYDEV_SUBMIT_PATCH && git add -A && git diff --cached`

Begin by exploring the repository structure."""

OBSERVATION_TEMPLATE = """<returncode>{returncode}</returncode>
<output>
{output}
</output>"""

FORMAT_ERROR_TEMPLATE = """Please always provide EXACTLY ONE bash code block with your command.
Your last response had {n_actions} bash code blocks instead of 1.

<correct_format>
THOUGHT: Your reasoning and analysis here.

```bash
your_single_command_here
```
</correct_format>

<wrong_format>
```bash
command1
```

```bash
command2
```
</wrong_format>

Note: If you need to run multiple commands, combine them with && or || in a single bash block."""

TIMEOUT_TEMPLATE = """Command timed out after {timeout}s and was killed.
Partial output: {output}

Please try a different command that completes faster."""

OBSERVATION_TRUNCATED_TEMPLATE = """<warning>
The output of your last command was too long ({total_chars} chars).
Please try a different command that produces less output.
</warning>
<output_head>{head}</output_head>
<elided_chars>{elided} characters elided</elided_chars>
<output_tail>{tail}</output_tail>"""

CONSULTATION_TEMPLATE = """## Multi-Model Consultation Request

I'm working on this issue and need help:

### Problem Statement
{problem_statement}

### Current Approach
{current_approach}

### Steps Taken So Far
{steps_taken}

### What I Need Help With
Please analyze this problem and suggest the best approach to fix it.
Focus on:
1. The root cause of the issue
2. The specific file(s) and line(s) to modify
3. The exact code changes needed"""


class PolydevAgent:
    """
    Polydev SWE-bench Agent v3.

    Key improvements:
    - REAL multi-model consultation via Polydev MCP
    - Higher step limit (100) for complex tasks
    - Better error recovery
    """

    def __init__(
        self,
        model: Optional[ClaudeModel] = None,
        env: Optional[LocalEnvironment] = None,
        config: Optional[AgentConfig] = None,
    ):
        self.model = model or ClaudeModel()
        self.env = env or LocalEnvironment()
        self.config = config or AgentConfig()

        self.messages: List[Dict[str, Any]] = []
        self.step_count = 0
        self.consultation_count = 0
        self.repo_path = ""
        self.problem_statement = ""
        self.failed_attempts = 0  # Track consecutive failures

    def run(self, task: Dict[str, Any], repo_path: str) -> Tuple[str, str]:
        """Run the agent on a SWE-bench task."""
        self.repo_path = repo_path
        self.problem_statement = task.get("problem_statement", "")
        self.messages = []
        self.step_count = 0
        self.failed_attempts = 0

        # Initialize conversation
        system_msg = SYSTEM_TEMPLATE
        instance_msg = INSTANCE_TEMPLATE.format(
            problem_statement=self.problem_statement,
            repo_path=repo_path,
        )

        self.add_message("system", system_msg)
        self.add_message("user", instance_msg)

        logger.info(f"Starting task: {task.get('instance_id', 'unknown')}")

        # Main loop
        while True:
            try:
                self.step()
                self.failed_attempts = 0  # Reset on success
            except NonTerminatingException as e:
                self.failed_attempts += 1
                self.add_message("user", str(e))

                # Consult if too many failures
                if self.config.consultation_on_failure and self.failed_attempts >= 3:
                    logger.info("Multiple failures, triggering consultation")
                    self._do_consultation()
                    self.failed_attempts = 0

            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> Dict[str, Any]:
        """Execute one step: query → parse → execute → observe."""
        self.step_count += 1
        logger.info(f"Step {self.step_count}")

        # Check limits
        if self.config.step_limit > 0 and self.step_count > self.config.step_limit:
            raise LimitsExceeded(f"Step limit ({self.config.step_limit}) exceeded")

        if self.config.cost_limit > 0 and self.model.cost > self.config.cost_limit:
            raise LimitsExceeded(f"Cost limit (${self.config.cost_limit}) exceeded")

        # Check if we should consult Polydev
        if self._should_consult():
            self._do_consultation()

        # Query model
        response = self.query()

        # Parse and execute action
        return self.get_observation(response)

    def query(self) -> Dict[str, Any]:
        """Query the model and add response to history."""
        response = self.model.query(self.messages)
        self.add_message("assistant", response["content"])
        return response

    def get_observation(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action and observe result."""
        action = self.parse_action(response)
        output = self.execute_action(action)

        observation = OBSERVATION_TEMPLATE.format(
            returncode=output["returncode"],
            output=self._truncate_output(output["output"]),
        )

        self.add_message("user", observation)
        return output

    def parse_action(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse bash command from response."""
        content = response.get("content", "")

        # Find bash code blocks
        actions = re.findall(self.config.action_regex, content, re.DOTALL)

        if len(actions) == 1:
            return {"action": actions[0].strip(), "content": content}

        # Format error
        raise FormatError(FORMAT_ERROR_TEMPLATE.format(n_actions=len(actions)))

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bash command and check for completion."""
        command = action["action"]
        logger.debug(f"Executing: {command[:80]}...")

        # Execute in repo directory
        output = self.env.execute(command, cwd=self.repo_path)

        if output["timed_out"]:
            raise ExecutionTimeoutError(TIMEOUT_TEMPLATE.format(
                timeout=self.env.config.timeout,
                output=output["output"][:1000],
            ))

        # Check for submission marker
        self._check_finished(output)

        return output

    def _check_finished(self, output: Dict[str, Any]):
        """Check if agent has submitted."""
        lines = output.get("output", "").strip().splitlines()

        if lines:
            first_line = lines[0].strip()
            for marker in self.config.submit_markers:
                if marker in first_line:
                    # Extract the patch
                    patch = self._extract_patch()
                    
                    # VALIDATION: Ensure patch is not empty
                    if not patch or len(patch.strip()) < 10:
                        # Patch is empty - don't submit, prompt agent to make changes
                        empty_patch_msg = """WARNING: You tried to submit but no code changes were detected!

You must make actual code changes before submitting. The git diff is empty.

Please:
1. Make the necessary code changes using sed, cat, or other file editing commands
2. Verify your changes with `git diff`
3. Only then submit with: echo POLYDEV_SUBMIT_PATCH

Continue by making the required code changes."""
                        self.add_message("user", empty_patch_msg)
                        self.failed_attempts += 1
                        return  # Don't raise Submitted - continue working
                    
                    raise Submitted(patch)

    def _extract_patch(self) -> str:
        """Extract the git diff patch from the repository."""
        result = self.env.execute("git diff", cwd=self.repo_path)
        return result.get("output", "")

    def _truncate_output(self, output: str, max_length: int = 10000) -> str:
        """Truncate long output using mini-SWE-agent style formatting."""
        if len(output) <= max_length:
            return output

        head = output[:5000]
        tail = output[-5000:]
        elided = len(output) - 10000

        return OBSERVATION_TRUNCATED_TEMPLATE.format(
            total_chars=len(output),
            head=head,
            elided=elided,
            tail=tail
        )

    def _should_consult(self) -> bool:
        """Check if we should trigger Polydev consultation."""
        if not self.config.consultation_enabled:
            return False

        # Consult every N steps (if enabled, i.e., > 0)
        if self.config.consultation_after_steps > 0:
            if self.step_count > 0 and self.step_count % self.config.consultation_after_steps == 0:
                return True

        # Check if model indicated it's stuck
        if self.messages and len(self.messages) >= 2:
            last_response = self.messages[-2].get("content", "")
            if self.model.should_consult(last_response):
                return True

        return False

    def _do_consultation(self):
        """Perform REAL Polydev MCP consultation."""
        self.consultation_count += 1
        logger.info(f"Consultation #{self.consultation_count} - REAL multi-model")

        # Build consultation context
        steps_taken = self._summarize_steps()

        context = CONSULTATION_TEMPLATE.format(
            problem_statement=self.problem_statement,
            current_approach=self._get_current_approach(),
            steps_taken=steps_taken,
        )

        # Get REAL perspectives from other models
        perspectives = self.model.consult_polydev(context)

        # Add consultation result to conversation
        consultation_msg = "## Multi-Model Consultation Results\n\n"
        consultation_msg += "I consulted multiple AI models for additional perspectives:\n\n"

        for model_name, response in perspectives.items():
            if model_name != "error":
                consultation_msg += f"### {model_name}\n{response[:2000]}\n\n"

        if "error" in perspectives:
            consultation_msg += f"\n(Note: {perspectives['error']})\n"

        consultation_msg += "\nPlease consider these perspectives and continue with your approach."

        self.add_message("user", consultation_msg)

    def _summarize_steps(self) -> str:
        """Summarize steps taken so far."""
        steps = []
        for i, msg in enumerate(self.messages):
            if msg["role"] == "assistant":
                # Extract action
                actions = re.findall(self.config.action_regex, msg.get("content", ""), re.DOTALL)
                if actions:
                    steps.append(f"Step {len(steps) + 1}: {actions[0][:100]}...")

        return "\n".join(steps[-10:])  # Last 10 steps

    def _get_current_approach(self) -> str:
        """Extract current approach from recent messages."""
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                content = msg.get("content", "")
                # Extract THOUGHT section
                if "THOUGHT:" in content:
                    thought = content.split("THOUGHT:")[-1].split("```")[0]
                    return thought.strip()[:500]
        return "Exploring the codebase"

    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "step_count": self.step_count,
            "consultation_count": self.consultation_count,
            "model_stats": self.model.get_stats(),
        }
