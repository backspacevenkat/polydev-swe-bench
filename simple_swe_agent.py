#!/usr/bin/env python3
"""
Simple SWE-bench Agent - Replicating Anthropic's 73.3% approach.

Key insight from Anthropic:
- Simple scaffold with 2 tools: bash + file editing
- 128K extended thinking budget
- Prompt: "use tools 100+ times" + "write tests first"
- No complex multi-agent orchestration

This uses Claude Code CLI directly with Haiku for cost efficiency.
"""

import subprocess
import json
import os
import sys
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Claude Code CLI path
CLAUDE_CLI = "/Users/venkat/.nvm/versions/node/v22.20.0/bin/claude"

# The key prompt from Anthropic's approach
SYSTEM_PROMPT = """You are an expert software engineer solving a GitHub issue.

IMPORTANT INSTRUCTIONS:
1. Use tools as much as possible, ideally more than 100 times
2. Implement your own tests FIRST before attempting the fix
3. Be thorough and persistent - explore the codebase deeply
4. When you find the issue, make minimal targeted changes
5. Test your fix before submitting

You have access to bash and file editing tools. Use them extensively.
"""

TASK_TEMPLATE = """## GitHub Issue to Solve

**Repository:** {repo}
**Issue:** {problem_statement}

## Instructions

1. First, explore the codebase to understand the structure
2. Write a test that reproduces the issue
3. Find the root cause
4. Implement the minimal fix
5. Verify your fix passes the test
6. When done, output: TASK_COMPLETE

## Workspace
You are in: {workspace_path}

Begin by exploring the repository structure.
"""


@dataclass
class TaskResult:
    instance_id: str
    status: str  # "submitted", "error", "timeout"
    patch: Optional[str] = None
    steps: int = 0
    duration_ms: int = 0
    error: Optional[str] = None


def run_claude_code(prompt: str, workspace: str, model: str = "haiku", timeout: int = 1200) -> Tuple[str, int]:
    """Run Claude Code CLI with the given prompt."""

    cmd = [
        CLAUDE_CLI,
        "--print",  # Non-interactive mode
        "--model", model,
        "--max-turns", "150",  # Allow many tool uses
        prompt
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1
    except Exception as e:
        return f"ERROR: {e}", -1


def extract_patch(workspace: str) -> Optional[str]:
    """Extract git diff from workspace."""
    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=workspace,
            capture_output=True,
            text=True
        )
        patch = result.stdout.strip()
        if patch and len(patch) > 10:
            return patch
    except:
        pass
    return None


def clone_repo(repo: str, commit: str, workspace: str) -> bool:
    """Clone and checkout specific commit."""
    repo_url = f"https://github.com/{repo}.git"

    try:
        # Clone
        subprocess.run(
            ["git", "clone", "--depth", "100", repo_url, workspace],
            check=True,
            capture_output=True
        )

        # Checkout specific commit
        subprocess.run(
            ["git", "checkout", commit],
            cwd=workspace,
            check=True,
            capture_output=True
        )
        return True
    except Exception as e:
        logger.error(f"Clone failed: {e}")
        return False


def run_single_task(task: dict, workspace_base: str, model: str = "haiku") -> TaskResult:
    """Run a single SWE-bench task."""

    instance_id = task["instance_id"]
    repo = task["repo"]
    base_commit = task["base_commit"]
    problem_statement = task["problem_statement"]

    logger.info(f"Starting task: {instance_id}")
    start_time = time.time()

    # Setup workspace
    workspace = os.path.join(workspace_base, instance_id.replace("/", "_").replace("-", "_"))
    os.makedirs(workspace_base, exist_ok=True)

    # Clone repo
    if os.path.exists(workspace):
        subprocess.run(["rm", "-rf", workspace], check=True)

    if not clone_repo(repo, base_commit, workspace):
        return TaskResult(
            instance_id=instance_id,
            status="error",
            error="Failed to clone repo"
        )

    # Build prompt
    prompt = TASK_TEMPLATE.format(
        repo=repo,
        problem_statement=problem_statement,
        workspace_path=workspace
    )

    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    # Run Claude Code
    output, returncode = run_claude_code(full_prompt, workspace, model=model)

    duration_ms = int((time.time() - start_time) * 1000)

    # Check for completion
    completed = "TASK_COMPLETE" in output
    patch = extract_patch(workspace)

    # Count approximate steps (tool uses)
    steps = output.count("Tool:") + output.count("bash") + output.count("edit")

    status = "submitted" if patch else ("timeout" if returncode == -1 else "error")

    logger.info(f"Completed {instance_id}: {status} ({steps} steps, {duration_ms}ms)")

    return TaskResult(
        instance_id=instance_id,
        status=status,
        patch=patch,
        steps=steps,
        duration_ms=duration_ms,
        error=None if status == "submitted" else output[-500:]
    )


def main():
    """Test with a simple task."""

    # Simple test task
    test_task = {
        "instance_id": "test__simple-1",
        "repo": "psf/requests",
        "base_commit": "v2.28.0",
        "problem_statement": """
        The requests library should handle None values in headers gracefully.
        Currently it may raise an error when a header value is None.
        Fix this to skip None values or convert them to empty strings.
        """
    }

    workspace_base = "/tmp/simple-swe-test"

    print("=" * 60)
    print("SIMPLE SWE-AGENT TEST")
    print("=" * 60)
    print(f"Model: haiku")
    print(f"Approach: Anthropic's simple scaffold (2 tools)")
    print("=" * 60)

    result = run_single_task(test_task, workspace_base, model="haiku")

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Status: {result.status}")
    print(f"Steps: {result.steps}")
    print(f"Duration: {result.duration_ms / 1000:.1f}s")
    print(f"Patch: {'Yes' if result.patch else 'No'}")
    if result.patch:
        print(f"\nPatch preview:\n{result.patch[:500]}")


if __name__ == "__main__":
    main()
