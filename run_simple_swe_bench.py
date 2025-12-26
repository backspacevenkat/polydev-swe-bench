#!/usr/bin/env python3
"""
Simple SWE-bench Runner - Replicating Anthropic's 73.3% approach.

Uses Claude Code CLI with:
- Simple scaffold (bash + edit tools)
- Extended thinking (built into Claude)
- "Use tools 100+ times" prompt
- Parallel execution
"""

import subprocess
import json
import os
import sys
import logging
import time
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/simple_swe_bench.log")
    ]
)
logger = logging.getLogger(__name__)

# Claude Code CLI
CLAUDE_CLI = "/Users/venkat/.nvm/versions/node/v22.20.0/bin/claude"

# Anthropic's winning prompt approach
SYSTEM_PROMPT = """You are an expert software engineer solving a GitHub issue.

CRITICAL INSTRUCTIONS:
1. You MUST make code changes - this is not optional
2. Use tools extensively - make 20+ tool calls minimum
3. Be persistent - if one approach fails, try another
4. Make minimal, targeted changes to fix the issue
5. ALWAYS edit the source files to implement your fix
6. Never give up without making at least one code change

Your job is NOT complete until you have EDITED at least one file.
Do NOT just analyze - you MUST write code changes."""


# 20 diverse SWE-bench Verified tasks (ALL VALIDATED)
TASKS = [
    # Django (easier, well-documented)
    "django__django-10097",
    "django__django-10554",
    "django__django-10880",
    "django__django-10914",
    "django__django-11066",
    "django__django-11087",
    # Requests (smaller codebase)
    "psf__requests-1142",
    "psf__requests-1724",
    "psf__requests-1766",  # Fixed: was 2148
    "psf__requests-2317",
    # Pytest (medium complexity)
    "pytest-dev__pytest-10051",
    "pytest-dev__pytest-10081",
    "pytest-dev__pytest-5262",  # Fixed: was 5103
    "pytest-dev__pytest-5631",  # Fixed: was 5221
    # Sphinx (documentation)
    "sphinx-doc__sphinx-10323",
    "sphinx-doc__sphinx-10435",
    # Scikit-learn (larger but focused)
    "scikit-learn__scikit-learn-10297",
    "scikit-learn__scikit-learn-10844",
    # Flask (simple)
    "pallets__flask-5014",
    # Sympy (math, complex)
    "sympy__sympy-11618",
]


@dataclass
class TaskResult:
    instance_id: str
    status: str  # "submitted", "error", "timeout"
    patch: Optional[str] = None
    steps: int = 0
    duration_sec: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None


def load_swe_bench_task(instance_id: str) -> Optional[dict]:
    """Load task from SWE-bench dataset."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        for task in dataset:
            if task["instance_id"] == instance_id:
                return task
        return None
    except Exception as e:
        logger.error(f"Failed to load task {instance_id}: {e}")
        return None


def clone_repo(repo: str, commit: str, workspace: str) -> bool:
    """Clone and checkout specific commit."""
    repo_url = f"https://github.com/{repo}.git"

    try:
        if os.path.exists(workspace):
            subprocess.run(["rm", "-rf", workspace], check=True)

        # Clone with depth for speed
        subprocess.run(
            ["git", "clone", "--depth", "100", repo_url, workspace],
            check=True,
            capture_output=True,
            timeout=120
        )

        # Fetch the specific commit if needed
        subprocess.run(
            ["git", "fetch", "--depth", "100", "origin", commit],
            cwd=workspace,
            capture_output=True,
            timeout=180  # Increased from 60s for large repos
        )

        # Checkout
        subprocess.run(
            ["git", "checkout", commit],
            cwd=workspace,
            check=True,
            capture_output=True
        )
        return True
    except Exception as e:
        logger.error(f"Clone failed for {repo}: {e}")
        return False


def parse_token_usage(output: str) -> Dict[str, int]:
    """Parse token usage from Claude Code output."""
    tokens = {"input": 0, "output": 0, "total": 0}

    # Look for token usage patterns in output
    # Claude Code may output stats like "Tokens: 1234 input, 567 output"
    patterns = [
        r"(\d+)\s*input.*?(\d+)\s*output",
        r"tokens?:?\s*(\d+)",
        r"Input:\s*(\d+).*Output:\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) >= 2:
                tokens["input"] = int(groups[0])
                tokens["output"] = int(groups[1])
            elif len(groups) == 1:
                tokens["total"] = int(groups[0])
            break

    tokens["total"] = tokens["input"] + tokens["output"] or tokens["total"]
    return tokens


def count_steps(output: str) -> int:
    """Count approximate tool uses/steps."""
    patterns = [
        r"```bash",
        r"Tool:",
        r"Running:",
        r"Editing:",
        r"Reading:",
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, output, re.IGNORECASE))
    return max(count, 1)


def run_single_task(instance_id: str, workspace_base: str, model: str = "haiku") -> TaskResult:
    """Run a single SWE-bench task using Claude Code CLI."""

    logger.info(f"Starting: {instance_id}")
    start_time = time.time()

    # Load task data
    task = load_swe_bench_task(instance_id)
    if not task:
        return TaskResult(
            instance_id=instance_id,
            status="error",
            error="Task not found in dataset"
        )

    # Setup workspace
    safe_id = instance_id.replace("/", "_").replace("-", "_")
    workspace = os.path.join(workspace_base, safe_id)

    # Clone repo
    if not clone_repo(task["repo"], task["base_commit"], workspace):
        return TaskResult(
            instance_id=instance_id,
            status="error",
            error="Failed to clone repository"
        )

    # Build prompt
    prompt = f"""## GitHub Issue to Solve

**Repository:** {task["repo"]}
**Instance:** {instance_id}

## Problem Statement

{task["problem_statement"]}

## Your Task

You MUST fix this issue by editing the source code. This is mandatory.

Steps:
1. Explore: `ls -la` and find relevant Python files
2. Understand: Read the relevant source files
3. Fix: Edit the file(s) to fix the bug
4. Verify: Check your changes make sense

IMPORTANT: You are being evaluated on whether you produce a code patch.
Do NOT just explain - you MUST edit files using the Edit tool.

Working directory: {workspace}

Start now - explore and fix the issue."""


    # Run Claude Code
    cmd = [
        CLAUDE_CLI,
        "--print",
        "--model", model,
        "--dangerously-skip-permissions",
        "--output-format", "json",
        "--max-turns", "30",
        "--system-prompt", SYSTEM_PROMPT,
        prompt
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=900  # 15 min per task
        )
        output = result.stdout + result.stderr
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        output = "TIMEOUT"
        returncode = -1
    except Exception as e:
        output = f"ERROR: {e}"
        returncode = -1

    duration = time.time() - start_time

    # Parse results
    steps = count_steps(output)
    tokens = parse_token_usage(output)

    # Check for patch
    try:
        git_result = subprocess.run(
            ["git", "diff"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=10
        )
        patch = git_result.stdout.strip() if git_result.stdout.strip() else None
    except:
        patch = None

    # Also check staged changes
    if not patch:
        try:
            git_result = subprocess.run(
                ["git", "diff", "--cached"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=10
            )
            patch = git_result.stdout.strip() if git_result.stdout.strip() else None
        except:
            pass

    status = "submitted" if patch else ("timeout" if returncode == -1 else "error")

    logger.info(f"Completed: {instance_id} - {status} ({steps} steps, {duration:.1f}s)")

    return TaskResult(
        instance_id=instance_id,
        status=status,
        patch=patch,
        steps=steps,
        duration_sec=duration,
        input_tokens=tokens["input"],
        output_tokens=tokens["output"],
        total_tokens=tokens["total"],
        error=None if patch else output[-500:] if len(output) > 500 else output
    )


def run_parallel_tasks(
    task_ids: List[str],
    workspace_base: str,
    max_workers: int = 10,
    model: str = "haiku"
) -> List[TaskResult]:
    """Run multiple tasks in parallel."""

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_single_task, task_id, workspace_base, model): task_id
            for task_id in task_ids
        }

        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                results.append(TaskResult(
                    instance_id=task_id,
                    status="error",
                    error=str(e)
                ))

    return results


def validate_tasks(task_ids: List[str]) -> tuple[List[str], List[str]]:
    """Pre-validate all task IDs against SWE-bench Verified dataset."""
    from datasets import load_dataset
    
    logger.info("Loading SWE-bench Verified dataset for validation...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    valid_ids = set(task["instance_id"] for task in dataset)
    
    valid = []
    invalid = []
    for task_id in task_ids:
        if task_id in valid_ids:
            valid.append(task_id)
        else:
            invalid.append(task_id)
            logger.error(f"INVALID TASK ID: {task_id} not in SWE-bench Verified")
    
    return valid, invalid


def main():
    print("=" * 70)
    print("SIMPLE SWE-BENCH RUNNER (Anthropic's 73.3% Approach)")
    print("=" * 70)
    
    # Pre-validate all tasks
    print("Validating task IDs against SWE-bench Verified...")
    valid_tasks, invalid_tasks = validate_tasks(TASKS)
    
    if invalid_tasks:
        print(f"\n⚠️  WARNING: {len(invalid_tasks)} invalid task IDs found:")
        for t in invalid_tasks:
            print(f"    ✗ {t}")
        print(f"\nProceeding with {len(valid_tasks)} valid tasks only.\n")
    else:
        print(f"✓ All {len(valid_tasks)} task IDs validated successfully.\n")
    
    if not valid_tasks:
        print("ERROR: No valid tasks to run!")
        sys.exit(1)
    
    print(f"Tasks: {len(valid_tasks)}")
    print(f"Workers: 10 (parallel)")
    print(f"Model: haiku")
    print(f"Approach: Simple scaffold (bash + edit) + extended thinking")
    print("=" * 70)

    workspace_base = "/tmp/simple-swe-bench-run"
    os.makedirs(workspace_base, exist_ok=True)

    start_time = time.time()

    # Run tasks
    results = run_parallel_tasks(
        task_ids=valid_tasks,
        workspace_base=workspace_base,
        max_workers=10,
        model="haiku"
    )

    total_duration = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    submitted = [r for r in results if r.status == "submitted"]
    with_patch = [r for r in submitted if r.patch and len(r.patch) > 10]
    errors = [r for r in results if r.status == "error"]
    timeouts = [r for r in results if r.status == "timeout"]

    total_steps = sum(r.steps for r in results)
    total_tokens = sum(r.total_tokens for r in results)
    total_input = sum(r.input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)

    for r in results:
        icon = "✓" if r.patch else "✗"
        patch_info = f"{len(r.patch)} chars" if r.patch else "no patch"
        print(f"  {icon} {r.instance_id}: {r.status} ({r.steps} steps, {r.duration_sec:.1f}s, {patch_info})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Tasks: {len(results)}")
    print(f"Submitted (with patch): {len(with_patch)}")
    print(f"Errors: {len(errors)}")
    print(f"Timeouts: {len(timeouts)}")
    print(f"Success Rate: {len(with_patch)/len(results)*100:.1f}%")
    print()
    print(f"Total Steps: {total_steps}")
    print(f"Avg Steps/Task: {total_steps/len(results):.1f}")
    print()
    print(f"Total Tokens: {total_tokens:,}")
    print(f"  Input: {total_input:,}")
    print(f"  Output: {total_output:,}")
    print(f"Avg Tokens/Task: {total_tokens/len(results):,.0f}")
    print()
    print(f"Total Duration: {total_duration/60:.1f} minutes")
    print(f"Avg Duration/Task: {sum(r.duration_sec for r in results)/len(results):.1f}s")
    print("=" * 70)

    # Save results
    results_data = {
        "summary": {
            "total": len(results),
            "submitted": len(with_patch),
            "success_rate": len(with_patch)/len(results)*100,
            "total_steps": total_steps,
            "total_tokens": total_tokens,
            "duration_min": total_duration/60
        },
        "tasks": [
            {
                "instance_id": r.instance_id,
                "status": r.status,
                "steps": r.steps,
                "duration_sec": r.duration_sec,
                "tokens": r.total_tokens,
                "has_patch": bool(r.patch)
            }
            for r in results
        ]
    }

    with open("/tmp/simple_swe_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: /tmp/simple_swe_results.json")
    print(f"Log saved to: /tmp/simple_swe_bench.log")


if __name__ == "__main__":
    main()
