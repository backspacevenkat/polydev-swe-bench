#!/usr/bin/env python3
"""Test extended thinking on a simpler SWE-bench instance with more turns."""

import json
import subprocess
import shutil
import os
from pathlib import Path
from datasets import load_dataset

MODEL = "claude-haiku-4-5-20251001"
THINKING_TOKENS = 128000
MAX_TURNS = 75  # More turns for proper test

TASK_TEMPLATE = """<issue_description>
{problem}
</issue_description>

<instructions>
The repository is at: {repo_dir}

Your task: Fix the issue described above by modifying the source code.

IMPORTANT: You should use tools as much as possible, ideally more than 100 times. You should also implement your own tests first before attempting the problem.

## Recommended Workflow:
1. **EXPLORE EXTENSIVELY**: Use bash commands (grep, find, ls, cat) to thoroughly explore the codebase.
2. **WRITE A TEST FIRST**: Before fixing anything, create a simple test script that reproduces the bug.
3. **LOCATE THE ROOT CAUSE**: Trace through the code to find exactly where the bug originates.
4. **IMPLEMENT THE FIX**: Make MINIMAL, targeted changes to fix the root cause.
5. **VERIFY WITH YOUR TEST**: Run your test again to confirm the fix works.

## Rules:
- MODIFY: Source code files only
- DO NOT MODIFY: Test files, setup.py, pyproject.toml, configuration files
- Make MINIMAL changes

When you've completed the fix, say: TASK_COMPLETE
</instructions>"""


def find_simpler_instance(ds):
    """Find a simpler instance - prefer smaller repos like astropy, sympy small issues."""
    # Look for smaller/simpler issues based on problem statement length and repo
    preferred_repos = ["astropy", "sympy", "pylint", "sphinx"]

    for repo_name in preferred_repos:
        for inst in ds:
            if repo_name in inst["instance_id"]:
                # Skip very long problem statements (usually complex)
                if len(inst["problem_statement"]) < 2000:
                    return inst

    # Fallback to any shorter problem
    for inst in ds:
        if len(inst["problem_statement"]) < 1500:
            return inst

    return ds[0]


def main():
    print("Testing Extended Thinking on SWE-bench instance")
    print(f"Thinking budget: {THINKING_TOKENS:,} tokens")
    print(f"Max turns: {MAX_TURNS}")
    print("="*60)

    # Load dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    # Find a simpler instance
    instance = find_simpler_instance(ds)

    print(f"Instance: {instance['instance_id']}")
    print(f"Repo: {instance['repo']}")
    print(f"Problem length: {len(instance['problem_statement'])} chars")

    # Clone repo
    repo_dir = Path(f"/tmp/swe_test_thinking/{instance['instance_id']}")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    print("\nCloning repository...")
    clone_result = subprocess.run(
        ["git", "clone", "--quiet", f"https://github.com/{instance['repo']}.git", str(repo_dir)],
        capture_output=True, timeout=300,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    )
    if clone_result.returncode != 0:
        print(f"Clone failed: {clone_result.stderr.decode()}")
        return

    subprocess.run(
        ["git", "checkout", "--quiet", instance['base_commit']],
        cwd=repo_dir, capture_output=True
    )
    print(f"Cloned to: {repo_dir}")

    # Build prompt
    prompt = TASK_TEMPLATE.format(repo_dir=str(repo_dir), problem=instance["problem_statement"])

    # Set up environment with extended thinking
    env = os.environ.copy()
    env["MAX_THINKING_TOKENS"] = str(THINKING_TOKENS)

    print(f"\nCalling Claude with extended thinking ({THINKING_TOKENS:,} tokens)...")
    print(f"Max turns: {MAX_TURNS}")
    print("-"*60)

    # Run Claude CLI
    result = subprocess.run(
        [
            "claude",
            "--model", MODEL,
            "--print",
            "--output-format", "json",
            "--dangerously-skip-permissions",
            "--max-turns", str(MAX_TURNS),
            "--add-dir", str(repo_dir),
            "-p", prompt
        ],
        capture_output=True,
        text=True,
        timeout=1200,  # 20 min timeout
        cwd=repo_dir,
        env=env
    )

    print("\nResult:")
    print("-"*60)

    try:
        output = json.loads(result.stdout)
        print(f"✓ Success!")
        print(f"  Turns: {output.get('num_turns', 0)}")
        print(f"  Cost: ${output.get('total_cost_usd', 0):.4f}")
        print(f"  Duration: {output.get('duration_ms', 0)/1000:.1f}s")

        # Check model usage for thinking tokens
        model_usage = output.get("modelUsage", {})
        total_output = 0
        for model, usage in model_usage.items():
            out_tokens = usage.get('outputTokens', 0)
            total_output += out_tokens
            print(f"  {model}:")
            print(f"    Input: {usage.get('inputTokens', 0):,}")
            print(f"    Output: {out_tokens:,}")
            print(f"    Cache Read: {usage.get('cacheReadInputTokens', 0):,}")
            print(f"    Cost: ${usage.get('costUSD', 0):.4f}")

        print(f"\n  Total output tokens: {total_output:,}")

        # Check if thinking is likely active (higher output count suggests more reasoning)
        if total_output > 5000:
            print("  ✓ High output token count - extended thinking likely active")
        else:
            print("  ⚠ Low output token count - extended thinking may not be active")

        # Check for patch
        diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True, cwd=repo_dir)
        if diff.stdout:
            print(f"\n✓ PATCH GENERATED ({len(diff.stdout)} chars)")
            print("-"*40)
            print(diff.stdout[:1000])
            if len(diff.stdout) > 1000:
                print(f"... ({len(diff.stdout) - 1000} more chars)")
        else:
            print("\n✗ No patch generated")

    except json.JSONDecodeError:
        print(f"✗ Failed to parse JSON")
        print(f"STDOUT: {result.stdout[:1000]}")
        print(f"STDERR: {result.stderr[:500]}")

    # Cleanup
    shutil.rmtree(repo_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
