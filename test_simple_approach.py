#!/usr/bin/env python3
"""
Test Anthropic's Simple SWE-bench Approach with Claude Code CLI.

Replicating their 73.3% approach:
- 2 tools: bash + file editing
- Prompt: "use tools 100+ times" + "write tests first"
- Extended thinking (built into Claude models)
"""

import subprocess
import os
import sys
import time

# Claude Code CLI
CLAUDE_CLI = "/Users/venkat/.nvm/versions/node/v22.20.0/bin/claude"

# Anthropic's key prompts
SYSTEM_PROMPT = """You are an expert software engineer solving a GitHub issue.

CRITICAL INSTRUCTIONS (from Anthropic's winning approach):
1. USE TOOLS AS MUCH AS POSSIBLE - ideally more than 100 tool calls
2. IMPLEMENT YOUR OWN TESTS FIRST before attempting the fix
3. Be extremely thorough - explore deeply, don't give up
4. Make minimal, targeted changes
5. Verify your fix works before declaring done

When complete, run: echo "POLYDEV_SUBMIT_PATCH" && git diff
"""


def test_real_swe_task():
    """Test with a real SWE-bench task."""

    # Setup workspace
    workspace = "/tmp/simple-swe-test-real"
    if os.path.exists(workspace):
        subprocess.run(["rm", "-rf", workspace], check=True)
    os.makedirs(workspace, exist_ok=True)

    # Clone a real repo (requests - simpler than Django)
    print("Cloning repository...")
    subprocess.run([
        "git", "clone", "--depth", "50",
        "https://github.com/psf/requests.git",
        workspace
    ], check=True, capture_output=True)

    # A realistic issue
    task_prompt = f"""## GitHub Issue

Repository: psf/requests
Working directory: {workspace}

## Problem Statement

When using `requests.get()` with a URL that has special characters in the path,
the library sometimes double-encodes the URL, leading to 404 errors.

For example:
```python
import requests
url = "https://example.com/path/with%20spaces"
r = requests.get(url)  # This may double-encode to %2520
```

## Your Task

1. First, write a test that reproduces this issue
2. Find where URL encoding happens in the codebase
3. Fix the double-encoding bug
4. Verify your fix with the test

{SYSTEM_PROMPT}

Begin by exploring the repository structure with `ls` and `find`.
"""

    print("=" * 60)
    print("TESTING ANTHROPIC'S SIMPLE APPROACH")
    print("=" * 60)
    print(f"Model: haiku")
    print(f"Workspace: {workspace}")
    print("=" * 60)

    start = time.time()

    # Run Claude Code
    cmd = [
        CLAUDE_CLI,
        "--print",
        "--model", "haiku",
        "--dangerously-skip-permissions",
        "--system-prompt", SYSTEM_PROMPT,
        task_prompt
    ]

    print("\nRunning Claude Code...")
    result = subprocess.run(
        cmd,
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=600  # 10 min timeout
    )

    duration = time.time() - start
    output = result.stdout + result.stderr

    # Check results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Duration: {duration:.1f}s")
    print(f"Exit code: {result.returncode}")

    # Count tool uses
    tool_count = output.count("Tool") + output.count("```bash") + output.count("edit")
    print(f"Approximate tool uses: {tool_count}")

    # Check for patch
    git_diff = subprocess.run(
        ["git", "diff"],
        cwd=workspace,
        capture_output=True,
        text=True
    )

    if git_diff.stdout.strip():
        print(f"Patch generated: YES ({len(git_diff.stdout)} chars)")
        print("\nPatch preview:")
        print(git_diff.stdout[:1000])
    else:
        print("Patch generated: NO")

    # Save output
    with open("/tmp/simple_swe_output.txt", "w") as f:
        f.write(output)
    print(f"\nFull output saved to: /tmp/simple_swe_output.txt")

    return git_diff.stdout.strip()


if __name__ == "__main__":
    try:
        patch = test_real_swe_task()
        sys.exit(0 if patch else 1)
    except subprocess.TimeoutExpired:
        print("TIMEOUT - Task took too long")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
