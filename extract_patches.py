#!/usr/bin/env python3
"""Extract patches from SWE-bench workspaces and format for evaluation."""

import json
import subprocess
import os

WORKSPACE_BASE = "/tmp/simple-swe-bench-run"
OUTPUT_FILE = "/tmp/swe_bench_patches.jsonl"

# Task IDs from our run
TASKS = [
    "django__django-10097",
    "django__django-10554",
    "django__django-10880",
    "django__django-10914",
    "django__django-11066",
    "django__django-11087",
    "psf__requests-1142",
    "psf__requests-1724",
    "psf__requests-1766",
    "psf__requests-2317",
    "pytest-dev__pytest-10051",
    "pytest-dev__pytest-10081",
    "pytest-dev__pytest-5262",
    "pytest-dev__pytest-5631",
    "sphinx-doc__sphinx-10323",
    "sphinx-doc__sphinx-10435",
    "scikit-learn__scikit-learn-10297",
    "scikit-learn__scikit-learn-10844",
    "pallets__flask-5014",
    "sympy__sympy-11618",
]

def get_patch(instance_id: str) -> str:
    """Get git diff from workspace."""
    safe_id = instance_id.replace("/", "_").replace("-", "_")
    workspace = os.path.join(WORKSPACE_BASE, safe_id)

    if not os.path.exists(workspace):
        return ""

    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=10
        )
        patch = result.stdout.strip()

        # Also check staged changes
        if not patch:
            result = subprocess.run(
                ["git", "diff", "--cached"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=10
            )
            patch = result.stdout.strip()

        return patch
    except Exception as e:
        print(f"Error getting patch for {instance_id}: {e}")
        return ""

def main():
    patches = []

    for task_id in TASKS:
        patch = get_patch(task_id)
        if patch:
            patches.append({
                "model_name_or_path": "claude-haiku-4.5",
                "instance_id": task_id,
                "model_patch": patch
            })
            print(f"✓ {task_id}: {len(patch)} chars")
        else:
            print(f"✗ {task_id}: no patch")

    # Write JSONL
    with open(OUTPUT_FILE, "w") as f:
        for p in patches:
            f.write(json.dumps(p) + "\n")

    print(f"\nTotal: {len(patches)}/{len(TASKS)} patches")
    print(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
