#!/usr/bin/env python3
"""
Pre-generate Polydev insights for SWE-bench instances.

This script extracts problem statements from the SWE-bench dataset
and writes them to stdout for processing by Claude Code with Polydev MCP.

Usage:
    python generate_polydev_insights.py > problems.json
    # Then use Claude Code to call Polydev for each problem
"""

import json
import sys
from datasets import load_dataset

def main():
    # Load the instance IDs
    with open('/Users/venkat/Documents/polydev-swe-bench/test_20_random_ids.json') as f:
        instance_ids = set(json.load(f))

    # Load SWE-bench dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    # Extract problems
    problems = []
    for inst in ds:
        if inst["instance_id"] in instance_ids:
            # Truncate problem to 3000 chars for faster Polydev processing
            problem_text = inst["problem_statement"][:3000]
            problems.append({
                "instance_id": inst["instance_id"],
                "repo": inst["repo"],
                "problem": problem_text
            })

    # Sort by instance_id for consistency
    problems.sort(key=lambda x: x["instance_id"])

    # Output as JSON
    print(json.dumps(problems, indent=2))

if __name__ == "__main__":
    main()
