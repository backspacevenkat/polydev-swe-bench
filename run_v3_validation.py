#!/usr/bin/env python3
"""
Validation run for the agent v3 fixes.

Tests 20 diverse tasks with 10 parallel workers to validate:
1. Updated prompts with mini-SWE-agent style
2. Increased step limit (10 iter × 50 steps = 500 max)
3. Format error handling with detailed feedback
4. Output truncation for long command outputs
"""

import sys
import os
import logging
import time

# Add agent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_v3 import SWEBenchRunner, RunConfig, AgentConfig

# 20 diverse tasks for validation (subset of 30 from original run)
TASKS = [
    "astropy__astropy-14508",
    "django__django-11119",
    "django__django-11740",
    "django__django-11790",
    "django__django-11964",
    "django__django-12125",
    "django__django-12774",
    "django__django-13028",
    "matplotlib__matplotlib-14623",
    "matplotlib__matplotlib-22719",
    "matplotlib__matplotlib-24026",
    "pallets__flask-5014",
    "psf__requests-1142",
    "psf__requests-1724",
    "pydata__xarray-3305",
    "pytest-dev__pytest-10051",
    "pytest-dev__pytest-5262",
    "sympy__sympy-12481",
    "sympy__sympy-13852",
    "sympy__sympy-17318",
]


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/tmp/agent_v3_validation.log"),
        ]
    )

    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("POLYDEV SWE-BENCH AGENT V3 - VALIDATION RUN")
    print("=" * 70)
    print(f"Tasks: {len(TASKS)}")
    print(f"Workers: 10 (parallel execution)")
    print(f"Step Limit: 500 (10 iterations × 50 steps)")
    print(f"MCP Consultation: ENABLED (REAL)")
    print("=" * 70)
    print("\nFixes being validated:")
    print("  1. Mini-SWE-agent style prompts")
    print("  2. Increased step limit (500 total)")
    print("  3. Format error handling")
    print("  4. Output truncation")
    print("=" * 70)
    print("\nTask Distribution:")

    # Count by repo
    from collections import Counter
    repos = Counter(t.split("__")[0] + "__" + t.split("__")[1].split("-")[0] for t in TASKS)
    for repo, count in sorted(repos.items(), key=lambda x: -x[1]):
        print(f"  {repo}: {count}")
    print("=" * 70)

    start_time = time.time()

    # Create configs
    run_config = RunConfig(
        output_dir="results/v3_validation_20tasks",
        workspace_dir="/tmp/polydev-v3-validation",
        max_workers=10,  # 10 parallel workers as requested
        stagger_delay=2.0,  # 2 second stagger to avoid thundering herd
        task_timeout=3600,  # 60 minutes per task
    )

    agent_config = AgentConfig(
        step_limit=500,  # 500 steps max (10 iter × 50 steps)
        consultation_enabled=True,
        consultation_after_steps=15,  # Consult every 15 steps
    )

    # Run
    runner = SWEBenchRunner(run_config, agent_config)
    results = runner.run_tasks(instance_ids=TASKS)

    # Print results
    duration = time.time() - start_time

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Group results by repo
    from collections import defaultdict
    by_repo = defaultdict(list)
    for r in results:
        repo = r.instance_id.split("__")[0] + "__" + r.instance_id.split("__")[1].split("-")[0]
        by_repo[repo].append(r)

    for repo, repo_results in sorted(by_repo.items()):
        print(f"\n{repo}:")
        for r in repo_results:
            patch_info = f"{len(r.patch)} chars" if r.patch else "no patch"
            consult_info = f", {r.consultations} consultations" if r.consultations > 0 else ""
            status_icon = "✓" if r.status == "submitted" and r.patch else "✗"
            print(f"  {status_icon} {r.instance_id}: {r.status} ({r.steps} steps, {patch_info}{consult_info})")

    # Summary stats
    submitted = [r for r in results if r.status == "submitted"]
    with_patch = [r for r in submitted if r.patch and len(r.patch.strip()) > 10]
    total_steps = sum(r.steps for r in results)
    total_consultations = sum(r.consultations for r in results)

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"Total Tasks: {len(results)}")
    print(f"Submitted: {len(submitted)} ({len(submitted)/len(results)*100:.1f}%)")
    print(f"With Valid Patches: {len(with_patch)} ({len(with_patch)/len(results)*100:.1f}%)")
    print(f"Total Steps: {total_steps}")
    print(f"Avg Steps/Task: {total_steps/len(results):.1f}")
    print(f"Total Consultations: {total_consultations}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Throughput: {len(results)/(duration/60):.1f} tasks/minute")
    print("=" * 70)

    # Comparison with baseline
    baseline_rate = 14  # Previous 14% resolution rate
    print(f"\n📊 BASELINE COMPARISON:")
    print(f"   Previous resolution rate: {baseline_rate}% (8/57 tasks)")
    print(f"   New patch submission rate: {len(with_patch)/len(results)*100:.1f}%")
    print("=" * 70)

    print(f"\n📊 Results saved to: {run_config.output_dir}")
    print(f"📊 Predictions: {run_config.output_dir}/predictions.jsonl")
    print(f"\n🔄 To submit for evaluation:")
    print(f"   sb-cli submit {run_config.output_dir}/predictions.jsonl --run_id polydev_v3_validation")


if __name__ == "__main__":
    main()
