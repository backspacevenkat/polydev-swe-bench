#!/usr/bin/env python3
"""
Quick 5-task test WITHOUT scheduled Polydev consultations.
Should complete in 10-15 minutes.
"""

import sys
import os
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_v3 import SWEBenchRunner, RunConfig, AgentConfig

# 5 diverse tasks - known to be solvable
TASKS = [
    "django__django-11119",  # Simple field issue
    "django__django-11790",  # Template issue
    "psf__requests-1142",    # HTTP header issue
    "sympy__sympy-12481",    # Math simplification
    "pytest-dev__pytest-5262", # Test fixture issue
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/tmp/quick_test.log"),
        ]
    )

    print("=" * 60)
    print("QUICK 5-TASK TEST (NO SCHEDULED POLYDEV)")
    print("=" * 60)
    print(f"Tasks: {len(TASKS)}")
    print(f"Workers: 5")
    print(f"Step Limit: 100")
    print(f"Polydev: Only when stuck (not scheduled)")
    print("=" * 60)

    start_time = time.time()

    run_config = RunConfig(
        output_dir="results/quick_test",
        workspace_dir="/tmp/polydev-quick-test",
        max_workers=5,
        stagger_delay=1.0,
        task_timeout=1200,  # 20 min per task max
    )

    agent_config = AgentConfig(
        step_limit=100,
        consultation_enabled=True,
        consultation_after_steps=0,  # DISABLED - only when stuck
    )

    runner = SWEBenchRunner(run_config, agent_config)
    results = runner.run_tasks(instance_ids=TASKS)

    duration = time.time() - start_time

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for r in results:
        patch_info = f"{len(r.patch)} chars" if r.patch else "no patch"
        status_icon = "✓" if r.status == "submitted" and r.patch else "✗"
        print(f"  {status_icon} {r.instance_id}: {r.status} ({r.steps} steps, {patch_info})")

    submitted = [r for r in results if r.status == "submitted"]
    with_patch = [r for r in submitted if r.patch and len(r.patch.strip()) > 10]
    total_steps = sum(r.steps for r in results)

    print("\n" + "=" * 60)
    print(f"Total: {len(results)} | Submitted: {len(submitted)} | With Patches: {len(with_patch)}")
    print(f"Total Steps: {total_steps} | Avg: {total_steps/len(results):.1f}")
    print(f"Duration: {duration/60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
