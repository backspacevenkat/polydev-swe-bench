#!/usr/bin/env python3
"""
Run Polydev SWE-bench Agent v3 on 10 test tasks.

Key improvements over v2:
- REAL Polydev MCP consultation
- 15-worker parallel execution
- 100 step limit (vs 30)
- Staggered starts to avoid thundering herd
"""

import sys
import os
import logging
import time

# Add agent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_v3 import SWEBenchRunner, RunConfig, AgentConfig

# Same 10 tasks as v2 test for comparison
TEST_TASKS = [
    "django__django-11099",
    "django__django-11163",
    "django__django-10097",
    "django__django-11179",
    "django__django-11292",
    "django__django-11433",
    "django__django-11451",
    "django__django-11555",
    "django__django-10554",
    "django__django-10914",
]


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/tmp/agent_v3_10tasks.log"),
        ]
    )

    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("POLYDEV SWE-BENCH AGENT V3 - PARALLEL TEST RUN")
    print("=" * 60)
    print(f"Tasks: {len(TEST_TASKS)}")
    print(f"Workers: 10 (for 10 tasks)")
    print(f"Step Limit: 100")
    print(f"MCP Consultation: ENABLED (REAL)")
    print("=" * 60)

    start_time = time.time()

    # Create configs - use 10 workers for 10 tasks
    run_config = RunConfig(
        output_dir="results/v3_test_10_parallel",
        workspace_dir="/tmp/polydev-v3-10tasks",
        max_workers=10,  # All 10 tasks in parallel
        stagger_delay=3.0,  # 3 second stagger
        task_timeout=3600,  # 60 minutes per task
    )

    agent_config = AgentConfig(
        step_limit=100,  # 100 steps max
        consultation_enabled=True,
        consultation_after_steps=15,  # Consult every 15 steps
    )

    # Run
    runner = SWEBenchRunner(run_config, agent_config)
    results = runner.run_tasks(instance_ids=TEST_TASKS)

    # Print results
    duration = time.time() - start_time

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for r in results:
        patch_info = f"{len(r.patch)} chars" if r.patch else "no patch"
        consult_info = f", {r.consultations} consultations" if r.consultations > 0 else ""
        print(f"{r.instance_id}: {r.status} ({r.steps} steps, {patch_info}{consult_info})")

    submitted = [r for r in results if r.status == "submitted" and r.patch]
    total_steps = sum(r.steps for r in results)
    total_consultations = sum(r.consultations for r in results)

    print(f"\nTotal: {len(submitted)}/{len(results)} submitted ({len(submitted)/len(results)*100:.0f}%)")
    print(f"Total Steps: {total_steps}")
    print(f"Total Consultations: {total_consultations}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"Speedup vs v2 (84 min): {84/(duration/60):.1f}x")


if __name__ == "__main__":
    main()
