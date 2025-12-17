#!/usr/bin/env python3
"""
Run Polydev SWE-bench Agent v3 on 30 diverse tasks.

Distribution:
- 13 Django tasks (baseline)
- 5 Matplotlib tasks (visualization)
- 3 Sympy tasks (symbolic math)
- 2 Requests tasks (HTTP)
- 2 Pytest tasks (testing)
- 2 Xarray tasks (data arrays)
- 1 Flask task (web)
- 1 Scikit-learn task (ML)
- 1 Sphinx task (docs)
- 1 Astropy task (astronomy)

Key improvements over v2:
- REAL Polydev MCP consultation
- 15-worker parallel execution
- 100 step limit
- Empty patch validation
- Comprehensive analytics
"""

import sys
import os
import logging
import time

# Add agent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_v3 import SWEBenchRunner, RunConfig, AgentConfig

# 30 diverse tasks (different from initial 10)
TASKS = [
    "astropy__astropy-14508",
    "django__django-11119",
    "django__django-11740",
    "django__django-11790",
    "django__django-11964",
    "django__django-12125",
    "django__django-12774",
    "django__django-13028",
    "django__django-13195",
    "django__django-14500",
    "django__django-15525",
    "django__django-16145",
    "django__django-16255",
    "matplotlib__matplotlib-14623",
    "matplotlib__matplotlib-22719",
    "matplotlib__matplotlib-24026",
    "matplotlib__matplotlib-24149",
    "matplotlib__matplotlib-24570",
    "pallets__flask-5014",
    "psf__requests-1142",
    "psf__requests-1724",
    "pydata__xarray-3305",
    "pydata__xarray-7233",
    "pytest-dev__pytest-10051",
    "pytest-dev__pytest-5262",
    "scikit-learn__scikit-learn-13142",
    "sphinx-doc__sphinx-8721",
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
            logging.FileHandler("/tmp/agent_v3_30tasks.log"),
        ]
    )

    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("POLYDEV SWE-BENCH AGENT V3 - 30 TASK EVALUATION")
    print("=" * 70)
    print(f"Tasks: {len(TASKS)}")
    print(f"Workers: 15 (parallel execution)")
    print(f"Step Limit: 100")
    print(f"MCP Consultation: ENABLED (REAL)")
    print(f"Empty Patch Validation: ENABLED")
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
        output_dir="results/v3_30tasks_diverse",
        workspace_dir="/tmp/polydev-v3-30tasks",
        max_workers=15,  # 15 parallel workers
        stagger_delay=2.0,  # 2 second stagger to avoid thundering herd
        task_timeout=3600,  # 60 minutes per task
    )

    agent_config = AgentConfig(
        step_limit=100,  # 100 steps max
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
    print("FINAL STATISTICS")
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

    print(f"\n📊 Results saved to: {run_config.output_dir}")
    print(f"📊 Predictions: {run_config.output_dir}/predictions.jsonl")
    print(f"\n🔄 To submit for evaluation:")
    print(f"   sb-cli submit {run_config.output_dir}/predictions.jsonl --run_id polydev_v3_30tasks")


if __name__ == "__main__":
    main()
