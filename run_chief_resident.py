#!/usr/bin/env python3
"""
Run Chief Resident Evaluation

This script runs the 4 experimental configurations on SWE-bench tasks:
- A: Base Alone (Claude only, no consultation)
- B: Base + Polydev Gated (main hypothesis - consult when triggered)
- C: Base + Polydev Always (consult every iteration)
- D: Base + Self-Reflect (control for "more compute")

Usage:
    # Run on all SWE-bench Verified tasks
    python run_chief_resident.py --all

    # Run on specific tasks
    python run_chief_resident.py --tasks task1,task2

    # Run with specific configs only
    python run_chief_resident.py --configs A,B

    # Resume from previous run
    python run_chief_resident.py --resume
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agent_v3 import (
    ChiefResidentAgent,
    ChiefResidentConfig,
    ExperimentConfig,
    EvaluationHarness,
    run_experiment
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_swe_bench_tasks(subset: str = "verified") -> list:
    """Load SWE-bench tasks from the datasets library."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        return list(dataset)
    except ImportError:
        logger.warning("datasets library not available, loading from local file")
        tasks_file = Path(__file__).parent / "data" / "swe_bench_verified.json"
        if tasks_file.exists():
            with open(tasks_file) as f:
                return json.load(f)
        return []


def get_repo_paths() -> dict:
    """Get mapping of instance_id to cloned repo path."""
    # Check for pre-cloned repos
    repos_dir = Path(__file__).parent / "repos"
    repo_paths = {}

    if repos_dir.exists():
        for repo_dir in repos_dir.iterdir():
            if repo_dir.is_dir():
                # Infer instance_id from directory name
                # Format: owner__repo__commit
                instance_id = repo_dir.name
                repo_paths[instance_id] = str(repo_dir)

    return repo_paths


def parse_configs(config_str: str) -> list:
    """Parse config string into ExperimentConfig list."""
    config_map = {
        "A": ExperimentConfig.BASE_ALONE,
        "B": ExperimentConfig.BASE_PLUS_POLYDEV_GATED,
        "C": ExperimentConfig.BASE_PLUS_POLYDEV_ALWAYS,
        "D": ExperimentConfig.BASE_PLUS_SELF_REFLECT,
    }

    if not config_str:
        return list(ExperimentConfig)

    configs = []
    for c in config_str.split(","):
        c = c.strip().upper()
        if c in config_map:
            configs.append(config_map[c])
        else:
            logger.warning(f"Unknown config: {c}")

    return configs


def run_single_task(
    task: dict,
    repo_path: str,
    config: ExperimentConfig
) -> dict:
    """Run a single task with a single configuration."""
    agent_config = ChiefResidentConfig(experiment_config=config)
    agent = ChiefResidentAgent(config=agent_config)

    success, patch, metrics = agent.solve(task, repo_path)

    return {
        "instance_id": task.get("instance_id"),
        "config": config.name,
        "success": success,
        "patch": patch,
        "metrics": metrics,
        "stats": agent.get_stats()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Chief Resident Evaluation on SWE-bench"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on all SWE-bench Verified tasks"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated list of task instance_ids"
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="A,B",
        help="Comma-separated list of configs (A, B, C, D)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/chief_resident",
        help="Output directory for results"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip completed tasks)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of tasks (0 for all)"
    )

    args = parser.parse_args()

    # Load tasks
    if args.all:
        tasks = load_swe_bench_tasks()
        logger.info(f"Loaded {len(tasks)} SWE-bench Verified tasks")
    elif args.tasks:
        all_tasks = load_swe_bench_tasks()
        task_ids = set(t.strip() for t in args.tasks.split(","))
        tasks = [t for t in all_tasks if t.get("instance_id") in task_ids]
        logger.info(f"Selected {len(tasks)} tasks from {len(task_ids)} requested")
    else:
        # Default: run on first few tasks for testing
        tasks = load_swe_bench_tasks()[:5]
        logger.info(f"Running on {len(tasks)} tasks (use --all for full evaluation)")

    # Apply limit
    if args.limit > 0:
        tasks = tasks[:args.limit]

    # Get repo paths
    repo_paths = get_repo_paths()
    logger.info(f"Found {len(repo_paths)} pre-cloned repositories")

    # Parse configs
    configs = parse_configs(args.configs)
    logger.info(f"Running configs: {[c.name for c in configs]}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    harness = EvaluationHarness(
        output_dir=str(output_dir),
        configs=configs,
        n_workers=args.workers
    )

    results = harness.run_evaluation(tasks, repo_paths, resume=args.resume)
    logger.info(f"Completed {len(results)} tasks")

    # Compute metrics
    metrics = harness.compute_metrics()
    harness.save_metrics(metrics)

    # Generate report
    report = harness.generate_report(metrics)
    harness.save_report(report)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # Summary
    print(f"\nResults saved to: {output_dir}")
    print(f"Metrics file: {output_dir}/metrics.json")
    print(f"Report file: {output_dir}/report.txt")


if __name__ == "__main__":
    main()
