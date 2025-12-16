#!/usr/bin/env python3
"""
Run Sample Evaluation

Quick validation run on a small number of tasks.
Runs both baseline and Polydev-enhanced configurations.

Usage:
    python evaluation/run_sample.py --tasks 10
    python evaluation/run_sample.py --tasks 10 --mode baseline
    python evaluation/run_sample.py --tasks 10 --mode polydev
    python evaluation/run_sample.py --tasks 10 --mode both --verbose
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import PolydevAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_swebench_tasks(
    num_tasks: int,
    dataset: str = "verified"
) -> List[Dict[str, Any]]:
    """
    Load SWE-bench tasks.

    Args:
        num_tasks: Number of tasks to load
        dataset: 'verified' or 'full'

    Returns:
        List of task dictionaries
    """
    # Try to load from local data directory
    data_dir = Path(__file__).parent.parent / "data" / f"swe-bench-{dataset}"
    tasks_file = data_dir / "tasks.json"

    if tasks_file.exists():
        logger.info(f"Loading tasks from {tasks_file}")
        with open(tasks_file) as f:
            all_tasks = json.load(f)
    else:
        # Try to load from SWE-bench package
        logger.info("Loading tasks from SWE-bench package")
        try:
            from swebench import get_eval_refs
            all_tasks = get_eval_refs(dataset)
        except ImportError:
            logger.error(
                "SWE-bench not installed and no local data found. "
                "Run: pip install swebench"
            )
            sys.exit(1)

    # Select subset
    tasks = all_tasks[:num_tasks]
    logger.info(f"Loaded {len(tasks)} tasks")

    return tasks


def run_evaluation(
    tasks: List[Dict[str, Any]],
    mode: str,
    output_dir: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run evaluation on tasks.

    Args:
        tasks: List of SWE-bench tasks
        mode: 'baseline' or 'polydev'
        output_dir: Directory for output
        verbose: Enable verbose logging

    Returns:
        Results dictionary
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create agent
    consultation_enabled = mode == "polydev"
    agent = PolydevAgent(
        consultation_enabled=consultation_enabled,
        confidence_threshold=8,
        log_dir=output_dir
    )

    logger.info(
        f"Starting {mode} evaluation on {len(tasks)} tasks "
        f"(consultation={'enabled' if consultation_enabled else 'disabled'})"
    )

    results = []
    start_time = time.time()

    for i, task in enumerate(tasks):
        logger.info(
            f"[{i+1}/{len(tasks)}] Processing {task['instance_id']}"
        )

        try:
            result = agent.solve_task(task)
            results.append({
                "instance_id": result.instance_id,
                "configuration": result.configuration,
                "initial_confidence": result.initial_confidence,
                "final_confidence": result.final_confidence,
                "consultation_triggered": result.consultation_triggered,
                "patch_generated": result.patch_generated,
                "duration_ms": result.total_duration_ms,
                "cost_usd": result.cost_usd,
                "error": result.error,
                "patch": result.patch
            })

            logger.info(
                f"  -> Confidence: {result.initial_confidence} -> {result.final_confidence}, "
                f"Consulted: {result.consultation_triggered}, "
                f"Time: {result.total_duration_ms}ms"
            )

        except Exception as e:
            logger.error(f"  -> Error: {e}")
            results.append({
                "instance_id": task["instance_id"],
                "configuration": mode,
                "error": str(e)
            })

    total_time = time.time() - start_time

    # Calculate statistics
    successful = [r for r in results if r.get("patch_generated")]
    consulted = [r for r in results if r.get("consultation_triggered")]

    stats = {
        "mode": mode,
        "total_tasks": len(tasks),
        "successful_patches": len(successful),
        "consultations": len(consulted),
        "total_time_seconds": total_time,
        "avg_time_per_task_seconds": total_time / len(tasks) if tasks else 0,
        "total_cost_usd": sum(r.get("cost_usd", 0) for r in results)
    }

    return {
        "stats": stats,
        "results": results
    }


def compare_results(
    baseline: Dict[str, Any],
    polydev: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare baseline and polydev results."""
    comparison = {
        "baseline": baseline["stats"],
        "polydev": polydev["stats"],
        "improvement": {
            "patches_delta": (
                polydev["stats"]["successful_patches"] -
                baseline["stats"]["successful_patches"]
            ),
            "cost_delta": (
                polydev["stats"]["total_cost_usd"] -
                baseline["stats"]["total_cost_usd"]
            )
        }
    }

    # Per-task comparison
    baseline_results = {r["instance_id"]: r for r in baseline["results"]}
    polydev_results = {r["instance_id"]: r for r in polydev["results"]}

    task_comparison = []
    for instance_id in baseline_results:
        b = baseline_results[instance_id]
        p = polydev_results.get(instance_id, {})

        task_comparison.append({
            "instance_id": instance_id,
            "baseline_confidence": b.get("final_confidence"),
            "polydev_confidence": p.get("final_confidence"),
            "consulted": p.get("consultation_triggered", False),
            "baseline_patch": bool(b.get("patch_generated")),
            "polydev_patch": bool(p.get("patch_generated"))
        })

    comparison["task_comparison"] = task_comparison

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench sample evaluation"
    )
    parser.add_argument(
        "--tasks", type=int, default=10,
        help="Number of tasks to evaluate (default: 10)"
    )
    parser.add_argument(
        "--mode", choices=["baseline", "polydev", "both"], default="both",
        help="Evaluation mode (default: both)"
    )
    parser.add_argument(
        "--output", type=str, default="results/sample",
        help="Output directory (default: results/sample)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create run-specific subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    tasks = load_swebench_tasks(args.tasks)

    # Run evaluations
    results = {}

    if args.mode in ["baseline", "both"]:
        logger.info("=" * 60)
        logger.info("Running BASELINE evaluation")
        logger.info("=" * 60)

        baseline_dir = run_dir / "baseline"
        baseline_dir.mkdir(exist_ok=True)

        results["baseline"] = run_evaluation(
            tasks, "baseline", baseline_dir, args.verbose
        )

        # Save baseline results
        with open(baseline_dir / "results.json", "w") as f:
            json.dump(results["baseline"], f, indent=2)

    if args.mode in ["polydev", "both"]:
        logger.info("=" * 60)
        logger.info("Running POLYDEV evaluation")
        logger.info("=" * 60)

        polydev_dir = run_dir / "polydev"
        polydev_dir.mkdir(exist_ok=True)

        results["polydev"] = run_evaluation(
            tasks, "polydev", polydev_dir, args.verbose
        )

        # Save polydev results
        with open(polydev_dir / "results.json", "w") as f:
            json.dump(results["polydev"], f, indent=2)

    # Generate comparison if both modes run
    if args.mode == "both":
        logger.info("=" * 60)
        logger.info("Generating comparison")
        logger.info("=" * 60)

        comparison = compare_results(
            results["baseline"],
            results["polydev"]
        )

        with open(run_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("SAMPLE EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Tasks evaluated: {args.tasks}")
        print()
        print(f"BASELINE:")
        print(f"  Patches generated: {comparison['baseline']['successful_patches']}")
        print(f"  Time: {comparison['baseline']['total_time_seconds']:.1f}s")
        print()
        print(f"POLYDEV:")
        print(f"  Patches generated: {comparison['polydev']['successful_patches']}")
        print(f"  Consultations: {comparison['polydev']['consultations']}")
        print(f"  Time: {comparison['polydev']['total_time_seconds']:.1f}s")
        print(f"  Cost: ${comparison['polydev']['total_cost_usd']:.4f}")
        print()
        print(f"IMPROVEMENT:")
        print(f"  +{comparison['improvement']['patches_delta']} patches")
        print("=" * 60)

    logger.info(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
