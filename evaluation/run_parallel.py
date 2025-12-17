#!/usr/bin/env python3
"""
Parallel SWE-bench Evaluation Runner

Runs multiple tasks concurrently for faster evaluation.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import PolydevAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_tasks(num_tasks: int) -> List[Dict[str, Any]]:
    """Load SWE-bench tasks from local cache or HuggingFace."""
    data_dir = Path(__file__).parent.parent / "data" / "swe-bench-verified"
    tasks_file = data_dir / "tasks.json"

    if tasks_file.exists():
        with open(tasks_file) as f:
            all_tasks = json.load(f)
    else:
        logger.info("Loading from HuggingFace...")
        from datasets import load_dataset
        hf_dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        all_tasks = [dict(task) for task in hf_dataset]
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(tasks_file, 'w') as f:
            json.dump(all_tasks, f, indent=2)

    return all_tasks[:num_tasks]


def run_single_task(
    task: Dict[str, Any],
    task_idx: int,
    output_dir: Path,
    stagger_delay: float
) -> Dict[str, Any]:
    """Run a single task with optional delay."""
    instance_id = task['instance_id']

    # Stagger start
    if stagger_delay > 0:
        time.sleep(task_idx * stagger_delay)

    logger.info(f"[{task_idx+1}] Starting {instance_id}")

    # Create task-specific output directory
    task_dir = output_dir / "tasks" / instance_id.replace("/", "__")
    task_dir.mkdir(parents=True, exist_ok=True)

    try:
        agent = PolydevAgent(
            consultation_enabled=True,
            confidence_threshold=8,
            log_dir=task_dir,
            mock_mode=False
        )

        result = agent.solve_task(task)

        result_data = {
            "instance_id": result.instance_id,
            "configuration": result.configuration,
            "initial_confidence": result.initial_confidence,
            "final_confidence": result.final_confidence,
            "consultation_triggered": result.consultation_triggered,
            "patch_generated": result.patch_generated,
            "duration_ms": result.total_duration_ms,
            "cost_usd": result.cost_usd,
            "error": result.error,
            "patch": result.patch,
            # Token usage per provider
            "token_usage": result.token_usage
        }

        # Save individual result
        with open(task_dir / "result.json", "w") as f:
            json.dump(result_data, f, indent=2)

        status = "✓" if result.patch_generated else "✗"
        tokens = result.token_usage
        logger.info(
            f"[{task_idx+1}] {status} {instance_id}: "
            f"conf={result.final_confidence}, "
            f"consulted={result.consultation_triggered}, "
            f"time={result.total_duration_ms}ms, "
            f"tokens=[claude:{tokens['claude']['input']}+{tokens['claude']['output']}, "
            f"gpt:{tokens['gpt']['input']}+{tokens['gpt']['output']}, "
            f"gemini:{tokens['gemini']['input']}+{tokens['gemini']['output']}]"
        )

        return result_data

    except Exception as e:
        logger.error(f"[{task_idx+1}] Error on {instance_id}: {e}")
        error_data = {
            "instance_id": instance_id,
            "configuration": "polydev",
            "error": str(e),
            "patch_generated": False,
            "token_usage": {
                "claude": {"input": 0, "output": 0},
                "gpt": {"input": 0, "output": 0},
                "gemini": {"input": 0, "output": 0}
            }
        }
        with open(task_dir / "result.json", "w") as f:
            json.dump(error_data, f, indent=2)
        return error_data


def aggregate_token_usage(results: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Aggregate token usage across all results."""
    totals = {
        "claude": {"input": 0, "output": 0, "total": 0},
        "gpt": {"input": 0, "output": 0, "total": 0},
        "gemini": {"input": 0, "output": 0, "total": 0}
    }
    
    for r in results:
        usage = r.get("token_usage", {})
        for provider in ["claude", "gpt", "gemini"]:
            if provider in usage:
                totals[provider]["input"] += usage[provider].get("input", 0)
                totals[provider]["output"] += usage[provider].get("output", 0)
                totals[provider]["total"] = totals[provider]["input"] + totals[provider]["output"]
    
    return totals


def update_progress(output_dir: Path, results: List[Dict], total: int, start_time: float):
    """Update progress file with current state."""
    completed = [r for r in results if r.get("patch_generated") is not None]
    successful = [r for r in completed if r.get("patch_generated")]
    consulted = [r for r in completed if r.get("consultation_triggered")]
    confidences = [r.get("final_confidence", 0) for r in completed if r.get("final_confidence")]
    
    # Aggregate token usage
    token_totals = aggregate_token_usage(completed)

    progress = {
        "mode": "polydev_parallel",
        "total_tasks": total,
        "completed": len(completed),
        "successful_patches": len(successful),
        "consultations": len(consulted),
        "avg_confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "token_usage": token_totals,
        "results": completed,
        "updated_at": datetime.now().isoformat()
    }

    with open(output_dir / "progress.json", "w") as f:
        json.dump(progress, f, indent=2)


def run_parallel_evaluation(
    tasks: List[Dict[str, Any]],
    output_dir: Path,
    max_workers: int = 10,
    stagger_seconds: float = 20.0
) -> Dict[str, Any]:
    """Run evaluation on all tasks in parallel."""

    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    logger.info(f"Starting parallel evaluation of {len(tasks)} tasks")
    logger.info(f"Max workers: {max_workers}, Stagger: {stagger_seconds}s")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_task, task, idx, output_dir, stagger_seconds
            ): task['instance_id']
            for idx, task in enumerate(tasks)
        }

        for future in as_completed(futures):
            instance_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                update_progress(output_dir, results, len(tasks), start_time)
            except Exception as e:
                logger.error(f"Future failed for {instance_id}: {e}")
                results.append({
                    "instance_id": instance_id,
                    "error": str(e),
                    "patch_generated": False,
                    "token_usage": {
                        "claude": {"input": 0, "output": 0},
                        "gpt": {"input": 0, "output": 0},
                        "gemini": {"input": 0, "output": 0}
                    }
                })

    # Final summary
    total_time = time.time() - start_time
    successful = [r for r in results if r.get("patch_generated")]
    consulted = [r for r in results if r.get("consultation_triggered")]
    confidences = [r.get("final_confidence", 0) for r in results if r.get("final_confidence")]
    
    # Aggregate token usage
    token_totals = aggregate_token_usage(results)

    summary = {
        "mode": "polydev_parallel",
        "total_tasks": len(tasks),
        "successful_patches": len(successful),
        "failed": len(tasks) - len(successful),
        "consultations": len(consulted),
        "consultation_rate": round(len(consulted) / len(tasks) * 100, 1) if tasks else 0,
        "avg_confidence": round(sum(confidences) / len(confidences), 2) if confidences else 0,
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_task": round(total_time / len(tasks), 1) if tasks else 0,
        "parallelism_speedup": round((sum(r.get("duration_ms", 0) for r in results) / 1000) / total_time, 1) if total_time > 0 else 0,
        "token_usage": token_totals,
        "timestamp": datetime.now().isoformat()
    }

    # Save final results
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    return summary


def format_tokens(num: int) -> str:
    """Format token count with K/M suffix."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)


def print_summary(summary: Dict[str, Any]):
    """Print formatted summary with token usage."""
    print("\n" + "=" * 70)
    print("PARALLEL EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Tasks: {summary['total_tasks']}")
    print(f"Successful patches: {summary['successful_patches']} ({100*summary['successful_patches']/summary['total_tasks']:.0f}%)")
    print(f"Failed: {summary['failed']}")
    print(f"Consultation rate: {summary['consultation_rate']}%")
    print(f"Avg confidence: {summary['avg_confidence']}/10")
    print(f"Total time: {summary['total_time_seconds']}s ({summary['total_time_seconds']/60:.1f} min)")
    print(f"Parallelism speedup: {summary['parallelism_speedup']}x")
    
    # Token usage breakdown
    print("\n" + "-" * 70)
    print("TOKEN USAGE BY PROVIDER")
    print("-" * 70)
    tokens = summary.get('token_usage', {})
    
    print(f"{'Provider':<12} {'Input':>12} {'Output':>12} {'Total':>12}")
    print("-" * 50)
    
    grand_total = 0
    for provider in ['claude', 'gpt', 'gemini']:
        if provider in tokens:
            inp = tokens[provider].get('input', 0)
            out = tokens[provider].get('output', 0)
            total = inp + out
            grand_total += total
            print(f"{provider.upper():<12} {format_tokens(inp):>12} {format_tokens(out):>12} {format_tokens(total):>12}")
    
    print("-" * 50)
    print(f"{'TOTAL':<12} {'':<12} {'':<12} {format_tokens(grand_total):>12}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Parallel SWE-bench evaluation")
    parser.add_argument("--tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--workers", type=int, default=10, help="Max parallel workers")
    parser.add_argument("--stagger", type=float, default=20.0, help="Seconds between task starts")
    parser.add_argument("--output", type=str, default="results/parallel", help="Output directory")

    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"run_{timestamp}"

    # Load and run
    tasks = load_tasks(args.tasks)
    summary = run_parallel_evaluation(
        tasks, output_dir,
        max_workers=args.workers,
        stagger_seconds=args.stagger
    )

    print_summary(summary)
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
