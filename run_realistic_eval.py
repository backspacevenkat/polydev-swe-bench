#!/usr/bin/env python3
"""
Realistic SWE-bench Evaluation Runner

Uses Claude CLI (Haiku 4.5) in agentic mode to generate patches from scratch.
Tracks all metrics: steps, tokens, time, cost.
Produces output suitable for leaderboard submission.
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import shutil
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Paths
CLAUDE_CLI = "claude"  # Use PATH
REPO_CACHE_DIR = Path("/tmp/swe_bench_repos")
RESULTS_DIR = Path("results/realistic_eval")

# The 20 new instances to evaluate
EVAL_INSTANCES = [
    "django__django-15973",
    "django__django-15572", 
    "django__django-13344",
    "pydata__xarray-7229",
    "django__django-13109",
    "django__django-13658",
    "django__django-13837",
    "django__django-12193",
    "sphinx-doc__sphinx-8595",
    "django__django-11477",
    "django__django-13925",
    "sphinx-doc__sphinx-7590",
    "django__django-15695",
    "matplotlib__matplotlib-25311",
    "scikit-learn__scikit-learn-14496",
    "scikit-learn__scikit-learn-25747",
    "matplotlib__matplotlib-25960",
    "django__django-11490",
    "pytest-dev__pytest-10356",
    "matplotlib__matplotlib-20676",
]


@dataclass
class InstanceMetrics:
    """Metrics for a single instance."""
    instance_id: str
    status: str = "pending"
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    time_seconds: float = 0.0
    patch: str = ""
    error_message: str = ""
    resolved: Optional[bool] = None
    model_used: str = ""


@dataclass 
class EvalResults:
    """Overall evaluation results."""
    model: str = "claude-haiku-4-5-20251001"
    timestamp: str = ""
    total_instances: int = 0
    completed: int = 0
    patches_generated: int = 0
    resolved: int = 0
    failed: int = 0
    errors: int = 0
    total_steps: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_time_seconds: float = 0.0
    avg_steps_per_instance: float = 0.0
    avg_tokens_per_instance: float = 0.0
    avg_cost_per_instance: float = 0.0
    avg_time_per_instance: float = 0.0
    instances: List[Dict] = field(default_factory=list)


class RealisticSWEBenchRunner:
    """
    Runs Claude CLI agent on SWE-bench instances using agentic mode.
    """

    def __init__(
        self,
        model: str = "haiku",
        max_turns: int = 30,
        timeout: int = 600,  # 10 minutes per instance
    ):
        self.model = model
        self.max_turns = max_turns
        self.timeout = timeout
        self.results = EvalResults(model=model)

        # Ensure directories exist
        REPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def load_instance(self, instance_id: str) -> Dict[str, Any]:
        """Load a SWE-bench instance from the dataset."""
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

        for item in ds:
            if item["instance_id"] == instance_id:
                return dict(item)

        raise ValueError(f"Instance {instance_id} not found")

    def setup_repo(self, instance: Dict[str, Any]) -> Path:
        """Clone and setup repository for an instance."""
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        base_commit = instance["base_commit"]

        repo_dir = REPO_CACHE_DIR / instance_id.replace("/", "__")

        if repo_dir.exists():
            shutil.rmtree(repo_dir)

        logger.info(f"Cloning {repo} at {base_commit[:8]}...")

        # Clone with limited depth
        result = subprocess.run(
            ["git", "clone", "--depth", "200", f"https://github.com/{repo}.git", str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            # Try full clone if shallow fails
            subprocess.run(
                ["git", "clone", f"https://github.com/{repo}.git", str(repo_dir)],
                capture_output=True,
                timeout=600
            )

        # Checkout the base commit
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=repo_dir,
            capture_output=True,
            timeout=60
        )

        # Reset to clean state
        subprocess.run(["git", "reset", "--hard", base_commit], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "clean", "-fd"], cwd=repo_dir, capture_output=True)

        return repo_dir

    def build_prompt(self, instance: Dict[str, Any], repo_dir: Path) -> str:
        """Build the task prompt for Claude."""
        return f"""Fix this GitHub issue in the repository at {repo_dir}

## Issue Description
{instance['problem_statement']}

## Instructions
1. Explore the repository to understand the codebase
2. Find the root cause of the issue
3. Make the minimal code changes needed to fix it
4. Do NOT modify any test files
5. When done, run: git diff

IMPORTANT: Only modify source code files. Make focused, minimal changes.
Start by exploring the repository structure."""

    def run_agent(self, instance: Dict[str, Any], repo_dir: Path) -> InstanceMetrics:
        """Run Claude CLI on an instance in agentic mode."""
        instance_id = instance["instance_id"]
        metrics = InstanceMetrics(instance_id=instance_id)
        metrics.status = "running"

        start_time = time.time()
        prompt = self.build_prompt(instance, repo_dir)

        try:
            # Run Claude CLI with agentic capabilities
            # --print makes it non-interactive
            # --dangerously-skip-permissions allows file operations
            # --max-turns limits iterations
            result = subprocess.run(
                [
                    CLAUDE_CLI,
                    "--model", self.model,
                    "--print",
                    "--output-format", "json",
                    "--max-turns", str(self.max_turns),
                    "--dangerously-skip-permissions",
                    "--add-dir", str(repo_dir),
                    "-p", prompt
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=repo_dir,  # Run in repo directory
            )

            metrics.time_seconds = time.time() - start_time

            if result.returncode != 0:
                metrics.status = "error"
                metrics.error_message = result.stderr[:1000]
                logger.error(f"[{instance_id}] CLI error: {result.stderr[:200]}")
                return metrics

            # Parse JSON response
            try:
                output = result.stdout.strip()
                # Handle potential multiple JSON lines
                for line in output.split('\n'):
                    if line.strip().startswith('{'):
                        try:
                            data = json.loads(line.strip())
                            if "usage" in data or "total_cost_usd" in data:
                                # Extract metrics from JSON
                                usage = data.get("usage", {})
                                metrics.input_tokens = usage.get("input_tokens", 0)
                                metrics.output_tokens = usage.get("output_tokens", 0)
                                metrics.cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
                                metrics.cache_read_tokens = usage.get("cache_read_input_tokens", 0)
                                metrics.total_tokens = metrics.input_tokens + metrics.output_tokens
                                metrics.cost_usd = data.get("total_cost_usd", 0.0)
                                metrics.steps = data.get("num_turns", 1)
                                metrics.model_used = self.model
                                break
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"[{instance_id}] Failed to parse metrics: {e}")

            # Extract patch from repository
            patch = self._extract_patch(repo_dir)
            metrics.patch = patch

            if patch and len(patch.strip()) > 20:
                metrics.status = "success"
                logger.info(f"[{instance_id}] Generated patch ({len(patch)} chars)")
            else:
                metrics.status = "failed"
                metrics.error_message = "No changes made"
                logger.warning(f"[{instance_id}] No patch generated")

        except subprocess.TimeoutExpired:
            metrics.status = "timeout"
            metrics.error_message = f"Timed out after {self.timeout}s"
            metrics.time_seconds = self.timeout
            logger.warning(f"[{instance_id}] Timed out")

        except Exception as e:
            metrics.status = "error"
            metrics.error_message = str(e)
            metrics.time_seconds = time.time() - start_time
            logger.error(f"[{instance_id}] Error: {e}")

        return metrics

    def _extract_patch(self, repo_dir: Path) -> str:
        """Extract git diff from repository."""
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout
        except:
            return ""

    def run_evaluation(self, instance_ids: List[str]) -> EvalResults:
        """Run evaluation on a list of instances."""
        self.results.timestamp = datetime.now().isoformat()
        self.results.total_instances = len(instance_ids)

        logger.info(f"="*60)
        logger.info(f"Starting REALISTIC evaluation")
        logger.info(f"Model: {self.model}")
        logger.info(f"Instances: {len(instance_ids)}")
        logger.info(f"Max turns: {self.max_turns}")
        logger.info(f"Timeout: {self.timeout}s")
        logger.info(f"="*60)

        for i, instance_id in enumerate(instance_ids, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{len(instance_ids)}] {instance_id}")
            logger.info(f"{'='*60}")

            try:
                # Load instance
                instance = self.load_instance(instance_id)

                # Setup repository
                repo_dir = self.setup_repo(instance)

                # Run agent
                metrics = self.run_agent(instance, repo_dir)

                # Store results
                self.results.instances.append(asdict(metrics))

                # Update totals
                self.results.total_steps += metrics.steps
                self.results.total_input_tokens += metrics.input_tokens
                self.results.total_output_tokens += metrics.output_tokens
                self.results.total_tokens += metrics.total_tokens
                self.results.total_cost_usd += metrics.cost_usd
                self.results.total_time_seconds += metrics.time_seconds

                if metrics.status == "success":
                    self.results.completed += 1
                    self.results.patches_generated += 1
                elif metrics.status in ["failed", "timeout"]:
                    self.results.failed += 1
                else:
                    self.results.errors += 1

                # Log progress
                logger.info(f"Status: {metrics.status}")
                logger.info(f"Steps: {metrics.steps} | Tokens: {metrics.total_tokens:,} | Cost: ${metrics.cost_usd:.4f} | Time: {metrics.time_seconds:.1f}s")

                # Save intermediate results
                self._save_results()

                # Cleanup repo to save space
                if repo_dir.exists():
                    shutil.rmtree(repo_dir, ignore_errors=True)

            except Exception as e:
                logger.error(f"[{instance_id}] Failed: {e}")
                self.results.errors += 1
                self.results.instances.append({
                    "instance_id": instance_id,
                    "status": "error",
                    "error_message": str(e)
                })
                self._save_results()

        # Calculate averages
        n = max(1, self.results.completed + self.results.failed)
        self.results.avg_steps_per_instance = self.results.total_steps / n
        self.results.avg_tokens_per_instance = self.results.total_tokens / n
        self.results.avg_cost_per_instance = self.results.total_cost_usd / n
        self.results.avg_time_per_instance = self.results.total_time_seconds / n

        # Save final results
        self._save_results()
        self._save_predictions()

        return self.results

    def _save_results(self):
        """Save results to JSON file."""
        results_file = RESULTS_DIR / f"results_{self.model}.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(self.results), f, indent=2)

    def _save_predictions(self):
        """Save predictions in SWE-bench format."""
        predictions_file = RESULTS_DIR / f"predictions_{self.model}.jsonl"
        with open(predictions_file, 'w') as f:
            for inst in self.results.instances:
                if inst.get("patch") and len(inst.get("patch", "").strip()) > 20:
                    pred = {
                        "model_name_or_path": self.model,
                        "instance_id": inst["instance_id"],
                        "model_patch": inst["patch"]
                    }
                    f.write(json.dumps(pred) + "\n")
        logger.info(f"Predictions saved to {predictions_file}")


def main():
    """Run the realistic evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description="Run realistic SWE-bench evaluation")
    parser.add_argument("--model", default="haiku", help="Model to use (haiku, sonnet, opus)")
    parser.add_argument("--max-turns", type=int, default=30, help="Max agent turns")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per instance (seconds)")
    parser.add_argument("--instances", type=int, default=20, help="Number of instances to run")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    args = parser.parse_args()

    runner = RealisticSWEBenchRunner(
        model=args.model,
        max_turns=args.max_turns,
        timeout=args.timeout,
    )

    # Select instances
    instances = EVAL_INSTANCES[args.start:args.start + args.instances]
    
    results = runner.run_evaluation(instances)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {results.model}")
    print(f"Total Instances: {results.total_instances}")
    print(f"Patches Generated: {results.patches_generated}")
    print(f"Failed/Timeout: {results.failed}")
    print(f"Errors: {results.errors}")
    print(f"\nMetrics:")
    print(f"  Total Steps: {results.total_steps}")
    print(f"  Total Tokens: {results.total_tokens:,}")
    print(f"  Total Cost: ${results.total_cost_usd:.2f}")
    print(f"  Total Time: {results.total_time_seconds/60:.1f} minutes")
    print(f"\nAverages per Instance:")
    print(f"  Steps: {results.avg_steps_per_instance:.1f}")
    print(f"  Tokens: {results.avg_tokens_per_instance:,.0f}")
    print(f"  Cost: ${results.avg_cost_per_instance:.4f}")
    print(f"  Time: {results.avg_time_per_instance:.1f}s")
    print("="*60)

    predictions_file = RESULTS_DIR / f"predictions_{results.model}.jsonl"
    if predictions_file.exists():
        print(f"\nPredictions saved to: {predictions_file}")
        print(f"\nTo evaluate, run:")
        print(f"cd /Users/venkat/Documents/polydev-swe-bench && python3 -m swebench.harness.run_evaluation \\")
        print(f"    -p {predictions_file} \\")
        print(f"    -d princeton-nlp/SWE-bench_Verified \\")
        print(f"    --report_dir ./logs/realistic_eval \\")
        print(f"    -t 1800")


if __name__ == "__main__":
    main()
