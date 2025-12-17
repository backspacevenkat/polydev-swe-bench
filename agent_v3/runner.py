"""
SWE-bench Task Runner v3 - PARALLEL EXECUTION

Key improvements:
- 15-worker parallel execution
- Staggered start to avoid thundering herd
- Pre-cloning of repositories
- Better error handling
"""

import os
import json
import time
import shutil
import subprocess
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

from datasets import load_dataset

from .agent import PolydevAgent, AgentConfig, Submitted, LimitsExceeded
from .environment import LocalEnvironment, EnvironmentConfig
from .model import ClaudeModel, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of running a single task."""
    instance_id: str
    status: str  # "submitted", "limits_exceeded", "error"
    patch: str = ""
    steps: int = 0
    consultations: int = 0
    duration_ms: int = 0
    error: Optional[str] = None
    token_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Configuration for a run - IMPROVED FOR PARALLEL."""
    output_dir: str
    workspace_dir: str = "/tmp/polydev-swe-bench-v3"
    max_workers: int = 15  # 15 parallel workers
    stagger_delay: float = 2.0  # 2 second delay between worker starts
    task_timeout: int = 3600  # 60 minutes per task (increased)
    cleanup_repos: bool = True
    pre_clone: bool = True  # Pre-clone repos for speed


class SWEBenchRunner:
    """
    Runs Polydev agent on SWE-bench tasks with PARALLEL EXECUTION.
    """

    def __init__(self, run_config: RunConfig, agent_config: Optional[AgentConfig] = None):
        self.run_config = run_config
        self.agent_config = agent_config or AgentConfig()

        # Ensure directories exist
        Path(run_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(run_config.workspace_dir).mkdir(parents=True, exist_ok=True)

        # Shared repo cache
        self.repo_cache_dir = Path(run_config.workspace_dir) / "repo_cache"
        self.repo_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        self.dataset = None
        self.dataset_dict = {}

    def load_dataset(self, dataset_name: str = "princeton-nlp/SWE-bench_Verified", split: str = "test"):
        """Load SWE-bench dataset."""
        logger.info(f"Loading dataset: {dataset_name} ({split})")
        self.dataset = load_dataset(dataset_name, split=split)

        # Index by instance_id for fast lookup
        self.dataset_dict = {item["instance_id"]: item for item in self.dataset}
        logger.info(f"Loaded {len(self.dataset_dict)} tasks")

    def run_tasks(
        self,
        instance_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[TaskResult]:
        """
        Run agent on specified tasks with PARALLEL EXECUTION.
        """
        if not self.dataset_dict:
            self.load_dataset()

        # Select tasks
        if instance_ids:
            tasks = [self.dataset_dict[id] for id in instance_ids if id in self.dataset_dict]
        else:
            tasks = list(self.dataset_dict.values())

        if limit:
            tasks = tasks[:limit]

        logger.info(f"Running {len(tasks)} tasks with {self.run_config.max_workers} workers")

        # Pre-clone unique repos if enabled
        if self.run_config.pre_clone:
            self._pre_clone_repos(tasks)

        results = []

        if self.run_config.max_workers == 1:
            # Sequential execution
            for task in tasks:
                result = self.run_single_task(task)
                results.append(result)
                self._save_result(result)
        else:
            # PARALLEL execution with staggered starts
            results = self._run_parallel_staggered(tasks)

        # Save summary
        self._save_summary(results)

        return results

    def _pre_clone_repos(self, tasks: List[Dict[str, Any]]):
        """Pre-clone unique repositories to cache."""
        unique_repos = set(task["repo"] for task in tasks)
        logger.info(f"Pre-cloning {len(unique_repos)} unique repositories...")

        def clone_repo(repo: str):
            repo_path = self.repo_cache_dir / repo.replace("/", "_")
            if repo_path.exists():
                logger.debug(f"Repo {repo} already cached")
                return

            repo_url = f"https://github.com/{repo}.git"
            try:
                subprocess.run(
                    ["git", "clone", "--quiet", "--bare", repo_url, str(repo_path)],
                    check=True,
                    capture_output=True,
                    timeout=300,
                )
                logger.info(f"Cached {repo}")
            except Exception as e:
                logger.warning(f"Failed to cache {repo}: {e}")

        # Clone in parallel (4 threads for git)
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(clone_repo, unique_repos))

        logger.info("Pre-cloning complete")

    def _run_parallel_staggered(self, tasks: List[Dict[str, Any]]) -> List[TaskResult]:
        """Run tasks in parallel with staggered starts."""
        results = []
        pending_futures = {}

        # Use ProcessPoolExecutor for isolation
        # But ThreadPoolExecutor works better with subprocess calls
        with ThreadPoolExecutor(max_workers=self.run_config.max_workers) as executor:
            # Submit tasks with staggered delays
            for i, task in enumerate(tasks):
                # Calculate stagger delay
                batch_num = i // 5  # Start 5 at a time
                delay = batch_num * self.run_config.stagger_delay

                # Submit with delay wrapper
                future = executor.submit(self._run_with_delay, task, delay)
                pending_futures[future] = task

                logger.info(f"Queued task {i+1}/{len(tasks)}: {task['instance_id']} (delay: {delay:.1f}s)")

            # Collect results as they complete
            for future in as_completed(pending_futures):
                task = pending_futures[future]
                try:
                    result = future.result(timeout=self.run_config.task_timeout)
                    results.append(result)
                    self._save_result(result)
                    logger.info(f"Completed: {result.instance_id} ({result.status}, {result.steps} steps)")
                except Exception as e:
                    logger.error(f"Task {task['instance_id']} failed: {e}")
                    result = TaskResult(
                        instance_id=task["instance_id"],
                        status="error",
                        error=str(e),
                    )
                    results.append(result)
                    self._save_result(result)

        return results

    def _run_with_delay(self, task: Dict[str, Any], delay: float) -> TaskResult:
        """Run a task with an initial delay (for staggering)."""
        if delay > 0:
            time.sleep(delay)
        return self.run_single_task(task)

    def run_single_task(self, task: Dict[str, Any]) -> TaskResult:
        """Run agent on a single task."""
        instance_id = task["instance_id"]
        logger.info(f"Starting task: {instance_id}")

        start_time = time.time()
        repo_path = None

        try:
            # Clone repository (from cache if available)
            repo_path = self._clone_repo(task)

            # Create agent with improved config
            model = ClaudeModel(ModelConfig(
                consultation_enabled=self.agent_config.consultation_enabled,
                timeout=300,  # 5 min per LLM call
            ))
            env = LocalEnvironment(EnvironmentConfig(
                cwd=repo_path,
                timeout=120,  # 2 min per command
            ))
            agent = PolydevAgent(model=model, env=env, config=self.agent_config)

            # Run agent
            exit_status, output = agent.run(task, repo_path)

            # Collect results
            duration_ms = int((time.time() - start_time) * 1000)
            stats = agent.get_stats()

            if exit_status == "Submitted":
                # Get the patch
                patch = output if output else self._get_patch(repo_path)

                return TaskResult(
                    instance_id=instance_id,
                    status="submitted",
                    patch=patch,
                    steps=stats["step_count"],
                    consultations=stats["consultation_count"],
                    duration_ms=duration_ms,
                    token_usage=stats["model_stats"]["token_usage"],
                )
            else:
                return TaskResult(
                    instance_id=instance_id,
                    status="limits_exceeded",
                    steps=stats["step_count"],
                    consultations=stats["consultation_count"],
                    duration_ms=duration_ms,
                    error=output,
                    token_usage=stats["model_stats"]["token_usage"],
                )

        except Exception as e:
            logger.error(f"Error running {instance_id}: {e}")
            import traceback
            traceback.print_exc()
            return TaskResult(
                instance_id=instance_id,
                status="error",
                duration_ms=int((time.time() - start_time) * 1000),
                error=str(e),
            )

        finally:
            # Cleanup
            if repo_path and self.run_config.cleanup_repos:
                self._cleanup_repo(repo_path)

    def _clone_repo(self, task: Dict[str, Any]) -> str:
        """Clone repository at specific commit (using cache if available)."""
        instance_id = task["instance_id"]
        repo = task["repo"]
        base_commit = task["base_commit"]

        # Create unique workspace
        repo_path = os.path.join(
            self.run_config.workspace_dir,
            "workspaces",
            instance_id.replace("/", "_").replace("__", "_")
        )

        # Clean up if exists
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        os.makedirs(repo_path, exist_ok=True)

        # Check if we have a cached clone
        cached_repo = self.repo_cache_dir / repo.replace("/", "_")

        if cached_repo.exists():
            # Clone from cache (much faster)
            logger.debug(f"Cloning {repo} from cache at {base_commit[:8]}...")
            subprocess.run(
                ["git", "clone", "--quiet", str(cached_repo), repo_path],
                check=True,
                capture_output=True,
            )
        else:
            # Clone from GitHub
            repo_url = f"https://github.com/{repo}.git"
            logger.info(f"Cloning {repo} from GitHub at {base_commit[:8]}...")
            subprocess.run(
                ["git", "clone", "--quiet", repo_url, repo_path],
                check=True,
                capture_output=True,
            )

        # Checkout specific commit
        subprocess.run(
            ["git", "checkout", "--quiet", base_commit],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        logger.debug(f"Repository ready at: {repo_path}")
        return repo_path

    def _get_patch(self, repo_path: str) -> str:
        """Get git diff from repository."""
        result = subprocess.run(
            ["git", "diff"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def _cleanup_repo(self, repo_path: str):
        """Remove cloned repository."""
        try:
            shutil.rmtree(repo_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup {repo_path}: {e}")

    def _save_result(self, result: TaskResult):
        """Save individual task result."""
        result_path = os.path.join(
            self.run_config.output_dir,
            "tasks",
            f"{result.instance_id.replace('/', '_')}.json"
        )

        Path(result_path).parent.mkdir(parents=True, exist_ok=True)

        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        logger.debug(f"Saved result: {result.instance_id} ({result.status})")

    def _save_summary(self, results: List[TaskResult]):
        """Save run summary with COMPREHENSIVE ANALYTICS."""
        submitted = [r for r in results if r.status == "submitted"]
        errors = [r for r in results if r.status == "error"]
        limits = [r for r in results if r.status == "limits_exceeded"]

        # Calculate totals
        total_tokens = {
            "claude": {"input": 0, "output": 0},
            "gpt": {"input": 0, "output": 0},
            "gemini": {"input": 0, "output": 0}
        }
        for r in results:
            for provider, usage in r.token_usage.items():
                if provider in total_tokens:
                    total_tokens[provider]["input"] += usage.get("input", 0)
                    total_tokens[provider]["output"] += usage.get("output", 0)

        # Calculate costs (approximate pricing)
        costs = {
            "claude": {
                "input_cost": total_tokens["claude"]["input"] * 0.003 / 1000,  # $3/1M input
                "output_cost": total_tokens["claude"]["output"] * 0.015 / 1000,  # $15/1M output
            },
            "gpt": {
                "input_cost": total_tokens["gpt"]["input"] * 0.005 / 1000,  # $5/1M input
                "output_cost": total_tokens["gpt"]["output"] * 0.015 / 1000,  # $15/1M output
            },
            "gemini": {
                "input_cost": total_tokens["gemini"]["input"] * 0.00025 / 1000,  # $0.25/1M input
                "output_cost": total_tokens["gemini"]["output"] * 0.0005 / 1000,  # $0.50/1M output
            },
        }
        total_cost = sum(c["input_cost"] + c["output_cost"] for c in costs.values())

        # Time analytics
        total_duration_ms = sum(r.duration_ms for r in results)
        avg_duration_ms = total_duration_ms / len(results) if results else 0
        
        summary = {
            "total_tasks": len(results),
            "submitted": len(submitted),
            "submitted_with_patch": len([r for r in submitted if r.patch]),
            "errors": len(errors),
            "limits_exceeded": len(limits),
            "submission_rate": len(submitted) / len(results) if results else 0,
            "patch_rate": len([r for r in submitted if r.patch]) / len(results) if results else 0,
            "total_consultations": sum(r.consultations for r in results),
            "avg_steps": sum(r.steps for r in results) / len(results) if results else 0,
            "max_steps": max((r.steps for r in results), default=0),
            "min_steps": min((r.steps for r in results if r.steps > 0), default=0),
            
            # Time analytics
            "total_duration_ms": total_duration_ms,
            "avg_duration_ms": avg_duration_ms,
            "avg_duration_min": avg_duration_ms / 60000,
            "wall_clock_estimate_min": avg_duration_ms * len(results) / self.run_config.max_workers / 60000,
            
            # Token analytics
            "token_usage": {
                provider: {
                    "input": usage["input"],
                    "output": usage["output"],
                    "total": usage["input"] + usage["output"],
                }
                for provider, usage in total_tokens.items()
            },
            "total_tokens": sum(u["input"] + u["output"] for u in total_tokens.values()),
            "avg_tokens_per_task": sum(u["input"] + u["output"] for u in total_tokens.values()) / len(results) if results else 0,
            
            # Cost analytics
            "cost_breakdown": costs,
            "total_cost_usd": total_cost,
            "avg_cost_per_task_usd": total_cost / len(results) if results else 0,
            
            "config": {
                "max_workers": self.run_config.max_workers,
                "step_limit": self.agent_config.step_limit,
                "consultation_enabled": self.agent_config.consultation_enabled,
            }
        }

        summary_path = os.path.join(self.run_config.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary: {len(submitted)}/{len(results)} submitted ({summary['submission_rate']:.1%})")
        logger.info(f"Total tokens: {summary['total_tokens']:,} | Cost: ${total_cost:.2f}")

        # Save predictions in SWE-bench format
        predictions = []
        for r in results:
            if r.patch:
                predictions.append({
                    "instance_id": r.instance_id,
                    "model_patch": r.patch,
                    "model_name_or_path": "polydev-agent-v3",
                })

        predictions_path = os.path.join(self.run_config.output_dir, "predictions.json")
        with open(predictions_path, "w") as f:
            json.dump(predictions, f, indent=2)

        # Also save as JSONL for sb-cli
        predictions_jsonl_path = os.path.join(self.run_config.output_dir, "predictions.jsonl")
        with open(predictions_jsonl_path, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        logger.info(f"Saved {len(predictions)} predictions")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Polydev SWE-bench Agent v3 (Parallel)")
    parser.add_argument("--output", "-o", default="results/v3_run", help="Output directory")
    parser.add_argument("--tasks", "-t", nargs="*", help="Specific task IDs to run")
    parser.add_argument("--limit", "-n", type=int, help="Max tasks to run")
    parser.add_argument("--workers", "-w", type=int, default=15, help="Parallel workers (default: 15)")
    parser.add_argument("--no-consultation", action="store_true", help="Disable Polydev consultation")
    parser.add_argument("--step-limit", type=int, default=100, help="Max steps per task (default: 100)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create configs
    run_config = RunConfig(
        output_dir=args.output,
        max_workers=args.workers,
    )

    agent_config = AgentConfig(
        consultation_enabled=not args.no_consultation,
        step_limit=args.step_limit,
    )

    # Run
    runner = SWEBenchRunner(run_config, agent_config)
    results = runner.run_tasks(instance_ids=args.tasks, limit=args.limit)

    # Print summary
    submitted = len([r for r in results if r.status == "submitted"])
    print(f"\nCompleted: {submitted}/{len(results)} submitted ({submitted/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
