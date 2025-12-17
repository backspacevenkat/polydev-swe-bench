"""
SWE-bench Task Runner

Handles:
- Loading tasks from SWE-bench dataset
- Cloning repositories at specific commits
- Running the agent on tasks
- Collecting results and patches
"""

import os
import json
import time
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """Configuration for a run."""
    output_dir: str
    workspace_dir: str = "/tmp/polydev-swe-bench"
    max_workers: int = 1  # Sequential by default for stability
    task_timeout: int = 1800  # 30 minutes per task
    cleanup_repos: bool = True


class SWEBenchRunner:
    """
    Runs Polydev agent on SWE-bench tasks.

    Usage:
        runner = SWEBenchRunner(config)
        results = runner.run_tasks(task_ids)
    """

    def __init__(self, run_config: RunConfig, agent_config: Optional[AgentConfig] = None):
        self.run_config = run_config
        self.agent_config = agent_config or AgentConfig()

        # Ensure directories exist
        Path(run_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(run_config.workspace_dir).mkdir(parents=True, exist_ok=True)

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
        Run agent on specified tasks.

        Args:
            instance_ids: Specific task IDs to run (None = all)
            limit: Max number of tasks to run

        Returns:
            List of TaskResult objects
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

        logger.info(f"Running {len(tasks)} tasks")

        results = []

        if self.run_config.max_workers == 1:
            # Sequential execution
            for task in tasks:
                result = self.run_single_task(task)
                results.append(result)
                self._save_result(result)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.run_config.max_workers) as executor:
                futures = {executor.submit(self.run_single_task, task): task for task in tasks}

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        self._save_result(result)
                    except Exception as e:
                        task = futures[future]
                        logger.error(f"Task {task['instance_id']} failed: {e}")

        # Save summary
        self._save_summary(results)

        return results

    def run_single_task(self, task: Dict[str, Any]) -> TaskResult:
        """Run agent on a single task."""
        instance_id = task["instance_id"]
        logger.info(f"Starting task: {instance_id}")

        start_time = time.time()
        repo_path = None

        try:
            # Clone repository
            repo_path = self._clone_repo(task)

            # Create agent
            model = ClaudeModel(ModelConfig(consultation_enabled=self.agent_config.consultation_enabled))
            env = LocalEnvironment(EnvironmentConfig(cwd=repo_path, timeout=120))
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
        """Clone repository at specific commit."""
        instance_id = task["instance_id"]
        repo = task["repo"]
        base_commit = task["base_commit"]

        # Create unique workspace
        repo_path = os.path.join(
            self.run_config.workspace_dir,
            instance_id.replace("/", "_").replace("__", "_")
        )

        # Clean up if exists
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        # Clone
        repo_url = f"https://github.com/{repo}.git"
        logger.info(f"Cloning {repo} at {base_commit[:8]}...")

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

        logger.info(f"Repository ready at: {repo_path}")
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

        logger.info(f"Saved result: {result.instance_id} ({result.status})")

    def _save_summary(self, results: List[TaskResult]):
        """Save run summary."""
        submitted = [r for r in results if r.status == "submitted"]
        errors = [r for r in results if r.status == "error"]
        limits = [r for r in results if r.status == "limits_exceeded"]

        # Calculate totals
        total_tokens = {"claude": {"input": 0, "output": 0}, "gpt": {"input": 0, "output": 0}, "gemini": {"input": 0, "output": 0}}
        for r in results:
            for provider, usage in r.token_usage.items():
                if provider in total_tokens:
                    total_tokens[provider]["input"] += usage.get("input", 0)
                    total_tokens[provider]["output"] += usage.get("output", 0)

        summary = {
            "total_tasks": len(results),
            "submitted": len(submitted),
            "errors": len(errors),
            "limits_exceeded": len(limits),
            "submission_rate": len(submitted) / len(results) if results else 0,
            "total_consultations": sum(r.consultations for r in results),
            "avg_steps": sum(r.steps for r in results) / len(results) if results else 0,
            "total_duration_ms": sum(r.duration_ms for r in results),
            "token_usage": total_tokens,
        }

        summary_path = os.path.join(self.run_config.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary: {len(submitted)}/{len(results)} submitted ({summary['submission_rate']:.1%})")

        # Save predictions in SWE-bench format
        predictions = []
        for r in results:
            if r.patch:
                predictions.append({
                    "instance_id": r.instance_id,
                    "model_patch": r.patch,
                    "model_name_or_path": "polydev-agent-v2",
                })

        predictions_path = os.path.join(self.run_config.output_dir, "predictions.json")
        with open(predictions_path, "w") as f:
            json.dump(predictions, f, indent=2)

        logger.info(f"Saved {len(predictions)} predictions")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Polydev SWE-bench Agent v2")
    parser.add_argument("--output", "-o", default="results/v2_run", help="Output directory")
    parser.add_argument("--tasks", "-t", nargs="*", help="Specific task IDs to run")
    parser.add_argument("--limit", "-n", type=int, help="Max tasks to run")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Parallel workers")
    parser.add_argument("--no-consultation", action="store_true", help="Disable Polydev consultation")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create configs
    run_config = RunConfig(
        output_dir=args.output,
        max_workers=args.workers,
    )

    agent_config = AgentConfig(
        consultation_enabled=not args.no_consultation,
    )

    # Run
    runner = SWEBenchRunner(run_config, agent_config)
    results = runner.run_tasks(instance_ids=args.tasks, limit=args.limit)

    # Print summary
    submitted = len([r for r in results if r.status == "submitted"])
    print(f"\nCompleted: {submitted}/{len(results)} submitted ({submitted/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
