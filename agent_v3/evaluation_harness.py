"""
Evaluation Harness for Chief Resident Experiments

Runs the 4 experimental configurations and computes:
- Unstuck Rate (primary metric)
- Hallucination Catch Rate
- Trigger Effectiveness
- Statistical Validation (McNemar's test)
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy import stats

from .chief_resident_agent import (
    ChiefResidentAgent,
    ChiefResidentConfig,
    ExperimentConfig
)

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result from running a task with one configuration."""
    instance_id: str
    config: str
    solved: bool
    patch: str
    duration_seconds: float
    metrics: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class ExperimentResult:
    """Results from running all configurations on a task."""
    instance_id: str
    results: Dict[str, TaskResult]

    @property
    def base_alone_solved(self) -> bool:
        return self.results.get("BASE_ALONE", TaskResult("", "", False, "", 0, {})).solved

    @property
    def base_plus_polydev_solved(self) -> bool:
        return self.results.get("BASE_PLUS_POLYDEV_GATED", TaskResult("", "", False, "", 0, {})).solved

    @property
    def was_unstuck(self) -> bool:
        """Task was solved with Polydev after base alone failed."""
        return not self.base_alone_solved and self.base_plus_polydev_solved


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all tasks."""
    total_tasks: int = 0

    # Per-configuration solve rates
    solve_rates: Dict[str, float] = field(default_factory=dict)
    solved_counts: Dict[str, int] = field(default_factory=dict)

    # Unstuck Rate (primary metric)
    unstuck_count: int = 0
    base_failed_count: int = 0
    unstuck_rate: float = 0.0

    # Consultation metrics
    total_consultations: int = 0
    consultations_that_helped: int = 0
    consultation_effectiveness: float = 0.0

    # Trigger breakdown
    trigger_effectiveness: Dict[str, float] = field(default_factory=dict)

    # Statistical tests
    mcnemar_chi2: float = 0.0
    mcnemar_p_value: float = 0.0
    is_significant: bool = False

    # Cost metrics
    avg_cost_base: float = 0.0
    avg_cost_polydev: float = 0.0
    cost_overhead: float = 0.0


class EvaluationHarness:
    """
    Harness for running Chief Resident experiments.

    Usage:
        harness = EvaluationHarness(output_dir="results")
        harness.run_evaluation(tasks, repo_paths)
        metrics = harness.compute_metrics()
    """

    def __init__(
        self,
        output_dir: str = "results",
        configs: Optional[List[ExperimentConfig]] = None,
        n_workers: int = 4
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default to all configs
        self.configs = configs or list(ExperimentConfig)
        self.n_workers = n_workers

        # Results storage
        self.experiment_results: List[ExperimentResult] = []
        self.task_results: Dict[str, Dict[str, TaskResult]] = {}

    def run_evaluation(
        self,
        tasks: List[Dict[str, Any]],
        repo_paths: Dict[str, str],
        resume: bool = True
    ) -> List[ExperimentResult]:
        """
        Run evaluation on all tasks with all configurations.

        Args:
            tasks: List of SWE-bench task dicts
            repo_paths: Mapping of instance_id -> repo_path
            resume: If True, skip already-completed tasks

        Returns:
            List of ExperimentResult
        """
        logger.info(f"Running evaluation on {len(tasks)} tasks with {len(self.configs)} configs")

        for task in tasks:
            instance_id = task.get("instance_id", "unknown")
            repo_path = repo_paths.get(instance_id)

            if not repo_path:
                logger.warning(f"No repo path for {instance_id}, skipping")
                continue

            # Check if already completed
            if resume and self._is_completed(instance_id):
                logger.info(f"Skipping {instance_id} (already completed)")
                self._load_result(instance_id)
                continue

            # Run all configurations for this task
            result = self._run_task_all_configs(task, repo_path)
            self.experiment_results.append(result)
            self._save_result(result)

        return self.experiment_results

    def _run_task_all_configs(
        self,
        task: Dict[str, Any],
        repo_path: str
    ) -> ExperimentResult:
        """Run all configurations on a single task."""
        instance_id = task.get("instance_id", "unknown")
        logger.info(f"Running task: {instance_id}")

        results = {}

        for config in self.configs:
            logger.info(f"  Config: {config.name}")
            start_time = time.time()

            try:
                # Create agent with this config
                agent_config = ChiefResidentConfig(experiment_config=config)
                agent = ChiefResidentAgent(config=agent_config)

                # Run the task
                solved, patch, metrics = agent.solve(task, repo_path)

                results[config.name] = TaskResult(
                    instance_id=instance_id,
                    config=config.name,
                    solved=solved,
                    patch=patch,
                    duration_seconds=time.time() - start_time,
                    metrics=metrics
                )

            except Exception as e:
                logger.error(f"Error running {config.name} on {instance_id}: {e}")
                results[config.name] = TaskResult(
                    instance_id=instance_id,
                    config=config.name,
                    solved=False,
                    patch="",
                    duration_seconds=time.time() - start_time,
                    metrics={},
                    error=str(e)
                )

            # Reset repo between configs
            self._reset_repo(repo_path)

        return ExperimentResult(instance_id=instance_id, results=results)

    def _reset_repo(self, repo_path: str):
        """Reset repo to clean state between experiments."""
        import subprocess
        try:
            subprocess.run(
                ["git", "checkout", "."],
                cwd=repo_path,
                capture_output=True,
                timeout=30
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=repo_path,
                capture_output=True,
                timeout=30
            )
        except Exception as e:
            logger.warning(f"Failed to reset repo: {e}")

    def compute_metrics(self) -> AggregateMetrics:
        """Compute aggregate metrics across all tasks."""
        metrics = AggregateMetrics()
        metrics.total_tasks = len(self.experiment_results)

        # Count solved per config
        config_solved = {c.name: 0 for c in self.configs}
        config_costs = {c.name: [] for c in self.configs}

        # For McNemar's test
        base_alone_results = []
        base_polydev_results = []

        # Consultation tracking
        total_consultations = 0
        consultations_helped = 0
        trigger_counts = {}
        trigger_helped = {}

        for exp_result in self.experiment_results:
            # Per-config solve rates
            for config_name, task_result in exp_result.results.items():
                if task_result.solved:
                    config_solved[config_name] += 1

                # Track costs
                cost = task_result.metrics.get("cost_usd", 0)
                if cost:
                    config_costs[config_name].append(cost)

            # Unstuck rate
            base_alone = exp_result.results.get("BASE_ALONE")
            base_polydev = exp_result.results.get("BASE_PLUS_POLYDEV_GATED")

            if base_alone and base_polydev:
                base_alone_results.append(base_alone.solved)
                base_polydev_results.append(base_polydev.solved)

                if not base_alone.solved:
                    metrics.base_failed_count += 1
                    if base_polydev.solved:
                        metrics.unstuck_count += 1

            # Consultation metrics
            if base_polydev:
                consult_count = base_polydev.metrics.get("total_consultations", 0)
                consult_helped = base_polydev.metrics.get("consultations_that_helped", 0)
                total_consultations += consult_count
                consultations_helped += consult_helped

                # Trigger breakdown
                breakdown = base_polydev.metrics.get("consultation_breakdown", {})
                for trigger, helped in breakdown.items():
                    trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
                    if helped:
                        trigger_helped[trigger] = trigger_helped.get(trigger, 0) + 1

        # Compute solve rates
        for config_name, solved_count in config_solved.items():
            metrics.solved_counts[config_name] = solved_count
            metrics.solve_rates[config_name] = (
                solved_count / metrics.total_tasks * 100
                if metrics.total_tasks > 0 else 0
            )

        # Compute unstuck rate
        if metrics.base_failed_count > 0:
            metrics.unstuck_rate = metrics.unstuck_count / metrics.base_failed_count * 100

        # Compute consultation effectiveness
        metrics.total_consultations = total_consultations
        metrics.consultations_that_helped = consultations_helped
        if total_consultations > 0:
            metrics.consultation_effectiveness = consultations_helped / total_consultations * 100

        # Compute trigger effectiveness
        for trigger, count in trigger_counts.items():
            helped = trigger_helped.get(trigger, 0)
            metrics.trigger_effectiveness[trigger] = helped / count * 100 if count > 0 else 0

        # McNemar's test
        if len(base_alone_results) >= 10:
            metrics.mcnemar_chi2, metrics.mcnemar_p_value = self._mcnemar_test(
                base_alone_results, base_polydev_results
            )
            metrics.is_significant = metrics.mcnemar_p_value < 0.05

        # Cost metrics
        if config_costs.get("BASE_ALONE"):
            metrics.avg_cost_base = np.mean(config_costs["BASE_ALONE"])
        if config_costs.get("BASE_PLUS_POLYDEV_GATED"):
            metrics.avg_cost_polydev = np.mean(config_costs["BASE_PLUS_POLYDEV_GATED"])
        if metrics.avg_cost_base > 0:
            metrics.cost_overhead = (
                (metrics.avg_cost_polydev - metrics.avg_cost_base) / metrics.avg_cost_base * 100
            )

        return metrics

    def _mcnemar_test(
        self,
        base_alone: List[bool],
        base_polydev: List[bool]
    ) -> Tuple[float, float]:
        """
        McNemar's test for paired comparisons.

        Tests if the proportion of "only A solved" differs from "only B solved".
        """
        # b = base alone solved, polydev failed
        # c = base alone failed, polydev solved
        b = sum(1 for a, p in zip(base_alone, base_polydev) if a and not p)
        c = sum(1 for a, p in zip(base_alone, base_polydev) if not a and p)

        if b + c == 0:
            return 0.0, 1.0

        # McNemar's chi-squared with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        return chi2, p_value

    def generate_report(self, metrics: AggregateMetrics) -> str:
        """Generate a text report of the results."""
        lines = [
            "=" * 60,
            "CHIEF RESIDENT EVALUATION REPORT",
            "=" * 60,
            "",
            f"Total Tasks: {metrics.total_tasks}",
            "",
            "SOLVE RATES BY CONFIGURATION:",
            "-" * 40,
        ]

        for config_name in sorted(metrics.solve_rates.keys()):
            rate = metrics.solve_rates[config_name]
            count = metrics.solved_counts[config_name]
            lines.append(f"  {config_name}: {rate:.1f}% ({count}/{metrics.total_tasks})")

        lines.extend([
            "",
            "PRIMARY METRIC: UNSTUCK RATE",
            "-" * 40,
            f"  Tasks where Base Alone failed: {metrics.base_failed_count}",
            f"  Tasks solved after Polydev consultation: {metrics.unstuck_count}",
            f"  UNSTUCK RATE: {metrics.unstuck_rate:.1f}%",
            "",
            "CONSULTATION METRICS:",
            "-" * 40,
            f"  Total consultations: {metrics.total_consultations}",
            f"  Consultations that helped: {metrics.consultations_that_helped}",
            f"  Consultation effectiveness: {metrics.consultation_effectiveness:.1f}%",
            "",
            "TRIGGER EFFECTIVENESS:",
            "-" * 40,
        ])

        for trigger, effectiveness in sorted(metrics.trigger_effectiveness.items()):
            lines.append(f"  {trigger}: {effectiveness:.1f}%")

        lines.extend([
            "",
            "STATISTICAL VALIDATION:",
            "-" * 40,
            f"  McNemar's chi-squared: {metrics.mcnemar_chi2:.2f}",
            f"  p-value: {metrics.mcnemar_p_value:.4f}",
            f"  Significant (p < 0.05): {'YES' if metrics.is_significant else 'NO'}",
            "",
            "COST ANALYSIS:",
            "-" * 40,
            f"  Average cost (Base Alone): ${metrics.avg_cost_base:.4f}",
            f"  Average cost (Base + Polydev): ${metrics.avg_cost_polydev:.4f}",
            f"  Cost overhead: {metrics.cost_overhead:.1f}%",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def _is_completed(self, instance_id: str) -> bool:
        """Check if task results already exist."""
        result_file = self.output_dir / f"{instance_id}.json"
        return result_file.exists()

    def _load_result(self, instance_id: str):
        """Load existing result from disk."""
        result_file = self.output_dir / f"{instance_id}.json"
        try:
            with open(result_file) as f:
                data = json.load(f)

            results = {}
            for config_name, result_data in data.get("results", {}).items():
                results[config_name] = TaskResult(**result_data)

            exp_result = ExperimentResult(
                instance_id=instance_id,
                results=results
            )
            self.experiment_results.append(exp_result)
        except Exception as e:
            logger.warning(f"Failed to load result for {instance_id}: {e}")

    def _save_result(self, result: ExperimentResult):
        """Save result to disk."""
        result_file = self.output_dir / f"{result.instance_id}.json"
        try:
            data = {
                "instance_id": result.instance_id,
                "results": {
                    config_name: asdict(task_result)
                    for config_name, task_result in result.results.items()
                }
            }
            with open(result_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save result for {result.instance_id}: {e}")

    def save_metrics(self, metrics: AggregateMetrics, filename: str = "metrics.json"):
        """Save aggregate metrics to disk."""
        metrics_file = self.output_dir / filename
        with open(metrics_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

    def save_report(self, report: str, filename: str = "report.txt"):
        """Save text report to disk."""
        report_file = self.output_dir / filename
        with open(report_file, "w") as f:
            f.write(report)


# === CLI Entry Point ===

def main():
    """Run evaluation from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Chief Resident Evaluation Harness")
    parser.add_argument("--tasks", required=True, help="Path to tasks JSON file")
    parser.add_argument("--repos", required=True, help="Path to repo paths JSON file")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    # Load tasks and repo paths
    with open(args.tasks) as f:
        tasks = json.load(f)
    with open(args.repos) as f:
        repo_paths = json.load(f)

    # Run evaluation
    harness = EvaluationHarness(
        output_dir=args.output,
        n_workers=args.workers
    )
    harness.run_evaluation(tasks, repo_paths)

    # Compute and save metrics
    metrics = harness.compute_metrics()
    harness.save_metrics(metrics)

    # Generate and save report
    report = harness.generate_report(metrics)
    harness.save_report(report)
    print(report)


if __name__ == "__main__":
    main()
