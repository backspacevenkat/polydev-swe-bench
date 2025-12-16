"""
Real-time Progress Monitor

Provides live updates on evaluation progress with rich terminal output.
"""

import time
import json
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class TaskProgress:
    """Progress for a single task."""
    instance_id: str
    status: str = "pending"  # pending, running, completed, failed
    phase: str = ""  # analysis, solution, consultation, patch
    confidence: int = 0
    consulted: bool = False
    patch_generated: bool = False
    duration_ms: int = 0
    error: Optional[str] = None
    start_time: Optional[float] = None


@dataclass
class EvaluationProgress:
    """Overall evaluation progress."""
    mode: str  # baseline or polydev
    total_tasks: int
    tasks: Dict[str, TaskProgress] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)

    @property
    def completed(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == "completed")

    @property
    def failed(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == "failed")

    @property
    def running(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == "running")

    @property
    def patches_generated(self) -> int:
        return sum(1 for t in self.tasks.values() if t.patch_generated)

    @property
    def consultations(self) -> int:
        return sum(1 for t in self.tasks.values() if t.consulted)

    @property
    def avg_confidence(self) -> float:
        completed = [t for t in self.tasks.values() if t.status == "completed"]
        if not completed:
            return 0.0
        return sum(t.confidence for t in completed) / len(completed)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time


class ProgressMonitor:
    """
    Real-time progress monitor for SWE-bench evaluation.

    Features:
    - Live terminal updates (if rich is available)
    - Progress file output for external monitoring
    - Callback support for custom integrations
    """

    def __init__(
        self,
        output_dir: Path,
        use_rich: bool = True,
        update_interval: float = 0.5
    ):
        """
        Initialize monitor.

        Args:
            output_dir: Directory for progress files
            use_rich: Use rich for terminal output
            update_interval: Seconds between display updates
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_rich = use_rich and RICH_AVAILABLE
        self.update_interval = update_interval

        self.progress: Optional[EvaluationProgress] = None
        self.callbacks: List[Callable] = []

        self._live = None
        self._console = Console() if self.use_rich else None

    def start_evaluation(self, mode: str, total_tasks: int) -> None:
        """Start tracking a new evaluation run."""
        self.progress = EvaluationProgress(
            mode=mode,
            total_tasks=total_tasks,
            start_time=time.time()
        )
        self._save_progress()

        if self.use_rich:
            self._console.print(f"\n[bold green]Starting {mode.upper()} evaluation[/bold green]")
            self._console.print(f"Tasks: {total_tasks}")
            self._console.print("")

    def start_task(self, instance_id: str) -> None:
        """Mark a task as started."""
        if not self.progress:
            return

        self.progress.tasks[instance_id] = TaskProgress(
            instance_id=instance_id,
            status="running",
            phase="analysis",
            start_time=time.time()
        )
        self._save_progress()
        self._notify_callbacks("task_started", instance_id)

        if self.use_rich:
            completed = self.progress.completed
            total = self.progress.total_tasks
            self._console.print(
                f"[cyan][{completed + 1}/{total}][/cyan] "
                f"Starting [bold]{instance_id}[/bold]..."
            )

    def update_phase(self, instance_id: str, phase: str) -> None:
        """Update the current phase of a task."""
        if not self.progress or instance_id not in self.progress.tasks:
            return

        self.progress.tasks[instance_id].phase = phase
        self._save_progress()

    def complete_task(
        self,
        instance_id: str,
        confidence: int,
        consulted: bool,
        patch_generated: bool,
        duration_ms: int,
        error: Optional[str] = None
    ) -> None:
        """Mark a task as completed."""
        if not self.progress or instance_id not in self.progress.tasks:
            return

        task = self.progress.tasks[instance_id]
        task.status = "failed" if error else "completed"
        task.confidence = confidence
        task.consulted = consulted
        task.patch_generated = patch_generated
        task.duration_ms = duration_ms
        task.error = error

        self._save_progress()
        self._notify_callbacks("task_completed", instance_id)

        if self.use_rich:
            status_icon = "✓" if not error else "✗"
            status_color = "green" if not error else "red"
            consult_str = " [yellow](consulted)[/yellow]" if consulted else ""

            self._console.print(
                f"  [{status_color}]{status_icon}[/{status_color}] "
                f"Confidence: {confidence}/10{consult_str}, "
                f"Patch: {'Yes' if patch_generated else 'No'}, "
                f"Time: {duration_ms}ms"
            )

    def finish_evaluation(self) -> Dict[str, Any]:
        """Complete the evaluation and return summary."""
        if not self.progress:
            return {}

        summary = {
            "mode": self.progress.mode,
            "total_tasks": self.progress.total_tasks,
            "completed": self.progress.completed,
            "failed": self.progress.failed,
            "patches_generated": self.progress.patches_generated,
            "consultations": self.progress.consultations,
            "avg_confidence": round(self.progress.avg_confidence, 2),
            "total_time_seconds": round(self.progress.elapsed_seconds, 1),
            "timestamp": datetime.now().isoformat()
        }

        # Save final summary
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        if self.use_rich:
            self._print_summary(summary)

        return summary

    def _save_progress(self) -> None:
        """Save current progress to file."""
        if not self.progress:
            return

        progress_file = self.output_dir / "progress.json"

        data = {
            "mode": self.progress.mode,
            "total_tasks": self.progress.total_tasks,
            "completed": self.progress.completed,
            "failed": self.progress.failed,
            "running": self.progress.running,
            "patches_generated": self.progress.patches_generated,
            "consultations": self.progress.consultations,
            "avg_confidence": round(self.progress.avg_confidence, 2),
            "elapsed_seconds": round(self.progress.elapsed_seconds, 1),
            "tasks": {
                tid: {
                    "status": t.status,
                    "phase": t.phase,
                    "confidence": t.confidence,
                    "consulted": t.consulted,
                    "patch_generated": t.patch_generated,
                    "duration_ms": t.duration_ms,
                    "error": t.error
                }
                for tid, t in self.progress.tasks.items()
            },
            "updated_at": datetime.now().isoformat()
        }

        with open(progress_file, "w") as f:
            json.dump(data, f, indent=2)

    def _notify_callbacks(self, event: str, instance_id: str) -> None:
        """Notify registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(event, instance_id, self.progress)
            except Exception:
                pass

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted summary using rich."""
        if not self.use_rich:
            return

        self._console.print("")
        self._console.print("=" * 60)
        self._console.print(
            f"[bold]{summary['mode'].upper()} EVALUATION COMPLETE[/bold]",
            justify="center"
        )
        self._console.print("=" * 60)

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Tasks Completed", f"{summary['completed']}/{summary['total_tasks']}")
        table.add_row("Failed", str(summary['failed']))
        table.add_row("Patches Generated", str(summary['patches_generated']))
        table.add_row("Consultations", str(summary['consultations']))
        table.add_row("Avg Confidence", f"{summary['avg_confidence']}/10")
        table.add_row("Total Time", f"{summary['total_time_seconds']}s")

        self._console.print(table)
        self._console.print("")

    def add_callback(self, callback: Callable) -> None:
        """Add a callback for progress events."""
        self.callbacks.append(callback)


def watch_progress(progress_file: Path, interval: float = 1.0) -> None:
    """
    Watch a progress file and print updates.

    Useful for monitoring from a separate terminal.

    Args:
        progress_file: Path to progress.json
        interval: Seconds between checks
    """
    console = Console() if RICH_AVAILABLE else None
    last_completed = 0

    print(f"Watching {progress_file}...")
    print("Press Ctrl+C to stop\n")

    while True:
        try:
            if progress_file.exists():
                with open(progress_file) as f:
                    data = json.load(f)

                completed = data.get("completed", 0)
                total = data.get("total_tasks", 0)

                if completed != last_completed:
                    last_completed = completed

                    if console:
                        console.print(
                            f"[{data['updated_at']}] "
                            f"[cyan]{completed}/{total}[/cyan] tasks, "
                            f"[green]{data['patches_generated']}[/green] patches, "
                            f"[yellow]{data['consultations']}[/yellow] consultations"
                        )
                    else:
                        print(
                            f"[{data['updated_at']}] "
                            f"{completed}/{total} tasks, "
                            f"{data['patches_generated']} patches, "
                            f"{data['consultations']} consultations"
                        )

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\nStopped watching.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        watch_progress(Path(sys.argv[1]))
    else:
        print("Usage: python monitor.py <progress.json>")
