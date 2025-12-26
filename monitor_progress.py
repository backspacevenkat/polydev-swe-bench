#!/usr/bin/env python3
"""Monitor SWE-bench evaluation progress for both baseline and polydev runs."""

import json
from pathlib import Path
from datetime import datetime

def load_progress(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def load_metrics(path):
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries

def main():
    base_dir = Path("/Users/venkat/Documents/polydev-swe-bench")

    print("=" * 70)
    print(f"SWE-BENCH FULL 500-INSTANCE EVALUATION PROGRESS")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Baseline progress
    baseline_progress = load_progress(base_dir / "results/baseline/progress.json")
    baseline_metrics = load_metrics(base_dir / "results/baseline/metrics.jsonl")

    baseline_completed = len(baseline_progress.get("completed", []))
    baseline_failed = len(baseline_progress.get("failed", []))
    baseline_total = baseline_completed + baseline_failed
    baseline_cost = sum(m['metrics']['cost_usd'] for m in baseline_metrics)

    print(f"\n📊 BASELINE (Claude Haiku 4.5)")
    print(f"   Completed: {baseline_completed}/500 ({baseline_completed/5:.1f}%)")
    print(f"   Failed: {baseline_failed}")
    print(f"   Cost so far: ${baseline_cost:.2f}")
    if baseline_metrics:
        avg_time = sum(m['metrics']['duration_sec'] for m in baseline_metrics) / len(baseline_metrics)
        remaining = 500 - baseline_total
        eta_hours = (remaining * avg_time) / 3600 / 8  # 8 workers
        print(f"   Avg time/instance: {avg_time:.0f}s")
        print(f"   ETA: ~{eta_hours:.1f} hours")

    # Polydev progress
    polydev_progress = load_progress(base_dir / "results/polydev/progress.json")
    polydev_metrics = load_metrics(base_dir / "results/polydev/metrics.jsonl")

    polydev_completed = len(polydev_progress.get("completed", []))
    polydev_failed = len(polydev_progress.get("failed", []))
    polydev_total = polydev_completed + polydev_failed
    polydev_cost = sum(m['metrics']['cost_usd'] for m in polydev_metrics)
    polydev_consultation_cost = sum(m['metrics'].get('polydev_cost', 0) for m in polydev_metrics)

    print(f"\n📊 POLYDEV (Multi-Model Consultation)")
    print(f"   Completed: {polydev_completed}/500 ({polydev_completed/5:.1f}%)")
    print(f"   Failed: {polydev_failed}")
    print(f"   Claude cost: ${polydev_cost:.2f}")
    print(f"   Polydev cost: ${polydev_consultation_cost:.2f}")
    print(f"   Total cost: ${polydev_cost + polydev_consultation_cost:.2f}")
    if polydev_metrics:
        avg_time = sum(m['metrics']['duration_sec'] for m in polydev_metrics) / len(polydev_metrics)
        remaining = 500 - polydev_total
        eta_hours = (remaining * avg_time) / 3600 / 8  # 8 workers
        print(f"   Avg time/instance: {avg_time:.0f}s")
        print(f"   ETA: ~{eta_hours:.1f} hours")

    # Combined stats
    total_cost = baseline_cost + polydev_cost + polydev_consultation_cost
    print(f"\n💰 TOTAL COST SO FAR: ${total_cost:.2f}")
    print(f"   Estimated final: ~${total_cost * 500 / max(1, baseline_total + polydev_total) * 2:.2f}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
