#!/usr/bin/env python3
"""
Pre-pull all SWE-bench Docker images to avoid rate limiting during evaluation.

This script:
1. Loads the SWE-bench Verified dataset to get all instance IDs
2. Pre-pulls all Docker images from the SWE-bench registry
3. Uses parallel workers with rate limiting
4. Tracks progress and can resume from where it left off

Usage:
    python pre_pull_docker_images.py [--workers 5] [--registry epoch]
"""

import json
import subprocess
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from datasets import load_dataset

# Configuration
OUTPUT_DIR = Path("/tmp/swe_docker_prepull")
PROGRESS_FILE = OUTPUT_DIR / "docker_pull_progress.json"

# Docker registries
REGISTRIES = {
    "epoch": "ghcr.io/epoch-research/swe-bench",  # Epoch AI pre-built (recommended)
    "swebench": "ghcr.io/swe-bench/swe-bench",    # Official SWE-bench
    "dockerhub": "swebench/swe-bench"              # Docker Hub (rate limited)
}

# Thread-safe stats
lock = threading.Lock()
stats = {
    "total": 0,
    "pulled": 0,
    "already_exists": 0,
    "failed": 0,
    "start_time": None
}

def get_image_name(instance_id: str, registry: str) -> str:
    """Generate Docker image name for an instance."""
    # SWE-bench image naming convention
    # Format: <registry>/<repo-name>:<instance-id>
    repo_name = instance_id.split("__")[0].replace("-", "_")

    if registry == "epoch":
        return f"{REGISTRIES['epoch']}/{repo_name}:{instance_id}"
    elif registry == "swebench":
        return f"{REGISTRIES['swebench']}/{repo_name}:{instance_id}"
    else:
        return f"{REGISTRIES['dockerhub']}/{repo_name}:{instance_id}"


def check_image_exists(image_name: str) -> bool:
    """Check if Docker image already exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True,
        text=True
    )
    return bool(result.stdout.strip())


def pull_image(instance_id: str, registry: str) -> dict:
    """Pull a single Docker image."""
    image_name = get_image_name(instance_id, registry)

    result = {
        "instance_id": instance_id,
        "image_name": image_name,
        "status": "unknown",
        "error": None,
        "duration_sec": 0
    }

    start_time = time.time()

    try:
        # Check if already exists
        if check_image_exists(image_name):
            result["status"] = "already_exists"
            with lock:
                stats["already_exists"] += 1
            return result

        # Pull image
        pull = subprocess.run(
            ["docker", "pull", image_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout per image
        )

        if pull.returncode == 0:
            result["status"] = "pulled"
            with lock:
                stats["pulled"] += 1
        else:
            result["status"] = "failed"
            result["error"] = pull.stderr[:500]
            with lock:
                stats["failed"] += 1

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Pull timed out after 600 seconds"
        with lock:
            stats["failed"] += 1
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        with lock:
            stats["failed"] += 1

    result["duration_sec"] = round(time.time() - start_time, 1)
    return result


def load_completed_instances() -> set:
    """Load already completed instances from progress file."""
    if PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text())
            return set(progress.get("completed_instances", []))
        except:
            pass
    return set()


def save_progress(completed_instances: list):
    """Save progress to file."""
    progress = {
        "timestamp": datetime.now().isoformat(),
        "stats": stats.copy(),
        "completed_instances": completed_instances
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def log_progress():
    """Log progress to console."""
    elapsed = time.time() - stats["start_time"] if stats["start_time"] else 0
    completed = stats["pulled"] + stats["already_exists"] + stats["failed"]
    rate = completed / (elapsed / 60) if elapsed > 0 else 0
    remaining = stats["total"] - completed
    eta_mins = remaining / rate if rate > 0 else 0

    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DOCKER PRE-PULL PROGRESS")
    print(f"  Completed: {completed}/{stats['total']} ({100*completed/max(1,stats['total']):.1f}%)")
    print(f"  Pulled: {stats['pulled']} | Already exists: {stats['already_exists']} | Failed: {stats['failed']}")
    print(f"  Rate: {rate:.1f}/min | ETA: {eta_mins:.0f} mins")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-pull SWE-bench Docker images")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--registry", choices=["epoch", "swebench", "dockerhub"],
                       default="epoch", help="Docker registry to use")
    parser.add_argument("--limit", type=int, help="Limit number of images to pull (for testing)")
    args = parser.parse_args()

    print(f"SWE-bench Docker Image Pre-Puller")
    print(f"Registry: {args.registry} ({REGISTRIES[args.registry]})")
    print(f"Workers: {args.workers}")
    print("="*70)

    # Setup
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load dataset
    print("Loading SWE-bench Verified dataset...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    all_instance_ids = [inst["instance_id"] for inst in ds]

    # Check for already completed
    completed = load_completed_instances()
    instance_ids = [iid for iid in all_instance_ids if iid not in completed]

    if args.limit:
        instance_ids = instance_ids[:args.limit]

    stats["total"] = len(instance_ids)
    stats["start_time"] = time.time()

    print(f"Total images to pull: {len(instance_ids)}")
    print(f"Already completed: {len(completed)}")
    print("Starting parallel download...")
    print("="*70)

    completed_list = list(completed)

    # Progress reporter thread
    def progress_reporter():
        while stats["pulled"] + stats["already_exists"] + stats["failed"] < stats["total"]:
            time.sleep(60)
            if stats["pulled"] + stats["already_exists"] + stats["failed"] < stats["total"]:
                log_progress()

    reporter = threading.Thread(target=progress_reporter, daemon=True)
    reporter.start()

    # Pull images in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(pull_image, iid, args.registry): iid for iid in instance_ids}

        for future in as_completed(futures):
            result = future.result()
            instance_id = result["instance_id"]

            completed_list.append(instance_id)

            # Save progress every 10 images
            if len(completed_list) % 10 == 0:
                save_progress(completed_list)

            # Print result
            status_emoji = {
                "pulled": "✓",
                "already_exists": "○",
                "failed": "✗",
                "timeout": "⏱",
                "error": "!"
            }.get(result["status"], "?")

            completed = stats["pulled"] + stats["already_exists"] + stats["failed"]
            print(f"[{completed}/{stats['total']}] {status_emoji} {instance_id[:50]} "
                  f"({result['status']} in {result['duration_sec']}s)")

    # Final save
    save_progress(completed_list)
    log_progress()

    print(f"\n{'='*70}")
    print("DOCKER PRE-PULL COMPLETE!")
    print(f"  Pulled: {stats['pulled']}")
    print(f"  Already existed: {stats['already_exists']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Progress saved: {PROGRESS_FILE}")
    print("="*70)


if __name__ == "__main__":
    main()
