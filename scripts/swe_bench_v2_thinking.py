#!/usr/bin/env python3
"""
SWE-bench v2 runner with Extended Thinking - targeting 73%+ resolution.
Based on Anthropic's EXACT methodology:
- 128K thinking budget via MAX_THINKING_TOKENS
- Prompt: "Use tools >100 times, implement tests first"
- Claude Haiku 4.5
"""

import json
import subprocess
import shutil
import time
import threading
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from datasets import load_dataset

# Configuration - matching Anthropic's benchmark
MODEL = "claude-haiku-4-5-20251001"
THINKING_TOKENS = 128000  # 128K thinking budget per Anthropic methodology
MAX_TURNS = 150  # Each turn has more thinking, so fewer turns needed
MAX_WORKERS = 10  # Reduced due to higher cost per instance
OUTPUT_DIR = Path("/tmp/swe_v2_thinking")
PREDICTIONS_FILE = OUTPUT_DIR / "predictions.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
LOGS_DIR = OUTPUT_DIR / "logs"
GIT_RETRY_COUNT = 3
GIT_RETRY_DELAY = 10

# Global counters
lock = threading.Lock()
stats = {
    "total": 0,
    "completed": 0,
    "patches": 0,
    "errors": 0,
    "total_cost": 0.0,
    "total_turns": 0,
    "start_time": None
}

# ANTHROPIC'S EXACT PROMPT from benchmark methodology:
# "You should use tools as much as possible, ideally more than 100 times.
#  You should also implement your own tests first before attempting the problem."
TASK_TEMPLATE = """<issue_description>
{problem}
</issue_description>

<instructions>
The repository is at: {repo_dir}

Your task: Fix the issue described above by modifying the source code.

IMPORTANT: You should use tools as much as possible, ideally more than 100 times. You should also implement your own tests first before attempting the problem.

## Recommended Workflow:
1. **EXPLORE EXTENSIVELY**: Use bash commands (grep, find, ls, cat) to thoroughly explore the codebase. Don't stop after finding one file - explore related files too.

2. **WRITE A TEST FIRST**: Before fixing anything, create a simple test script that reproduces the bug. Run it to confirm the bug exists.

3. **LOCATE THE ROOT CAUSE**: Trace through the code to find exactly where the bug originates. Read multiple files if needed.

4. **IMPLEMENT THE FIX**: Make MINIMAL, targeted changes to fix the root cause.

5. **VERIFY WITH YOUR TEST**: Run your test again to confirm the fix works.

## Rules:
- MODIFY: Source code files only
- DO NOT MODIFY: Test files, setup.py, pyproject.toml, configuration files
- Make MINIMAL changes - the smallest fix that solves the issue
- Use tools extensively - explore before you fix

When you've completed the fix, say: TASK_COMPLETE
</instructions>"""


def log_progress():
    """Log progress to file and console."""
    elapsed = time.time() - stats["start_time"] if stats["start_time"] else 0
    rate = stats["completed"] / (elapsed / 60) if elapsed > 0 else 0
    eta_mins = (stats["total"] - stats["completed"]) / rate if rate > 0 else 0
    avg_turns = stats["total_turns"] / max(1, stats["completed"])
    patch_rate = 100 * stats["patches"] / max(1, stats["completed"])

    progress = {
        "timestamp": datetime.now().isoformat(),
        "completed": stats["completed"],
        "total": stats["total"],
        "patches": stats["patches"],
        "patch_rate_pct": round(patch_rate, 1),
        "errors": stats["errors"],
        "total_cost": round(stats["total_cost"], 2),
        "avg_turns": round(avg_turns, 1),
        "elapsed_mins": round(elapsed / 60, 1),
        "rate_per_min": round(rate, 2),
        "eta_mins": round(eta_mins, 1)
    }

    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))

    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] PROGRESS UPDATE (Extended Thinking)")
    print(f"  Completed: {stats['completed']}/{stats['total']} ({100*stats['completed']/max(1,stats['total']):.1f}%)")
    print(f"  Patches: {stats['patches']} ({patch_rate:.1f}%) | Errors: {stats['errors']}")
    print(f"  Cost: ${stats['total_cost']:.2f} | Avg turns: {avg_turns:.1f}")
    print(f"  Rate: {rate:.1f}/min | ETA: {eta_mins:.0f} mins")
    print(f"{'='*70}\n")


def progress_reporter():
    """Report progress every 5 minutes."""
    while stats["completed"] < stats["total"]:
        time.sleep(300)
        if stats["completed"] < stats["total"]:
            log_progress()


def git_clone_with_retry(repo, repo_dir, base_commit):
    """Clone and checkout with retries."""
    for attempt in range(GIT_RETRY_COUNT):
        try:
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            repo_dir.parent.mkdir(parents=True, exist_ok=True)

            clone = subprocess.run(
                ["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(repo_dir)],
                capture_output=True, timeout=600,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            )
            if clone.returncode != 0:
                raise Exception(f"Clone failed: {clone.stderr.decode()[:200]}")

            checkout = subprocess.run(
                ["git", "checkout", "--quiet", base_commit],
                cwd=repo_dir, capture_output=True, timeout=120
            )
            if checkout.returncode != 0:
                subprocess.run(
                    ["git", "fetch", "--quiet", "--depth=100", "origin"],
                    cwd=repo_dir, capture_output=True, timeout=300,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
                )
                checkout = subprocess.run(
                    ["git", "checkout", "--quiet", base_commit],
                    cwd=repo_dir, capture_output=True, timeout=120
                )
                if checkout.returncode != 0:
                    raise Exception(f"Checkout failed: {checkout.stderr.decode()[:200]}")

            return True

        except Exception as e:
            if attempt < GIT_RETRY_COUNT - 1:
                time.sleep(GIT_RETRY_DELAY)
            else:
                raise e

    return False


def run_instance(instance: dict) -> dict:
    """Run Claude CLI with extended thinking on a single instance."""
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    problem = instance["problem_statement"]

    repo_dir = Path(f"/tmp/swe_repos/{instance_id}")
    log_file = LOGS_DIR / f"{instance_id}.log"

    result = {
        "instance_id": instance_id,
        "model_name_or_path": MODEL,
        "model_patch": "",
        "turns": 0,
        "cost": 0,
        "error": None
    }

    try:
        # Clone with retry
        git_clone_with_retry(repo, repo_dir, base_commit)

        # Build prompt
        prompt = TASK_TEMPLATE.format(repo_dir=str(repo_dir), problem=problem)

        # Run Claude with extended thinking via environment variable
        env = os.environ.copy()
        env["MAX_THINKING_TOKENS"] = str(THINKING_TOKENS)

        claude = subprocess.run(
            [
                "claude",
                "--model", MODEL,
                "--print",
                "--output-format", "json",
                "--dangerously-skip-permissions",
                "--max-turns", str(MAX_TURNS),
                "--add-dir", str(repo_dir),
                "-p", prompt
            ],
            capture_output=True,
            text=True,
            timeout=2400,  # 40 min timeout (extended thinking takes longer)
            cwd=repo_dir,
            env=env
        )

        # Save log
        with open(log_file, "w") as f:
            f.write(f"STDOUT:\n{claude.stdout}\n\nSTDERR:\n{claude.stderr}")

        # Parse output
        try:
            output = json.loads(claude.stdout)
            result["turns"] = output.get("num_turns", 0)
            result["cost"] = output.get("total_cost_usd", 0)
        except:
            pass

        # Get patch
        diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True, cwd=repo_dir)
        result["model_patch"] = diff.stdout

        # Update stats
        with lock:
            stats["completed"] += 1
            stats["total_cost"] += result["cost"]
            stats["total_turns"] += result["turns"]
            if result["model_patch"]:
                stats["patches"] += 1

    except Exception as e:
        result["error"] = str(e)[:500]
        with open(log_file, "w") as f:
            f.write(f"ERROR:\n{result['error']}")
        with lock:
            stats["completed"] += 1
            stats["errors"] += 1

    finally:
        shutil.rmtree(repo_dir, ignore_errors=True)

    return result


def main():
    print(f"SWE-bench v2 with Extended Thinking - Targeting 73%+ Resolution")
    print(f"Model: {MODEL}")
    print(f"Thinking Budget: {THINKING_TOKENS:,} tokens")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Max turns: {MAX_TURNS}")
    print("="*70)
    print("Using Anthropic's exact prompt:")
    print('  "You should use tools as much as possible, ideally more than 100 times.')
    print('   You should also implement your own tests first before attempting the problem."')
    print("="*70)

    # Setup
    OUTPUT_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    if PREDICTIONS_FILE.exists():
        PREDICTIONS_FILE.unlink()

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    instances = list(ds)

    stats["total"] = len(instances)
    stats["start_time"] = time.time()

    print(f"Total instances: {len(instances)}")
    print("Starting parallel execution with extended thinking...")
    print("="*70)

    # Start progress reporter
    reporter = threading.Thread(target=progress_reporter, daemon=True)
    reporter.start()

    # Run in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_instance, inst): inst for inst in instances}

        for future in as_completed(futures):
            result = future.result()

            # Save prediction
            with lock:
                with open(PREDICTIONS_FILE, "a") as f:
                    pred = {
                        "instance_id": result["instance_id"],
                        "model_name_or_path": result["model_name_or_path"],
                        "model_patch": result["model_patch"]
                    }
                    f.write(json.dumps(pred) + "\n")

            # Print result
            status = "✓" if result.get("model_patch") else "✗"
            err = f" [{result.get('error', '')[:40]}]" if result.get("error") else ""
            patch_rate = 100 * stats["patches"] / max(1, stats["completed"])
            print(f"[{stats['completed']}/{stats['total']}] {status} {result['instance_id'][:45]} "
                  f"(turns:{result.get('turns', 0)} ${result.get('cost', 0):.2f}) "
                  f"[Patches: {stats['patches']} ({patch_rate:.0f}%)]{err}")

    # Final summary
    log_progress()
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"Predictions: {PREDICTIONS_FILE}")
    print(f"Patches: {stats['patches']}/{stats['total']} ({100*stats['patches']/stats['total']:.1f}%)")
    print(f"Cost: ${stats['total_cost']:.2f}")
    print(f"\nTo evaluate locally:")
    print(f"python -m swebench.harness.run_evaluation \\")
    print(f"  --predictions_path {PREDICTIONS_FILE} \\")
    print(f"  --dataset princeton-nlp/SWE-bench_Verified \\")
    print(f"  --run_id haiku-v2-thinking")


if __name__ == "__main__":
    main()
