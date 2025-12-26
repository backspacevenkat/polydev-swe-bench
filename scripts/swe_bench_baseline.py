#!/usr/bin/env python3
"""
SWE-bench Baseline Runner - Exact Anthropic Methodology
=========================================================

Produces LEADERBOARD-COMPATIBLE output for SWE-bench submission.

Features:
- Exact Anthropic methodology (128K thinking, their prompt)
- Full metrics tracking for paper (tokens, time, steps, cost)
- Checkpointing and resume capability
- Test mode (--test N for N samples)
- Leaderboard-compatible JSONL output
- RETRY LOGIC for empty patches (ensures 100% patch generation)

Usage:
    # Test on 20 samples with 5 workers
    python swe_bench_baseline.py --test 20 --workers 5

    # Full run
    python swe_bench_baseline.py --workers 10

    # Resume interrupted run
    python swe_bench_baseline.py --resume
"""

import json
import subprocess
import shutil
import time
import threading
import argparse
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from datasets import load_dataset

# =============================================================================
# CONFIGURATION - Matching Anthropic's EXACT methodology
# =============================================================================
MODEL = "claude-haiku-4-5-20251001"
THINKING_TOKENS = 128000  # 128K thinking budget (Anthropic's setting)
MAX_TURNS = 250  # Allow more turns for complex issues
TIMEOUT = 3000  # 50 min timeout per instance
MAX_RETRIES = 2  # Retry empty patches up to 2 times

# Output directories
BASE_OUTPUT_DIR = Path("/Users/venkat/Documents/polydev-swe-bench/results/baseline")
PREDICTIONS_FILE = BASE_OUTPUT_DIR / "all_preds.jsonl"
METRICS_FILE = BASE_OUTPUT_DIR / "metrics.jsonl"
PROGRESS_FILE = BASE_OUTPUT_DIR / "progress.json"
LOGS_DIR = BASE_OUTPUT_DIR / "logs"
TRAJS_DIR = BASE_OUTPUT_DIR / "trajs"

# Git settings
GIT_RETRY_COUNT = 3
GIT_RETRY_DELAY = 10

# =============================================================================
# ANTHROPIC'S EXACT PROMPT (verbatim from their methodology)
# =============================================================================
ANTHROPIC_PROMPT_ADDITION = """You should use tools as much as possible, ideally more than 100 times. You should also implement your own tests first before attempting the problem."""

TASK_TEMPLATE = """<issue_description>
{problem}
</issue_description>

<instructions>
The repository is cloned at: {repo_dir}

Your task: Fix the issue described above by modifying the source code.

IMPORTANT: {anthropic_prompt}

## Recommended Workflow:
1. **EXPLORE EXTENSIVELY**: Use bash commands (grep, find, ls, cat) to thoroughly explore the codebase. Don't stop after finding one file - explore related files too.

2. **WRITE A TEST FIRST**: Before fixing anything, create a simple test script that reproduces the bug. Run it to confirm the bug exists.

3. **LOCATE THE ROOT CAUSE**: Trace through the code to find exactly where the bug originates. Read multiple files if needed.

4. **IMPLEMENT THE FIX**: Make MINIMAL, targeted changes to fix the root cause. USE THE EDIT TOOL to make changes - do not just describe what to change.

5. **VERIFY WITH YOUR TEST**: Run your test again to confirm the fix works.

## CRITICAL Rules:
- MODIFY: Source code files only - USE THE EDIT TOOL to make actual changes
- DO NOT MODIFY: Test files, setup.py, pyproject.toml, configuration files
- Make MINIMAL changes - the smallest fix that solves the issue
- Use tools extensively - explore before you fix
- ACTUALLY EDIT FILES - do not just describe the changes, use the Edit tool to make them

When you've completed the fix AND verified it works, say: TASK_COMPLETE
</instructions>"""

# =============================================================================
# THREAD-SAFE STATISTICS
# =============================================================================
lock = threading.Lock()
stats = {
    "run_id": "",
    "model": MODEL,
    "thinking_tokens": THINKING_TOKENS,
    "total": 0,
    "completed": 0,
    "patches_generated": 0,
    "errors": 0,
    "total_cost": 0.0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_turns": 0,
    "total_time_sec": 0,
    "start_time": None,
    "instances": {}
}


def log_progress():
    """Log progress to file and console."""
    elapsed = time.time() - stats["start_time"] if stats["start_time"] else 0
    rate = stats["completed"] / (elapsed / 60) if elapsed > 0 else 0
    eta_mins = (stats["total"] - stats["completed"]) / rate if rate > 0 else 0
    avg_turns = stats["total_turns"] / max(1, stats["completed"])
    avg_time = stats["total_time_sec"] / max(1, stats["completed"])
    patch_rate = 100 * stats["patches_generated"] / max(1, stats["completed"])

    progress = {
        "timestamp": datetime.now().isoformat(),
        "run_id": stats["run_id"],
        "model": MODEL,
        "thinking_tokens": THINKING_TOKENS,
        "completed": stats["completed"],
        "total": stats["total"],
        "patches_generated": stats["patches_generated"],
        "patch_rate_pct": round(patch_rate, 1),
        "errors": stats["errors"],
        "total_cost_usd": round(stats["total_cost"], 4),
        "total_input_tokens": stats["total_input_tokens"],
        "total_output_tokens": stats["total_output_tokens"],
        "avg_turns": round(avg_turns, 1),
        "avg_time_sec": round(avg_time, 1),
        "elapsed_mins": round(elapsed / 60, 1),
        "rate_per_min": round(rate, 2),
        "eta_mins": round(eta_mins, 1)
    }

    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))

    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] BASELINE RUN PROGRESS")
    print(f"  Run ID: {stats['run_id']}")
    print(f"  Completed: {stats['completed']}/{stats['total']} ({100*stats['completed']/max(1,stats['total']):.1f}%)")
    print(f"  Patches: {stats['patches_generated']} ({patch_rate:.1f}%) | Errors: {stats['errors']}")
    print(f"  Cost: ${stats['total_cost']:.4f} | Tokens: {stats['total_input_tokens']:,} in / {stats['total_output_tokens']:,} out")
    print(f"  Avg: {avg_turns:.1f} turns, {avg_time:.0f}s per instance")
    print(f"  Rate: {rate:.2f}/min | ETA: {eta_mins:.0f} mins")
    print(f"{'='*80}\n")


def progress_reporter():
    """Report progress every 2 minutes."""
    while stats["completed"] < stats["total"]:
        time.sleep(120)
        if stats["completed"] < stats["total"]:
            log_progress()


def git_clone_with_retry(repo: str, repo_dir: Path, base_commit: str) -> bool:
    """Clone and checkout with retries. Uses shallow clone for speed."""
    for attempt in range(GIT_RETRY_COUNT):
        try:
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            repo_dir.parent.mkdir(parents=True, exist_ok=True)

            # Shallow clone (faster for large repos)
            clone = subprocess.run(
                ["git", "clone", "--quiet", "--depth=1", f"https://github.com/{repo}.git", str(repo_dir)],
                capture_output=True, timeout=300,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            )
            if clone.returncode != 0:
                raise Exception(f"Clone failed: {clone.stderr.decode()[:200]}")

            # Fetch the specific commit we need
            fetch = subprocess.run(
                ["git", "fetch", "--quiet", "--depth=1", "origin", base_commit],
                cwd=repo_dir, capture_output=True, timeout=300,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
            )
            if fetch.returncode != 0:
                # If specific commit fetch fails, try fetching more history
                subprocess.run(
                    ["git", "fetch", "--quiet", "--unshallow"],
                    cwd=repo_dir, capture_output=True, timeout=600,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
                )

            # Checkout specific commit
            checkout = subprocess.run(
                ["git", "checkout", "--quiet", base_commit],
                cwd=repo_dir, capture_output=True, timeout=120
            )
            if checkout.returncode != 0:
                raise Exception(f"Checkout failed: {checkout.stderr.decode()[:200]}")

            return True

        except Exception as e:
            if attempt < GIT_RETRY_COUNT - 1:
                time.sleep(GIT_RETRY_DELAY * (attempt + 1))
            else:
                raise e

    return False


def parse_claude_output(stdout: str, stderr: str) -> dict:
    """Parse Claude CLI output to extract metrics."""
    result = {
        "turns": 0,
        "cost": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "duration_sec": 0,
        "duration_api_sec": 0
    }

    try:
        # Try parsing as JSON first
        output = json.loads(stdout)
        result["turns"] = output.get("num_turns", 0)
        result["cost"] = output.get("total_cost_usd", 0)
        result["input_tokens"] = output.get("input_tokens", 0)
        result["output_tokens"] = output.get("output_tokens", 0)
        result["duration_sec"] = output.get("duration_seconds", 0)
        result["duration_api_sec"] = output.get("duration_api_seconds", 0)
    except json.JSONDecodeError:
        # Fallback: parse from text output
        # Look for patterns like "Cost: $0.0315" or "Turns: 39"
        cost_match = re.search(r'\$(\d+\.?\d*)', stdout + stderr)
        if cost_match:
            result["cost"] = float(cost_match.group(1))

        turns_match = re.search(r'(\d+)\s*turns?', stdout + stderr, re.IGNORECASE)
        if turns_match:
            result["turns"] = int(turns_match.group(1))

    return result


def run_instance(instance: dict) -> dict:
    """Run Claude CLI with extended thinking on a single instance.
    
    Includes RETRY LOGIC: If no patch is generated, retries up to MAX_RETRIES times.
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    problem = instance["problem_statement"]

    print(f"\n🔄 Starting: {instance_id} (repo: {repo})", flush=True)
    print(f"   📥 Cloning repo...", flush=True)

    repo_dir = Path(f"/tmp/swe_repos/{instance_id}")
    log_file = LOGS_DIR / f"{instance_id}.log"
    traj_file = TRAJS_DIR / f"{instance_id}.json"

    start_time = time.time()

    # Result structure for metrics tracking
    result = {
        "instance_id": instance_id,
        "model_name_or_path": MODEL,
        "model_patch": "",
        "success": False,
        "error": None,
        "retries": 0,
        "metrics": {
            "turns": 0,
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "duration_sec": 0,
            "git_clone_sec": 0,
            "claude_sec": 0,
            "retries": 0
        }
    }

    # Retry loop for empty patches
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Clone repository (fresh clone for each retry)
            clone_start = time.time()
            git_clone_with_retry(repo, repo_dir, base_commit)
            clone_sec = round(time.time() - clone_start, 1)
            result["metrics"]["git_clone_sec"] = clone_sec
            print(f"   ✅ Cloned in {clone_sec}s", flush=True)

            # Build prompt - add retry context if this is a retry
            retry_context = ""
            if attempt > 0:
                retry_context = f"""

IMPORTANT: This is retry attempt {attempt + 1}. The previous attempt did NOT produce a working patch.
You MUST use the Edit tool to actually modify files. Do not just describe changes - make them!
After making changes, verify with 'git diff' that your changes are saved."""

            prompt = TASK_TEMPLATE.format(
                repo_dir=str(repo_dir),
                problem=problem,
                anthropic_prompt=ANTHROPIC_PROMPT_ADDITION
            ) + retry_context

            # Run Claude with extended thinking
            print(f"   🤖 Starting Claude...", flush=True)
            env = os.environ.copy()
            env["MAX_THINKING_TOKENS"] = str(THINKING_TOKENS)

            claude_start = time.time()
            claude = subprocess.run(
                [
                    "claude",
                    "--model", MODEL,
                    "--print",
                    "--output-format", "json",
                    "--dangerously-skip-permissions",
                    "--max-turns", str(MAX_TURNS),
                    "-p", prompt
                ],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                cwd=repo_dir,
                env=env
            )
            result["metrics"]["claude_sec"] += round(time.time() - claude_start, 1)

            # Save log (append for retries)
            log_mode = "w" if attempt == 0 else "a"
            with open(log_file, log_mode) as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== ATTEMPT {attempt + 1} ===\n")
                f.write(f"=== INSTANCE: {instance_id} ===\n")
                f.write(f"=== TIMESTAMP: {datetime.now().isoformat()} ===\n")
                f.write(f"=== REPO: {repo} @ {base_commit} ===\n\n")
                f.write(f"=== STDOUT ===\n{claude.stdout}\n\n")
                f.write(f"=== STDERR ===\n{claude.stderr}\n")

            # Parse output for metrics (accumulate across retries)
            parsed = parse_claude_output(claude.stdout, claude.stderr)
            result["metrics"]["turns"] += parsed["turns"]
            result["metrics"]["cost_usd"] += parsed["cost"]
            result["metrics"]["input_tokens"] += parsed["input_tokens"]
            result["metrics"]["output_tokens"] += parsed["output_tokens"]

            # Get patch via git diff
            diff = subprocess.run(
                ["git", "diff", "HEAD"],
                capture_output=True,
                text=True,
                cwd=repo_dir
            )

            patch = diff.stdout.strip()
            # Ensure patch ends with newline (required for proper git apply)
            if patch and not patch.endswith("\n"):
                patch += "\n"

            # Check if we got a valid patch
            if patch:
                result["model_patch"] = patch
                result["success"] = True
                result["retries"] = attempt
                result["metrics"]["retries"] = attempt
                break  # Success! Exit retry loop
            else:
                # No patch generated
                if attempt < MAX_RETRIES:
                    print(f"  ⚠️  {instance_id}: Empty patch on attempt {attempt + 1}, retrying...")
                    # Clean up for retry
                    shutil.rmtree(repo_dir, ignore_errors=True)
                    time.sleep(5)  # Brief pause before retry
                else:
                    # All retries exhausted
                    result["error"] = f"No patch generated after {MAX_RETRIES + 1} attempts"
                    result["retries"] = attempt
                    result["metrics"]["retries"] = attempt

        except subprocess.TimeoutExpired:
            result["error"] = f"Timeout after {TIMEOUT} seconds (attempt {attempt + 1})"
            if attempt < MAX_RETRIES:
                shutil.rmtree(repo_dir, ignore_errors=True)
                continue
            break

        except Exception as e:
            result["error"] = str(e)[:500]
            if attempt < MAX_RETRIES:
                shutil.rmtree(repo_dir, ignore_errors=True)
                continue
            break

    # Final cleanup and stats update
    result["metrics"]["duration_sec"] = round(time.time() - start_time, 1)
    
    # Save trajectory
    trajectory = {
        "instance_id": instance_id,
        "model": MODEL,
        "thinking_tokens": THINKING_TOKENS,
        "retries": result["retries"],
        "metrics": result["metrics"],
        "patch_generated": result["success"],
        "patch_length": len(result["model_patch"]),
        "error": result.get("error"),
        "timestamp": datetime.now().isoformat()
    }
    with open(traj_file, "w") as f:
        json.dump(trajectory, f, indent=2)

    # Update global stats
    with lock:
        stats["completed"] += 1
        stats["total_cost"] += result["metrics"]["cost_usd"]
        stats["total_input_tokens"] += result["metrics"]["input_tokens"]
        stats["total_output_tokens"] += result["metrics"]["output_tokens"]
        stats["total_turns"] += result["metrics"]["turns"]
        stats["total_time_sec"] += result["metrics"]["duration_sec"]
        if result["success"]:
            stats["patches_generated"] += 1
        else:
            stats["errors"] += 1
        stats["instances"][instance_id] = {
            "status": "completed" if result["success"] else "failed",
            "success": result["success"],
            "retries": result["retries"],
            "metrics": result["metrics"]
        }

    # Cleanup
    shutil.rmtree(repo_dir, ignore_errors=True)

    return result


def load_completed_instances() -> set:
    """Load already completed instances for resume capability."""
    completed = set()
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                    completed.add(pred["instance_id"])
                except:
                    pass
    return completed


def save_metadata():
    """Save metadata.yaml for leaderboard submission."""
    metadata = {
        "name": f"Polydev-Baseline-{stats['run_id']}",
        "model": MODEL,
        "thinking_tokens": THINKING_TOKENS,
        "methodology": "Anthropic's exact methodology with 128K thinking budget",
        "prompt": ANTHROPIC_PROMPT_ADDITION,
        "oss": True,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "stats": {
            "total_instances": stats["total"],
            "patches_generated": stats["patches_generated"],
            "total_cost_usd": round(stats["total_cost"], 2),
            "total_tokens": stats["total_input_tokens"] + stats["total_output_tokens"]
        }
    }

    metadata_file = BASE_OUTPUT_DIR / "metadata.yaml"
    import yaml
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)


def save_readme():
    """Save README.md for leaderboard submission."""
    readme = f"""# SWE-bench Baseline Submission

## Model
- **Name**: Claude Haiku 4.5
- **Model ID**: {MODEL}
- **Extended Thinking**: {THINKING_TOKENS:,} tokens

## Methodology
Exact replication of Anthropic's SWE-bench methodology:
- Simple scaffold with bash + file editing tools
- 128K thinking budget
- Default sampling parameters
- Prompt: "{ANTHROPIC_PROMPT_ADDITION}"

## Results
- **Total Instances**: {stats['total']}
- **Patches Generated**: {stats['patches_generated']} ({100*stats['patches_generated']/max(1,stats['total']):.1f}%)
- **Total Cost**: ${stats['total_cost']:.2f}
- **Total Tokens**: {stats['total_input_tokens'] + stats['total_output_tokens']:,}

## Run Details
- **Run ID**: {stats['run_id']}
- **Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Duration**: {(time.time() - stats['start_time'])/60:.0f} minutes

## Submission
```bash
sb-cli submit swe-bench_verified test --predictions_path all_preds.jsonl --run_id {stats['run_id']}
```
"""

    readme_file = BASE_OUTPUT_DIR / "README.md"
    readme_file.write_text(readme)


def main():
    parser = argparse.ArgumentParser(description="SWE-bench Baseline Runner")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--test", type=int, help="Test mode: run on N samples only")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--run-id", type=str, help="Custom run ID")
    parser.add_argument("--instances", type=str, help="JSON file with list of instance IDs to run")
    args = parser.parse_args()

    # Setup
    run_id = args.run_id or f"baseline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    stats["run_id"] = run_id

    print("="*80)
    print("SWE-BENCH BASELINE RUNNER")
    print("Exact Anthropic Methodology - Leaderboard Compatible")
    print("="*80)
    print(f"  Run ID: {run_id}")
    print(f"  Model: {MODEL}")
    print(f"  Thinking Budget: {THINKING_TOKENS:,} tokens")
    print(f"  Workers: {args.workers}")
    print(f"  Max Turns: {MAX_TURNS}")
    print(f"  Prompt: \"{ANTHROPIC_PROMPT_ADDITION}\"")
    if args.test:
        print(f"  TEST MODE: {args.test} samples only")
    if args.resume:
        print(f"  RESUME MODE: Continuing from previous run")
    print("="*80)

    # Create output directories
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    TRAJS_DIR.mkdir(exist_ok=True)

    # Load dataset
    print("\nLoading SWE-bench Verified dataset...")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    all_instances = list(ds)

    # Handle resume
    completed_instances = set()
    if args.resume:
        completed_instances = load_completed_instances()
        print(f"  Found {len(completed_instances)} already completed instances")

    # Filter instances
    instances = [inst for inst in all_instances if inst["instance_id"] not in completed_instances]

    # Filter by specific instance IDs if provided
    if args.instances:
        with open(args.instances) as f:
            target_ids = set(json.load(f))
        instances = [inst for inst in instances if inst["instance_id"] in target_ids]
        print(f"  Filtered to {len(instances)} specific instances from {args.instances}")

    # Test mode
    if args.test:
        instances = instances[:args.test]

    stats["total"] = len(instances)
    stats["start_time"] = time.time()

    print(f"\nInstances to process: {len(instances)}")
    print("Starting parallel execution...")
    print("="*80 + "\n")

    # Start progress reporter
    reporter = threading.Thread(target=progress_reporter, daemon=True)
    reporter.start()

    # Run in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_instance, inst): inst for inst in instances}

        for future in as_completed(futures):
            result = future.result()

            # Save prediction (leaderboard format)
            with lock:
                with open(PREDICTIONS_FILE, "a") as f:
                    pred = {
                        "instance_id": result["instance_id"],
                        "model_name_or_path": result["model_name_or_path"],
                        "model_patch": result["model_patch"]
                    }
                    f.write(json.dumps(pred) + "\n")

                # Save metrics
                with open(METRICS_FILE, "a") as f:
                    metrics_entry = {
                        "instance_id": result["instance_id"],
                        "success": result["success"],
                        "metrics": result["metrics"],
                        "error": result.get("error"),
                        "timestamp": datetime.now().isoformat()
                    }
                    f.write(json.dumps(metrics_entry) + "\n")

            # Print status
            status = "✓" if result["success"] else "✗"
            err = f" [{result.get('error', '')[:30]}]" if result.get("error") else ""
            m = result["metrics"]
            patch_rate = 100 * stats["patches_generated"] / max(1, stats["completed"])
            print(f"[{stats['completed']}/{stats['total']}] {status} {result['instance_id'][:45]:<45} "
                  f"turns:{m['turns']:>3} ${m['cost_usd']:.3f} {m['duration_sec']:>4.0f}s "
                  f"[{stats['patches_generated']} patches {patch_rate:.0f}%]{err}")

    # Final summary
    log_progress()

    # Save metadata and README for submission
    try:
        save_metadata()
        save_readme()
    except Exception as e:
        print(f"Warning: Could not save metadata/readme: {e}")

    print(f"\n{'='*80}")
    print("BASELINE RUN COMPLETE!")
    print(f"{'='*80}")
    print(f"  Predictions: {PREDICTIONS_FILE}")
    print(f"  Metrics: {METRICS_FILE}")
    print(f"  Logs: {LOGS_DIR}")
    print(f"  Trajectories: {TRAJS_DIR}")
    print(f"\n  Patches: {stats['patches_generated']}/{stats['total']} ({100*stats['patches_generated']/max(1,stats['total']):.1f}%)")
    print(f"  Cost: ${stats['total_cost']:.4f}")
    print(f"  Tokens: {stats['total_input_tokens']:,} in / {stats['total_output_tokens']:,} out")
    print(f"\n  To submit to leaderboard:")
    print(f"  sb-cli submit swe-bench_verified test --predictions_path {PREDICTIONS_FILE} --run_id {run_id}")
    print("="*80)


if __name__ == "__main__":
    main()
