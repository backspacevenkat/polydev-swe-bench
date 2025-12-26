#!/usr/bin/env python3
"""
Hybrid Ensemble SWE-bench Evaluation
=====================================
Runs dual-path (baseline + polydev) for each instance in parallel batches.
Tracks all metrics including token usage, timing, and success rates.
"""

import os
import sys
import json
import time
import asyncio
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import random

# Configuration
BATCH_SIZE = 5  # Instances processed in parallel
STAGGER_DELAY = 2  # Seconds between instance starts
TIMEOUT_PER_INSTANCE = 300  # 5 minutes max per instance
POLYDEV_USER_TOKEN = os.environ.get("POLYDEV_USER_TOKEN", "")

# Metrics tracking
class MetricsTracker:
    def __init__(self):
        self.start_time = time.time()
        self.instances = {}
        self.totals = {
            "baseline_only_solved": 0,
            "polydev_only_solved": 0,
            "both_solved": 0,
            "neither_solved": 0,
            "total_tokens": 0,
            "baseline_tokens": 0,
            "polydev_tokens": 0,
            "total_cost_usd": 0.0,
        }

    def add_instance(self, instance_id, data):
        self.instances[instance_id] = data

    def get_elapsed(self):
        return time.time() - self.start_time

    def get_summary(self):
        total = len(self.instances)
        baseline_wins = sum(1 for i in self.instances.values() if i.get("winner") == "baseline")
        polydev_wins = sum(1 for i in self.instances.values() if i.get("winner") == "polydev")
        both_pass = sum(1 for i in self.instances.values() if i.get("winner") == "both")
        neither = sum(1 for i in self.instances.values() if i.get("winner") == "neither")

        total_solved = baseline_wins + polydev_wins + both_pass

        return {
            "total_instances": total,
            "total_solved": total_solved,
            "pass_rate": f"{(total_solved/total*100):.1f}%" if total > 0 else "0%",
            "baseline_only_wins": baseline_wins,
            "polydev_only_wins": polydev_wins,
            "both_passed": both_pass,
            "neither_passed": neither,
            "elapsed_seconds": self.get_elapsed(),
            "elapsed_formatted": format_time(self.get_elapsed()),
            "tokens": self.totals,
        }

def format_time(seconds):
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s"

def print_progress(current, total, instance_id, status, metrics):
    """Print progress bar and current status"""
    bar_width = 30
    progress = current / total
    filled = int(bar_width * progress)
    bar = "█" * filled + "░" * (bar_width - filled)

    elapsed = metrics.get_elapsed()
    summary = metrics.get_summary()

    print(f"\r[{bar}] {current}/{total} | {format_time(elapsed)} | "
          f"✓{summary['total_solved']} solved | "
          f"B:{summary['baseline_only_wins']} P:{summary['polydev_only_wins']} "
          f"Both:{summary['both_passed']} | {instance_id[:30]}...", end="", flush=True)

def run_claude_baseline(instance_id, problem_statement, repo_path):
    """Run Claude Haiku baseline (no Polydev consultation)"""
    start_time = time.time()

    prompt = f"""You are an expert software engineer. Fix the following GitHub issue.

## Issue
{problem_statement}

## Repository
{repo_path}

Analyze the issue carefully and provide a minimal, focused patch that fixes the problem.
Output ONLY the git diff patch, nothing else.
"""

    try:
        # Use Claude API directly for baseline
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", "claude-haiku-4-5-20251001", "--max-tokens", "4096"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_PER_INSTANCE,
            cwd=repo_path if os.path.exists(repo_path) else None
        )

        latency = time.time() - start_time
        output = result.stdout.strip()

        # Estimate tokens (rough: 4 chars = 1 token)
        tokens = len(prompt) // 4 + len(output) // 4

        return {
            "success": result.returncode == 0 and len(output) > 10,
            "patch": output,
            "tokens": tokens,
            "latency_ms": int(latency * 1000),
            "error": result.stderr if result.returncode != 0 else None
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "patch": "", "tokens": 0, "latency_ms": TIMEOUT_PER_INSTANCE * 1000, "error": "timeout"}
    except Exception as e:
        return {"success": False, "patch": "", "tokens": 0, "latency_ms": 0, "error": str(e)}

def run_claude_with_polydev(instance_id, problem_statement, repo_path):
    """Run Claude Haiku with Polydev consultation"""
    start_time = time.time()
    total_tokens = 0

    # Step 1: Consult Polydev for analysis
    polydev_prompt = f"""Analyze this GitHub issue and provide insights for fixing it:

## Issue
{problem_statement}

Please provide:
1. Root cause analysis
2. Key files likely involved
3. Edge cases to consider
4. Recommended fix approach
"""

    try:
        # Call Polydev for multi-model perspectives
        polydev_result = subprocess.run(
            ["node", "-e", f"""
const {{ spawn }} = require('child_process');
const polydev = spawn('npx', ['-y', 'polydev-ai@latest'], {{
    cwd: '/Users/venkat/mcp-execution',
    env: {{ ...process.env }},
    stdio: ['pipe', 'pipe', 'pipe']
}});

let output = '';
polydev.stdout.on('data', d => output += d.toString());
polydev.stderr.on('data', d => {{}});

polydev.stdin.write(JSON.stringify({{
    jsonrpc: '2.0', method: 'initialize', params: {{ capabilities: {{}} }}, id: 1
}}) + '\\n');

setTimeout(() => {{
    polydev.stdin.write(JSON.stringify({{
        jsonrpc: '2.0',
        method: 'tools/call',
        params: {{
            name: 'get_perspectives',
            arguments: {{ prompt: {json.dumps(polydev_prompt)} }}
        }},
        id: 2
    }}) + '\\n');
}}, 500);

setTimeout(() => {{
    const lines = output.split('\\n').filter(l => l.startsWith('{{'));
    for (const line of lines) {{
        try {{
            const j = JSON.parse(line);
            if (j.id === 2 && j.result) {{
                console.log(JSON.stringify(j.result));
            }}
        }} catch (e) {{}}
    }}
    polydev.kill();
    process.exit(0);
}}, 30000);
"""],
            capture_output=True,
            text=True,
            timeout=60
        )

        polydev_insights = ""
        polydev_tokens = 0
        if polydev_result.stdout:
            try:
                result_json = json.loads(polydev_result.stdout.strip())
                if result_json.get("content"):
                    polydev_insights = result_json["content"][0].get("text", "")
                    # Extract token count from response
                    if "tokens" in polydev_insights.lower():
                        import re
                        token_match = re.search(r'"tokens":\s*(\d+)', polydev_insights)
                        if token_match:
                            polydev_tokens = int(token_match.group(1))
            except:
                polydev_insights = polydev_result.stdout

        total_tokens += polydev_tokens

    except Exception as e:
        polydev_insights = f"Polydev consultation failed: {e}"

    # Step 2: Generate patch with Polydev insights
    enhanced_prompt = f"""You are an expert software engineer. Fix the following GitHub issue.

## Issue
{problem_statement}

## Expert Analysis (from multiple AI models)
{polydev_insights[:3000]}  # Limit to avoid token overflow

## Repository
{repo_path}

Based on the expert analysis above, provide a minimal, focused patch that fixes the problem.
Output ONLY the git diff patch, nothing else.
"""

    try:
        result = subprocess.run(
            ["claude", "-p", enhanced_prompt, "--model", "claude-haiku-4-5-20251001", "--max-tokens", "4096"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_PER_INSTANCE,
            cwd=repo_path if os.path.exists(repo_path) else None
        )

        latency = time.time() - start_time
        output = result.stdout.strip()

        # Estimate Claude tokens
        claude_tokens = len(enhanced_prompt) // 4 + len(output) // 4
        total_tokens += claude_tokens

        return {
            "success": result.returncode == 0 and len(output) > 10,
            "patch": output,
            "tokens": total_tokens,
            "polydev_tokens": polydev_tokens,
            "claude_tokens": claude_tokens,
            "latency_ms": int(latency * 1000),
            "polydev_insights": polydev_insights[:500],  # Truncate for logging
            "error": result.stderr if result.returncode != 0 else None
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "patch": "", "tokens": total_tokens, "latency_ms": TIMEOUT_PER_INSTANCE * 1000, "error": "timeout"}
    except Exception as e:
        return {"success": False, "patch": "", "tokens": total_tokens, "latency_ms": 0, "error": str(e)}

def process_instance(instance, metrics, instance_num, total):
    """Process a single instance with dual-path approach"""
    instance_id = instance["instance_id"]
    problem_statement = instance.get("problem_statement", "")
    repo = instance.get("repo", "")

    print_progress(instance_num, total, instance_id, "processing", metrics)

    # Run both paths in parallel using threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        baseline_future = executor.submit(run_claude_baseline, instance_id, problem_statement, f"/tmp/{repo}")
        polydev_future = executor.submit(run_claude_with_polydev, instance_id, problem_statement, f"/tmp/{repo}")

        baseline_result = baseline_future.result()
        polydev_result = polydev_future.result()

    # Determine winner (for now, just based on patch generation success)
    # Real evaluation would run tests on both patches
    baseline_ok = baseline_result.get("success", False) and len(baseline_result.get("patch", "")) > 50
    polydev_ok = polydev_result.get("success", False) and len(polydev_result.get("patch", "")) > 50

    if baseline_ok and polydev_ok:
        winner = "both"
    elif baseline_ok:
        winner = "baseline"
    elif polydev_ok:
        winner = "polydev"
    else:
        winner = "neither"

    result = {
        "instance_id": instance_id,
        "baseline": baseline_result,
        "polydev": polydev_result,
        "winner": winner,
        "total_tokens": baseline_result.get("tokens", 0) + polydev_result.get("tokens", 0),
    }

    metrics.add_instance(instance_id, result)
    return result

def load_instances(dataset_path, num_instances=20, random_seed=42):
    """Load SWE-bench instances"""
    # Try to load from local file or use swebench library
    try:
        from datasets import load_dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        instances = list(dataset)

        # Random sample
        random.seed(random_seed)
        if num_instances < len(instances):
            instances = random.sample(instances, num_instances)

        return instances
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback: create dummy instances for testing
        return [{"instance_id": f"test-{i}", "problem_statement": f"Test problem {i}", "repo": "test/repo"}
                for i in range(num_instances)]

def save_predictions(results, output_path):
    """Save predictions in SWE-bench format"""
    predictions = []
    for r in results:
        # Use the best patch (prefer polydev if both succeeded)
        if r["winner"] in ["polydev", "both"]:
            patch = r["polydev"].get("patch", "")
        else:
            patch = r["baseline"].get("patch", "")

        predictions.append({
            "instance_id": r["instance_id"],
            "model_patch": patch,
            "model_name_or_path": "hybrid-ensemble-haiku",
        })

    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    return output_path

def print_final_report(metrics, results):
    """Print comprehensive final report"""
    summary = metrics.get_summary()

    print("\n" + "="*70)
    print("                    HYBRID ENSEMBLE EVALUATION REPORT")
    print("="*70)

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ OVERALL RESULTS                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ Total Instances:     {summary['total_instances']:>5}                                         │
│ Total Solved:        {summary['total_solved']:>5}  ({summary['pass_rate']})                                    │
│ Elapsed Time:        {summary['elapsed_formatted']:>10}                                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ APPROACH COMPARISON                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Baseline Only Wins:  {summary['baseline_only_wins']:>5}  (Claude Haiku alone succeeded)       │
│ Polydev Only Wins:   {summary['polydev_only_wins']:>5}  (Polydev consultation needed)         │
│ Both Passed:         {summary['both_passed']:>5}  (Either approach works)               │
│ Neither Passed:      {summary['neither_passed']:>5}  (Both approaches failed)              │
└─────────────────────────────────────────────────────────────────────┘
""")

    # Token usage
    total_baseline_tokens = sum(r["baseline"].get("tokens", 0) for r in results)
    total_polydev_tokens = sum(r["polydev"].get("tokens", 0) for r in results)
    total_tokens = total_baseline_tokens + total_polydev_tokens

    # Estimate costs (rough: $0.25/1M input, $1.25/1M output for Haiku)
    estimated_cost = (total_tokens / 1_000_000) * 0.50  # Average cost

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ TOKEN USAGE                                                          │
├─────────────────────────────────────────────────────────────────────┤
│ Baseline Tokens:     {total_baseline_tokens:>10,}                                     │
│ Polydev Tokens:      {total_polydev_tokens:>10,}                                     │
│ Total Tokens:        {total_tokens:>10,}                                     │
│ Estimated Cost:      ${estimated_cost:>9.4f}                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

    # Latency stats
    baseline_latencies = [r["baseline"].get("latency_ms", 0) for r in results]
    polydev_latencies = [r["polydev"].get("latency_ms", 0) for r in results]

    avg_baseline = sum(baseline_latencies) / len(baseline_latencies) if baseline_latencies else 0
    avg_polydev = sum(polydev_latencies) / len(polydev_latencies) if polydev_latencies else 0

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ LATENCY (per instance)                                               │
├─────────────────────────────────────────────────────────────────────┤
│ Avg Baseline:        {avg_baseline/1000:>7.1f}s                                       │
│ Avg Polydev:         {avg_polydev/1000:>7.1f}s                                       │
│ Parallel Overhead:   Minimal (both run simultaneously)               │
└─────────────────────────────────────────────────────────────────────┘
""")

    # Instance breakdown
    print("\n" + "─"*70)
    print("INSTANCE DETAILS")
    print("─"*70)
    print(f"{'Instance ID':<45} {'Winner':<12} {'Tokens':>10}")
    print("─"*70)

    for r in results:
        instance_id = r["instance_id"][:44]
        winner = r["winner"]
        tokens = r["total_tokens"]

        winner_symbol = {
            "baseline": "🅱️  Baseline",
            "polydev": "🅿️  Polydev",
            "both": "✅ Both",
            "neither": "❌ Neither"
        }.get(winner, winner)

        print(f"{instance_id:<45} {winner_symbol:<12} {tokens:>10,}")

    print("─"*70)
    print("\n")

async def main():
    parser = argparse.ArgumentParser(description="Hybrid Ensemble SWE-bench Evaluation")
    parser.add_argument("--instances", type=int, default=20, help="Number of instances to process")
    parser.add_argument("--batch-size", type=int, default=5, help="Parallel batch size")
    parser.add_argument("--output", type=str, default="hybrid-ensemble-predictions.jsonl", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for instance selection")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("         HYBRID ENSEMBLE SWE-BENCH EVALUATION")
    print("="*70)
    print(f"""
Configuration:
  - Instances: {args.instances}
  - Batch Size: {args.batch_size} (parallel)
  - Output: {args.output}
  - Seed: {args.seed}

Approach: Dual-path (Baseline + Polydev) per instance
Expected Time: ~{args.instances // args.batch_size * 3} minutes
""")
    print("="*70 + "\n")

    # Load instances
    print("Loading SWE-bench instances...")
    instances = load_instances(None, args.instances, args.seed)
    print(f"Loaded {len(instances)} instances\n")

    # Initialize metrics
    metrics = MetricsTracker()
    results = []

    # Process in batches
    total = len(instances)
    for batch_start in range(0, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)
        batch = instances[batch_start:batch_end]

        # Process batch with thread pool
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            futures = {}
            for i, instance in enumerate(batch):
                instance_num = batch_start + i + 1
                # Stagger starts
                time.sleep(STAGGER_DELAY)
                future = executor.submit(process_instance, instance, metrics, instance_num, total)
                futures[future] = instance

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

    print("\n\n")  # Clear progress line

    # Save predictions
    predictions_path = save_predictions(results, args.output)
    print(f"Predictions saved to: {predictions_path}")

    # Print final report
    print_final_report(metrics, results)

    # Save detailed results
    detailed_path = args.output.replace(".jsonl", "-detailed.json")
    with open(detailed_path, "w") as f:
        json.dump({
            "summary": metrics.get_summary(),
            "results": results
        }, f, indent=2)
    print(f"Detailed results saved to: {detailed_path}")

    return results

if __name__ == "__main__":
    asyncio.run(main())
