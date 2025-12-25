#!/usr/bin/env python3
"""
Pre-download all SWE-bench data to avoid issues during full run.
"""
import json
from pathlib import Path
from datasets import load_dataset
import subprocess
import os
import time

print("=" * 60)
print("SWE-BENCH DATA PRE-DOWNLOAD")
print("=" * 60)

# 1. Download and cache the dataset
print("\n[1/3] Downloading SWE-bench Verified dataset...")
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
instances = list(ds)
print(f"  ✓ Downloaded {len(instances)} instances")

# 2. Save all instance IDs and repos
print("\n[2/3] Extracting unique repositories...")
repos = set()
for inst in instances:
    repos.add(inst["repo"])
print(f"  ✓ Found {len(repos)} unique repositories")

# 3. Save instance list for reference
output_dir = Path("/Users/venkat/Documents/polydev-swe-bench/data")
output_dir.mkdir(parents=True, exist_ok=True)

instances_file = output_dir / "all_instances.json"
with open(instances_file, "w") as f:
    json.dump([{
        "instance_id": inst["instance_id"],
        "repo": inst["repo"],
        "base_commit": inst["base_commit"],
        "problem_length": len(inst["problem_statement"])
    } for inst in instances], f, indent=2)
print(f"  ✓ Saved instance list to {instances_file}")

repos_file = output_dir / "repos.txt"
with open(repos_file, "w") as f:
    for repo in sorted(repos):
        f.write(f"{repo}\n")
print(f"  ✓ Saved repo list to {repos_file}")

# 4. Summary stats
print("\n[3/3] Summary:")
print(f"  Total instances: {len(instances)}")
print(f"  Unique repos: {len(repos)}")
print(f"  Repos: {', '.join(sorted(repos)[:10])}...")

# Estimate time and cost
avg_time_sec = 370  # From our test
avg_cost = 0.043  # From our test
total_time_hrs = (len(instances) * avg_time_sec / 5) / 3600  # with 5 workers
total_cost = len(instances) * avg_cost

print(f"\n  Estimated time (5 workers): {total_time_hrs:.1f} hours per run")
print(f"  Estimated cost per run: ${total_cost:.2f}")
print(f"  Total for both runs: ${total_cost * 2:.2f}")

print("\n" + "=" * 60)
print("DATA PRE-DOWNLOAD COMPLETE")
print("=" * 60)
