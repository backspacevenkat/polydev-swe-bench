#!/usr/bin/env python3
"""
Select 30 diverse tasks from SWE-bench_Verified.
Criteria:
1. Different repositories (not just Django)
2. Mix of difficulty levels
3. Exclude already-tested tasks
"""

from datasets import load_dataset
import random
from collections import defaultdict

# Tasks already tested in v3
ALREADY_TESTED = {
    "django__django-11099",
    "django__django-11163",
    "django__django-10097",
    "django__django-11179",
    "django__django-11292",
    "django__django-11433",
    "django__django-11451",
    "django__django-11555",
    "django__django-10554",
    "django__django-10914",
}

def main():
    print("Loading SWE-bench_Verified dataset...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    print(f"Total tasks: {len(dataset)}")

    # Group by repo
    repo_tasks = defaultdict(list)
    for item in dataset:
        instance_id = item["instance_id"]
        if instance_id not in ALREADY_TESTED:
            repo = instance_id.split("__")[0] + "__" + instance_id.split("__")[1].split("-")[0]
            repo_tasks[repo].append(instance_id)

    print(f"\nRepositories available (excluding tested):")
    for repo, tasks in sorted(repo_tasks.items(), key=lambda x: -len(x[1])):
        print(f"  {repo}: {len(tasks)} tasks")

    # Selection strategy:
    # - 10 Django tasks (familiar, good baseline)
    # - 5 Scikit-learn tasks (ML library)
    # - 5 Matplotlib tasks (visualization)
    # - 3 Sympy tasks (symbolic math)
    # - 3 Flask/Requests tasks (web)
    # - 4 Other repos (diversity)

    selected = []

    # Select from each repo
    selections = {
        "django__django": 10,
        "scikit-learn__scikit-learn": 5,
        "matplotlib__matplotlib": 5,
        "sympy__sympy": 3,
        "psf__requests": 2,
        "pallets__flask": 1,
        "pytest-dev__pytest": 2,
        "pydata__xarray": 2,
    }

    for repo, count in selections.items():
        if repo in repo_tasks and len(repo_tasks[repo]) >= count:
            # Random sample from this repo
            random.seed(42)  # Reproducible
            sampled = random.sample(repo_tasks[repo], min(count, len(repo_tasks[repo])))
            selected.extend(sampled)
            print(f"\nSelected {len(sampled)} from {repo}:")
            for t in sampled:
                print(f"  - {t}")

    # If we don't have 30, add more from any repo
    remaining = 30 - len(selected)
    if remaining > 0:
        all_remaining = []
        for repo, tasks in repo_tasks.items():
            for t in tasks:
                if t not in selected:
                    all_remaining.append(t)

        random.seed(42)
        extra = random.sample(all_remaining, min(remaining, len(all_remaining)))
        selected.extend(extra)
        print(f"\nAdded {len(extra)} extra tasks to reach 30")

    print(f"\n{'='*60}")
    print(f"FINAL SELECTION: {len(selected)} tasks")
    print(f"{'='*60}")

    # Print as Python list for copy-paste
    print("\nTASKS = [")
    for t in sorted(selected):
        print(f'    "{t}",')
    print("]")

if __name__ == "__main__":
    main()
