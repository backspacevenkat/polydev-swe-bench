#!/usr/bin/env python3
"""Run SWE-bench evaluation on generated patches."""

import os
os.makedirs("/tmp/swebench_reports", exist_ok=True)

from swebench.harness.run_evaluation import main

# Only run the 6 instances that need evaluation
# (14 already resolved successfully)
missing_instances = [
    "django__django-10554",  # New patch, OOM on build  
    "django__django-11087",  # Fixed patch
    "psf__requests-1766",    # Needs re-run
    "psf__requests-2317",    # Needs re-run
    "pytest-dev__pytest-10051",  # Needs re-run
    "sphinx-doc__sphinx-10435",  # Fixed patch
]

# Call main with all required arguments
# Using max_workers=1 to avoid OOM issues during Docker builds
main(
    dataset_name="princeton-nlp/SWE-bench_Verified",
    split="test",
    instance_ids=missing_instances,  # Only run missing instances
    predictions_path="/tmp/swe_bench_patches.jsonl",
    max_workers=1,  # Reduced from 2 to prevent OOM (error 137)
    force_rebuild=False,
    cache_level="env",
    clean=False,
    open_file_limit=4096,
    run_id="claude_haiku_eval",
    timeout=1800,
    namespace=None,
    rewrite_reports=False,
    modal=False,
    instance_image_tag="latest",
    env_image_tag="latest",
    report_dir="/tmp/swebench_reports",
)
