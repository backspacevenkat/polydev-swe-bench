# Monitoring & Logging Guide

This document describes the monitoring and logging infrastructure for Polydev SWE-bench evaluation runs.

## Overview

Every evaluation run produces comprehensive logs and metrics for:
- **Debugging**: Understand why specific tasks pass or fail
- **Analysis**: Deep-dive into consultation effectiveness
- **Reproducibility**: Complete audit trail of all decisions
- **Publication**: Evidence for claims made in results

## Log Structure

```
logs/
├── current -> run_20241216_143022/  # Symlink to latest run
└── run_20241216_143022/
    ├── run_config.yaml              # Configuration snapshot
    ├── agent.log                    # Main agent log
    ├── consultations.log            # All consultation details
    ├── errors.log                   # Errors and warnings
    ├── metrics.json                 # Aggregated metrics
    ├── tasks/
    │   ├── django__django-11099.json    # Per-task detailed log
    │   ├── django__django-11133.json
    │   └── ...
    └── summary.json                 # Run summary
```

## Log Levels

| Level | Description | Use |
|-------|-------------|-----|
| DEBUG | Detailed trace info | Development only |
| INFO | Normal operation | Default for runs |
| WARNING | Potential issues | Always logged |
| ERROR | Failures | Always logged |

Configure via environment:
```bash
export LOG_LEVEL=INFO  # or DEBUG, WARNING, ERROR
```

## Agent Log (`agent.log`)

Main log showing agent decision flow:

```
2024-12-16 14:30:22 INFO  [Task: django__django-11099] Starting analysis
2024-12-16 14:30:25 INFO  [Task: django__django-11099] Read 5 files, identified root cause
2024-12-16 14:30:28 INFO  [Task: django__django-11099] Generated solution, confidence: 6/10
2024-12-16 14:30:28 INFO  [Task: django__django-11099] Confidence < 8, triggering consultation
2024-12-16 14:30:35 INFO  [Task: django__django-11099] Received 2 perspectives
2024-12-16 14:30:38 INFO  [Task: django__django-11099] Synthesis complete, final confidence: 8/10
2024-12-16 14:30:42 INFO  [Task: django__django-11099] Generated patch (12 lines changed)
2024-12-16 14:30:45 INFO  [Task: django__django-11099] Result: PASS (41.1s)
```

## Consultations Log (`consultations.log`)

Detailed record of all consultations:

```
================================================================================
CONSULTATION: django__django-11099
Timestamp: 2024-12-16T14:30:28Z
Triggered because: confidence (6) < threshold (8)
================================================================================

--- CONSULTATION REQUEST ---
I'm working on a software engineering task and would like expert perspectives.

## Task Information
- Repository: django/django
- Issue ID: django__django-11099
...

--- GPT-5.2 RESPONSE ---
Looking at this issue, I notice that the problem is in the filter chain...
[Full response]
Latency: 5.2s

--- GEMINI-3-PRO RESPONSE ---
The root cause appears to be in the query compilation order...
[Full response]
Latency: 3.8s

--- SYNTHESIS ---
Points of agreement: Both models agree the issue is in filter chain ordering.
Points of disagreement: GPT suggests modifying filter(), Gemini suggests compile().
Final decision: Combine both insights - fix filter chain and add compilation check.
Approach changed from original: YES

================================================================================
```

## Per-Task Logs (`tasks/*.json`)

Complete record for each task:

```json
{
  "instance_id": "django__django-11099",
  "timestamp_start": "2024-12-16T14:30:22Z",
  "timestamp_end": "2024-12-16T14:31:03Z",
  "configuration": "polydev",

  "phases": {
    "ingestion": {
      "duration_ms": 523,
      "task_loaded": true
    },

    "analysis": {
      "duration_ms": 3245,
      "files_read": [
        "django/db/models/query.py",
        "django/db/models/sql/query.py",
        "tests/queries/test_count.py"
      ],
      "lines_read": 847,
      "root_cause": "Query compilation order causes count() to use stale filter state",
      "affected_files_identified": ["django/db/models/sql/query.py"]
    },

    "solution": {
      "duration_ms": 2134,
      "initial_approach": "Modify the count() method to refresh filter state",
      "confidence_score": 6,
      "confidence_reasoning": "Multiple valid approaches exist. The issue could be in count() or in the filter chain. Not 100% certain which is the better fix location.",
      "uncertainty_markers": ["could be", "multiple approaches"]
    },

    "consultation": {
      "triggered": true,
      "trigger_reason": "confidence_below_threshold",
      "duration_ms": 9023,

      "request_sent": "[Full consultation request text]",

      "responses": {
        "gpt-5.2": {
          "text": "[Full GPT response]",
          "latency_ms": 5234,
          "tokens": 342,
          "cost_usd": 0.0
        },
        "gemini-3-pro": {
          "text": "[Full Gemini response]",
          "latency_ms": 3789,
          "tokens": 287,
          "cost_usd": 0.00087
        }
      },

      "synthesis": {
        "agreement_points": ["Filter chain is the issue", "Need to ensure state consistency"],
        "disagreement_points": ["Location of fix: filter() vs compile()"],
        "models_incorporated": ["gpt-5.2", "gemini-3-pro"],
        "approach_changed": true,
        "original_approach": "Modify count() method",
        "final_approach": "Fix filter chain ordering in compile() with count() refresh",
        "final_confidence": 8
      }
    },

    "patch_generation": {
      "duration_ms": 2567,
      "files_modified": ["django/db/models/sql/query.py"],
      "lines_added": 8,
      "lines_removed": 2,
      "patch_size_bytes": 523
    },

    "evaluation": {
      "patch_applies": true,
      "tests_run": true,
      "fail_to_pass": {
        "test_count_distinct_annotate": "PASS"
      },
      "pass_to_pass": {
        "test_count_basic": "PASS",
        "test_count_filter": "PASS"
      },
      "regression_tests": "ALL_PASS"
    }
  },

  "result": {
    "status": "PASS",
    "total_duration_ms": 41123,
    "total_cost_usd": 0.00087,
    "consultation_helped": true
  },

  "patch": "diff --git a/django/db/models/sql/query.py b/django/db/models/sql/query.py\n..."
}
```

## Metrics Collection

### Real-Time Metrics (`metrics.json`)

Updated after each task:

```json
{
  "run_id": "run_20241216_143022",
  "status": "running",
  "progress": {
    "total_tasks": 500,
    "completed": 127,
    "remaining": 373,
    "percent_complete": 25.4
  },

  "results_so_far": {
    "passed": 42,
    "failed": 85,
    "current_pass_rate": 0.331
  },

  "consultations": {
    "total_triggered": 45,
    "rate": 0.354,
    "effectiveness": {
      "helped": 15,
      "neutral": 28,
      "hurt": 2
    }
  },

  "performance": {
    "avg_time_per_task_ms": 38234,
    "total_time_elapsed_s": 4855,
    "estimated_time_remaining_s": 14235
  },

  "costs": {
    "total_usd": 0.045,
    "avg_per_consultation_usd": 0.001
  },

  "errors": {
    "total": 3,
    "types": {
      "timeout": 2,
      "parse_error": 1
    }
  }
}
```

### Summary (`summary.json`)

Final run summary:

```json
{
  "run_id": "run_20241216_143022",
  "configuration": "polydev",
  "completed_at": "2024-12-16T22:45:33Z",

  "results": {
    "total_tasks": 500,
    "passed": 185,
    "failed": 315,
    "pass_rate": 0.370
  },

  "consultation_stats": {
    "total_consultations": 147,
    "consultation_rate": 0.294,
    "consultations_that_helped": 52,
    "consultations_neutral": 89,
    "consultations_that_hurt": 6,
    "effectiveness_rate": 0.354
  },

  "performance_stats": {
    "total_duration_seconds": 19234,
    "avg_time_per_task_seconds": 38.5,
    "fastest_task_seconds": 12.3,
    "slowest_task_seconds": 245.7
  },

  "cost_stats": {
    "total_cost_usd": 0.147,
    "avg_cost_per_task_usd": 0.0003,
    "avg_cost_per_consultation_usd": 0.001
  },

  "breakdown_by_repo": {
    "django/django": {"total": 143, "passed": 58, "rate": 0.406},
    "scikit-learn/scikit-learn": {"total": 89, "passed": 31, "rate": 0.348},
    ...
  }
}
```

## Real-Time Monitoring

### Watch Progress

```bash
# Follow main log
tail -f logs/current/agent.log

# Watch metrics update
watch -n 5 cat logs/current/metrics.json | jq '.progress'
```

### Dashboard Script

```bash
# Launch monitoring dashboard
python monitoring/dashboard.py --run logs/current/

# Output:
# ┌─────────────────────────────────────────────────────────────┐
# │ Polydev SWE-bench Evaluation - RUNNING                      │
# ├─────────────────────────────────────────────────────────────┤
# │ Progress: [████████████░░░░░░░░░░░░░░] 127/500 (25.4%)     │
# │ Pass Rate: 33.1% (42/127)                                   │
# │ Consultations: 45 triggered (35.4%), 15 helped              │
# │ Time: 1h 21m elapsed, ~4h remaining                         │
# │ Cost: $0.045                                                │
# │                                                             │
# │ Recent Tasks:                                               │
# │ ✓ django__django-11099 (PASS, 41s, consulted)              │
# │ ✗ django__django-11133 (FAIL, 38s)                         │
# │ ✓ sklearn__sklearn-12345 (PASS, 52s)                       │
# └─────────────────────────────────────────────────────────────┘
```

## Error Handling

### Error Log (`errors.log`)

```
2024-12-16 15:23:45 ERROR [Task: sympy__sympy-14024] Model timeout after 120s
  Retrying (attempt 2/3)...
2024-12-16 15:25:52 ERROR [Task: sympy__sympy-14024] Model timeout after 120s
  Retrying (attempt 3/3)...
2024-12-16 15:27:59 ERROR [Task: sympy__sympy-14024] Max retries exceeded
  Marking as FAIL (timeout)

2024-12-16 16:45:12 WARNING [Task: requests__requests-5287] Patch applies but
  introduces new test failure: test_redirect_history
  Marking as FAIL (regression)
```

### Error Categories

| Category | Description | Handling |
|----------|-------------|----------|
| `timeout` | Model didn't respond in time | Retry 3x, then FAIL |
| `parse_error` | Couldn't parse model output | Log and FAIL |
| `patch_error` | Patch doesn't apply | FAIL |
| `test_timeout` | Tests took too long | Use SWE-bench default |
| `regression` | Patch breaks other tests | FAIL |

## Post-Run Analysis

### Generate Report

```bash
python scripts/generate_report.py --run logs/run_20241216_143022/

# Outputs:
# - results/comparison/report.md (human-readable)
# - results/comparison/report.json (machine-readable)
# - results/comparison/visualizations/ (charts)
```

### Consultation Deep Dive

```bash
# Analyze all consultations
python scripts/analyze_consultations.py --run logs/current/

# Output:
# Consultations that changed outcome to PASS: 52
# Consultations that changed outcome to FAIL: 6
# Tasks where GPT insight was key: 23
# Tasks where Gemini insight was key: 18
# Tasks where both contributed: 11
```

## Archiving

After a run completes:

```bash
# Compress logs for storage
tar -czvf archives/run_20241216_143022.tar.gz logs/run_20241216_143022/

# Upload to cloud storage (optional)
aws s3 cp archives/run_20241216_143022.tar.gz s3://polydev-swebench-runs/
```

## Alerts (Optional)

Configure alerts in `monitoring/alerts.yaml`:

```yaml
alerts:
  - name: "High Error Rate"
    condition: "error_rate > 0.05"
    action: "log_warning"

  - name: "Run Complete"
    condition: "progress.percent_complete == 100"
    action: "send_notification"
    notification:
      type: "slack"
      channel: "#swebench-runs"
```
