# SWE-bench Realistic Evaluation Results

> **Date**: December 21, 2024
> **Model**: Claude Haiku 4.5 (via Claude CLI)
> **Instances**: 20 (new subset from SWE-bench Verified)

## Executive Summary

| Metric | Value |
|--------|-------|
| **Instances Evaluated** | 20 |
| **Patches Generated** | 10 (50%) |
| **Patches Applied** | 4 (40% of generated) |
| **Issues Resolved** | **0 (0%)** |

This is a **realistic baseline** using Claude CLI's agentic mode to generate patches from scratch, without any gold patches or hints.

## Agent Performance Metrics

### Overall Statistics

| Metric | Total | Per Instance |
|--------|-------|--------------|
| Steps | 566 | 28.3 |
| Tokens | 138,599 | 6,930 |
| Cost | $6.86 | $0.34 |
| Time | 51.5 min | 2.6 min |

### Instance Breakdown

| Instance | Patch Generated | Patch Applied | Resolved | Steps | Cost | Time |
|----------|-----------------|---------------|----------|-------|------|------|
| django__django-15973 | ✅ 4168 chars | ❌ Error | ❌ | 30 | $0.28 | 104s |
| django__django-15572 | ❌ | - | - | 30 | $0.25 | 165s |
| django__django-13344 | ✅ 686 chars | ✅ | ❌ | 30 | $0.42 | 234s |
| pydata__xarray-7229 | ❌ | - | - | 30 | $0.38 | 238s |
| django__django-13109 | ❌ | - | - | 30 | $0.28 | 86s |
| django__django-13658 | ❌ | - | - | 30 | $0.22 | 101s |
| django__django-13837 | ✅ 968 chars | ❌ Error | ❌ | 30 | $0.21 | 103s |
| django__django-12193 | ✅ 891 chars | ❌ Error | ❌ | 14 | $0.16 | 104s |
| sphinx-doc__sphinx-8595 | ❌ | - | - | 30 | $0.44 | 156s |
| django__django-11477 | ❌ | - | - | 30 | $0.24 | 90s |
| django__django-13925 | ❌ | - | - | 30 | $0.45 | 237s |
| sphinx-doc__sphinx-7590 | ❌ | - | - | 30 | $0.54 | 188s |
| django__django-15695 | ✅ 2154 chars | ✅ | ❌ | 24 | $0.42 | 196s |
| matplotlib__matplotlib-25311 | ✅ 492KB | ❌ Error | ❌ | 30 | $0.32 | 218s |
| scikit-learn__scikit-learn-14496 | ✅ 1215 chars | ❌ Error | ❌ | 18 | $0.40 | 108s |
| scikit-learn__scikit-learn-25747 | ✅ 835 chars | ❌ Error | ❌ | 30 | $0.27 | 133s |
| matplotlib__matplotlib-25960 | ✅ 5176 chars | ✅ | ❌ | 30 | $0.38 | 177s |
| django__django-11490 | ✅ 6070 chars | ✅ | ❌ | 30 | $0.46 | 184s |
| pytest-dev__pytest-10356 | ❌ | - | - | 30 | $0.34 | 143s |
| matplotlib__matplotlib-20676 | ❌ | - | - | 30 | $0.40 | 127s |

## Failure Analysis

### Why Patches Failed to Apply (6/10)

1. **Wrong file paths**: Agent used incorrect paths (e.g., `sklearn/cluster/_optics.py` didn't exist at that commit)
2. **Wrong line numbers**: Patches targeted wrong line locations
3. **Huge patches**: matplotlib__matplotlib-25311 generated 492KB - likely included entire files instead of targeted changes
4. **Version mismatches**: Patches referenced code that didn't exist in the base commit

### Why Tests Failed (4/10 that applied)

1. Patches fixed the wrong thing
2. Incomplete fixes that missed edge cases
3. Breaking changes to other functionality

## Comparison

| Approach | Resolve Rate | Notes |
|----------|--------------|-------|
| Gold Patches (infrastructure validation) | 100% (20/20) | Known correct answers |
| **Claude Haiku 4.5 Realistic (this run)** | **0% (0/10)** | Agent-generated |
| SWE-bench Leaderboard (best open-source) | ~20-30% | Various approaches |
| SWE-bench Leaderboard (best overall) | ~50% | Often with RAG/retrieval |

## Key Insights

### 1. Patch Generation vs Resolution

- **50% patch generation** doesn't mean **50% resolution**
- Most patches have structural issues (wrong paths, wrong line numbers)
- Even correctly applied patches often don't fix the actual issue

### 2. Agent Limitations

- 30 turns is often not enough for complex debugging
- Agent doesn't verify patches before generating
- No test-driven development approach

### 3. Cost Efficiency

- Average $0.34 per instance with Haiku
- Would be ~$3.40/instance with Sonnet, ~$13.70/instance with Opus
- At 0% resolution, any cost is wasted

## Recommendations for Improvement

1. **Add patch validation step** before outputting
2. **Run tests within agent loop** to verify fixes
3. **Increase max turns** for complex issues
4. **Use Polydev multi-model consultation** for uncertain cases
5. **Add retrieval augmentation** for codebase understanding

## Reproduction

```bash
# Run the evaluation
cd /Users/venkat/Documents/polydev-swe-bench
python3 run_realistic_eval.py --model haiku --instances 20 --max-turns 30 --timeout 600

# Evaluate generated patches
python3 -m swebench.harness.run_evaluation \
    -p results/realistic_eval/predictions_haiku.jsonl \
    -d princeton-nlp/SWE-bench_Verified \
    -id haiku-realistic \
    -t 1800
```

## Files

- **Runner**: `run_realistic_eval.py`
- **Predictions**: `results/realistic_eval/predictions_haiku.jsonl`
- **Metrics**: `results/realistic_eval/results_haiku.json`
- **Logs**: `logs/run_evaluation/haiku-realistic/`

---

*This is a baseline for comparison with Polydev-enhanced approaches.*
