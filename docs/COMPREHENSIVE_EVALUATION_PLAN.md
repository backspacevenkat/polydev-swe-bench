# Comprehensive SWE-bench Evaluation Plan

## Executive Summary

This plan outlines a robust, token-efficient approach to generate **two sets of patches** for the SWE-bench Verified benchmark:

1. **Baseline Run**: Claude Haiku 4.5 with extended thinking (targeting 73%+)
2. **Polydev-Enhanced Run**: Baseline + multi-model consultation (targeting 75%+)

This dual approach demonstrates Polydev's value for the paper.

---

## Part 1: Docker Rate Limiting Solutions

### Problem
- Docker Hub limits: **10 pulls/hour** (unauthenticated), **100 pulls/hour** (authenticated)
- Our previous run failed 51 evaluations due to 429 errors

### Solutions (In Order of Preference)

#### Option A: Use Epoch AI's Pre-built Registry (RECOMMENDED)
```bash
# Epoch AI maintains pre-built SWE-bench images
# Repository: github.com/orgs/epoch-research/packages?repo_name=SWE-bench
# Size: 30 GiB for all 500 SWE-bench Verified images
# Speed: Can evaluate all 500 in ~62 minutes!

# Configure Docker to use their registry
docker pull ghcr.io/epoch-research/swe-bench/<image-name>
```

#### Option B: Docker Hub Authentication
```bash
# Login to Docker Hub (100 pulls/hour vs 10)
docker login -u <username> -p <token>

# Export credentials for evaluation
export DOCKER_USERNAME=<username>
export DOCKER_PASSWORD=<token>
```

#### Option C: Use Google's Docker Mirror
```bash
# Configure Docker to use mirror.gcr.io
# Edit /etc/docker/daemon.json:
{
  "registry-mirrors": ["https://mirror.gcr.io"]
}

# Restart Docker
sudo systemctl restart docker
```

#### Option D: Pre-pull All Images (Local Caching)
```bash
# Pre-pull all 500 images before evaluation
python pre_pull_images.py --dataset princeton-nlp/SWE-bench_Verified

# Images cached locally, no rate limiting during evaluation
```

### Recommended Approach
1. **Primary**: Use Epoch AI's registry (fastest, no rate limits)
2. **Fallback**: Docker Hub authenticated + rate limiting in our code
3. **Backup**: Pre-pull images before running evaluation

---

## Part 2: Robust Patch Generation Pipeline

### Design Principles
1. **Checkpointing**: Save progress after each instance
2. **Resume capability**: Continue from where we left off
3. **Retry logic**: Handle transient failures
4. **Rate limiting**: Respect API limits
5. **Validation**: Verify patches before saving

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROBUST EVALUATION PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. LOAD & CHECKPOINT                                           │
│     ├── Load dataset (500 instances)                            │
│     ├── Load existing progress (if resuming)                    │
│     └── Skip already completed instances                        │
│                                                                  │
│  2. PARALLEL EXECUTION (10 workers)                             │
│     ├── Clone repo with retry (3 attempts)                      │
│     ├── Run Claude CLI with extended thinking                   │
│     ├── Validate patch (non-empty, valid diff)                  │
│     └── Save immediately after each instance                    │
│                                                                  │
│  3. PROGRESS TRACKING                                           │
│     ├── Real-time progress file (JSON)                          │
│     ├── Per-instance log files                                  │
│     └── Summary statistics                                      │
│                                                                  │
│  4. ERROR HANDLING                                              │
│     ├── Git clone failures → retry with delay                   │
│     ├── API timeouts → retry once                               │
│     ├── Rate limiting → exponential backoff                     │
│     └── Fatal errors → log and continue                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Checkpointing Strategy

```python
# Progress saved after EACH instance completion
progress = {
    "run_id": "haiku-thinking-v1",
    "started_at": "2024-12-22T23:00:00",
    "completed_instances": ["instance_1", "instance_2", ...],
    "failed_instances": {"instance_3": "error message"},
    "stats": {
        "total": 500,
        "completed": 245,
        "patches_generated": 240,
        "errors": 5,
        "total_cost": 5.25
    }
}

# Resume logic
def should_skip(instance_id, progress):
    return instance_id in progress["completed_instances"]
```

---

## Part 3: Two-Patch Strategy for Paper

### Goal
Demonstrate that Polydev's multi-model consultation improves SWE-bench results.

### Approach

#### Run 1: Baseline (Extended Thinking Only)
```
Model: claude-haiku-4-5-20251001
Thinking: 128K tokens
Prompt: Anthropic's methodology
Expected: ~73% resolution
Cost: ~$10-15
```

#### Run 2: Polydev-Enhanced
```
Model: claude-haiku-4-5-20251001
Thinking: 128K tokens
Polydev: Multi-model consultation for complex instances
Expected: ~75%+ resolution
Cost: ~$15-20 (additional Polydev calls)
```

### Polydev Integration Strategy

```python
# Token-efficient Polydev integration
def solve_with_polydev(instance):
    # Step 1: Initial analysis with Claude (cheap)
    initial_analysis = claude_analyze(instance)

    # Step 2: Consult Polydev only for complex cases
    if is_complex(initial_analysis):
        # Get perspectives from GPT-4, Claude, Gemini, Grok
        perspectives = polydev.get_perspectives(
            f"SWE-bench issue analysis: {instance['problem_statement'][:2000]}"
        )

        # Synthesize insights
        enhanced_prompt = synthesize_insights(initial_analysis, perspectives)
    else:
        enhanced_prompt = initial_analysis

    # Step 3: Generate patch with enhanced context
    patch = claude_generate_patch(enhanced_prompt)
    return patch

def is_complex(analysis):
    """Determine if instance needs multi-model consultation"""
    indicators = [
        len(analysis.get('files_to_modify', [])) > 3,
        'architectural' in analysis.lower(),
        'multiple components' in analysis.lower(),
        analysis.get('confidence', 1.0) < 0.7
    ]
    return sum(indicators) >= 2
```

### Token Efficiency

| Component | Tokens/Instance | Cost/Instance |
|-----------|-----------------|---------------|
| Extended Thinking | ~15K in, 1K out | $0.02 |
| Polydev (20% of instances) | ~2K per model | $0.01 avg |
| **Total Baseline** | ~16K | **$0.02** |
| **Total Polydev** | ~20K | **$0.03** |

### Expected Improvements

| Metric | Baseline | Polydev-Enhanced | Improvement |
|--------|----------|------------------|-------------|
| Resolution Rate | 73% | 75%+ | +2-3% |
| Complex Instance Rate | 60% | 70%+ | +10% |
| Cost | $10-15 | $15-20 | +$5 |

---

## Part 4: Execution Plan

### Phase 1: Preparation (30 mins)
1. ✓ Document plan
2. □ Configure Docker (use Epoch AI registry or authenticate)
3. □ Test single instance with both approaches
4. □ Verify checkpointing works

### Phase 2: Baseline Run (3-4 hours)
1. □ Run baseline evaluation with extended thinking
2. □ Monitor progress every 30 mins
3. □ Handle any failures/retries
4. □ Save predictions to `baseline_predictions.jsonl`

### Phase 3: Polydev-Enhanced Run (4-5 hours)
1. □ Identify complex instances from baseline run
2. □ Run Polydev consultation for complex instances
3. □ Generate enhanced patches
4. □ Save predictions to `polydev_predictions.jsonl`

### Phase 4: Evaluation (2-3 hours)
1. □ Configure Docker with Epoch AI registry
2. □ Run SWE-bench evaluation on baseline patches
3. □ Run SWE-bench evaluation on Polydev patches
4. □ Compare results and document

### Phase 5: Paper Results
1. □ Generate comparison tables
2. □ Document methodology
3. □ Create visualizations

---

## Part 5: Scripts Required

### 1. swe_bench_robust_baseline.py
- Extended thinking enabled
- Full checkpointing
- Resume capability
- Rate limiting

### 2. swe_bench_polydev_enhanced.py
- Same as baseline + Polydev integration
- Token-efficient multi-model consultation
- Only for complex instances

### 3. pre_pull_docker_images.py
- Pre-pull all 500 Docker images
- Use authenticated pulls
- Local caching

### 4. compare_results.py
- Compare baseline vs Polydev results
- Generate tables for paper
- Statistical analysis

---

## Part 6: Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Docker rate limiting | Medium | High | Use Epoch AI registry + auth |
| API timeouts | Low | Medium | Retry logic + checkpointing |
| Extended thinking not working | Low | High | Already verified in tests |
| Polydev API failures | Low | Medium | Fallback to baseline |
| Cost overrun | Medium | Medium | Monitor costs, set limits |

---

## Appendix: Quick Start Commands

```bash
# 1. Configure Docker
docker login  # Use your Docker Hub credentials

# 2. Run baseline
cd /Users/venkat/Documents/polydev-swe-bench
python scripts/swe_bench_robust_baseline.py

# 3. Monitor progress
tail -f /tmp/swe_robust_baseline/progress.json

# 4. Run Polydev-enhanced
python scripts/swe_bench_polydev_enhanced.py

# 5. Evaluate results
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path predictions/baseline_predictions.jsonl \
    --run_id baseline_eval

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path predictions/polydev_predictions.jsonl \
    --run_id polydev_eval
```

---

## Summary

This plan ensures:
1. ✅ No Docker rate limiting issues (Epoch registry + auth)
2. ✅ All 500 patches generated reliably (checkpointing + retries)
3. ✅ Token-efficient execution (targeted Polydev usage)
4. ✅ Two sets of patches for paper comparison
5. ✅ Clear methodology for reproducibility
