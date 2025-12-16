# Methodology

## Table of Contents

1. [Overview](#1-overview)
2. [Experimental Design](#2-experimental-design)
3. [Agent Architecture](#3-agent-architecture)
4. [Confidence Detection](#4-confidence-detection)
5. [Consultation Protocol](#5-consultation-protocol)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Statistical Analysis](#7-statistical-analysis)
8. [Reproducibility](#8-reproducibility)

---

## 1. Overview

### 1.1 Research Question

**Does multi-model consultation improve performance on software engineering tasks compared to single-model inference?**

### 1.2 Approach

We evaluate two configurations on SWE-bench Verified (500 tasks):

1. **Baseline**: Claude Opus 4.5 alone
2. **Enhanced**: Claude Opus 4.5 + Polydev MCP consultation (GPT-5.2, Gemini 3 Pro)

### 1.3 Key Innovation

Unlike ensemble methods that use multiple instances of the same model, we leverage **diverse models** with different training data, architectures, and capabilities:

| Model | Strengths |
|-------|-----------|
| Claude Opus 4.5 | Nuanced reasoning, careful analysis, strong code understanding |
| GPT-5.2 | Broad knowledge, practical solutions, code generation |
| Gemini 3 Pro | Technical depth, alternative perspectives, fast inference |

---

## 2. Experimental Design

### 2.1 Independent Variable

**Consultation Mode**:
- OFF: Single model (Claude alone)
- ON: Multi-model consultation when confidence < 8

### 2.2 Dependent Variable

**Pass Rate**: Percentage of tasks where the generated patch passes all tests

### 2.3 Control Variables

| Variable | Control |
|----------|---------|
| Base model | Claude Opus 4.5 (same version) |
| Task set | SWE-bench Verified (same 500 tasks) |
| Prompts | Identical prompts (see `agent/prompts/`) |
| Evaluation | Official SWE-bench harness |
| Temperature | 0.0 for reproducibility |
| Max tokens | 4096 per response |

### 2.4 Sample Size

- **SWE-bench Verified**: 500 human-verified tasks
- **Power analysis**: With 500 tasks, we can detect a 5% improvement with 80% power at α=0.05

---

## 3. Agent Architecture

### 3.1 Overview

```python
class PolydevAgent:
    """
    Lightweight agent for SWE-bench evaluation.

    Flow:
    1. Receive task (issue + repo)
    2. Analyze codebase and issue
    3. Generate solution hypothesis with confidence
    4. If low confidence, consult Polydev MCP
    5. Synthesize perspectives (if consulted)
    6. Generate patch
    """
```

### 3.2 Components

#### 3.2.1 Task Ingestion

```python
def ingest_task(self, task: SWEBenchTask) -> TaskContext:
    """
    Prepare task context for analysis.

    Returns:
        TaskContext with:
        - issue_description: str
        - repository: str
        - base_commit: str
        - relevant_files: List[str]
        - test_files: List[str]
    """
```

#### 3.2.2 Code Analysis

```python
def analyze_codebase(self, context: TaskContext) -> Analysis:
    """
    Analyze the codebase to understand the issue.

    Steps:
    1. Read issue description
    2. Identify potentially affected files
    3. Read relevant source files
    4. Trace code paths
    5. Identify root cause hypothesis

    Returns:
        Analysis with:
        - root_cause: str
        - affected_files: List[str]
        - relevant_code_snippets: Dict[str, str]
    """
```

#### 3.2.3 Solution Generation

```python
def generate_solution(self, analysis: Analysis) -> Solution:
    """
    Generate solution with confidence assessment.

    Returns:
        Solution with:
        - approach: str
        - confidence: int (1-10)
        - confidence_reasoning: str
        - proposed_changes: List[Change]
    """
```

#### 3.2.4 Consultation (if needed)

```python
def consult_polydev(self, context: TaskContext, analysis: Analysis,
                    solution: Solution) -> ConsultationResult:
    """
    Consult other models via Polydev MCP.

    Only called when solution.confidence < 8.

    Returns:
        ConsultationResult with:
        - gpt_perspective: str
        - gemini_perspective: str
        - all_perspectives: List[Perspective]
    """
```

#### 3.2.5 Synthesis

```python
def synthesize(self, solution: Solution,
               consultation: ConsultationResult) -> FinalSolution:
    """
    Claude synthesizes all perspectives and decides.

    Returns:
        FinalSolution with:
        - final_approach: str
        - reasoning: str
        - incorporated_from: List[str]  # Which models influenced
        - final_confidence: int
    """
```

#### 3.2.6 Patch Generation

```python
def generate_patch(self, solution: FinalSolution) -> str:
    """
    Generate unified diff patch.

    Returns:
        Patch in unified diff format:
        diff --git a/path/to/file.py b/path/to/file.py
        --- a/path/to/file.py
        +++ b/path/to/file.py
        @@ -10,5 +10,6 @@
        ...
    """
```

---

## 4. Confidence Detection

### 4.1 Confidence Scale

| Score | Level | Description | Action |
|-------|-------|-------------|--------|
| 9-10 | Very High | Clear problem, obvious solution | Proceed alone |
| 8 | High | Confident, minor uncertainties | Proceed alone |
| 7 | Moderate-High | Good understanding, some questions | Consult (recommended) |
| 5-6 | Moderate | Multiple approaches possible | Consult (recommended) |
| 3-4 | Low | Significant uncertainties | Consult (required) |
| 1-2 | Very Low | Largely guessing | Consult (required) |

### 4.2 Confidence Assessment Prompt

```
After analyzing this software engineering task, assess your confidence
in your proposed solution on a scale of 1-10.

Consider these factors:
1. PROBLEM CLARITY: Is the issue well-defined? (unclear = lower confidence)
2. ROOT CAUSE CERTAINTY: Are you sure about what's causing the issue?
3. SOLUTION UNIQUENESS: Is there one clear fix, or multiple approaches?
4. DOMAIN FAMILIARITY: How well do you know this codebase/library?
5. SIDE EFFECT RISK: Could your fix break other things?
6. EDGE CASES: Might there be edge cases you're missing?

Rate your confidence:
- 9-10: Very confident. Clear problem, clear solution, familiar domain.
- 7-8: Confident. Good understanding, minor uncertainties acceptable.
- 5-6: Moderate. Would benefit from a second opinion.
- 3-4: Low. Significant uncertainties about approach.
- 1-2: Very low. Largely uncertain, multiple unknowns.

Output your confidence score and brief reasoning.
```

### 4.3 Confidence Extraction

```python
def extract_confidence(self, response: str) -> Tuple[int, str]:
    """
    Extract confidence score from model response.

    Looks for patterns like:
    - "Confidence: 7/10"
    - "confidence score: 7"
    - "<confidence>7</confidence>"

    Returns:
        (score: int, reasoning: str)
    """
```

### 4.4 Threshold Selection

We use **confidence < 8** as the consultation threshold based on:

1. **Conservative**: 8+ indicates high confidence; below suggests uncertainty
2. **Practical**: Avoids over-consultation while catching genuine uncertainty
3. **Validated**: Tested on pilot tasks to ensure reasonable trigger rate (~30%)

---

## 5. Consultation Protocol

### 5.1 When to Consult

Consultation is triggered when:
- Confidence score < 8
- OR explicit uncertainty markers in response:
  - "I'm not sure"
  - "Multiple approaches possible"
  - "This could be done in several ways"
  - "I'm unfamiliar with this library"

### 5.2 Consultation Request Format

```
I'm working on a software engineering task and would like expert perspectives.

## Task Information
- Repository: {repo_name}
- Issue ID: {instance_id}
- Issue Description:
{issue_description}

## Relevant Code
{relevant_code_snippets}

## My Analysis
Root cause: {root_cause_hypothesis}
Affected files: {affected_files}

## My Proposed Solution
{proposed_solution}

## Why I'm Uncertain
{confidence_reasoning}

## Questions for Consultation
1. Is my diagnosis of the root cause correct?
2. Is my proposed approach sound?
3. Are there better alternatives?
4. What edge cases or risks should I consider?

Please provide your perspective on the best approach.
```

### 5.3 Models Consulted

| Model | Access Method | Cost |
|-------|--------------|------|
| GPT-5.2 | Codex CLI | Free |
| Gemini 3 Pro | Polydev MCP API | ~$0.001/query |

### 5.4 Response Collection

```python
def collect_perspectives(self, consultation_request: str) -> Dict[str, str]:
    """
    Query all consultation models in parallel.

    Uses Polydev MCP to query:
    - GPT-5.2 (via Codex CLI)
    - Gemini 3 Pro (via API)

    Returns:
        {
            "gpt-5.2": "GPT's perspective...",
            "gemini-3-pro": "Gemini's perspective..."
        }
    """
```

### 5.5 Synthesis Protocol

After receiving perspectives, Claude synthesizes:

```
I've received perspectives from multiple expert models.

## Original Proposal
{my_original_proposal}

## GPT-5.2 Perspective
{gpt_response}

## Gemini 3 Pro Perspective
{gemini_response}

## Synthesis

### Points of Agreement
[What all models agree on]

### Points of Disagreement
[Where models differ]

### Evaluation of Each Perspective
- GPT-5.2: [Strengths/weaknesses of this perspective]
- Gemini 3 Pro: [Strengths/weaknesses of this perspective]
- My original: [Strengths/weaknesses]

### Final Decision
[My chosen approach and why]

### Incorporated Insights
[What I'm taking from each model]

Proceeding with: [final_approach]
```

---

## 6. Evaluation Protocol

### 6.1 SWE-bench Harness

We use the official SWE-bench evaluation harness:

```bash
# Official evaluation command
python -m swebench.harness.run_evaluation \
    --predictions_path results/predictions.json \
    --swe_bench_tasks swe-bench-verified \
    --log_dir logs/evaluation/ \
    --testbed /tmp/swebench_testbed \
    --timeout 900 \
    --verbose
```

### 6.2 Evaluation Criteria

A task is considered **PASSED** if:
1. Patch applies cleanly to the repository
2. All tests in `FAIL_TO_PASS` now pass
3. All tests in `PASS_TO_PASS` still pass
4. No new test failures introduced

### 6.3 Per-Task Logging

For each task, we record:

```json
{
  "instance_id": "django__django-11099",
  "timestamp": "2024-12-16T14:30:22Z",
  "configuration": "polydev",

  "analysis_phase": {
    "time_seconds": 12.3,
    "files_read": ["django/db/models/query.py", "..."],
    "root_cause_identified": "Query compilation order"
  },

  "solution_phase": {
    "time_seconds": 8.5,
    "initial_confidence": 6,
    "confidence_reasoning": "Multiple approaches possible",
    "proposed_approach": "Fix filter chain ordering"
  },

  "consultation_phase": {
    "triggered": true,
    "time_seconds": 15.2,
    "models_queried": ["gpt-5.2", "gemini-3-pro"],
    "gpt_summary": "Suggested alternative approach...",
    "gemini_summary": "Highlighted edge case...",
    "synthesis_reasoning": "Combined insights from...",
    "final_confidence": 8,
    "approach_changed": true
  },

  "patch_phase": {
    "time_seconds": 5.1,
    "files_modified": ["django/db/models/query.py"],
    "lines_changed": 12
  },

  "evaluation": {
    "patch_applies": true,
    "tests_passed": true,
    "fail_to_pass": ["test_count_distinct_annotate"],
    "pass_to_pass_maintained": true
  },

  "totals": {
    "time_seconds": 41.1,
    "cost_usd": 0.0012,
    "result": "PASS"
  }
}
```

---

## 7. Statistical Analysis

### 7.1 Primary Analysis

**McNemar's Test** for paired binary outcomes:

- Null hypothesis (H₀): Consultation doesn't affect pass rate
- Alternative (H₁): Consultation improves pass rate
- Significance level: α = 0.05

```python
from statsmodels.stats.contingency_tables import mcnemar

# Contingency table
#                  Polydev Pass  Polydev Fail
# Baseline Pass        a            b
# Baseline Fail        c            d

# Test statistic
result = mcnemar([[a, b], [c, d]], exact=True)
print(f"p-value: {result.pvalue}")
```

### 7.2 Secondary Analyses

1. **By Repository**: Pass rate comparison per repository
2. **By Difficulty**: Easy/Medium/Hard task breakdown
3. **Consultation Effectiveness**: % of consultations that improved outcome
4. **Cost Analysis**: Cost per additional task solved

### 7.3 Effect Size

**Odds Ratio** for consultation effectiveness:

```
OR = (tasks_helped_by_consultation * tasks_not_needing_consultation) /
     (tasks_hurt_by_consultation * tasks_needing_but_not_helped)
```

---

## 8. Reproducibility

### 8.1 Version Pinning

All dependencies are pinned in `requirements.txt`:

```
anthropic==0.40.0
openai==1.55.0
google-generativeai==0.8.0
swebench==1.0.0
...
```

### 8.2 Random Seed

Set via environment variable:

```bash
export POLYDEV_SEED=42
```

Applied to:
- Any sampling operations
- Model temperature (set to 0)
- Task ordering

### 8.3 Prompt Versioning

All prompts are stored in `agent/prompts/` with version headers:

```
# Version: 1.0.0
# Last Modified: 2024-12-16
# Description: Issue analysis prompt

[Prompt content...]
```

### 8.4 Run Verification

Each run produces a verification hash:

```python
def compute_run_hash(config, prompts, results) -> str:
    """
    Compute deterministic hash of run configuration.

    Includes:
    - Configuration parameters
    - Prompt file hashes
    - Model versions
    - Task IDs processed
    """
```

### 8.5 Artifact Storage

All artifacts are stored with metadata:

```
results/
├── run_20241216_143022/
│   ├── config.yaml          # Exact configuration used
│   ├── prompts_hash.txt     # Hash of all prompts
│   ├── results.json         # Full results
│   ├── verification.json    # Reproducibility info
│   └── logs/               # Complete logs
```

---

## Appendix A: Task Selection

SWE-bench Verified contains 500 tasks from these repositories:

| Repository | Tasks | Description |
|------------|-------|-------------|
| django/django | 143 | Web framework |
| scikit-learn/scikit-learn | 89 | ML library |
| matplotlib/matplotlib | 67 | Plotting library |
| sympy/sympy | 52 | Symbolic math |
| pytest-dev/pytest | 41 | Testing framework |
| astropy/astropy | 38 | Astronomy library |
| sphinx-doc/sphinx | 35 | Documentation |
| pallets/flask | 18 | Web framework |
| psf/requests | 17 | HTTP library |

## Appendix B: Prompt Templates

See `agent/prompts/` for complete prompt templates:

- `analysis.txt`: Initial issue analysis
- `confidence.txt`: Confidence assessment
- `consultation.txt`: Consultation request
- `synthesis.txt`: Response synthesis
- `patch.txt`: Patch generation

## Appendix C: Error Handling

| Error Type | Handling |
|------------|----------|
| Model timeout | Retry 3x with exponential backoff |
| Parse error | Log and mark task as "error" |
| Patch apply failure | Record as "fail" with reason |
| Test timeout | Use SWE-bench default (900s) |
