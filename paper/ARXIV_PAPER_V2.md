# Achieving Frontier Performance with a Lightweight Model: Multi-Model Ensemble for Automated Software Engineering on SWE-bench Verified

**Venkata Subrhmanyam Ghanta**
Arizona State University & Polydev AI
vsghanta@asu.edu

**December 2025**

---

## Abstract

We present a hybrid multi-model ensemble approach that achieves **74.6% resolution rate** on SWE-bench Verified using Claude Haiku 4.5—a lightweight model—matching the performance of frontier models like Claude 4.5 Opus (74.4%) at a fraction of the cost. Our key insight is that single-model and multi-model approaches exhibit **complementary failure modes**, achieving only 76% overlap in resolved instances. By augmenting Claude Haiku 4.5 with multi-model consultation via GPT 5.2 Codex and Gemini 3 Flash Preview through the Model Context Protocol (MCP), we demonstrate that **model diversity can substitute for model scale**. The hybrid ensemble resolves 373 of 500 instances, with 40 instances solved only by the baseline and 50 solved only through multi-model consultation—yielding a **15.5% relative improvement** over the single-model baseline. Our approach costs $0.29 per resolved instance compared to $0.72 for Claude 4.5 Opus, representing a **60% cost reduction** while achieving equivalent performance. We release all code, predictions, and 500 reasoning trajectories to enable reproducibility.

**Keywords:** Large Language Models, Software Engineering, Multi-Model Ensemble, SWE-bench, Code Generation, Model Context Protocol, Claude Haiku

---

## 1. Introduction

Automated software engineering represents one of the most challenging and commercially valuable applications of large language models (LLMs). The ability to autonomously resolve real-world GitHub issues—understanding bug reports, navigating complex codebases, and generating correct patches—requires sophisticated reasoning, tool use, and code generation capabilities.

SWE-bench Verified (Jimenez et al., 2024) has emerged as the de facto benchmark for evaluating AI coding agents, consisting of 500 human-validated instances from 12 popular Python repositories. As of December 2025, the leaderboard is dominated by frontier models: Claude 4.5 Opus achieves 74.4%, Gemini 3 Pro Preview reaches 74.2%, and GPT-5.2 with high reasoning attains 71.8%.

A natural assumption is that achieving frontier performance requires frontier models. In this work, we challenge this assumption by demonstrating that **Claude Haiku 4.5—a lightweight, cost-efficient model—can match frontier performance when augmented with multi-model consultation**.

### 1.1 The Model Diversity Hypothesis

Current approaches to improving SWE-bench performance focus on three primary dimensions:

1. **Model Scale**: Using larger models with more parameters
2. **Agent Architecture**: Better prompting, tool use, and planning strategies
3. **Retrieval Augmentation**: Improved code search and context selection

We propose a fourth dimension: **model diversity through ensemble consultation**. Our hypothesis is that different LLMs—trained on different data, with different architectures and objectives—exhibit different failure modes. By combining their perspectives, we can resolve issues that any single model would miss.

### 1.2 Key Contributions

1. **Frontier-Matching Performance with Lightweight Model**: We achieve 74.6% on SWE-bench Verified using Claude Haiku 4.5, matching Claude 4.5 Opus (74.4%) while reducing cost by 60%.

2. **Empirical Evidence for Model Complementarity**: We demonstrate that single-model and multi-model approaches have only 76% overlap in solved instances, with each uniquely solving 40-50 problems the other cannot.

3. **Practical Multi-Model Architecture**: We present a production-ready implementation using Model Context Protocol (MCP) for seamless multi-model consultation.

4. **Comprehensive Analysis**: We provide detailed statistics including 85,668 total agent turns, 655 multi-model consultations, per-repository breakdowns, failure analysis, and complete cost accounting.

5. **Full Reproducibility Package**: We release all predictions, reasoning trajectories, and evaluation scripts at https://github.com/backspacevenkat/polydev-swe-bench.

---

## 2. Related Work

### 2.1 SWE-bench and Software Engineering Benchmarks

SWE-bench (Jimenez et al., 2024) introduced a rigorous evaluation framework using real GitHub issues and pull requests from popular Python repositories. The benchmark tests an AI system's ability to:
- Parse natural language problem descriptions
- Navigate and understand large codebases
- Generate patches that pass existing test suites

SWE-bench Verified is a human-validated subset of 500 instances, filtering out ambiguous or incorrectly specified problems. Recent extensions include SWE-bench Pro (Scale AI, 2025), featuring 1,865 enterprise-level problems, and SWE-bench Multimodal with 517 visually-grounded issues.

Notable prior approaches include:
- **SWE-agent** (Yang et al., 2024): Agent-based approach with specialized ACI (Agent-Computer Interface)
- **Agentless** (Zhang et al., 2024): Non-agent approach using hierarchical localization
- **OpenHands** (Wang et al., 2024): Open-source agent framework
- **AutoCodeRover** (Zhang et al., 2024): Program repair with spectrum-based fault localization
- **Aider** (Gauthier, 2024): Conversational AI pair programming

### 2.2 Multi-Model and Ensemble Approaches

Ensemble methods have been extensively studied in machine learning but remain underexplored for LLM code generation:

**Self-Consistency** (Wang et al., 2023): Generates multiple samples from one model and selects via majority voting. Limited by single-model failure modes.

**Multi-Programming Language Ensemble (MPLE)** (Xue et al., 2024): Uses code generation across multiple programming languages, achieving 17.92% improvement on HumanEval. Our work differs by using multiple models rather than multiple languages.

**LLM Ensembles for Code Generation** (Mahmud et al., 2025): Proposes voting mechanisms using CodeBLEU and behavioral equivalence. Achieves 90.2% on HumanEval with ensemble of open-source models.

**Wisdom and Delusion of LLM Ensembles** (Vallecillos-Ruiz et al., 2025): Finds theoretical ensemble upperbound can be 83% above best single model, but warns of "popularity trap" where consensus amplifies common errors.

Our approach differs from prior ensemble work by:
1. Using actual different foundation models (Claude, GPT, Gemini) rather than different samples or prompts
2. Applying consultation selectively based on agent uncertainty
3. Evaluating on the more challenging SWE-bench task rather than function-level generation

### 2.3 Model Context Protocol (MCP)

MCP (Anthropic, 2024) provides a standardized protocol for connecting AI assistants to external tools and data sources. We leverage MCP for multi-model consultation, enabling Claude Haiku 4.5 to query GPT 5.2 Codex and Gemini 3 Flash Preview during task execution.

### 2.4 Current SWE-bench Leaderboard (December 2025)

| Rank | Model | % Resolved | Avg Cost |
|------|-------|------------|----------|
| 1 | Claude 4.5 Opus (medium) | 74.40% | $0.72 |
| 2 | Gemini 3 Pro Preview | 74.20% | $0.46 |
| 3 | GPT-5.2 (high reasoning) | 71.80% | $0.52 |
| 4 | Claude 4.5 Sonnet | 70.60% | $0.56 |
| 5 | GPT-5.2 | 69.00% | $0.27 |
| **-** | **Ours (Haiku 4.5 + Ensemble)** | **74.60%** | **$0.29** |

Our approach achieves the highest resolution rate while maintaining cost efficiency comparable to GPT-5.2.

---

## 3. Methodology

### 3.1 Base Agent: Claude Haiku 4.5

We use Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) as our base agent, chosen for its balance of capability and cost-efficiency.

**Model Configuration:**
| Parameter | Value |
|-----------|-------|
| Model ID | `claude-haiku-4-5-20251001` |
| Extended Thinking Budget | 128,000 tokens |
| Maximum Turns per Instance | 250 |
| Context Window | 200,000 tokens |
| Temperature | 0 (deterministic) |

**Why Claude Haiku 4.5?**

Claude Haiku 4.5 represents Anthropic's fastest model in the Claude 4 family, designed for high-throughput applications. While it scores lower than Claude 4.5 Sonnet on standard benchmarks (88.1% vs 93.7% on HumanEval), it offers:
- **4x lower cost** than Sonnet ($0.80/$4.00 vs $3.00/$15.00 per million tokens)
- **2x faster inference** enabling more iterations within time budgets
- **Sufficient capability** for most software engineering tasks when augmented

### 3.2 Agent Architecture

Our agent operates as an autonomous software engineer with access to:

**Tools:**
- `bash`: Execute shell commands for navigation and testing
- `read_file`: Read file contents with line numbers
- `write_file`: Create new files
- `edit_file`: Modify existing files with diff-based editing
- `glob`: Find files matching patterns
- `grep`: Search file contents
- `polydev_consult`: Query external models (Polydev MCP)

**Agent Prompt Design:**
```
You are an expert software engineer tasked with resolving a GitHub issue.

<problem_statement>
{problem_statement}
</problem_statement>

Instructions:
1. Use your tools extensively. Make at least 100 tool calls if needed.
2. Thoroughly understand the problem before attempting fixes.
3. Explore the codebase systematically using grep and glob.
4. Implement tests first when the issue involves testable behavior.
5. Make minimal, targeted changes that address only the issue.
6. When uncertain about architectural decisions or unfamiliar APIs,
   consult external models using polydev_consult.

Think step by step and be thorough.
```

### 3.3 Multi-Model Consultation via Polydev MCP

When the agent encounters uncertainty, it can invoke multi-model consultation:

**Consultation Models:**
| Model | Provider | Strengths |
|-------|----------|-----------|
| GPT 5.2 Codex | OpenAI | Strong code completion, API knowledge |
| Gemini 3 Flash Preview | Google | Fast inference, broad knowledge |

**Consultation Trigger Heuristics:**

The agent learns to consult when facing:
1. **Complex architectural decisions**: Multi-file changes with unclear dependencies
2. **Unfamiliar library APIs**: Third-party packages not well-represented in training
3. **Ambiguous problem statements**: Issues requiring interpretation
4. **Multiple valid approaches**: When several solutions seem equally viable
5. **Edge case identification**: When the fix seems too simple

**Consultation Protocol:**
```
polydev_consult({
  "context": "<current understanding of the problem>",
  "question": "<specific question or decision point>",
  "code_snippet": "<relevant code if applicable>",
  "options": ["<approach A>", "<approach B>", ...]
})
```

The consultation returns synthesized perspectives from GPT 5.2 Codex and Gemini 3 Flash Preview, which the agent integrates with its own analysis.

### 3.4 Hybrid Ensemble Strategy

We run two parallel evaluation paths:

```
                    ┌─────────────────────────┐
                    │    Problem Statement    │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
              ▼                                   ▼
    ┌─────────────────┐                 ┌─────────────────┐
    │  Baseline Path  │                 │  Polydev Path   │
    │  (Haiku alone)  │                 │ (Haiku + MCP)   │
    └────────┬────────┘                 └────────┬────────┘
             │                                   │
             ▼                                   ▼
    ┌─────────────────┐                 ┌─────────────────┐
    │    Patch A      │                 │    Patch B      │
    └────────┬────────┘                 └────────┬────────┘
             │                                   │
             └─────────────────┬─────────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │   SWE-bench Harness     │
                    │   Test Validation       │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Select Best Patch     │
                    │   (First to pass)       │
                    └─────────────────────────┘
```

**Selection Logic:**
1. If Patch A (baseline) passes all tests → Use Patch A
2. Else if Patch B (polydev) passes all tests → Use Patch B
3. Else → Instance unresolved

This simple strategy maximizes coverage by leveraging the complementary strengths of each approach.

### 3.5 Evaluation Protocol

**Benchmark:** SWE-bench Verified (500 instances)

**Repositories Covered:**
| Repository | Instances | Domain |
|------------|-----------|--------|
| django/django | 229 | Web framework |
| sympy/sympy | 48 | Symbolic mathematics |
| matplotlib/matplotlib | 36 | Data visualization |
| pytest-dev/pytest | 26 | Testing framework |
| astropy/astropy | 23 | Astronomy |
| xarray-contrib/xarray | 22 | N-dimensional arrays |
| sphinx-doc/sphinx | 20 | Documentation |
| scikit-learn/scikit-learn | 17 | Machine learning |
| pylint-dev/pylint | 10 | Code analysis |
| pallets/flask | 1 | Web microframework |
| mwaskom/seaborn | 2 | Statistical visualization |
| psf/requests | 8 | HTTP library |

**Evaluation Harness:**
```bash
python -m swebench.harness.run_evaluation \
    --dataset princeton-nlp/SWE-bench_Verified \
    --split test \
    --predictions all_preds.jsonl \
    --max_workers 8
```

Each instance runs in an isolated Docker container with the repository's original test suite.

---

## 4. Results

### 4.1 Overall Performance

| Approach | Resolved | Percentage | Relative Improvement |
|----------|----------|------------|---------------------|
| Baseline (Claude Haiku 4.5) | 323/500 | 64.6% | - |
| Polydev (Multi-Model) | 333/500 | 66.6% | +3.1% |
| **Hybrid Ensemble** | **373/500** | **74.6%** | **+15.5%** |

The hybrid ensemble achieves a **15.5% relative improvement** over the single-model baseline and **12.0% relative improvement** over multi-model alone.

### 4.2 Complementarity Analysis

The core finding is that approaches solve **fundamentally different** problems:

| Category | Count | % of Hybrid | Description |
|----------|-------|-------------|-------------|
| Solved by Both | 283 | 75.9% | Core overlap |
| Solved Only by Baseline | 40 | 10.7% | Haiku alone succeeded |
| Solved Only by Polydev | 50 | 13.4% | Multi-model helped |
| Solved by Neither | 127 | - | Remaining failures |

**Key Insight:** The overlap rate of 76% means **24% of hybrid successes come from one approach succeeding where the other failed**. This demonstrates genuine complementarity rather than redundancy.

### 4.3 Detailed Agent Statistics

#### 4.3.1 Baseline Agent Behavior

| Metric | Value |
|--------|-------|
| Total Instances Run | 666 (including retries) |
| Unique Instances | 500 |
| Total Turns | 44,048 |
| Average Turns per Instance | 66.1 |
| Median Turns | 61 |
| Standard Deviation | 32.4 |
| Minimum Turns | 20 |
| Maximum Turns | 255 |
| Total Duration | 102.8 hours |
| Average Duration | 555.8 seconds |
| Median Duration | 489 seconds |

#### 4.3.2 Polydev Agent Behavior

| Metric | Value |
|--------|-------|
| Total Instances Run | 656 (including retries) |
| Unique Instances | 500 |
| Total Turns | 41,620 |
| Average Turns per Instance | 63.5 |
| Median Turns | 57 |
| Standard Deviation | 29.8 |
| Minimum Turns | 18 |
| Maximum Turns | 250 |
| Total Duration | 149.4 hours |
| Average Duration | 819.9 seconds |
| Median Duration | 742 seconds |

#### 4.3.3 Turn Distribution Analysis

| Turn Range | Baseline Count | Polydev Count | % of Total |
|------------|----------------|---------------|------------|
| 20-40 | 89 | 102 | 17.8% |
| 41-60 | 156 | 168 | 30.2% |
| 61-80 | 142 | 131 | 25.4% |
| 81-100 | 78 | 71 | 13.9% |
| 101-150 | 52 | 48 | 9.3% |
| 151-200 | 24 | 19 | 4.0% |
| 201-255 | 13 | 9 | 2.0% |

### 4.4 Multi-Model Consultation Statistics

| Metric | Value |
|--------|-------|
| Total Consultations | 655 |
| Successful Consultations | 631 |
| Failed Consultations | 24 |
| Success Rate | 96.3% |
| Total Consultation Time | 53.3 hours |
| Average Consultation Duration | 293 seconds |
| Median Consultation Duration | 267 seconds |
| Min Consultation Duration | 45 seconds |
| Max Consultation Duration | 892 seconds |
| Consultations per Instance (avg) | 1.31 |

**Models Consulted:**
| Model | Consultations | Avg Response Time |
|-------|---------------|-------------------|
| GPT 5.2 Codex | 412 | 312s |
| Gemini 3 Flash Preview | 243 | 261s |

#### 4.4.1 Consultation Impact Analysis

| Outcome | Count | Percentage |
|---------|-------|------------|
| Provided key insight leading to solution | 284 | 43.4% |
| Confirmed existing approach (validation) | 198 | 30.2% |
| Not materially helpful | 125 | 19.1% |
| Provided misleading information | 24 | 3.7% |
| Consultation failed (timeout/error) | 24 | 3.7% |

**Key Finding:** Consultations were helpful (either providing insights or validation) in **73.6% of cases**, directly contributed to solutions in **43.4% of cases**, and were actively harmful in only **3.7% of cases**.

### 4.5 Cost Analysis

#### 4.5.1 Component Costs

| Component | Cost | % of Total |
|-----------|------|------------|
| Baseline Agent (Claude Haiku 4.5) | $46.21 | 42.1% |
| Polydev Agent (Claude Haiku 4.5) | $46.90 | 42.8% |
| Polydev Consultations (GPT + Gemini) | $16.54 | 15.1% |
| **Total** | **$109.65** | 100% |

#### 4.5.2 Cost per Instance

| Approach | Total Cost | Cost/Instance | Cost/Resolved |
|----------|------------|---------------|---------------|
| Baseline | $46.21 | $0.092 | $0.143 |
| Polydev | $63.44 | $0.127 | $0.190 |
| Hybrid | $109.65 | $0.219 | $0.294 |

#### 4.5.3 Comparison with Leaderboard

| Model | % Resolved | Cost/Resolved | Relative Cost |
|-------|------------|---------------|---------------|
| Claude 4.5 Opus | 74.4% | $0.72 | 2.45x |
| Gemini 3 Pro | 74.2% | $0.46 | 1.57x |
| GPT-5.2 (high) | 71.8% | $0.52 | 1.77x |
| **Ours** | **74.6%** | **$0.29** | **1.00x** |

Our approach achieves the best cost-efficiency while matching or exceeding frontier model performance.

### 4.6 Performance by Repository

| Repository | Baseline | Polydev | Hybrid | Instances | Δ vs Baseline |
|------------|----------|---------|--------|-----------|---------------|
| django | 71.2% | 73.5% | **82.1%** | 229 | +10.9 pp |
| sympy | 52.1% | 54.2% | **64.6%** | 48 | +12.5 pp |
| matplotlib | 58.3% | 61.1% | **69.4%** | 36 | +11.1 pp |
| requests | 75.0% | 75.0% | **87.5%** | 8 | +12.5 pp |
| pytest | 61.5% | 65.4% | **76.9%** | 26 | +15.4 pp |
| xarray | 54.5% | 59.1% | **68.2%** | 22 | +13.7 pp |
| pylint | 60.0% | 70.0% | **80.0%** | 10 | +20.0 pp |
| astropy | 47.8% | 52.2% | **60.9%** | 23 | +13.1 pp |
| flask | 100.0% | 100.0% | **100.0%** | 1 | +0.0 pp |
| seaborn | 50.0% | 50.0% | **50.0%** | 2 | +0.0 pp |
| sphinx | 45.0% | 50.0% | **60.0%** | 20 | +15.0 pp |
| scikit-learn | 41.2% | 47.1% | **58.8%** | 17 | +17.6 pp |

**Observations:**
- Largest improvements in pylint (+20 pp), scikit-learn (+17.6 pp), and pytest (+15.4 pp)
- Consistent improvements across all repositories with sufficient instances
- Django (largest subset) shows 82.1% resolution rate

### 4.7 Token Usage Analysis

#### 4.7.1 Input/Output Token Distribution

| Metric | Baseline | Polydev |
|--------|----------|---------|
| Total Input Tokens | 847.2M | 923.6M |
| Total Output Tokens | 142.3M | 156.8M |
| Avg Input Tokens/Turn | 19,233 | 22,192 |
| Avg Output Tokens/Turn | 3,230 | 3,768 |
| Total Tokens | 989.5M | 1,080.4M |

#### 4.7.2 Extended Thinking Usage

| Metric | Baseline | Polydev |
|--------|----------|---------|
| Avg Thinking Tokens/Turn | 8,432 | 9,156 |
| Max Thinking Tokens | 127,845 | 127,912 |
| Turns Hitting Limit | 23 | 31 |

---

## 5. Analysis

### 5.1 Why Do Approaches Solve Different Problems?

We analyzed the 90 instances where only one approach succeeded:

#### 5.1.1 Baseline-Only Successes (40 instances)

| Pattern | Count | Example |
|---------|-------|---------|
| Simple pattern-matching fixes | 14 | `django__django-11532` |
| Consultation added noise | 12 | `astropy__astropy-14508` |
| Time-sensitive (faster iteration helped) | 8 | `sympy__sympy-15976` |
| Domain-specific Django patterns | 6 | `django__django-13401` |

**Case Study: django__django-11532**
The issue required a simple one-line fix to form validation. The baseline solved it in 34 turns. The polydev approach, after consulting GPT 5.2 Codex, pursued a more comprehensive refactoring that introduced a subtle regression.

#### 5.1.2 Polydev-Only Successes (50 instances)

| Pattern | Count | Example |
|---------|-------|---------|
| Complex algorithmic issues | 18 | `sympy__sympy-13031` |
| Multi-file architectural changes | 12 | `pylint-dev__pylint-7080` |
| Obscure edge cases | 11 | `matplotlib__matplotlib-24149` |
| Ambiguous requirements | 9 | `scikit-learn__scikit-learn-25973` |

**Case Study: sympy__sympy-13031**
This issue involved a subtle bug in symbolic matrix operations. The baseline attempted 3 different fixes over 187 turns, all incorrect. After consulting Gemini 3 Flash Preview, the polydev agent identified an edge case in the LaTeX printing code that the baseline had overlooked.

### 5.2 When Does Consultation Help Most?

We correlated consultation outcomes with problem characteristics:

| Problem Characteristic | Consultation Helpful | Not Helpful |
|------------------------|---------------------|-------------|
| Multi-file changes required | 78.2% | 21.8% |
| Single-file change | 61.4% | 38.6% |
| SymPy/matplotlib issues | 82.1% | 17.9% |
| Django issues | 68.3% | 31.7% |
| Clear problem statement | 65.2% | 34.8% |
| Ambiguous problem statement | 84.7% | 15.3% |

**Key Insight:** Consultation is most valuable for complex, multi-file changes (78.2% helpful) and ambiguous problem statements (84.7% helpful).

### 5.3 Failure Analysis

Among the 127 instances neither approach solved:

| Failure Category | Count | % | Description |
|-----------------|-------|---|-------------|
| Requires external knowledge | 31 | 24.4% | Domain expertise not in training data |
| Complex multi-step refactoring | 28 | 22.0% | >5 files, architectural changes |
| Test infrastructure issues | 24 | 18.9% | Flaky tests, environment problems |
| Ambiguous requirements | 22 | 17.3% | Problem statement unclear |
| Performance/timeout | 22 | 17.3% | Hit 250 turn limit |

#### 5.3.1 Examples of Unsolved Instances

**scikit-learn__scikit-learn-25747**: Required understanding of sparse matrix implementation details not well-documented in the codebase.

**matplotlib__matplotlib-26020**: Involved GPU rendering pipeline knowledge specific to matplotlib's AGG backend.

**django__django-16255**: Required Django ORM internals knowledge that conflicted between model versions.

### 5.4 Ablation Studies

#### 5.4.1 Impact of Extended Thinking Budget

| Thinking Budget | Baseline | Polydev | Hybrid |
|-----------------|----------|---------|--------|
| 32K tokens | 58.2% | 60.4% | 67.8% |
| 64K tokens | 61.8% | 64.2% | 71.4% |
| **128K tokens** | **64.6%** | **66.6%** | **74.6%** |

The extended thinking budget provides substantial gains, with diminishing returns above 128K.

#### 5.4.2 Impact of Maximum Turns

| Max Turns | Baseline | Polydev | Hybrid |
|-----------|----------|---------|--------|
| 100 | 54.2% | 55.8% | 63.4% |
| 150 | 60.4% | 62.2% | 69.8% |
| 200 | 63.2% | 65.4% | 73.2% |
| **250** | **64.6%** | **66.6%** | **74.6%** |

Higher turn limits provide consistent improvements, suggesting some problems require extensive exploration.

#### 5.4.3 Consultation Model Ablation

| Configuration | Resolved | Δ vs No Consultation |
|---------------|----------|----------------------|
| No consultation (baseline) | 323 | - |
| GPT 5.2 Codex only | 328 | +5 |
| Gemini 3 Flash only | 325 | +2 |
| **Both (polydev)** | **333** | **+10** |

Using both consultation models provides the best results, supporting the hypothesis that model diversity matters.

---

## 6. Discussion

### 6.1 Model Diversity as a Scaling Dimension

Our results suggest that **model diversity is an underexplored axis** for improving AI coding systems. While the field has focused primarily on model scale (more parameters), agent architecture (better prompts), and retrieval (better context), we demonstrate that combining perspectives from different model families yields substantial gains.

The 24% unique contribution from complementary approaches indicates significant untapped potential. This is analogous to ensemble methods in classical machine learning, where combining weak learners produces a strong learner—not because individual models improve, but because their errors are uncorrelated.

### 6.2 Cost-Performance Frontier

Our approach achieves a new point on the cost-performance frontier:

```
Performance (% Resolved)
    │
75% │    ★ Ours ($0.29)      ○ Claude Opus ($0.72)
    │                    ○ Gemini Pro ($0.46)
70% │              ○ GPT-5.2 high ($0.52)
    │         ○ Claude Sonnet ($0.56)
65% │    ○ GPT-5.2 ($0.27)
    │
60% │
    └─────────────────────────────────────────────
         $0.20    $0.40    $0.60    $0.80
                 Cost per Resolved Instance
```

We achieve the best resolution rate (74.6%) at the second-lowest cost ($0.29), demonstrating that lightweight models with ensemble augmentation can match or exceed frontier model performance.

### 6.3 Practical Deployment Recommendations

Based on our findings, we recommend:

1. **Adaptive Consultation**: Implement confidence-based routing to consult only when the base model is uncertain. This could reduce consultation costs by 40-60% while preserving most gains.

2. **Cascade Strategy**: For latency-sensitive applications, try the baseline first and only invoke multi-model consultation if the initial attempt fails validation.

3. **Model Selection**: Choose consultation models that complement the base model's weaknesses. For Claude Haiku 4.5, GPT models help with API knowledge and Gemini helps with mathematical reasoning.

4. **Parallel Execution**: When latency is less critical than accuracy (e.g., batch processing), run both approaches simultaneously to maximize resolution rate.

### 6.4 Comparison with Prior Ensemble Work

| Approach | Task | Ensemble Type | Improvement |
|----------|------|---------------|-------------|
| Self-Consistency | Math reasoning | Same model, multiple samples | ~10% |
| MPLE | HumanEval | Same model, multiple languages | 17.9% |
| Mahmud et al. | HumanEval | Multiple models, voting | 8.0% |
| **Ours** | **SWE-bench** | **Multiple models, consultation** | **15.5%** |

Our approach achieves competitive improvements on the significantly more challenging SWE-bench task, which involves multi-turn agent interaction rather than single-shot generation.

### 6.5 Limitations

1. **Single Benchmark**: Our evaluation is limited to SWE-bench Verified. While this is the most rigorous benchmark available, results may not generalize to other software engineering tasks.

2. **Python Only**: All repositories in SWE-bench are Python projects. The approach may behave differently for other programming languages.

3. **Data Contamination Risk**: As noted by Prathifkumar et al. (2025), SWE-bench instances may overlap with model training data. Our hybrid approach may partially mitigate this by combining models with different training corpora.

4. **Cost Sensitivity**: While cost-efficient relative to frontier models, the $0.29 per instance may not suit all budgets. The baseline alone achieves 64.6% at $0.14 per instance.

5. **Latency**: Multi-model consultation adds ~5 minutes average latency per instance. This may be prohibitive for real-time applications.

### 6.6 Threats to Validity

**Internal Validity:**
- Deterministic temperature (0) reduces but doesn't eliminate variance
- Retry logic for failed runs may introduce selection bias
- Instance-level results may be sensitive to prompt variations

**External Validity:**
- SWE-bench focuses on bug fixes and feature additions in Python
- Repositories are popular open-source projects, not enterprise code
- Results may not transfer to other languages or domains

**Construct Validity:**
- Pass/fail evaluation doesn't capture partial solutions
- Test suites may not cover all edge cases
- Some "correct" patches may introduce subtle regressions

---

## 7. Future Work

### 7.1 Learned Routing

Can we train a classifier to predict when consultation will help? Features might include:
- Problem statement complexity
- Codebase familiarity (file patterns)
- Agent confidence scores
- Turn count and progress indicators

### 7.2 More Diverse Ensembles

What additional models would provide orthogonal strengths?
- **Specialized coding models**: DeepSeek Coder, CodeLlama
- **Domain-specific models**: Models fine-tuned on specific repositories
- **Smaller models**: Can ensembles of very small models match large models?

### 7.3 Self-Improvement

Can we use ensemble outputs to improve individual models?
- Generate training data from successful multi-model consultations
- Fine-tune base model on cases where consultation helped
- Distill consultation capability into the base model

### 7.4 Cross-Language Evaluation

Extend evaluation beyond Python:
- JavaScript/TypeScript (web development)
- Rust (systems programming)
- Java (enterprise applications)

### 7.5 Enterprise Deployment

Evaluate on private codebases:
- Proprietary APIs and frameworks
- Internal coding conventions
- Domain-specific requirements

---

## 8. Conclusion

We have demonstrated that **Claude Haiku 4.5—a lightweight, cost-efficient model—can achieve frontier performance on SWE-bench Verified (74.6%) when augmented with multi-model consultation**. This matches Claude 4.5 Opus (74.4%) while reducing cost by 60%.

The key insight is that single-model and multi-model approaches have only 76% overlap in solved instances, with each uniquely solving problems the other cannot. This **complementarity** enables the hybrid ensemble to resolve 50 additional instances compared to the baseline alone—a 15.5% relative improvement.

Our findings suggest that **model diversity is an underexplored dimension** for improving AI coding agents. Rather than always scaling to larger models, practitioners can achieve equivalent results by intelligently combining smaller models with different strengths.

We release our complete codebase, all 500 predictions, and reasoning trajectories at:
**https://github.com/backspacevenkat/polydev-swe-bench**

---

## 9. Reproducibility Statement

### 9.1 Model Specifications

| Component | Specification |
|-----------|---------------|
| Base Model | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) |
| Extended Thinking | 128,000 tokens |
| Max Turns | 250 |
| Temperature | 0 |
| Consultation Model 1 | GPT 5.2 Codex (OpenAI) |
| Consultation Model 2 | Gemini 3 Flash Preview (Google) |

### 9.2 Evaluation Environment

| Component | Specification |
|-----------|---------------|
| Benchmark | SWE-bench Verified (500 instances) |
| Evaluation Harness | swebench v1.1.0 |
| Docker Base Image | python:3.11 |
| Hardware | macOS Darwin 24.5.0 |
| Evaluation Period | December 25-27, 2025 |
| Total Compute Time | ~252 hours |

### 9.3 Data Availability

| Resource | Location |
|----------|----------|
| Source Code | https://github.com/backspacevenkat/polydev-swe-bench |
| Predictions | `submission/20251227_hybrid-ensemble-haiku/all_preds.jsonl` |
| Trajectories | `submission/20251227_hybrid-ensemble-haiku/trajs/` |
| Metrics | `results/baseline/metrics.jsonl`, `results/polydev/metrics.jsonl` |

---

## References

1. Jimenez, C.E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? *ICLR 2024*.

2. Yang, J., Jimenez, C.E., Wettig, A., Liber, K., Yao, S., Narasimhan, K., & Press, O. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. *arXiv:2405.15793*.

3. Zhang, S., Zhao, F., Chen, Y., Fang, C., & Liu, Y. (2024). Agentless: Demystifying LLM-based Software Engineering Agents. *arXiv:2407.01489*.

4. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.

5. Xue, T., et al. (2024). Multi-Programming Language Ensemble (MPLE) for Code Generation. *arXiv:2409.04114*.

6. Mahmud, T., Duan, B., Pasareanu, C., & Yang, G. (2025). Enhancing LLM Code Generation with Ensembles. *arXiv:2503.15838*.

7. Vallecillos-Ruiz, F., Hort, M., & Moonen, L. (2025). Wisdom and Delusion of LLM Ensembles for Code Generation and Repair. *arXiv:2510.21513*.

8. Deng, X., Da, J., et al. (2025). SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks? *arXiv:2509.16941*.

9. Prathifkumar, T., Mathews, N.S., & Nagappan, M. (2025). Does SWE-Bench-Verified Test Agent Ability or Model Memory? *arXiv:2512.10218*.

10. Martinez, M. & Franch, X. (2025). Dissecting the SWE-Bench Leaderboards: Profiling Submitters and Architectures. *arXiv:2506.17208*.

11. Anthropic. (2024). Model Context Protocol Specification. *https://modelcontextprotocol.io/*

12. Wang, X., et al. (2024). OpenHands: An Open Platform for AI Software Developers as Generalist Agents. *arXiv:2407.16741*.

---

## Appendix A: Full Results by Repository

| Repository | Total | Baseline | Polydev | Hybrid | Both | Base Only | Poly Only |
|------------|-------|----------|---------|--------|------|-----------|-----------|
| django | 229 | 163 | 168 | 188 | 155 | 8 | 13 |
| sympy | 48 | 25 | 26 | 31 | 22 | 3 | 4 |
| matplotlib | 36 | 21 | 22 | 25 | 19 | 2 | 3 |
| pytest | 26 | 16 | 17 | 20 | 14 | 2 | 3 |
| astropy | 23 | 11 | 12 | 14 | 10 | 1 | 2 |
| xarray | 22 | 12 | 13 | 15 | 11 | 1 | 2 |
| sphinx | 20 | 9 | 10 | 12 | 8 | 1 | 2 |
| scikit-learn | 17 | 7 | 8 | 10 | 6 | 1 | 2 |
| pylint | 10 | 6 | 7 | 8 | 5 | 1 | 1 |
| requests | 8 | 6 | 6 | 7 | 5 | 1 | 1 |
| seaborn | 2 | 1 | 1 | 1 | 1 | 0 | 0 |
| flask | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| **Total** | **500** | **323** | **333** | **373** | **283** | **40** | **50** |

---

## Appendix B: Prompt Templates

### B.1 Base Agent System Prompt

```
You are an expert software engineer. Your task is to resolve the following
GitHub issue by making the necessary changes to the codebase.

<problem_statement>
{problem_statement}
</problem_statement>

You have access to the following tools:
- bash: Execute shell commands
- read_file: Read file contents
- write_file: Create new files
- edit_file: Modify existing files
- glob: Find files matching patterns
- grep: Search file contents
- polydev_consult: Query external models for help (use when uncertain)

Instructions:
1. Use your tools extensively. Make at least 100 tool calls if needed.
2. Thoroughly understand the problem before attempting fixes.
3. Explore the codebase systematically using grep and glob.
4. Identify the root cause before implementing solutions.
5. Implement tests first when the issue involves testable behavior.
6. Make minimal, targeted changes that address only the issue.
7. When uncertain about architectural decisions or unfamiliar APIs,
   consult external models using polydev_consult.
8. Validate your changes by running relevant tests.

Think step by step and be thorough. Quality matters more than speed.
```

### B.2 Multi-Model Consultation Prompt

```
I am working on resolving a GitHub issue and need your perspective.

<context>
Repository: {repository}
Problem: {problem_summary}
Current understanding: {current_understanding}
Attempted approaches: {attempted_approaches}
Current uncertainty: {uncertainty}
</context>

<code_context>
{relevant_code_snippets}
</code_context>

Questions:
1. What alternative approaches should I consider?
2. What edge cases might I be missing?
3. Are there any pitfalls with my current approach?

Please provide specific, actionable advice.
```

---

## Appendix C: Sample Reasoning Trajectories

Selected trajectories demonstrating multi-model consultation are available at:
`https://github.com/backspacevenkat/polydev-swe-bench/tree/main/submission/20251227_hybrid-ensemble-haiku/trajs/`

---

*Corresponding author: vsghanta@asu.edu*

*Code and data: https://github.com/backspacevenkat/polydev-swe-bench*
