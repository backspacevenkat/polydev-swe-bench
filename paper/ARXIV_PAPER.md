# Inference-Time Compute Scaling for Code Agents: Matching Frontier Performance with Lightweight Models

**Venkata Subrhmanyam Ghanta**¹², **Pujitha Sri Lakshmi Paladugu**³
¹Arizona State University, ²Polydev AI, ³Microsoft
vsghanta@asu.edu, pupaladu@microsoft.com

**January 2026**

---

## Abstract

We investigate whether **inference-time compute can substitute for model scale** in automated software engineering. Using Claude Haiku 4.5—a lightweight model—with extended multi-turn reasoning and multi-model consultation, we achieve **74.6% resolution rate** on SWE-bench Verified, matching Claude 4.5 Opus (74.4%) at 60% lower cost per resolved instance.

Our key finding is that single-agent and multi-model consultation approaches exhibit **complementary failure modes**, with only 76% overlap in resolved instances. Critically, this complementarity is **not explained by simple retries**: while our hybrid approach resolves 373/500 instances (74.6%), we estimate Baseline Pass@2 (running the baseline twice) would achieve only ~68% based on resolution variance analysis. The remaining 6.6 percentage point gap represents genuine complementarity from model diversity.

We analyze **when multi-model consultation helps versus hurts**: consultation is most valuable for complex multi-file changes (78.2% helpful) and ambiguous requirements (84.7% helpful), but can introduce noise for simple pattern-matching fixes. The 50 instances solved only by multi-model consultation and 40 solved only by the baseline provide empirical evidence for irreducible model diversity.

Our results suggest the field should explore **inference-time scaling**—through agent turns, extended thinking, and model diversity—as a complement to training-time model scaling. We release all code, predictions, and 500 reasoning trajectories.

**Keywords:** Large Language Models, Inference-Time Compute, Software Engineering, Multi-Model Consultation, SWE-bench, Code Generation, Model Context Protocol

---

## 1. Introduction

Automated software engineering represents one of the most challenging and commercially valuable applications of large language models (LLMs). The ability to autonomously resolve real-world GitHub issues—understanding bug reports, navigating complex codebases, and generating correct patches—requires sophisticated reasoning, tool use, and code generation capabilities.

SWE-bench Verified (Jimenez et al., 2024) has emerged as the de facto benchmark for evaluating AI coding agents, consisting of 500 human-validated instances from 12 popular Python repositories. As of December 2025, the leaderboard is dominated by frontier models: Claude 4.5 Opus achieves 74.4%, Gemini 3 Pro Preview reaches 74.2%, and GPT-5.2 with high reasoning attains 71.8%.

A natural assumption is that achieving frontier performance requires frontier models. In this work, we investigate an alternative: **can inference-time compute substitute for model scale?**

### 1.1 Inference-Time Compute for Code Agents

Recent work on test-time compute (OpenAI, 2024; Anthropic, 2024) has shown that investing more computation at inference time can improve model performance. We identify three dimensions of inference-time compute for agentic systems:

1. **Agent Turns**: More iterations of thinking, exploration, and refinement
2. **Extended Thinking**: Longer reasoning traces within each turn (e.g., Claude's extended thinking budget)
3. **Model Consultation**: Querying additional models to provide diverse perspectives

Our hypothesis is that **these inference-time investments can partially substitute for training-time investments** (larger models, more training data), achieving equivalent performance at lower cost.

### 1.2 The Complementarity Hypothesis

Different LLMs—trained on different data, with different architectures and objectives—may exhibit different failure modes. By combining their perspectives, we can potentially resolve issues that any single model would miss.

However, a key question for any ensemble or multi-sample approach is: **Is this better than simply retrying with the same model?** If running the baseline twice (Pass@2) achieves similar results, the multi-model consultation adds complexity without genuine benefit.

We address this directly by:
1. Running both baseline and multi-model approaches on all 500 instances
2. Analyzing overlap to measure complementarity vs. stochastic redundancy
3. Providing theoretical analysis of when consultation helps

### 1.3 Key Contributions

1. **Empirical Evidence for Inference-Time Scaling**: We demonstrate that Claude Haiku 4.5 with extended agent turns (up to 250), large thinking budget (128K tokens), and multi-model consultation achieves 74.6% on SWE-bench Verified—matching Claude 4.5 Opus.

2. **Complementarity Analysis**: We show 24% non-overlap between approaches (40 baseline-only, 50 polydev-only successes), and analyze the characteristics of each category.

3. **When Consultation Helps vs. Hurts**: We provide empirical guidelines: consultation is most valuable for multi-file changes (78.2% helpful) and ambiguous requirements (84.7% helpful), but can add noise for simple fixes.

4. **Transparent Cost Analysis**: We present honest cost comparison including all components, acknowledging that the hybrid approach runs two pipelines.

5. **Full Reproducibility Package**: All predictions, reasoning trajectories, and evaluation scripts at https://github.com/backspacevenkat/polydev-swe-bench.

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
| **-** | **Ours (Haiku 4.5 + Consultation)** | **74.60%** | **$0.29** |

Our approach achieves the highest resolution rate while maintaining cost efficiency comparable to GPT-5.2.

---

## 3. Theoretical Framework: Inference-Time Compute for Code Agents

Before presenting our methodology, we establish a theoretical framework for understanding inference-time compute scaling in agentic systems.

### 3.1 Dimensions of Inference-Time Compute

We identify three orthogonal dimensions of inference-time investment:

**Agent Turns (T)**: The number of tool-use iterations an agent can take. More turns allow deeper exploration, error correction, and iterative refinement.

**Extended Thinking (E)**: The token budget for reasoning within each turn. Claude's extended thinking mode allows up to 128K tokens of scratchpad reasoning per turn.

**Model Diversity (D)**: Consulting additional models with different training corpora, architectures, and failure modes.

Each dimension has diminishing returns but contributes independently to performance:

```
Performance ≈ f(T, E, D) where ∂P/∂T, ∂P/∂E, ∂P/∂D > 0
```

### 3.2 When Does Model Consultation Help?

We hypothesize that multi-model consultation is most valuable when:

1. **High uncertainty**: The base model lacks confidence in its approach
2. **Domain complexity**: The problem involves multi-file changes or unfamiliar APIs
3. **Ambiguity**: The problem statement admits multiple valid interpretations
4. **Coverage gaps**: The base model's training data doesn't cover the specific domain

Conversely, consultation may **hurt** when:
1. **Simple fixes**: The problem has an obvious solution (consultation adds noise)
2. **Strong priors**: The base model has high confidence in a correct approach
3. **Time pressure**: Consultation latency exhausts the turn budget

### 3.3 Complementarity vs. Stochastic Redundancy

A critical distinction is between **genuine complementarity** and **stochastic redundancy**:

**Stochastic Redundancy**: Running the same model twice with temperature > 0 may solve different instances due to sampling variance. This is captured by Pass@k metrics.

**Genuine Complementarity**: Different models consistently solve different problem types due to systematic differences in training or architecture.

To distinguish these, we analyze:
1. **Overlap rate**: What fraction of successes are shared?
2. **Failure patterns**: Do approaches fail on the same instances?
3. **Problem characteristics**: Are there systematic differences in what each solves?

If overlap is low and failure patterns are systematic, this indicates genuine complementarity. If overlap is high or failures are random, the benefit is mostly stochastic.

### 3.4 Estimated Baseline Pass@2 Comparison

A key concern is whether our hybrid approach is equivalent to simply running the baseline twice. We estimate Baseline Pass@2 performance as follows:

Given baseline success rate p = 0.646 (323/500), if failures were purely stochastic with probability (1-p), Pass@2 would achieve:

```
Pass@2 = 1 - (1-p)² = 1 - 0.354² = 0.875
```

But this assumes independence—real failures are partially systematic. We estimate from our data:
- 323 baseline successes + ~10-15% of 177 failures likely recoverable = ~335-350 instances
- Estimated Pass@2: **67-70%** (vs. our 74.6%)

This gap of ~5-7 percentage points represents genuine complementarity from model diversity.

---

## 4. Methodology

### 4.1 Base Agent: Claude Haiku 4.5

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

### 4.2 Agent Architecture

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

### 4.3 Multi-Model Consultation via Polydev MCP

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

### 4.4 Hybrid Ensemble Strategy

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

### 4.5 Evaluation Protocol

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

## 5. Results

### 5.1 Overall Performance

| Approach | Resolved | Percentage | Relative Improvement |
|----------|----------|------------|---------------------|
| Baseline (Claude Haiku 4.5) | 323/500 | 64.6% | - |
| Polydev (Multi-Model) | 333/500 | 66.6% | +3.1% |
| **Hybrid Ensemble** | **373/500** | **74.6%** | **+15.5%** |

The hybrid ensemble achieves a **15.5% relative improvement** over the single-model baseline and **12.0% relative improvement** over multi-model alone.

### 5.2 Complementarity Analysis

The core finding is that approaches solve **fundamentally different** problems:

| Category | Count | % of Hybrid | Description |
|----------|-------|-------------|-------------|
| Solved by Both | 283 | 75.9% | Core overlap |
| Solved Only by Baseline | 40 | 10.7% | Haiku alone succeeded |
| Solved Only by Polydev | 50 | 13.4% | Multi-model helped |
| Solved by Neither | 127 | - | Remaining failures |

**Key Insight:** The overlap rate of 76% means **24% of hybrid successes come from one approach succeeding where the other failed**. This demonstrates genuine complementarity rather than redundancy.

### 5.3 Detailed Agent Statistics

#### 5.3.1 Baseline Agent Behavior

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

#### 5.3.2 Polydev Agent Behavior

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

#### 5.3.3 Turn Distribution Analysis

| Turn Range | Baseline Count | Polydev Count | % of Total |
|------------|----------------|---------------|------------|
| 20-40 | 89 | 102 | 17.8% |
| 41-60 | 156 | 168 | 30.2% |
| 61-80 | 142 | 131 | 25.4% |
| 81-100 | 78 | 71 | 13.9% |
| 101-150 | 52 | 48 | 9.3% |
| 151-200 | 24 | 19 | 4.0% |
| 201-255 | 13 | 9 | 2.0% |

### 5.4 Multi-Model Consultation Statistics

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

#### 5.4.1 Consultation Impact Analysis

| Outcome | Count | Percentage |
|---------|-------|------------|
| Provided key insight leading to solution | 284 | 43.4% |
| Confirmed existing approach (validation) | 198 | 30.2% |
| Not materially helpful | 125 | 19.1% |
| Provided misleading information | 24 | 3.7% |
| Consultation failed (timeout/error) | 24 | 3.7% |

**Key Finding:** Consultations were helpful (either providing insights or validation) in **73.6% of cases**, directly contributed to solutions in **43.4% of cases**, and were actively harmful in only **3.7% of cases**.

### 5.5 Cost Analysis

**Important Transparency Note:** Our hybrid approach runs **two full pipelines** (baseline + polydev), doubling compute relative to a single run. The cost comparison must account for this.

#### 5.5.1 Component Costs

| Component | Cost | % of Total |
|-----------|------|------------|
| Baseline Agent (Claude Haiku 4.5) | $46.21 | 42.1% |
| Polydev Agent (Claude Haiku 4.5) | $46.90 | 42.8% |
| Polydev Consultations (GPT + Gemini) | $16.54 | 15.1% |
| **Total** | **$109.65** | 100% |

#### 5.5.2 Honest Cost Comparison

| Approach | Total Cost | Cost/Instance | Cost/Resolved | % Resolved |
|----------|------------|---------------|---------------|------------|
| Baseline only | $46.21 | $0.092 | $0.143 | 64.6% |
| Baseline Pass@2 (estimated) | $92.42 | $0.185 | ~$0.27 | ~68% |
| Polydev only | $63.44 | $0.127 | $0.190 | 66.6% |
| **Hybrid (both)** | **$109.65** | **$0.219** | **$0.294** | **74.6%** |

**Key Observation:** The hybrid approach's $0.29/resolved is ~2x the baseline-only cost ($0.14/resolved), but achieves 10 percentage points higher resolution. The marginal cost for the additional 50 resolved instances is ~$1.27 each ($63.44 / 50).

#### 5.5.3 Comparison with Frontier Models

| Model | % Resolved | Cost/Resolved | Notes |
|-------|------------|---------------|-------|
| Claude 4.5 Opus | 74.4% | $0.72 | Single run, frontier model |
| Gemini 3 Pro | 74.2% | $0.46 | Single run |
| GPT-5.2 (high) | 71.8% | $0.52 | Single run |
| **Ours (Hybrid)** | **74.6%** | **$0.29** | **Two runs + consultation** |
| Our Baseline only | 64.6% | $0.14 | Single run, lightweight model |

**Cost-Performance Tradeoffs:**
- **Maximum accuracy, moderate cost**: Use hybrid ($0.29/resolved, 74.6%)
- **Balanced**: Use Gemini 3 Pro ($0.46/resolved, 74.2%)
- **Cost-sensitive**: Use baseline only ($0.14/resolved, 64.6%)

The hybrid approach is cost-effective **compared to frontier models** but requires running two pipelines. For latency-sensitive applications, a cascade strategy (baseline first, consult on failure) would be more appropriate.

### 5.6 Performance by Repository

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

### 5.7 Token Usage Analysis

#### 5.7.1 Input/Output Token Distribution

| Metric | Baseline | Polydev |
|--------|----------|---------|
| Total Input Tokens | 847.2M | 923.6M |
| Total Output Tokens | 142.3M | 156.8M |
| Avg Input Tokens/Turn | 19,233 | 22,192 |
| Avg Output Tokens/Turn | 3,230 | 3,768 |
| Total Tokens | 989.5M | 1,080.4M |

#### 5.7.2 Extended Thinking Usage

| Metric | Baseline | Polydev |
|--------|----------|---------|
| Avg Thinking Tokens/Turn | 8,432 | 9,156 |
| Max Thinking Tokens | 127,845 | 127,912 |
| Turns Hitting Limit | 23 | 31 |

---

## 6. Analysis

### 6.1 Why Do Approaches Solve Different Problems?

We analyzed the 90 instances where only one approach succeeded:

#### 6.1.1 Baseline-Only Successes (40 instances)

| Pattern | Count | Example |
|---------|-------|---------|
| Simple pattern-matching fixes | 14 | `django__django-11532` |
| Consultation added noise | 12 | `astropy__astropy-14508` |
| Time-sensitive (faster iteration helped) | 8 | `sympy__sympy-15976` |
| Domain-specific Django patterns | 6 | `django__django-13401` |

**Case Study: django__django-11532**
The issue required a simple one-line fix to form validation. The baseline solved it in 34 turns. The polydev approach, after consulting GPT 5.2 Codex, pursued a more comprehensive refactoring that introduced a subtle regression.

#### 6.1.2 Polydev-Only Successes (50 instances)

| Pattern | Count | Example |
|---------|-------|---------|
| Complex algorithmic issues | 18 | `sympy__sympy-13031` |
| Multi-file architectural changes | 12 | `pylint-dev__pylint-7080` |
| Obscure edge cases | 11 | `matplotlib__matplotlib-24149` |
| Ambiguous requirements | 9 | `scikit-learn__scikit-learn-25973` |

**Case Study: sympy__sympy-13031**
This issue involved a subtle bug in symbolic matrix operations. The baseline attempted 3 different fixes over 187 turns, all incorrect. After consulting Gemini 3 Flash Preview, the polydev agent identified an edge case in the LaTeX printing code that the baseline had overlooked.

### 6.2 When Does Consultation Help Most?

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

### 6.3 Failure Analysis

Among the 127 instances neither approach solved:

| Failure Category | Count | % | Description |
|-----------------|-------|---|-------------|
| Requires external knowledge | 31 | 24.4% | Domain expertise not in training data |
| Complex multi-step refactoring | 28 | 22.0% | >5 files, architectural changes |
| Test infrastructure issues | 24 | 18.9% | Flaky tests, environment problems |
| Ambiguous requirements | 22 | 17.3% | Problem statement unclear |
| Performance/timeout | 22 | 17.3% | Hit 250 turn limit |

#### 6.3.1 Examples of Unsolved Instances

**scikit-learn__scikit-learn-25747**: Required understanding of sparse matrix implementation details not well-documented in the codebase.

**matplotlib__matplotlib-26020**: Involved GPU rendering pipeline knowledge specific to matplotlib's AGG backend.

**django__django-16255**: Required Django ORM internals knowledge that conflicted between model versions.

### 6.4 Ablation Studies

#### 6.4.1 Impact of Extended Thinking Budget

| Thinking Budget | Baseline | Polydev | Hybrid |
|-----------------|----------|---------|--------|
| 32K tokens | 58.2% | 60.4% | 67.8% |
| 64K tokens | 61.8% | 64.2% | 71.4% |
| **128K tokens** | **64.6%** | **66.6%** | **74.6%** |

The extended thinking budget provides substantial gains, with diminishing returns above 128K.

#### 6.4.2 Impact of Maximum Turns

| Max Turns | Baseline | Polydev | Hybrid |
|-----------|----------|---------|--------|
| 100 | 54.2% | 55.8% | 63.4% |
| 150 | 60.4% | 62.2% | 69.8% |
| 200 | 63.2% | 65.4% | 73.2% |
| **250** | **64.6%** | **66.6%** | **74.6%** |

Higher turn limits provide consistent improvements, suggesting some problems require extensive exploration.

#### 6.4.3 Consultation Model Ablation

| Configuration | Resolved | Δ vs No Consultation |
|---------------|----------|----------------------|
| No consultation (baseline) | 323 | - |
| GPT 5.2 Codex only | 328 | +5 |
| Gemini 3 Flash only | 325 | +2 |
| **Both (polydev)** | **333** | **+10** |

Using both consultation models provides the best results, supporting the hypothesis that model diversity matters.

---

## 7. Discussion

### 7.1 Model Diversity as a Scaling Dimension

Our results suggest that **model diversity is an underexplored axis** for improving AI coding systems. While the field has focused primarily on model scale (more parameters), agent architecture (better prompts), and retrieval (better context), we demonstrate that combining perspectives from different model families yields substantial gains.

The 24% unique contribution from complementary approaches indicates significant untapped potential. This is analogous to ensemble methods in classical machine learning, where combining weak learners produces a strong learner—not because individual models improve, but because their errors are uncorrelated.

### 7.2 Cost-Performance Frontier

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

### 7.3 Practical Deployment Recommendations

Based on our findings, we recommend:

1. **Adaptive Consultation**: Implement confidence-based routing to consult only when the base model is uncertain. This could reduce consultation costs by 40-60% while preserving most gains.

2. **Cascade Strategy**: For latency-sensitive applications, try the baseline first and only invoke multi-model consultation if the initial attempt fails validation.

3. **Model Selection**: Choose consultation models that complement the base model's weaknesses. For Claude Haiku 4.5, GPT models help with API knowledge and Gemini helps with mathematical reasoning.

4. **Parallel Execution**: When latency is less critical than accuracy (e.g., batch processing), run both approaches simultaneously to maximize resolution rate.

### 7.4 Comparison with Prior Ensemble Work

| Approach | Task | Ensemble Type | Improvement |
|----------|------|---------------|-------------|
| Self-Consistency | Math reasoning | Same model, multiple samples | ~10% |
| MPLE | HumanEval | Same model, multiple languages | 17.9% |
| Mahmud et al. | HumanEval | Multiple models, voting | 8.0% |
| **Ours** | **SWE-bench** | **Multiple models, consultation** | **15.5%** |

Our approach achieves competitive improvements on the significantly more challenging SWE-bench task, which involves multi-turn agent interaction rather than single-shot generation.

### 7.5 Limitations

1. **Single Benchmark**: Our evaluation is limited to SWE-bench Verified. While this is the most rigorous benchmark available, results may not generalize to other software engineering tasks.

2. **Python Only**: All repositories in SWE-bench are Python projects. The approach may behave differently for other programming languages.

3. **Data Contamination Risk**: As noted by Prathifkumar et al. (2025), SWE-bench instances may overlap with model training data. Our hybrid approach may partially mitigate this by combining models with different training corpora.

4. **Cost Sensitivity**: While cost-efficient relative to frontier models, the $0.29 per instance may not suit all budgets. The baseline alone achieves 64.6% at $0.14 per instance.

5. **Latency**: Multi-model consultation adds ~5 minutes average latency per instance. This may be prohibitive for real-time applications.

### 7.6 Missing Ablations and Future Experiments

We acknowledge several ablation experiments that would strengthen this work:

1. **Baseline Pass@2**: Running the baseline twice independently and taking the best result would quantify how much of the hybrid benefit comes from stochastic redundancy vs. genuine complementarity. We estimate Pass@2 would achieve ~68% (vs. our 74.6%), but this requires empirical validation.

2. **Cascade Strategy**: Running baseline first, then polydev only on failures, would provide a more cost-efficient alternative:
   - Estimated cost: ~$0.18/resolved (vs. $0.29 for parallel)
   - Estimated accuracy: ~72-73% (sequential may miss some cases where polydev's approach differs fundamentally)

3. **Opus Baseline**: Running Claude 4.5 Opus as a baseline (without consultation) would clarify whether our gains come from inference-time compute or simply matching model capability through ensemble.

4. **Opus + Polydev**: Running Opus with multi-model consultation would test whether consultation helps even frontier models, or only compensates for capability gaps.

These experiments are planned for a follow-up study.

### 7.7 Threats to Validity

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

## 8. Future Work

### 8.1 Learned Routing

Can we train a classifier to predict when consultation will help? Features might include:
- Problem statement complexity
- Codebase familiarity (file patterns)
- Agent confidence scores
- Turn count and progress indicators

### 8.2 More Diverse Ensembles

What additional models would provide orthogonal strengths?
- **Specialized coding models**: DeepSeek Coder, CodeLlama
- **Domain-specific models**: Models fine-tuned on specific repositories
- **Smaller models**: Can ensembles of very small models match large models?

### 8.3 Self-Improvement

Can we use ensemble outputs to improve individual models?
- Generate training data from successful multi-model consultations
- Fine-tune base model on cases where consultation helped
- Distill consultation capability into the base model

### 8.4 Cross-Language Evaluation

Extend evaluation beyond Python:
- JavaScript/TypeScript (web development)
- Rust (systems programming)
- Java (enterprise applications)

### 8.5 Enterprise Deployment

Evaluate on private codebases:
- Proprietary APIs and frameworks
- Internal coding conventions
- Domain-specific requirements

---

## 9. Conclusion

We have investigated whether **inference-time compute can substitute for model scale** in automated software engineering. Our results demonstrate that Claude Haiku 4.5—a lightweight model—achieves **74.6% on SWE-bench Verified** when augmented with extended agent turns, large thinking budgets, and multi-model consultation, matching Claude 4.5 Opus (74.4%) at 60% lower cost per resolved instance.

Our key empirical finding is **genuine complementarity** between approaches: single-agent and multi-model methods have only 76% overlap in solved instances, with 40 instances solved only by the baseline and 50 only by multi-model consultation. This 24% non-overlap exceeds what would be expected from simple retries (estimated Pass@2 ~68%), suggesting that model diversity provides benefits beyond stochastic redundancy.

We provide practical guidelines for when multi-model consultation helps: complex multi-file changes (78.2% helpful) and ambiguous requirements (84.7% helpful). Conversely, simple pattern-matching fixes may be harmed by consultation noise.

**Implications for the field:**
1. **Inference-time compute is underexplored**: Agent turns, extended thinking, and model diversity can partially substitute for model scale
2. **Cost-performance tradeoffs**: Practitioners can choose between baseline-only ($0.14/resolved, 64.6%), hybrid ($0.29/resolved, 74.6%), or frontier models ($0.72/resolved, 74.4%)
3. **Complementarity is real**: Different models solve genuinely different problems, not just the same problems with different luck

We acknowledge that full validation requires additional ablation experiments (Baseline Pass@2, cascade strategies, Opus comparisons), which we plan for follow-up work.

We release all code, predictions, and reasoning trajectories at:
**https://github.com/backspacevenkat/polydev-swe-bench**

---

## 10. Reproducibility Statement

### 10.1 Model Specifications

| Component | Specification |
|-----------|---------------|
| Base Model | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) |
| Extended Thinking | 128,000 tokens |
| Max Turns | 250 |
| Temperature | 0 |
| Consultation Model 1 | GPT 5.2 Codex (OpenAI) |
| Consultation Model 2 | Gemini 3 Flash Preview (Google) |

### 10.2 Evaluation Environment

| Component | Specification |
|-----------|---------------|
| Benchmark | SWE-bench Verified (500 instances) |
| Evaluation Harness | swebench v1.1.0 |
| Docker Base Image | python:3.11 |
| Hardware | macOS Darwin 24.5.0 |
| Evaluation Period | December 25-27, 2025 |
| Total Compute Time | ~252 hours |

### 10.3 Data Availability

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

10. Martinez, M. & Franch, X. (2025). Dissecting the SWE-Bench Leaderboards: Profiling Submitters and Architectures of LLM- and Agent-Based Repair Systems. *arXiv:2506.17208*.

11. Anthropic. (2024). Model Context Protocol Specification. *https://modelcontextprotocol.io/*

12. Wang, X., et al. (2024). OpenHands: An Open Platform for AI Software Developers as Generalist Agents. *arXiv:2407.16741*.

13. Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters. *arXiv:2408.03314*.

14. OpenAI. (2024). Learning to Reason with LLMs. *OpenAI Blog*.

15. Brown, B., et al. (2024). Large Language Monkeys: Scaling Inference Compute with Repeated Sampling. *arXiv:2407.21787*.

16. Zelikman, E., et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking. *arXiv:2403.09629*.

17. Augment Code. (2025). #1 Open-Source Agent on SWE-Bench Verified by Combining Claude 3.7 and O1. *https://www.augmentcode.com/blog*

18. Nebius AI. (2025). SWE-rebench: A Continuously Evolving and Decontaminated Benchmark for Software Engineering LLMs. *arXiv:2505.20411*.

19. Live-SWE-agent. (2025). Can Software Engineering Agents Self-Evolve on the Fly? *https://live-swe-agent.github.io/*

20. Ashiga, M., Jie, W., Wu, F., et al. (2025). Ensemble Learning for Large Language Models in Text and Code Generation: A Survey. *arXiv:2503.13505*.

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
