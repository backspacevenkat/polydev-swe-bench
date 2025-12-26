# Polydev arXiv Paper: Comprehensive Scientific Strategy
## "Selective Cross-Provider Consultation for LLM Coding Agents"

---

## Executive Summary

**The Problem**: Every IDE (Claude Code, Cursor, Windsurf) has a base model doing the work. When that model gets stuck or is uncertain, it has no way to get a "second opinion" from models with different training and perspectives.

**Our Solution**: Polydev is a **consultation layer** that augments the base model. The base model calls `consult_polydev` when it needs help - Polydev provides cross-provider perspectives, and the base model synthesizes the final solution.

**Key Distinction**:
- **NOT**: Multi-model system replacing the base model
- **YES**: Base model + on-demand cross-provider consultation

**Core Claim**:
> "Base Model + Polydev Consultation > Base Model Alone"

---

## Part 1: The "Chief Resident" Architecture

### 1.1 The Mental Model

Think of it like a hospital:
- **Base Model** = Chief Resident (does the work, makes decisions)
- **Polydev** = Board of Specialists (consulted for second opinions)
- The Chief Resident is ALWAYS in charge
- Specialists are only called when needed

### 1.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IDE (Claude Code, Cursor, Windsurf)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │           BASE MODEL (e.g., Claude Opus 4.5)                │   │
│   │           ══════════════════════════════════                │   │
│   │                                                             │   │
│   │   ┌─────────────────────────────────────────────────────┐   │   │
│   │   │                 SOLVE → TEST → REFLECT              │   │   │
│   │   │                                                     │   │   │
│   │   │   1. Analyze task                                   │   │   │
│   │   │   2. Generate solution                              │   │   │
│   │   │   3. Run tests                                      │   │   │
│   │   │   4. REFLECT: Am I stuck? Uncertain? Need help?     │   │   │
│   │   │              │                                      │   │   │
│   │   │              ▼                                      │   │   │
│   │   │      ┌───────────────────┐                          │   │   │
│   │   │      │ TRIGGER CHECK     │                          │   │   │
│   │   │      │ - Stuck loop?     │                          │   │   │
│   │   │      │ - Low confidence? │                          │   │   │
│   │   │      │ - Repeated error? │                          │   │   │
│   │   │      └────────┬──────────┘                          │   │   │
│   │   │               │                                     │   │   │
│   │   │       NO ◄────┴────► YES                            │   │   │
│   │   │       │               │                             │   │   │
│   │   │       ▼               ▼                             │   │   │
│   │   │   Continue        CALL consult_polydev()            │   │   │
│   │   │   alone                   │                         │   │   │
│   │   │                           ▼                         │   │   │
│   │   │               ┌─────────────────────────┐           │   │   │
│   │   │               │   POLYDEV CONSULTATION  │           │   │   │
│   │   │               │                         │           │   │   │
│   │   │               │  ┌─────┐ ┌─────┐       │           │   │   │
│   │   │               │  │GPT  │ │Gemini│       │           │   │   │
│   │   │               │  │5.2  │ │3.0  │        │           │   │   │
│   │   │               │  └──┬──┘ └──┬──┘       │           │   │   │
│   │   │               │     └───┬───┘          │           │   │   │
│   │   │               │         ▼              │           │   │   │
│   │   │               │  Aggregated Response   │           │   │   │
│   │   │               └────────────────────────┘           │   │   │
│   │   │                           │                         │   │   │
│   │   │                           ▼                         │   │   │
│   │   │   5. Base model SYNTHESIZES final solution          │   │   │
│   │   │   6. Submit patch                                   │   │   │
│   │   └─────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Critical Design Principle

**Polydev is only callable from the REFLECT step.**

This prevents:
- "Prompt hopping" as default behavior
- Unnecessary API costs
- Loss of base model agency

The base model remains the intelligent filter - it can accept, reject, or synthesize Polydev's advice.

---

## Part 2: The `consult_polydev` Tool Interface

### 2.1 Tool Definition (MCP/Function Calling)

```json
{
  "name": "consult_polydev",
  "description": "Consults external expert models for diverse perspectives. Use this when you are stuck, uncertain about an obscure library, facing repeated errors, or planning a high-risk architectural change.",
  "parameters": {
    "type": "object",
    "properties": {
      "context_summary": {
        "type": "string",
        "description": "Brief summary of the task and relevant code context"
      },
      "current_hypothesis": {
        "type": "string",
        "description": "What you currently believe is the issue/solution"
      },
      "what_you_tried": {
        "type": "string",
        "description": "What approaches you've already attempted"
      },
      "failure_signals": {
        "type": "string",
        "description": "Error messages, test failures, or unexpected behavior"
      },
      "specific_questions": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Specific questions you need answered"
      },
      "consult_type": {
        "type": "string",
        "enum": ["debug", "design_alternative", "edge_cases", "api_verification", "architecture"],
        "description": "Type of consultation needed"
      }
    },
    "required": ["context_summary", "current_hypothesis", "specific_questions", "consult_type"]
  }
}
```

### 2.2 Polydev Response Format

Polydev returns structured advice, NOT raw code to paste:

```json
{
  "consensus_view": "Both GPT and Gemini agree that the issue is X...",
  "dissenting_views": [
    {"provider": "gpt-5.2", "view": "Suggests approach A because..."},
    {"provider": "gemini-3.0", "view": "Suggests approach B because..."}
  ],
  "blind_spot_alert": "You may have missed edge case X...",
  "suggested_next_steps": [
    "Try adding a null check before line 42",
    "Consider using library Y instead of Z"
  ],
  "confidence": 0.85,
  "providers_consulted": ["gpt-5.2-codex-thinking", "gemini-3.0-pro"]
}
```

### 2.3 Base Model Synthesis

The base model receives Polydev's response as an **observation**, then:
1. Evaluates which advice is valid
2. Rejects hallucinations or bad suggestions
3. Synthesizes the final solution itself

**Key Instruction to Base Model**:
> "You have received a consultation. You are the lead engineer. Accept valid criticism, ignore hallucinations in the consultation, and synthesize the final solution yourself."

---

## Part 3: Trigger Conditions (When to Consult)

### 3.1 Two-Stage Gating

**Stage A: Deterministic Rule-Based Triggers** (cheap, fast)
Observable signals that reliably indicate need for help.

**Stage B: Model-Based Self-Diagnostic** (if Stage A passes)
Short prompt asking base model to assess "Do I need consultation?"

### 3.2 Trigger Taxonomy

| Trigger | Description | Observable Signal |
|---------|-------------|-------------------|
| **Stuck Loop** | Same error after 2+ fix attempts | Identical error signature |
| **Stagnation** | No progress in 2+ iterations | Same failing tests, no new evidence |
| **Contradiction** | Model's own assertions conflict | Generated invariants contradict |
| **Low Confidence** | Model expresses uncertainty | "I assume", "might work", "not sure" |
| **Search Explosion** | Too many plausible causes | >3 hypotheses without ranking |
| **API Uncertainty** | Unsure about library usage | Unfamiliar library, breaking changes |
| **Repeated Compile Error** | Same type/syntax error 2+ times | Same compiler error class |
| **Security Sensitive** | Auth, crypto, sandboxing changes | File paths match security patterns |

### 3.3 Implementation

```python
class ConsultationTrigger:
    """Determines when base model should call consult_polydev."""

    def should_consult(self, state: AgentState) -> tuple[bool, str]:
        """
        Returns (should_consult, trigger_reason).
        """
        # Stage A: Deterministic checks
        if self._is_stuck_loop(state):
            return True, "stuck_loop"

        if self._is_stagnating(state):
            return True, "stagnation"

        if self._has_repeated_error(state):
            return True, "repeated_error"

        if self._is_security_sensitive(state):
            return True, "security_sensitive"

        # Stage B: Model-based self-assessment
        confidence = self._get_model_confidence(state)
        if confidence < 0.7:
            return True, "low_confidence"

        return False, None

    def _is_stuck_loop(self, state: AgentState) -> bool:
        """Same error signature after 2+ attempts."""
        if len(state.error_history) < 2:
            return False
        return state.error_history[-1] == state.error_history[-2]

    def _is_stagnating(self, state: AgentState) -> bool:
        """No reduction in failing tests for 2+ iterations."""
        if len(state.test_results) < 2:
            return False
        return state.test_results[-1].failures >= state.test_results[-2].failures

    def _has_repeated_error(self, state: AgentState) -> bool:
        """Same error type 2+ times."""
        recent_errors = state.error_history[-3:]
        return len(set(e.type for e in recent_errors)) == 1 and len(recent_errors) >= 2
```

### 3.4 Budget Controls

To prevent over-consultation:
- **Max consults per task**: 1-3
- **Cooldown**: Must attempt N new actions before consulting again
- **Justification required**: Base model must explain why it's consulting

---

## Part 4: Harness Implementation

### 4.1 The Main Loop

```python
async def solve_swe_task(
    task: SWETask,
    base_model: str = "claude-opus-4.5",
    max_iterations: int = 5,
    max_consults: int = 2
) -> Patch:
    """
    Base model solves task, consulting Polydev when triggered.
    """
    state = AgentState(task=task)
    trigger = ConsultationTrigger()
    consults_used = 0

    for iteration in range(max_iterations):
        # SOLVE: Generate/refine patch
        patch = await base_model.generate_patch(state)

        # TEST: Run tests
        test_result = await run_tests(patch, task)
        state.add_test_result(test_result)

        if test_result.passed:
            return patch  # Success!

        # REFLECT: Check if consultation needed
        should_consult, trigger_reason = trigger.should_consult(state)

        if should_consult and consults_used < max_consults:
            # CONSULT: Call Polydev
            perspectives = await consult_polydev(
                context_summary=state.get_context(),
                current_hypothesis=patch.rationale,
                what_you_tried=state.get_attempt_history(),
                failure_signals=test_result.error_message,
                specific_questions=[
                    "What alternative approaches should I consider?",
                    "What am I missing about this codebase?"
                ],
                consult_type=_map_trigger_to_type(trigger_reason)
            )

            # INCORPORATE: Base model synthesizes advice
            state.add_perspectives(perspectives)
            consults_used += 1

        state.add_error(test_result.error)

    # Return best attempt
    return state.best_patch
```

### 4.2 Worker Allocation (16 Workers)

| Pool | Workers | Role |
|------|---------|------|
| Primary solving | 12 | Base model attempts |
| Polydev queries | 4 | Cross-provider consultation (shared) |

### 4.3 Cost Model

| Event | Cost | Frequency |
|-------|------|-----------|
| Base model attempt | ~$0.05 | Every task |
| Polydev consultation | ~$0.10 | ~30% of tasks (triggered) |
| **Expected average** | ~$0.08/task | |

---

## Part 5: Experimental Design

### 5.1 Primary Experiment Configurations

| Config | Description | Purpose |
|--------|-------------|---------|
| **A: Base Alone** | Claude Opus 4.5, no consultation | Baseline |
| **B: Base + Polydev (Gated)** | Claude + Polydev when triggered | Main hypothesis |
| **C: Base + Polydev (Always)** | Claude + Polydev every iteration | Test gating value |
| **D: Base + Self-Reflect** | Claude + same tokens for self-critique | Control for "more compute" |

### 5.2 Key Experiments

#### Experiment 1: The "Unstuck Rate" (Primary Metric)

**Setup**: Take the hardest 20% of SWE-bench Verified tasks.

**Protocol**:
1. Run Base Alone, allow 3 iterations
2. Run Base + Polydev, allow 3 iterations (consult on failure)

**Metric**:
```
Unstuck Rate = (Tasks solved after Polydev consultation) / (Tasks where Base Alone failed)
```

**Hypothesis**: Unstuck Rate > 40%

#### Experiment 2: Hallucination Mitigation

**Setup**: Tasks involving deprecated APIs or libraries with breaking changes.

**Protocol**:
1. Track when base model generates hallucinated API calls
2. Track when Polydev catches these via cross-provider verification

**Metric**:
```
Hallucination Catch Rate = (Hallucinations caught by Polydev) / (Total hallucinations)
```

**Hypothesis**: Polydev reduces API hallucinations by >30%

#### Experiment 3: The "Ego Test"

**Setup**: Intentionally feed Polydev incorrect advice.

**Goal**: Prove base model is smart enough to REJECT bad Polydev advice.

**Metric**:
```
Rejection Rate = (Bad advice correctly rejected) / (Bad advice given)
```

**Hypothesis**: Rejection Rate > 80% (proves augmentation, not replacement)

#### Experiment 4: Cross-Provider vs Same-Provider

**Setup**: When base model fails, compare:
- A: Retry with same model (Claude → Claude again)
- B: Consult different provider (Claude → ask GPT via Polydev)

**Metric**: Recovery rate for each condition

**Hypothesis**: Cross-provider consultation > Same-provider retry

#### Experiment 5: Trigger Effectiveness

**Setup**: Analyze which triggers predict successful consultations.

**Metrics**:
- Precision: P(consultation helps | trigger fired)
- Recall: P(trigger fired | consultation would have helped)

**Goal**: Identify optimal trigger conditions

### 5.3 Statistical Validation

**McNemar's Test**: Paired comparison on same 500 tasks
```python
def mcnemar_test(base_alone: List[bool], base_plus_polydev: List[bool]):
    b = sum(1 for a, p in zip(base_alone, base_plus_polydev) if a and not p)
    c = sum(1 for a, p in zip(base_alone, base_plus_polydev) if not a and p)
    chi2 = (abs(b - c) - 1)**2 / (b + c)
    return chi2, p_value
```

**Bootstrap CIs**: 95% confidence intervals on improvement

---

## Part 6: Paper Positioning

### 6.1 Title Options

**Primary (Recommended)**:
> "Polydev: Selective Cross-Provider Consultation for LLM Coding Agents"

**Alternative 1**:
> "When Your AI Gets Stuck: Cross-Provider Perspectives for Software Engineering"

**Alternative 2**:
> "Breaking the Monolith: Cross-Provider Consultation as an Augmentation Layer for AI Agents"

### 6.2 Abstract (Draft)

> While LLM reasoning capabilities continue to scale, single-model coding agents remain susceptible to specific blind spots and hallucinations. We introduce Polydev, a consultation layer that allows a base model (e.g., Claude Opus 4.5) to dynamically query peer models (e.g., GPT 5.2, Gemini 3.0) when uncertainty arises. Unlike ensemble methods that run all models in parallel, Polydev uses selective triggering to consult only when the base model is stuck or uncertain. On SWE-bench Verified, we demonstrate that Claude Opus 4.5 + Polydev achieves an "Unstuck Rate" of X%, recovering Y% of tasks where the base model alone failed. Polydev reduces API hallucinations by Z% through cross-provider verification, while adding only 30% to inference cost. We release the trigger taxonomy, consultation protocol, and evaluation harness to enable future research on cross-provider augmentation.

### 6.3 Core Contributions

| # | Contribution | Why Citable |
|---|--------------|-------------|
| 1 | **Selective Consultation Framework** | First principled approach to when/how to consult other models |
| 2 | **Trigger Taxonomy** | Reproducible conditions for consultation |
| 3 | **Unstuck Rate Metric** | New metric for measuring consultation value |
| 4 | **Cross-Provider Complementarity Analysis** | Which providers help with which error types |
| 5 | **MCP-Based Tool Interface** | Portable, works with any IDE |

### 6.4 Positioning vs Related Work

| Approach | How Polydev Differs |
|----------|---------------------|
| **Ensemble (voting)** | Polydev is selective, not always-on |
| **Multi-agent systems** | Base model stays in charge, agents don't |
| **Self-critique** | Cross-provider provides genuinely different perspectives |
| **Tree-of-thought** | Polydev consults experts, not just branches |

### 6.5 Key Narrative Points

1. **The Silo Problem**: If Claude has a blind spot, the user is stuck. No second opinion available.

2. **The Second Opinion Standard**: In high-stakes coding (production), a single model is a single point of failure.

3. **Augmentation, Not Replacement**: Polydev makes the base model better, doesn't replace it.

4. **Selective = Efficient**: Smart triggers achieve 90% of always-consult gains at 30% of cost.

---

## Part 7: Paper Structure

```
1. Introduction (1.5 pages)
   - The single-model limitation
   - The "second opinion" insight
   - Contribution summary

2. Background (1.5 pages)
   - LLM coding assistants
   - Uncertainty in language models
   - Related work: ensembles, multi-agent, self-critique

3. The Polydev Framework (2 pages)
   - Chief Resident architecture
   - consult_polydev tool interface
   - Trigger taxonomy
   - Response synthesis

4. Experimental Setup (1.5 pages)
   - SWE-bench Verified
   - Configurations (A, B, C, D)
   - Metrics

5. Results (3 pages)
   - Main results table
   - Unstuck Rate analysis
   - Hallucination mitigation
   - Trigger effectiveness
   - Cost analysis

6. Analysis (2 pages)
   - When does consultation help?
   - Provider complementarity
   - The Ego Test: rejection of bad advice

7. Discussion (1 page)
   - Implications for IDE design
   - Limitations
   - Future work

8. Conclusion (0.5 pages)
```

---

## Part 8: Key Figures

### Figure 1: Chief Resident Architecture
System diagram showing base model with consult_polydev tool

### Figure 2: Trigger Decision Tree
Flowchart of when to consult

### Figure 3: Unstuck Rate by Trigger Type
Bar chart showing which triggers lead to successful recovery

### Figure 4: Provider Complementarity Heatmap
Which providers catch which error types

### Figure 5: Cost vs Accuracy Pareto Frontier
Showing gated consultation is cost-effective

### Figure 6: The Ego Test Results
Showing base model correctly rejects bad advice

---

## Part 9: Comparison with State-of-the-Art (Dec 2025)

| Agent | Score | Architecture | How Polydev Differs |
|-------|-------|--------------|---------------------|
| Gemini 3 Flash | 76.20% | Single model | Polydev augments any base |
| GPT 5.2 | 75.40% | Single model | Polydev augments any base |
| Trae Agent | 75.20% | Complex ensemble | Polydev is simple tool call |
| Refact.ai | 74.40% | Single + scaffolding | Polydev adds cross-provider |
| Augment Code | 65.4% | Dual-model | Polydev is selective, not always-on |
| **Claude + Polydev** | **Target: >77%** | **Base + Selective Consultation** | **Simple, portable, efficient** |

---

## Part 10: Implementation Checklist

### Harness Components
- [ ] ConsultationTrigger class with all trigger types
- [ ] consult_polydev MCP tool definition
- [ ] Response aggregation logic
- [ ] Budget controls (max consults, cooldown)
- [ ] Logging for trigger analysis

### Evaluation
- [ ] SWE-bench Verified setup with sb-cli
- [ ] 4 experimental configurations
- [ ] Metrics collection (unstuck rate, hallucination catch rate, etc.)
- [ ] Statistical tests (McNemar's, bootstrap)

### Paper
- [ ] LaTeX template
- [ ] All figures
- [ ] Code release repository
- [ ] Leaderboard submission materials

---

## Part 11: Why This Paper Will Be Cited

1. **Novel Framework**: First principled "selective consultation" approach
2. **Practical Tool**: Works with any IDE, any base model
3. **Trigger Taxonomy**: Reproducible, others will build on it
4. **New Metrics**: Unstuck Rate becomes standard measure
5. **MCP Standard**: Proposes consultation as MCP tool pattern

---

## Appendix A: Research Insights Sources

### From GPT-5.2 Consultation:
- Single-loop controller with consult only in reflect step
- Two-stage gating (deterministic + model-based)
- Structured consult API contract
- Budget controls and hysteresis

### From Gemini-3-Pro Consultation:
- "Chief Resident" architecture metaphor
- Tool-use paradigm (not voting)
- Trigger types: Stuck Loop, Low Confidence, Hallucination Check
- Experiments: Unstuck Rate, Ego Test

### From Exa Research:
- Uncertainty Highlighting in AI Code Completions (Vasconcelos et al.)
- Tools in the Loop: Uncertainty Quantification (Lymperopoulos)
- Curiosity by Design: Asking Clarification Questions

---

*Document Version: 4.0*
*Architecture: Chief Resident (Base Model + Selective Polydev Consultation)*
*Core Claim: Base Model + Polydev > Base Model Alone*
*Models: Claude Opus 4.5, GPT 5.2 Codex Thinking High, Gemini 3.0 Pro*
