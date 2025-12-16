# Prompts Documentation

This document contains all prompts used in the Polydev SWE-bench evaluation. These prompts are designed to be reproducible and are versioned for publication.

## Prompt Design Principles

1. **Clarity**: Clear instructions without ambiguity
2. **Consistency**: Same format across all phases
3. **Structured Output**: Request specific output formats for parsing
4. **Minimal Bias**: Avoid leading the model toward specific solutions

---

## 1. Issue Analysis Prompt

**File**: `agent/prompts/analysis.txt`
**Version**: 1.0.0

```
You are a senior software engineer analyzing a GitHub issue.

## Task
Analyze the following issue and identify:
1. The root cause of the problem
2. Which files are likely affected
3. What behavior is expected vs. actual

## Issue Information
Repository: {repository}
Issue ID: {instance_id}

### Issue Description
{problem_statement}

### Hints (if available)
{hints_text}

## Instructions
1. Read the issue description carefully
2. Identify what the user is reporting
3. Hypothesize about the root cause
4. List files that likely need to be examined
5. Note any edge cases or special considerations

## Output Format
<analysis>
<root_cause>
[Your hypothesis about what's causing this issue]
</root_cause>

<affected_files>
- file1.py: [why this file is relevant]
- file2.py: [why this file is relevant]
</affected_files>

<expected_behavior>
[What should happen]
</expected_behavior>

<actual_behavior>
[What's happening instead]
</actual_behavior>

<edge_cases>
[Any edge cases to consider]
</edge_cases>
</analysis>
```

---

## 2. Code Reading Prompt

**File**: `agent/prompts/code_reading.txt`
**Version**: 1.0.0

```
You are examining source code to understand how to fix an issue.

## Context
Repository: {repository}
Issue: {issue_summary}
Root cause hypothesis: {root_cause}

## File Contents
{file_contents}

## Instructions
1. Read the code carefully
2. Identify the specific location of the bug
3. Understand the surrounding code context
4. Note any dependencies or related functions

## Output Format
<code_analysis>
<bug_location>
File: [filename]
Lines: [line numbers]
Function/Class: [name]
</bug_location>

<bug_description>
[What exactly is wrong in the code]
</bug_description>

<context>
[How this code relates to the issue]
</context>

<dependencies>
[Other code that depends on or is called by this]
</dependencies>
</code_analysis>
```

---

## 3. Solution Generation Prompt

**File**: `agent/prompts/solution.txt`
**Version**: 1.0.0

```
You are a senior software engineer proposing a fix for a bug.

## Issue Context
Repository: {repository}
Issue: {instance_id}
Root cause: {root_cause}
Bug location: {bug_location}

## Relevant Code
{relevant_code}

## Instructions
Generate a solution that:
1. Fixes the root cause
2. Doesn't break existing functionality
3. Follows the codebase's coding style
4. Handles edge cases appropriately

## Output Format
<solution>
<approach>
[High-level description of your fix approach]
</approach>

<changes>
<change file="path/to/file.py">
<description>[What this change does]</description>
<before>
[Code before the change]
</before>
<after>
[Code after the change]
</after>
</change>
</changes>

<rationale>
[Why this approach is correct]
</rationale>

<risks>
[Potential risks or side effects]
</risks>
</solution>
```

---

## 4. Confidence Assessment Prompt

**File**: `agent/prompts/confidence.txt`
**Version**: 1.0.0

```
You have proposed a solution to a software engineering task. Now assess your confidence in this solution.

## Your Proposed Solution
{proposed_solution}

## Confidence Assessment Criteria
Rate your confidence on a scale of 1-10 based on these factors:

1. **Problem Clarity** (Is the issue well-defined?)
   - 10: Crystal clear problem statement
   - 5: Some ambiguity
   - 1: Very unclear what's being asked

2. **Root Cause Certainty** (Are you sure about the cause?)
   - 10: Definitely identified the root cause
   - 5: Likely but not certain
   - 1: Just guessing

3. **Solution Uniqueness** (Is there one clear fix?)
   - 10: Only one reasonable approach
   - 5: A few valid approaches exist
   - 1: Many possible solutions, unclear which is best

4. **Domain Familiarity** (Do you know this codebase?)
   - 10: Very familiar with this library/pattern
   - 5: Somewhat familiar
   - 1: Completely unfamiliar

5. **Side Effect Risk** (Could this break things?)
   - 10: Very low risk, isolated change
   - 5: Moderate risk, some dependencies
   - 1: High risk, touches core functionality

6. **Edge Case Coverage** (Have you considered edge cases?)
   - 10: All edge cases handled
   - 5: Most common cases handled
   - 1: Probably missing important cases

## Output Format
<confidence>
<score>[1-10]</score>
<breakdown>
- Problem Clarity: [score]
- Root Cause Certainty: [score]
- Solution Uniqueness: [score]
- Domain Familiarity: [score]
- Side Effect Risk: [score]
- Edge Case Coverage: [score]
</breakdown>
<reasoning>
[Explain your confidence level in 2-3 sentences]
</reasoning>
<uncertainties>
[List specific things you're uncertain about]
</uncertainties>
</confidence>
```

---

## 5. Consultation Request Prompt

**File**: `agent/prompts/consultation.txt`
**Version**: 1.0.0

```
I'm working on a software engineering task and would like expert perspectives before finalizing my approach.

## Task Information
- **Repository**: {repository}
- **Issue ID**: {instance_id}

### Issue Description
{problem_statement}

## Relevant Code
```{language}
{relevant_code}
```

## My Analysis
### Root Cause
{root_cause}

### Affected Files
{affected_files}

## My Proposed Solution
{proposed_solution}

## Why I'm Seeking Consultation
Confidence Score: {confidence_score}/10
{confidence_reasoning}

### Specific Uncertainties
{uncertainties}

## Questions for You
1. Is my diagnosis of the root cause correct?
2. Is my proposed solution approach sound?
3. Are there better alternatives I should consider?
4. What edge cases or risks might I be missing?
5. Any other concerns about my approach?

Please provide your expert perspective on the best way to solve this issue.
```

---

## 6. Synthesis Prompt

**File**: `agent/prompts/synthesis.txt`
**Version**: 1.0.0

```
I've received perspectives from multiple expert models on a software engineering task. I need to synthesize these views and make a final decision.

## Original Context
Repository: {repository}
Issue: {instance_id}
My original confidence: {original_confidence}/10

## My Original Proposal
{original_proposal}

## Expert Perspectives

### GPT-5.2 Analysis
{gpt_response}

### Gemini 3 Pro Analysis
{gemini_response}

## Synthesis Task
Analyze all perspectives and decide on the best approach.

### Instructions
1. Identify points where all models agree
2. Identify points of disagreement
3. Evaluate the reasoning quality of each perspective
4. Decide on the final approach
5. Explain what you're incorporating from each model

## Output Format
<synthesis>
<agreement>
[Points all models agree on]
</agreement>

<disagreement>
[Points where models differ and why]
</disagreement>

<evaluation>
<original>
Strengths: [...]
Weaknesses: [...]
</original>
<gpt>
Strengths: [...]
Weaknesses: [...]
</gpt>
<gemini>
Strengths: [...]
Weaknesses: [...]
</gemini>
</evaluation>

<final_decision>
<approach>
[The approach I'm going with]
</approach>
<incorporated_from>
- From original: [what I'm keeping]
- From GPT: [what insights I'm using]
- From Gemini: [what insights I'm using]
</incorporated_from>
<rationale>
[Why this combined approach is best]
</rationale>
<new_confidence>[1-10]</new_confidence>
</final_decision>
</synthesis>
```

---

## 7. Patch Generation Prompt

**File**: `agent/prompts/patch.txt`
**Version**: 1.0.0

```
Generate a patch to fix the following issue.

## Final Approach
{final_approach}

## Files to Modify
{files_to_modify}

## Current File Contents
{current_contents}

## Instructions
Generate a unified diff patch that:
1. Implements the specified fix
2. Follows the existing code style
3. Includes minimal changes (don't refactor unrelated code)
4. Preserves existing functionality

## Output Format
Output ONLY the patch in unified diff format:

```diff
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,count +line,count @@
 context line
-removed line
+added line
 context line
```

Do not include any explanation before or after the patch.
```

---

## Prompt Versioning

All prompts include a version header:

```
# Prompt: [name]
# Version: [major.minor.patch]
# Last Modified: [YYYY-MM-DD]
# Description: [brief description]
#
# Changelog:
# - 1.0.0 (2024-12-16): Initial version
```

## Prompt Testing

Before using prompts in evaluation:

```bash
# Test all prompts
python tests/test_prompts.py

# Validates:
# - All placeholders are valid
# - Output format is parseable
# - No syntax errors in templates
```

## Modification Guidelines

When modifying prompts:

1. **Increment version number** (semver)
2. **Document changes** in changelog
3. **Run tests** to ensure parseability
4. **Re-run sample evaluation** to verify behavior
5. **Do not modify during active evaluation runs**
