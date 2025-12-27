# arXiv Submission Guide

## Paper Details

**Title:** Multi-Model Ensemble for Automated Software Engineering: Achieving 74.6% on SWE-bench Verified

**Author:** Venkata Subrhmanyam Ghanta (vsghanta@asu.edu)

**Affiliation:** Arizona State University & Polydev AI

**Category:** cs.SE (Software Engineering), cs.LG (Machine Learning), cs.AI (Artificial Intelligence)

---

## Step-by-Step Submission Process

### Step 1: Create arXiv Account

1. Go to https://arxiv.org/user/register
2. Fill in your details:
   - Username: vsghanta (or your preferred username)
   - Email: vsghanta@asu.edu
   - First Name: Venkata Subrhmanyam
   - Last Name: Ghanta
   - Affiliation: Arizona State University
3. Verify your email address
4. Wait for account approval (may take 1-2 days for new accounts)

### Step 2: Request Endorsement (If Needed)

New arXiv users need endorsement for cs.SE, cs.LG, or cs.AI categories.

**Options:**
1. **Academic Email**: Using vsghanta@asu.edu should help with faster endorsement
2. **Request Endorsement**: Find endorsers at https://arxiv.org/auth/endorse
3. **Cross-list**: If endorsed for one category, you can cross-list to others

### Step 3: Prepare Files

#### Required Files:

1. **Main Paper (LaTeX recommended)**
   - Convert `ARXIV_PAPER.md` to LaTeX format
   - Use arXiv template: https://arxiv.org/help/submit

2. **Figures** (if any)
   - PNG or PDF format
   - High resolution (300 DPI minimum)

3. **Supplementary Materials** (optional)
   - Code repository link
   - Additional results

#### Converting Markdown to LaTeX:

```bash
# Install pandoc if not already installed
brew install pandoc

# Convert markdown to LaTeX
cd /Users/venkat/Documents/polydev-swe-bench/paper
pandoc ARXIV_PAPER.md -o paper.tex --standalone

# Or use this template structure:
```

### Step 4: LaTeX Template

Create `paper.tex` with this structure:

```latex
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{listings}

\title{Multi-Model Ensemble for Automated Software Engineering: Achieving 74.6\% on SWE-bench Verified}

\author{
  Venkata Subrhmanyam Ghanta\\
  Arizona State University \& Polydev AI\\
  \texttt{vsghanta@asu.edu}
}

\date{December 2025}

\begin{document}
\maketitle

\begin{abstract}
We present a hybrid ensemble approach for automated software engineering...
\end{abstract}

% Copy content from ARXIV_PAPER.md, converting markdown to LaTeX
% Tables use \begin{tabular}
% Code blocks use \begin{lstlisting}
% Links use \href{url}{text}

\end{document}
```

### Step 5: Submit to arXiv

1. **Login** to https://arxiv.org
2. Click **"Submit"** in the top menu
3. **Choose Category**:
   - Primary: `cs.SE` (Software Engineering)
   - Cross-list: `cs.LG`, `cs.AI`
4. **Upload Files**:
   - Upload `paper.tex` and any figures
   - arXiv will compile the LaTeX
5. **Add Metadata**:
   - Title: Multi-Model Ensemble for Automated Software Engineering: Achieving 74.6% on SWE-bench Verified
   - Authors: Venkata Subrhmanyam Ghanta
   - Abstract: (copy from paper)
   - Comments: "Code available at https://github.com/backspacevenkat/polydev-swe-bench"
   - ACM-class: I.2.2 (Automatic Programming)
   - MSC-class: 68T05 (Learning and adaptive systems)
6. **Preview** the compiled PDF
7. **Submit** if everything looks correct

### Step 6: After Submission

1. **Processing**: arXiv reviews submissions (1-2 business days)
2. **Announcement**: Paper appears on arXiv after next announcement cycle
3. **arXiv ID**: You'll receive an ID like `arXiv:2501.XXXXX`
4. **Update Citation**: Update GitHub README with actual arXiv ID

---

## Quick Reference: Metadata

```yaml
Title: Multi-Model Ensemble for Automated Software Engineering: Achieving 74.6% on SWE-bench Verified

Authors: Venkata Subrhmanyam Ghanta

Abstract: |
  We present a hybrid ensemble approach for automated software engineering
  that achieves 74.6% resolution rate on SWE-bench Verified, a benchmark of
  500 real-world GitHub issues. Our key finding is that single-model and
  multi-model approaches solve fundamentally different types of problems,
  achieving only 76% overlap in resolved instances. By combining Claude
  Haiku 4.5 as the base agent with multi-model consultation via GPT 5.2
  Codex and Gemini 3 Flash Preview, we demonstrate a 15.5% relative
  improvement over the baseline alone. This work provides evidence that
  model diversity, rather than model scale, may be an underexplored
  dimension for improving AI coding agents.

Categories:
  - Primary: cs.SE (Software Engineering)
  - Secondary: cs.LG (Machine Learning), cs.AI (Artificial Intelligence)

Keywords:
  - Large Language Models
  - Software Engineering
  - Multi-Model Ensemble
  - SWE-bench
  - Code Generation

License: arXiv.org perpetual, non-exclusive license

Code: https://github.com/backspacevenkat/polydev-swe-bench
```

---

## Timeline

| Day | Action |
|-----|--------|
| Day 1 | Create arXiv account, request endorsement if needed |
| Day 2-3 | Wait for account/endorsement approval |
| Day 3-4 | Prepare LaTeX files, upload and submit |
| Day 4-5 | arXiv review and processing |
| Day 5-6 | Paper appears on arXiv |
| Day 6+ | Share on social media, update GitHub |

---

## Checklist

- [ ] Create arXiv account with vsghanta@asu.edu
- [ ] Get endorsement for cs.SE (if needed)
- [ ] Convert paper to LaTeX format
- [ ] Compile and verify PDF looks correct
- [ ] Prepare supplementary materials
- [ ] Submit to arXiv
- [ ] Update GitHub with arXiv link after publication
- [ ] Share on Twitter/LinkedIn
- [ ] Add to personal/lab website

---

## Need Help?

- arXiv Help: https://arxiv.org/help
- LaTeX Help: https://www.overleaf.com/learn
- Category Descriptions: https://arxiv.org/category_taxonomy

**Note:** The paper markdown file is at `/Users/venkat/Documents/polydev-swe-bench/paper/ARXIV_PAPER.md`
