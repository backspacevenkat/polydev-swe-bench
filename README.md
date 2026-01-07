# Matching Frontier Code Agents with Lightweight Models via Multi-Model Consultation

[![SWE-bench Verified](https://img.shields.io/badge/SWE--bench%20Verified-66.6%25%20(single)%20%7C%2074.6%25%20(Resolve@2)-brightgreen)](https://www.swebench.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2501.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Can inference-time compute substitute for model scale?** We demonstrate that Claude Haiku 4.5 (a lightweight model) achieves **66.6% on SWE-bench Verified** as a single policy, and **74.6% Resolve@2 (oracle)** when taking the best of two independent policies—matching Claude 4.5 Opus (74.4%)—when augmented with extended agent turns, large thinking budgets, and multi-model consultation.

## Key Results

| Approach | Resolution Rate | Cost/Resolved |
|----------|-----------------|---------------|
| Baseline (Claude Haiku 4.5) | 64.6% | $0.18 |
| Polydev (+ Multi-Model) | 66.6% | $0.24 |
| **Resolve@2 (oracle)†** | **74.6%** | **$0.37** |
| Claude 4.5 Opus (reference) | 74.4% | $0.97 |

†Resolve@2 (oracle): Best result from two independent Haiku 4.5 policies (baseline + Polydev). This is an upper bound showing complementarity, not a single-policy result.

**Key Finding:** Single-agent and multi-model approaches have only **76% overlap** in solved instances—24% of Resolve@2 successes come from one approach succeeding where the other failed.

## Quick Start

```bash
# Clone and install
git clone https://github.com/backspacevenkat/polydev-swe-bench.git
cd polydev-swe-bench
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="your-key"

# Run evaluation (validation set)
python scripts/swe_bench_baseline.py --mode validation --workers 4
```

## Repository Structure

```
polydev-swe-bench/
├── paper/                          # Research paper
│   ├── arxiv_paper.tex             # Main paper (LaTeX, for arXiv submission)
│   └── ARXIV_PAPER.md              # Paper in Markdown format (readable)
├── scripts/                        # Evaluation scripts
│   ├── swe_bench_baseline.py       # Baseline agent
│   └── swe_bench_polydev.py        # Polydev-enhanced agent
├── agent_v3/                       # Agent implementation
│   ├── agent.py                    # Main agent logic
│   └── consultation.py             # Multi-model consultation
├── results/                        # Evaluation results
│   ├── baseline/                   # Baseline predictions & trajectories
│   └── polydev/                    # Polydev predictions & trajectories
├── submission/                     # SWE-bench leaderboard submission
│   └── 20251227_hybrid-ensemble-haiku/
│       ├── all_preds.jsonl         # Final predictions
│       └── trajs/                  # Reasoning trajectories
└── FINAL_RESULTS.json              # Complete evaluation results
```

## Methodology

### Inference-Time Compute Dimensions

We identify three dimensions of inference-time investment:

1. **Agent Turns** (up to 250): More iterations for exploration and refinement
2. **Extended Thinking** (128K tokens): Large reasoning budget per turn
3. **Model Consultation**: Querying GPT 5.2 Codex and Gemini 3 Flash Preview

### Dual-Policy Evaluation Strategy

```
Problem Statement ─┬─► [Baseline Path] ─► Patch A ─┐
                   │   (Haiku alone)               ├─► Test Validation ─► Best Patch
                   └─► [Polydev Path] ──► Patch B ─┘
                       (Haiku + MCP)
```

## When Does Multi-Model Consultation Help?

| Problem Characteristic | Consultation Helpful |
|------------------------|---------------------|
| Multi-file changes | 78.2% |
| Ambiguous requirements | 84.7% |
| Single-file change | 61.4% |
| Clear problem statement | 65.2% |

**Takeaway:** Consultation is most valuable for complex, multi-file changes and ambiguous problem statements.

## Complementarity Analysis

| Category | Count | Description |
|----------|-------|-------------|
| Both solved | 283 | Core overlap (76%) |
| **Baseline only** | 40 | Simple fixes where consultation added noise |
| **Polydev only** | 50 | Complex issues where consultation helped |
| Neither solved | 127 | Remaining failures |

## Cost Analysis

| Approach | Total Cost | Cost/Instance | Cost/Resolved |
|----------|------------|---------------|---------------|
| Baseline only | $57.76 | $0.12 | $0.18 |
| Polydev only | $78.58 | $0.16 | $0.24 |
| **Resolve@2 (both)** | **$136.34** | **$0.27** | **$0.37** |

**Note:** Resolve@2 runs two full pipelines. Cost-effective vs. frontier models ($0.97 for Opus) but requires double compute. Effective input rates are lower than list prices due to prompt caching (~90% cache hit rate).

## Reproducibility

### Environment
- **Benchmark:** SWE-bench Verified (500 instances)
- **Base Model:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)
- **Thinking Budget:** 128,000 tokens
- **Max Turns:** 250
- **Evaluation Period:** December 25-27, 2025

### Data Availability
- **Predictions:** `submission/20251227_hybrid-ensemble-haiku/all_preds.jsonl`
- **Trajectories:** `submission/20251227_hybrid-ensemble-haiku/trajs/`
- **Metrics:** `results/baseline/metrics.jsonl`, `results/polydev/metrics.jsonl`

## Citation

```bibtex
@article{ghanta2026matching,
  title={Matching Frontier Code Agents with Lightweight Models
         via Multi-Model Consultation},
  author={Ghanta, Venkata Subrhmanyam and Paladugu, Pujitha Sri Lakshmi},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2026},
  url={https://github.com/backspacevenkat/polydev-swe-bench}
}
```

## Authors

- **Venkata Subrhmanyam Ghanta** - Arizona State University & Polydev AI (vsghanta@asu.edu)
- **Pujitha Sri Lakshmi Paladugu** - Microsoft (pupaladu@microsoft.com)

## Related Work

- [SWE-bench](https://www.swebench.com/) - The benchmark
- [Polydev](https://polydev.ai) - Multi-model consultation platform
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Anthropic for Claude API access
- Princeton NLP for SWE-bench benchmark
- OpenAI and Google for consultation model APIs
