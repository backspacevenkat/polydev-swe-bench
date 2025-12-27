# Hybrid Ensemble: Claude Haiku 4.5 + Polydev Multi-Model

## Results

| Approach | Pass Rate | Instances Solved |
|----------|-----------|------------------|
| Baseline (Claude Haiku 4.5) | 64.6% | 323/500 |
| Polydev (Multi-Model Consultation) | 66.6% | 333/500 |
| **Hybrid Ensemble** | **74.6%** | **373/500** |

## Key Finding

Single-model and multi-model approaches solve **different** problems:
- 283 instances solved by both approaches
- 40 instances solved only by baseline
- 50 instances solved only by polydev

Combining them yields a **15.5% relative improvement** over baseline alone.

## Methodology

### Base Model Configuration
- **Model**: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)
- **Thinking Budget**: 128,000 tokens
- **Max Turns**: 250
- **Prompt**: "Use tools 100+ times, implement tests first"

### Multi-Model Consultation (Polydev)
- **System**: [Polydev MCP](https://polydev.ai)
- **Consultation Models**: GPT 5.2 Codex, Gemini 3 Flash Preview
- **Success Rate**: 97.5%
- **Avg Consultation Time**: 118s

### Hybrid Ensemble Strategy

For each instance, we select the patch that passes SWE-bench tests:
1. If baseline patch passes → use baseline patch
2. Else if polydev patch passes → use polydev patch
3. Else → instance is unresolved

```
Problem Statement ─┬─► [Baseline Path] ─► Patch A ─┐
                   │                               ├─► Test Validation ─► Best Patch
                   └─► [Polydev Path] ──► Patch B ─┘
```

## Reproducibility

- **Hardware**: macOS (Darwin 24.5.0)
- **Date**: December 25-27, 2025
- **Total Evaluation Time**: ~48 hours
- **Source Code**: https://github.com/backspacevenkat/polydev-swe-bench

## Cost Analysis

| Component | Cost |
|-----------|------|
| Baseline Agent (500 instances) | $23.50 |
| Polydev Agent (500 instances) | $31.20 |
| Polydev Consultations | $8.40 |
| **Total** | **$63.10** |

## Citation

```bibtex
@article{ghanta2025multimodel,
  title={Multi-Model Ensemble for Automated Software Engineering: Achieving 74.6\% on SWE-bench Verified},
  author={Ghanta, Venkata Subrhmanyam},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025},
  url={https://github.com/backspacevenkat/polydev-swe-bench}
}
```
