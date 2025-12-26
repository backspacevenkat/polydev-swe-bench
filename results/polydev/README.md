# SWE-bench Polydev-Enhanced Submission

## Model
- **Name**: Claude Haiku 4.5 + Polydev Multi-Model Consultation
- **Model ID**: claude-haiku-4-5-20251001
- **Extended Thinking**: 128,000 tokens

## Methodology
Anthropic's SWE-bench methodology enhanced with Polydev:
1. **Polydev Consultation**: Every instance receives analysis from GPT-4, Claude, Gemini, Grok
2. **Insights Synthesis**: Multi-model perspectives combined for enhanced context
3. **Enhanced Generation**: Claude uses synthesized insights for patch generation
4. Extended thinking with 128K token budget
5. Default sampling parameters

## Results
- **Total Instances**: 20
- **Patches Generated**: 20 (100.0%)
- **Polydev Calls**: 20
- **Claude Cost**: $2.32
- **Polydev Cost**: $0.53
- **Total Cost**: $2.85
- **Total Tokens**: 0

## Run Details
- **Run ID**: validation-polydev-20
- **Date**: 2025-12-25
- **Duration**: 46 minutes

## Submission
```bash
sb-cli submit swe-bench_verified test --predictions_path all_preds.jsonl --run_id validation-polydev-20
```
