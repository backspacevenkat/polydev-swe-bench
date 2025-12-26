# SWE-bench Baseline Submission

## Model
- **Name**: Claude Haiku 4.5
- **Model ID**: claude-haiku-4-5-20251001
- **Extended Thinking**: 128,000 tokens

## Methodology
Exact replication of Anthropic's SWE-bench methodology:
- Simple scaffold with bash + file editing tools
- 128K thinking budget
- Default sampling parameters
- Prompt: "You should use tools as much as possible, ideally more than 100 times. You should also implement your own tests first before attempting the problem."

## Results
- **Total Instances**: 20
- **Patches Generated**: 20 (100.0%)
- **Total Cost**: $0.88
- **Total Tokens**: 0

## Run Details
- **Run ID**: validation-baseline-20
- **Date**: 2025-12-25
- **Duration**: 29 minutes

## Submission
```bash
sb-cli submit swe-bench_verified test --predictions_path all_preds.jsonl --run_id validation-baseline-20
```
