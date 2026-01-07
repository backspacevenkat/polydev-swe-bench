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
- **Total Instances**: 33
- **Patches Generated**: 33 (100.0%)
- **Total Cost**: $3.31
- **Total Tokens**: 0

## Run Details
- **Run ID**: baseline-20251226-005019
- **Date**: 2025-12-26
- **Duration**: 60 minutes

## Submission
```bash
sb-cli submit swe-bench_verified test --predictions_path all_preds.jsonl --run_id baseline-20251226-005019
```
