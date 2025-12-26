# SWE-bench Leaderboard Submission Guide

## Key Requirements (from Exa Research)

### 1. Required Files

```
experiments/
└── evaluation/
    └── verified/
        └── YYYYMMDD_hybrid-ensemble-haiku/
            ├── all_preds.jsonl      # Model predictions
            ├── metadata.yaml        # Submission metadata
            ├── README.md            # Model/approach description
            ├── logs/                # Evaluation logs per instance
            └── trajs/               # Reasoning traces (required since 7/29/2024)
```

### 2. Prediction Format (all_preds.jsonl)

Each line must be valid JSON:
```json
{
  "instance_id": "repo__repo-12345",
  "model_name_or_path": "hybrid-ensemble-haiku",
  "model_patch": "diff --git a/file.py b/file.py\n..."
}
```

### 3. metadata.yaml Format

```yaml
name: "Hybrid Ensemble (Claude Haiku 4.5 + Polydev)"
oss: true                    # Open source system
date: "2025-12-25"
verified: false              # Request via issue
trajs: true                  # Trajectories included
logs: true                   # Logs included
split: "verified"            # Which benchmark split
model: "claude-haiku-4-5-20251001"
```

### 4. Reasoning Traces (trajs/)

- **Required** for leaderboard since July 2024
- Must be human-readable
- Show intermediate steps leading to solution
- Formats: .md, .json, .yaml, .txt all acceptable
- Filename must include instance_id

Example: `trajs/django__django-16100.md`

### 5. Evaluation Logs (logs/)

Per-instance folders containing:
- `patch.diff` - Applied patch
- `report.json` - Test results
- `test_output.txt` - Test execution output
- `run_instance.log` - Full execution log

### 6. Submission Process

```bash
# 1. Fork the experiments repo
git clone https://github.com/YOUR_USERNAME/experiments.git

# 2. Create submission directory
mkdir -p experiments/evaluation/verified/20251225_hybrid-ensemble-haiku

# 3. Copy required files
cp all_preds.jsonl experiments/evaluation/verified/20251225_hybrid-ensemble-haiku/
cp metadata.yaml experiments/evaluation/verified/20251225_hybrid-ensemble-haiku/
cp README.md experiments/evaluation/verified/20251225_hybrid-ensemble-haiku/
cp -r logs/ experiments/evaluation/verified/20251225_hybrid-ensemble-haiku/
cp -r trajs/ experiments/evaluation/verified/20251225_hybrid-ensemble-haiku/

# 4. Run cleanup script
cd experiments
python -m analysis.get_results evaluation/verified/20251225_hybrid-ensemble-haiku

# 5. Create PR
git add .
git commit -m "Add Hybrid Ensemble (Claude Haiku + Polydev) submission"
git push origin main

# 6. Open PR on GitHub
# 7. Create issue requesting "verified" checkmark
```

### 7. Important Policy Notes (as of Nov 2025)

⚠️ **New Policy**: SWE-bench Verified and Multilingual now only accept 
submissions from **academic/research institutions** with **open publications**.

- Commercial submissions may be restricted
- Contact support@swebench.com for clarification
- Need to demonstrate research/academic affiliation

### 8. Using sb-cli (Alternative)

```bash
# Install
pip install sb-cli

# Authenticate
sb login

# Submit (for cloud evaluation)
sb-cli submit swe-bench_verified test \
  --predictions_path all_preds.jsonl \
  --run_id hybrid-ensemble-haiku

# Check quotas
sb-cli get-quotas
```

### 9. Our Submission Checklist

- [ ] Generate all_preds.jsonl with hybrid ensemble patches
- [ ] Create metadata.yaml with correct fields
- [ ] Write README.md describing approach
- [ ] Collect logs/ from evaluation runs
- [ ] Generate trajs/ from Claude execution logs
- [ ] Run analysis.get_results cleanup
- [ ] Fork SWE-bench/experiments
- [ ] Create PR with submission
- [ ] Open issue requesting verification

### 10. Expected Timeline

1. **Complete validation run**: ~1 hour (ongoing)
2. **Analyze results**: ~30 min
3. **Prepare submission files**: ~1 hour
4. **Create PR**: ~30 min
5. **Review/verification**: 1-2 weeks (SWE-bench team)

