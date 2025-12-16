# Setup Guide

This guide walks you through setting up the Polydev SWE-bench evaluation environment.

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Runtime |
| Docker | 24.0+ | SWE-bench test execution |
| Git | 2.40+ | Repository operations |
| Node.js | 18+ | Claude Code CLI |

### Required CLI Tools

| Tool | Purpose | Installation |
|------|---------|--------------|
| Claude Code | Claude Opus 4.5 access | `npm install -g @anthropic-ai/claude-code` |
| Codex CLI | GPT-5.2 access | `npm install -g @openai/codex` |

### API Access

| Service | Required | Notes |
|---------|----------|-------|
| Anthropic API | No | Using Claude Code CLI (free) |
| OpenAI API | No | Using Codex CLI (free) |
| Google AI API | Yes | For Gemini 3 Pro (~$0.001/query) |

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/backspacevenkat/polydev-swe-bench.git
cd polydev-swe-bench
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install SWE-bench

```bash
# Clone SWE-bench
git clone https://github.com/princeton-nlp/SWE-bench.git ../SWE-bench

# Install SWE-bench
cd ../SWE-bench
pip install -e .
cd ../polydev-swe-bench
```

### Step 5: Download Dataset

```bash
./scripts/download_swebench.sh
```

This downloads:
- SWE-bench Verified task definitions (500 tasks)
- Test specifications
- Repository metadata

### Step 6: Verify CLI Tools

```bash
# Verify Claude Code
claude-code --version

# Verify Codex
codex --version

# Verify Polydev MCP (from polydev-ai)
python -c "from polydev_mcp import PolydevMCP; print('Polydev OK')"
```

### Step 7: Configure Environment

Create `.env` file:

```bash
# .env
GOOGLE_AI_API_KEY=your_gemini_api_key_here
POLYDEV_SEED=42
LOG_LEVEL=INFO
```

### Step 8: Verify Setup

```bash
python scripts/verify_setup.py
```

Expected output:
```
[✓] Python 3.10+ detected
[✓] Docker available
[✓] Claude Code CLI working
[✓] Codex CLI working
[✓] Polydev MCP configured
[✓] SWE-bench dataset available (500 tasks)
[✓] Google AI API key valid

Setup complete! Ready to run evaluation.
```

## Directory Structure After Setup

```
polydev-swe-bench/
├── venv/                    # Virtual environment
├── data/
│   └── swe-bench-verified/  # Downloaded dataset
│       ├── tasks.json       # 500 task definitions
│       └── repos/           # Cached repository data
├── .env                     # Environment variables
└── ...
```

## Docker Setup (for SWE-bench)

SWE-bench uses Docker for isolated test execution:

```bash
# Pull required images
docker pull python:3.9-slim
docker pull python:3.10-slim
docker pull python:3.11-slim

# Verify Docker
docker run --rm python:3.10-slim python --version
```

## Troubleshooting

### Claude Code Not Found

```bash
# Check npm global path
npm config get prefix

# Add to PATH if needed
export PATH="$(npm config get prefix)/bin:$PATH"
```

### Docker Permission Denied

```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker
```

### SWE-bench Import Error

```bash
# Ensure SWE-bench is installed in editable mode
cd ../SWE-bench
pip install -e .
```

### Polydev MCP Connection Failed

```bash
# Check Polydev MCP server is running
# In polydev-ai directory:
npm run mcp-server
```

## Configuration Options

### `evaluation/config.yaml`

```yaml
# Evaluation configuration
evaluation:
  dataset: "swe-bench-verified"
  tasks: "all"  # or list of task IDs
  timeout_per_task: 900  # seconds
  parallel_workers: 1  # Sequential for reproducibility

agent:
  base_model: "claude-opus-4.5"
  consultation_threshold: 8  # Consult if confidence < 8
  max_retries: 3
  temperature: 0.0

consultation:
  enabled: true
  models:
    - name: "gpt-5.2"
      method: "codex-cli"
    - name: "gemini-3-pro"
      method: "polydev-mcp"

logging:
  level: "INFO"
  save_consultations: true
  save_reasoning: true

output:
  directory: "results/"
  format: "json"
```

## Running a Quick Test

After setup, verify everything works:

```bash
# Run on 3 sample tasks
python evaluation/run_sample.py --tasks 3 --verbose

# Check output
cat results/sample/summary.json
```

Expected output:
```json
{
  "tasks_run": 3,
  "baseline_passed": 1,
  "polydev_passed": 2,
  "consultations_triggered": 2
}
```

## Next Steps

1. **Run sample evaluation**: `python evaluation/run_sample.py --tasks 10`
2. **Review results**: Check `results/sample/`
3. **Run full baseline**: `python evaluation/run_baseline.py`
4. **Run full Polydev**: `python evaluation/run_polydev.py`
5. **Generate report**: `python scripts/generate_report.py`
