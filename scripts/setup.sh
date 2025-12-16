#!/bin/bash
# Setup script for Polydev SWE-bench Evaluation
set -e

echo "========================================"
echo "Polydev SWE-bench Evaluation Setup"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}[✓]${NC} $1 found"
        return 0
    else
        echo -e "${RED}[✗]${NC} $1 not found"
        return 1
    fi
}

# Check prerequisites
echo "Checking prerequisites..."
echo ""

check_command python3
check_command pip
check_command docker
check_command git

# Check for Claude Code CLI
if check_command claude; then
    echo "    Claude Code version: $(claude --version 2>/dev/null || echo 'unknown')"
else
    echo -e "${YELLOW}    Install: npm install -g @anthropic-ai/claude-code${NC}"
fi

# Check for Codex CLI
if check_command codex; then
    echo "    Codex version: $(codex --version 2>/dev/null || echo 'unknown')"
else
    echo -e "${YELLOW}    Install: npm install -g @openai/codex${NC}"
fi

echo ""

# Create virtual environment
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}[✓]${NC} Virtual environment created"
else
    echo -e "${YELLOW}[!]${NC} Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Installing SWE-bench..."
if [ -d "../SWE-bench" ]; then
    echo "SWE-bench directory found, installing..."
    pip install -e ../SWE-bench
else
    echo "Cloning SWE-bench..."
    git clone https://github.com/princeton-nlp/SWE-bench.git ../SWE-bench
    pip install -e ../SWE-bench
fi

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/swe-bench-verified
mkdir -p results/baseline
mkdir -p results/polydev
mkdir -p results/comparison
mkdir -p logs

# Download SWE-bench Verified if not present
echo ""
echo "Checking SWE-bench Verified dataset..."
if [ ! -f "data/swe-bench-verified/tasks.json" ]; then
    echo "Downloading dataset..."
    python3 -c "
from swebench import get_eval_refs
import json
tasks = get_eval_refs('verified')
with open('data/swe-bench-verified/tasks.json', 'w') as f:
    json.dump(tasks, f, indent=2)
print(f'Downloaded {len(tasks)} tasks')
"
else
    echo -e "${GREEN}[✓]${NC} Dataset already present"
fi

# Create .env template
echo ""
echo "Creating .env template..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# Polydev SWE-bench Environment Variables

# Google AI API key (for Gemini 3 Pro)
# Get from: https://makersuite.google.com/app/apikey
GOOGLE_AI_API_KEY=your_api_key_here

# Random seed for reproducibility
POLYDEV_SEED=42

# Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Timeouts (seconds)
MODEL_TIMEOUT=120
TEST_TIMEOUT=900
EOF
    echo -e "${GREEN}[✓]${NC} .env template created"
    echo -e "${YELLOW}[!]${NC} Edit .env and add your GOOGLE_AI_API_KEY"
else
    echo -e "${YELLOW}[!]${NC} .env already exists"
fi

# Verify setup
echo ""
echo "Verifying setup..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from agent import PolydevAgent
    print('  Agent module: OK')
except ImportError as e:
    print(f'  Agent module: FAILED ({e})')

try:
    from swebench import get_eval_refs
    tasks = get_eval_refs('verified')
    print(f'  SWE-bench: OK ({len(tasks)} tasks)')
except Exception as e:
    print(f'  SWE-bench: FAILED ({e})')
"

echo ""
echo "========================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your GOOGLE_AI_API_KEY"
echo "  2. Activate environment: source venv/bin/activate"
echo "  3. Run sample: python evaluation/run_sample.py --tasks 10"
echo ""
