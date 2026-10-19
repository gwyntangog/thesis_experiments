#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=phase1_baseline
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/phase1_baseline_%j.log

set -e

echo "========================================"
echo "Phase 1: Baseline Consistency & Fidelity"
echo "========================================"

module load miniforge/24.3.0-0
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
pip install --quiet transformers torch scipy

mkdir -p logs data/results

# Check configs
if [ ! -f "config/personas.json" ]; then
    echo "[✗] personas.json not found!"
    exit 1
fi

# Step 1: Generate baseline responses
echo "[Step 1/3] Running baseline survey..."
python scripts/01_run_baseline_llama.py --num_runs 10

# Get latest baseline file
BASELINE=$(ls -t data/results/baseline_responses_*.json | head -1)
echo "Using: $(basename $BASELINE)"

# Step 2: Measure consistency
echo "[Step 2/3] Measuring consistency..."
python scripts/02_measure_consistency.py --results_file "$BASELINE"

# Step 3: Measure fidelity
echo "[Step 3/3] Measuring fidelity..."
python scripts/03_measure_fidelity.py \
    --ai_results "$BASELINE" \
    --human_dists data/human_survey/anes_distributions.json

echo ""
echo "========================================"
echo "Phase 1 Complete!"
echo "========================================"
