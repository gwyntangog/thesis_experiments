#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=phase3_priorities
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=01:30:00
#SBATCH --output=logs/phase3_priorities_%j.log

set -e

echo "========================================"
echo "Phase 3: Priority Elicitation"
echo "========================================"

module load miniforge/24.3.0-0
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
pip install --quiet transformers torch scipy

mkdir -p logs data/results

# Step 1: Elicit priorities
echo "[Step 1/3] Eliciting priorities..."
python scripts/10_elicit_priorities_llama.py

# Get latest priorities file
PRIORITIES=$(ls -t data/results/elicited_priorities_*.json | head -1)
echo "Using: $(basename $PRIORITIES)"

# Step 2: Run baseline with priorities
echo "[Step 2/3] Running baseline with priorities..."
python scripts/11_run_baseline_with_priorities.py --priorities_file "$PRIORITIES"

# Get baseline with priorities
BASELINE_PRIORITIES=$(ls -t data/results/baseline_responses_with_priorities_*.json | head -1)

# Step 3: Measure consistency & fidelity
echo "[Step 3/3] Measuring consistency & fidelity..."
python scripts/02_measure_consistency.py --results_file "$BASELINE_PRIORITIES"
python scripts/03_measure_fidelity.py \
    --ai_results "$BASELINE_PRIORITIES" \
    --human_dists data/human_survey/anes_distributions.json

echo ""
echo "========================================"
echo "Phase 3 Complete!"
echo "========================================"
