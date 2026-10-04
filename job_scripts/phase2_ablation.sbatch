#!/bin/bash
#
# Phase 2: Trait Ablation Experiment
# Run: sbatch job_scripts/phase2_ablation.sbatch
#

#SBATCH -p mit_normal_gpu
#SBATCH --job-name=persona_phase2_ablation
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=03:00:00git
#SBATCH --output=logs/phase2_ablation_%j.log

set -e

echo "========================================"
echo "Phase 2: Trait Ablation Experiment"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

module load miniforge/24.3.0-0
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
pip install --quiet transformers torch peft

mkdir -p logs

# Verify configs
if [ ! -f "config/ablation_config.json" ]; then
    echo "[✗] ERROR: config/ablation_config.json not found!"
    exit 1
fi

echo "[✓] Running trait ablation experiment..."
echo "    (3 personas × 4 traits × 5 runs = 60 queries)"
echo ""

python scripts/05_run_trait_ablation_llama.py --num_runs 5

# Analyze results
echo ""
echo "[Analyzing ablation results...]"
LATEST=$(ls -t data/results/ablation_results_*.json | head -1)
python scripts/06_analyze_ablation.py --results_file "$LATEST"

echo ""
echo "========================================"
echo "Phase 2 Complete!"
echo "========================================"
echo "End time: $(date)"
