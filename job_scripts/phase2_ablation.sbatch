#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=persona_phase2_ablation
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --output=logs/phase2_ablation_%j.log

set -e

echo "========================================"
echo "Phase 2: Trait Ablation Experiment"
echo "========================================"

module load miniforge/24.3.0-0
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
pip install --quiet transformers torch peft

mkdir -p logs

python scripts/05_run_trait_ablation_llama.py --num_runs 5
LATEST=$(ls -t data/results/ablation_results_*.json | head -1)
python scripts/06_analyze_ablation.py --results_file "$LATEST"

echo "Phase 2 Complete!"
EOF
