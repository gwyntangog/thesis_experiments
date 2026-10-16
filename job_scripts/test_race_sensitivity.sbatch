#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=race_sensitivity
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/race_sensitivity_%j.log

module load miniforge/24.3.0-0
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

python scripts/07_test_race_across_personas.py
LATEST=$(ls -t data/results/race_sensitivity_test_*.json | head -1)
python scripts/08_analyze_race_sensitivity.py --results_file "$LATEST"
