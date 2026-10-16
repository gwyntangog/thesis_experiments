#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=test_race
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_race_%j.log

module load miniforge/24.3.0-0
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
python scripts/07_test_race_defaults.py
