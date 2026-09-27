#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --job-name=ai_persona_phase1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/phase1_%j.log

module load miniforge/24.3.0-0

# Create virtual environment if needed
# python -m venv venv
# source venv/bin/activate

# Install dependencies
pip install openai pandas scipy numpy --quiet

# Set API key (store securely, don't commit!)
export OPENAI_API_KEY=$(<~/.openai_key)

# Run Phase 1: Generate LLM responses
echo "Phase 1: Generating persona responses..."
python scripts/02_run_llm_survey.py --num_runs 10

# Get the latest results file
LATEST_RESULTS=$(ls -t data/results/survey_responses_*.json | head -1)

# Run Phase 1: Measure consistency
echo "Phase 1: Analyzing consistency..."
python scripts/03_measure_consistency.py --results_file "$LATEST_RESULTS"

echo "Phase 1 complete! Check logs/ for output."
