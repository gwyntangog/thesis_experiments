import json
import numpy as np
import pandas as pd
from scipy.stats import variation
import argparse

def load_results(results_file):
    with open(results_file) as f:
        return json.load(f)

def compute_consistency_metrics(results_df):
    """
    Compute consistency metrics:
    - Intra-cluster variance (within persona)
    - Test-retest correlation (same question repeated)
    - Coefficient of variation
    """

    metrics = {}

    # Group by persona and question
    for persona_id in results_df['persona_id'].unique():
        persona_data = results_df[results_df['persona_id'] == persona_id]

        persona_metrics = {
            'persona_id': persona_id,
            'n_runs': len(persona_data['run'].unique()),
            'consistency_scores': {}
        }

        for q_id in persona_data['q_id'].unique():
            q_data = persona_data[persona_data['q_id'] == q_id]

            # Extract numeric answers
            try:
                answers = pd.to_numeric(q_data['answer'], errors='coerce').dropna()
            except:
                continue

            if len(answers) > 1:
                # Coefficient of variation (std/mean)
                cv = variation(answers)

                # Within-group standard deviation
                std_dev = np.std(answers)

                persona_metrics['consistency_scores'][q_id] = {
                    'mean': np.mean(answers),
                    'std': std_dev,
                    'cv': cv,
                    'min': np.min(answers),
                    'max': np.max(answers),
                    'range': np.max(answers) - np.min(answers)
                }

        metrics[persona_id] = persona_metrics

    return metrics

def print_summary(metrics):
    """Print human-readable summary"""
    print("\n" + "="*60)
    print("CONSISTENCY METRICS SUMMARY")
    print("="*60)

    for persona_id, m in metrics.items():
        print(f"\n{m['persona_id']} ({m['n_runs']} runs):")
        print("-" * 40)

        for q_id, q_metrics in m['consistency_scores'].items():
            print(f"  {q_id}:")
            print(f"    Mean: {q_metrics['mean']:.2f}")
            print(f"    Std Dev: {q_metrics['std']:.2f}")
            print(f"    Coeff of Var: {q_metrics['cv']:.3f}")
            print(f"    Range: [{q_metrics['min']:.0f}, {q_metrics['max']:.0f}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True)
    args = parser.parse_args()

    results = load_results(args.results_file)
    results_df = pd.DataFrame(results)

    metrics = compute_consistency_metrics(results_df)
    print_summary(metrics)

    # Save metrics
    with open("data/results/consistency_metrics.json", 'w') as f:
        # Convert numpy types for JSON serialization
        metrics_serializable = json.loads(json.dumps(metrics, default=str))
        json.dump(metrics_serializable, f, indent=2)
