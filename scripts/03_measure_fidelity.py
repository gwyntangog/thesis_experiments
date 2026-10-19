#!/usr/bin/env python3
"""Measure fidelity: AI vs human distributions"""

import json
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, entropy
import argparse

def measure_fidelity(ai_results_file, human_dists_file):
    """Compare AI responses to human ANES distributions"""

    with open(ai_results_file) as f:
        ai_results = json.load(f)

    with open(human_dists_file) as f:
        human_dists = json.load(f)

    ai_df = pd.DataFrame(ai_results)

    print("\n" + "=" * 80)
    print("FIDELITY ANALYSIS: AI vs HUMAN")
    print("=" * 80)

    fidelity_metrics = []

    for persona_name in sorted(ai_df['persona_name'].unique()):
        persona_data = ai_df[ai_df['persona_name'] == persona_name]

        print(f"\n{persona_name}:")
        print("-" * 80)

        for q_id in sorted(persona_data['q_id'].unique()):
            q_data = persona_data[persona_data['q_id'] == q_id]
            ai_responses = q_data['response'].values

            # Get human distribution
            if q_id not in human_dists:
                print(f"  {q_id}: No human data available")
                continue

            human_responses = np.array(human_dists[q_id])

            # Metrics
            ai_mean = np.mean(ai_responses)
            ai_std = np.std(ai_responses)
            human_mean = np.mean(human_responses)
            human_std = np.std(human_responses)

            # Wasserstein distance
            wasserstein = wasserstein_distance(ai_responses, human_responses)

            # KL divergence (binned)
            ai_hist, edges = np.histogram(ai_responses, bins=20, range=(0, 100))
            human_hist, _ = np.histogram(human_responses, bins=20, range=(0, 100))

            # Normalize
            ai_hist = ai_hist / ai_hist.sum()
            human_hist = human_hist / human_hist.sum()

            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            kl_div = entropy(ai_hist + epsilon, human_hist + epsilon)

            mean_diff = abs(ai_mean - human_mean)

            print(f"  {q_id}:")
            print(f"    AI:    mean={ai_mean:6.1f}, std={ai_std:6.1f}")
            print(f"    Human: mean={human_mean:6.1f}, std={human_std:6.1f}")
            print(f"    Wasserstein Distance: {wasserstein:6.2f}")
            print(f"    KL Divergence: {kl_div:6.2f}")
            print(f"    Mean Difference: {mean_diff:6.2f}")

            fidelity_metrics.append({
                "persona_name": persona_name,
                "q_id": q_id,
                "ai_mean": float(ai_mean),
                "ai_std": float(ai_std),
                "human_mean": float(human_mean),
                "human_std": float(human_std),
                "wasserstein_distance": float(wasserstein),
                "kl_divergence": float(kl_div),
                "mean_difference": float(mean_diff),
                "n_ai_samples": len(ai_responses),
                "n_human_samples": len(human_responses)
            })

    # Summary
    print("\n\n" + "=" * 80)
    print("FIDELITY SUMMARY")
    print("=" * 80)

    fidelity_df = pd.DataFrame(fidelity_metrics)

    print("\nAverage Wasserstein Distance by Persona:")
    persona_wasserstein = fidelity_df.groupby('persona_name')['wasserstein_distance'].mean().sort_values()
    for persona, w in persona_wasserstein.items():
        print(f"  {persona:40s}: {w:6.2f}")

    print("\nAverage Wasserstein Distance by Question:")
    q_wasserstein = fidelity_df.groupby('q_id')['wasserstein_distance'].mean()
    for q_id, w in q_wasserstein.items():
        print(f"  {q_id:20s}: {w:6.2f}")

    # Save metrics
    output_file = ai_results_file.replace('baseline_responses', 'fidelity_metrics')
    with open(output_file, 'w') as f:
        json.dump(fidelity_metrics, f, indent=2)

    print(f"\n[✓] Fidelity metrics saved to: {output_file}")

    return fidelity_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai_results", required=True)
    parser.add_argument("--human_dists", required=True)
    args = parser.parse_args()

    measure_fidelity(args.ai_results, args.human_dists)
