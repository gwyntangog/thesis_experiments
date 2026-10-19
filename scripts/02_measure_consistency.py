#!/usr/bin/env python3
"""Measure consistency of baseline responses"""

import json
import pandas as pd
import numpy as np
import argparse

def measure_consistency(results_file):
    """Measure intra-cluster consistency"""

    with open(results_file) as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("CONSISTENCY ANALYSIS")
    print("=" * 80)

    consistency_metrics = []

    for persona_name in sorted(df['persona_name'].unique()):
        persona_data = df[df['persona_name'] == persona_name]

        print(f"\n{persona_name}:")
        print("-" * 80)

        for q_id in sorted(persona_data['q_id'].unique()):
            q_data = persona_data[persona_data['q_id'] == q_id]
            responses = q_data['response'].values

            mean = np.mean(responses)
            std = np.std(responses)
            cv = (std / mean * 100) if mean != 0 else 0  # Coefficient of variation
            min_resp = np.min(responses)
            max_resp = np.max(responses)

            print(f"  {q_id:20s}: Mean={mean:6.1f}, Std={std:6.1f}, CV={cv:6.1f}%, Range=[{min_resp:3.0f}, {max_resp:3.0f}]")

            consistency_metrics.append({
                "persona_name": persona_name,
                "q_id": q_id,
                "mean": float(mean),
                "std": float(std),
                "cv": float(cv),
                "min": float(min_resp),
                "max": float(max_resp),
                "n_runs": len(responses)
            })

    # Summary
    print("\n\n" + "=" * 80)
    print("CONSISTENCY SUMMARY")
    print("=" * 80)

    consistency_df = pd.DataFrame(consistency_metrics)

    print("\nBy Persona (average CV across questions):")
    persona_cv = consistency_df.groupby('persona_name')['cv'].mean().sort_values()
    for persona, cv in persona_cv.items():
        print(f"  {persona:40s}: CV = {cv:6.1f}%")

    print("\nBy Question (average CV across personas):")
    q_cv = consistency_df.groupby('q_id')['cv'].mean()
    for q_id, cv in q_cv.items():
        print(f"  {q_id:20s}: CV = {cv:6.1f}%")

    # Save metrics
    output_file = results_file.replace('baseline_responses', 'consistency_metrics')
    with open(output_file, 'w') as f:
        json.dump(consistency_metrics, f, indent=2)

    print(f"\n[✓] Consistency metrics saved to: {output_file}")

    return consistency_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True)
    args = parser.parse_args()

    measure_consistency(args.results_file)
