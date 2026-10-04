#!/usr/bin/env python3
"""
Analyze trait ablation results
Which traits matter most?
"""

import json
import pandas as pd
import numpy as np
import argparse

def analyze_ablation_results(results_file):
    """Analyze which traits have biggest effect"""

    with open(results_file) as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("TRAIT ABLATION ANALYSIS")
    print("=" * 80)

    # Summary by trait
    print("\nTrait Effect Sizes (Average across personas and questions):")
    print("-" * 80)

    trait_effects = df.groupby('trait_removed')['effect_size'].agg(['mean', 'std', 'min', 'max', 'count'])
    trait_effects = trait_effects.sort_values('mean', ascending=False)

    print(trait_effects.to_string())

    # By persona and trait
    print("\n\nEffect Sizes by Persona and Trait:")
    print("-" * 80)

    for persona_name in df['persona_name'].unique():
        persona_data = df[df['persona_name'] == persona_name]
        print(f"\n{persona_name}:")

        trait_summary = persona_data.groupby('trait_removed')['effect_size'].agg(['mean', 'std'])
        trait_summary = trait_summary.sort_values('mean', ascending=False)

        for trait, row in trait_summary.iterrows():
            print(f"  {trait:20s} effect: {row['mean']:6.2f} ± {row['std']:6.2f}")

    # By question
    print("\n\nEffect Sizes by Question:")
    print("-" * 80)

    for q_id in df['q_id'].unique():
        q_data = df[df['q_id'] == q_id]
        print(f"\n{q_id}:")

        trait_summary = q_data.groupby('trait_removed')['effect_size'].agg(['mean', 'std'])
        trait_summary = trait_summary.sort_values('mean', ascending=False)

        for trait, row in trait_summary.iterrows():
            print(f"  {trait:20s} effect: {row['mean']:6.2f} ± {row['std']:6.2f}")

    # Top ablations (largest effects)
    print("\n\nTop 10 Largest Effects (Trait Ablation):")
    print("-" * 80)

    top_effects = df.nlargest(10, 'effect_size')[['persona_name', 'trait_removed', 'q_id', 'effect_size', 'percent_change']]

    for idx, row in top_effects.iterrows():
        print(f"{row['persona_name']:30s} | Remove {row['trait_removed']:12s} | {row['q_id']:20s} | "
              f"Effect: {row['effect_size']:6.2f} ({row['percent_change']:6.1f}%)")

    # Key insight
    print("\n\nKey Insights:")
    print("-" * 80)

    most_important_trait = trait_effects.index[0]
    avg_effect = trait_effects.iloc[0]['mean']

    print(f"Most Important Trait: {most_important_trait}")
    print(f"  Average effect when removed: {avg_effect:.2f} points (0-100 scale)")

    least_important_trait = trait_effects.index[-1]
    avg_effect = trait_effects.iloc[-1]['mean']

    print(f"\nLeast Important Trait: {least_important_trait}")
    print(f"  Average effect when removed: {avg_effect:.2f} points")

    print("\n" + "=" * 80)

    return df

def create_ablation_summary(df, output_file="data/results/ablation_summary.json"):
    """Create summary JSON for visualization"""

    summary = {
        "by_trait": {},
        "by_persona": {},
        "by_question": {}
    }

    # By trait
    for trait in df['trait_removed'].unique():
        trait_data = df[df['trait_removed'] == trait]
        summary["by_trait"][trait] = {
            "mean_effect": float(trait_data['effect_size'].mean()),
            "std_effect": float(trait_data['effect_size'].std()),
            "n_tests": len(trait_data)
        }

    # By persona
    for persona in df['persona_name'].unique():
        persona_data = df[df['persona_name'] == persona]
        summary["by_persona"][persona] = {
            "traits_tested": persona_data['trait_removed'].unique().tolist(),
            "effects": persona_data.groupby('trait_removed')['effect_size'].mean().to_dict()
        }

    # By question
    for q_id in df['q_id'].unique():
        q_data = df[df['q_id'] == q_id]
        summary["by_question"][q_id] = {
            "mean_effect": float(q_data['effect_size'].mean()),
            "effects_by_trait": q_data.groupby('trait_removed')['effect_size'].mean().to_dict()
        }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[✓] Summary saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True, help="Path to ablation_results_*.json")
    args = parser.parse_args()

    df = analyze_ablation_results(args.results_file)
    create_ablation_summary(df)
