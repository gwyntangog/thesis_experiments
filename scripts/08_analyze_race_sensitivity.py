#!/usr/bin/env python3
"""Analyze race sensitivity across personas"""

import json
import pandas as pd
import argparse
import os

def analyze_race_sensitivity(results_file):
    """Analyze how much race affects each persona"""

    with open(results_file) as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("RACE SENSITIVITY ANALYSIS")
    print("=" * 80)

    # By persona
    print("\nResponse Range by Persona (all races, all questions):")
    print("-" * 80)

    for persona_name in df['persona_name'].unique():
        persona_data = df[df['persona_name'] == persona_name]
        responses = persona_data['response'].values

        min_resp = responses.min()
        max_resp = responses.max()
        range_resp = max_resp - min_resp
        mean_resp = responses.mean()

        print(f"{persona_name:40s} | Range: {min_resp:3.0f}-{max_resp:3.0f} (Δ={range_resp:3.0f}) | Mean: {mean_resp:5.1f}")

    # By persona and question
    print("\n\nBy Persona and Question:")
    print("-" * 80)

    for persona_name in df['persona_name'].unique():
        print(f"\n{persona_name}:")
        persona_data = df[df['persona_name'] == persona_name]

        for q_id in persona_data['q_id'].unique():
            q_data = persona_data[persona_data['q_id'] == q_id]

            print(f"  {q_id}:")
            for race in q_data['race_tested'].unique():
                race_data = q_data[q_data['race_tested'] == race]
                response = race_data['response'].values[0]
                print(f"    {race:30s}: {response:3.0f}")

    # NEW: Table comparing unspecified vs specified races
    print("\n\n" + "=" * 80)
    print("DELTA FROM 'UNSPECIFIED' RACE")
    print("=" * 80)
    print("(Positive = response increases when race is specified)")
    print("(Negative = response decreases when race is specified)")
    print("-" * 80)

    delta_results = []

    for persona_name in sorted(df['persona_name'].unique()):
        persona_data = df[df['persona_name'] == persona_name]

        for q_id in sorted(persona_data['q_id'].unique()):
            q_data = persona_data[persona_data['q_id'] == q_id]

            # Get unspecified response
            unspec_data = q_data[q_data['race_tested'] == 'of unspecified race/ethnicity']
            if unspec_data.empty:
                continue
            unspec_response = unspec_data['response'].values[0]

            # Get other races
            for race in ['non-Hispanic white', 'non-Hispanic black', 'Hispanic']:
                race_data = q_data[q_data['race_tested'] == race]
                if race_data.empty:
                    continue

                race_response = race_data['response'].values[0]
                delta = race_response - unspec_response

                delta_results.append({
                    "persona": persona_name,
                    "question": q_id,
                    "race": race,
                    "unspecified_response": unspec_response,
                    "race_response": race_response,
                    "delta": delta
                })

                sign = "+" if delta >= 0 else ""
                print(f"{persona_name:40s} | {q_id:20s} | {race:20s} | Δ = {sign}{delta:5.1f}")

    # Summary: Which races cause biggest shifts?
    print("\n\n" + "=" * 80)
    print("AVERAGE DELTA BY RACE (across all personas & questions)")
    print("=" * 80)

    delta_df = pd.DataFrame(delta_results)
    race_averages = delta_df.groupby('race')['delta'].agg(['mean', 'std', 'min', 'max'])
    race_averages = race_averages.sort_values('mean', ascending=False)

    for race, row in race_averages.iterrows():
        print(f"{race:20s} | Mean Δ: {row['mean']:+6.2f} | Std: {row['std']:6.2f} | Range: [{row['min']:6.1f}, {row['max']:6.1f}]")

    # Summary: Which personas most affected by race?
    print("\n\n" + "=" * 80)
    print("RACE SENSITIVITY BY PERSONA (average absolute delta)")
    print("=" * 80)

    persona_sensitivity = delta_df.copy()
    persona_sensitivity['abs_delta'] = persona_sensitivity['delta'].abs()
    persona_averages = persona_sensitivity.groupby('persona')['abs_delta'].mean().sort_values(ascending=False)

    for persona, avg_delta in persona_averages.items():
        print(f"{persona:40s} | Avg |Δ|: {avg_delta:6.2f}")

    # SAVE RESULTS
    # SAVE RESULTS
    print("\n\n[Saving results...]")
    output_dir = os.path.dirname(results_file)

    # Convert pandas types to Python native types for JSON serialization
    delta_results_serializable = []
    for item in delta_results:
        delta_results_serializable.append({
            k: int(v) if isinstance(v, (pd.Int64Dtype, type(pd.NA))) or str(type(v)) == "<class 'numpy.int64'>" else v
            for k, v in item.items()
        })

    # Save delta table as JSON
    delta_file = results_file.replace('race_sensitivity_test', 'race_delta_analysis')
    with open(delta_file, 'w') as f:
        json.dump({
            "analysis_type": "race_delta_from_unspecified",
            "deltas": delta_results_serializable,
            "race_averages": {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in race_averages.to_dict().items()},
            "persona_sensitivity": {k: float(v) for k, v in persona_averages.to_dict().items()}
        }, f, indent=2)

    print(f"[✓] Saved to: {delta_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True)
    args = parser.parse_args()

    analyze_race_sensitivity(args.results_file)
