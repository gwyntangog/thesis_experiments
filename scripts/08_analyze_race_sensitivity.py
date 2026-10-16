#!/usr/bin/env python3
"""Analyze race sensitivity across personas"""

import json
import pandas as pd
import argparse

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

    # Summary: Which personas are most sensitive to race?
    print("\n\nRace Sensitivity Ranking (by response range):")
    print("-" * 80)

    sensitivity = []
    for persona_name in df['persona_name'].unique():
        persona_data = df[df['persona_name'] == persona_name]
        range_resp = persona_data['response'].max() - persona_data['response'].min()
        sensitivity.append({
            "persona": persona_name,
            "range": range_resp
        })

    sensitivity_df = pd.DataFrame(sensitivity).sort_values('range', ascending=False)

    for idx, row in sensitivity_df.iterrows():
        print(f"{row['persona']:40s} | Range: {row['range']:3.0f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True)
    args = parser.parse_args()

    analyze_race_sensitivity(args.results_file)
