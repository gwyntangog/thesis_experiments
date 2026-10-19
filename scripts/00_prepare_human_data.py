#!/usr/bin/env python3
"""Prepare human ANES reference distributions"""

import pandas as pd
import json
import os

def prepare_human_distributions(anes_csv="data/human_survey/anes_cdf_2024.csv",
                               output_dir="data/human_survey"):
    """Extract distributions for our survey questions"""

    print("[Loading ANES data...]")
    df = pd.read_csv(anes_csv, low_memory=False)

    print(f"[✓] Loaded {len(df)} respondents")

    # ANES 2024 column names
    column_mapping = {
        "dem_thermometer": "V241166",      # Democratic Party feeling thermometer
        "rep_thermometer": "V241167",      # Republican Party feeling thermometer
    }

    distributions = {}

    for q_id, col_name in column_mapping.items():
        if col_name not in df.columns:
            print(f"[✗] Error: Column {col_name} not found for {q_id}")
            exit(1)

        # Get responses (handle missing values)
        responses = df[col_name].dropna()

        # Filter to 0-100 range
        responses = responses[(responses >= 0) & (responses <= 100)]

        print(f"\n{q_id} ({col_name}):")
        print(f"  Valid responses: {len(responses)}")
        print(f"  Mean: {responses.mean():.1f}")
        print(f"  Std: {responses.std():.1f}")
        print(f"  Range: [{responses.min():.0f}, {responses.max():.0f}]")

        # Store as list
        distributions[q_id] = responses.tolist()

    # Save distributions
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "anes_distributions.json")

    with open(output_file, 'w') as f:
        json.dump(distributions, f, indent=2)

    print(f"\n[✓] Human distributions saved to: {output_file}")

    return distributions

if __name__ == "__main__":
    prepare_human_distributions()
