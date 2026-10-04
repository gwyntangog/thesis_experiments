#!/usr/bin/env python3
"""
Visualize trait ablation effects
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

def plot_ablation_effects(results_file, output_file="data/results/ablation_effects.png"):
    """Create heatmap of trait effects"""

    with open(results_file) as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    # Create pivot table: traits × questions
    pivot = df.pivot_table(
        values='effect_size',
        index='trait_removed',
        columns='q_id',
        aggfunc='mean'
    )

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add values in cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            text = ax.text(j, i, f'{value:.1f}', ha="center", va="center", color="black", fontsize=10)

    ax.set_title("Trait Ablation Effects: How Much Do Responses Change?", fontsize=14, fontweight='bold')
    ax.set_xlabel("Survey Question", fontsize=12)
    ax.set_ylabel("Trait Removed", fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Effect Size (0-100 scale)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file}")

    # Bar plot: trait importance ranking
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    trait_importance = df.groupby('trait_removed')['effect_size'].mean().sort_values(ascending=True)

    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(trait_importance)))

    trait_importance.plot(kind='barh', ax=ax2, color=colors)

    ax2.set_xlabel("Average Effect Size When Removed", fontsize=12)
    ax2.set_ylabel("Trait", fontsize=12)
    ax2.set_title("Trait Importance Ranking\n(Higher = More Important)", fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_file2 = output_file.replace('.png', '_ranking.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True)
    args = parser.parse_args()

    plot_ablation_effects(args.results_file)
