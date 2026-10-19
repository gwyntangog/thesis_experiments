#!/usr/bin/env python3
"""Visualize race sensitivity across personas"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def visualize_race_sensitivity(results_file, output_dir="data/results"):
    """Create visualizations for race sensitivity"""

    with open(results_file) as f:
        data = json.load(f)

    deltas = pd.DataFrame(data["deltas"])

    # Set style
    sns.set_style("whitegrid")

    # ===== PLOT 1: Heatmap by Persona × Race =====
    print("[Creating plot 1: Heatmap...]")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create pivot table: personas × races
    pivot_data = deltas.pivot_table(
        values='delta',
        index='persona',
        columns='race',
        aggfunc='mean'  # Average across both questions
    )

    # Sort by sensitivity
    persona_sensitivity = data["persona_sensitivity"]
    pivot_data = pivot_data.reindex(sorted(pivot_data.index,
                                           key=lambda x: persona_sensitivity.get(x, 0),
                                           reverse=True))

    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdBu_r', center=0,
                ax=ax, cbar_kws={'label': 'Delta (points)'}, vmin=-30, vmax=30)
    ax.set_title('Race Sensitivity Heatmap\n(Average Delta by Persona × Race)',
                  fontsize=14, fontweight='bold')
    ax.set_xlabel('Race Specification', fontsize=12)
    ax.set_ylabel('Persona', fontsize=12)

    plt.tight_layout()
    output_file1 = f"{output_dir}/race_sensitivity_heatmap.png"
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file1}")
    plt.close()

    # ===== PLOT 2: Persona Sensitivity Ranking =====
    print("[Creating plot 2: Ranking...]")
    fig, ax = plt.subplots(figsize=(10, 8))

    sensitivity_df = pd.DataFrame([
        {"persona": k, "sensitivity": v}
        for k, v in data["persona_sensitivity"].items()
    ]).sort_values('sensitivity', ascending=True)

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sensitivity_df)))
    bars = ax.barh(sensitivity_df['persona'], sensitivity_df['sensitivity'], color=colors)

    ax.set_xlabel('Average Absolute Delta', fontsize=12)
    ax.set_title('Persona Sensitivity Ranking\n(How much race affects responses)',
                  fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {width:.2f}', ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    output_file2 = f"{output_dir}/race_sensitivity_ranking.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file2}")
    plt.close()

    # ===== PLOT 3: Delta by Question =====
    print("[Creating plot 3: By question...]")
    fig, ax = plt.subplots(figsize=(11, 7))

    # Get deltas by question and race
    question_race = deltas.groupby(['question', 'race'])['delta'].mean().unstack()

    question_race.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Average Delta by Question × Race\n(Delta = Response when race specified - Response when unspecified)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Question', fontsize=12)
    ax.set_ylabel('Delta (points)', fontsize=12)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, label='No effect (delta=0)')
    ax.legend(title='Race', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file3 = f"{output_dir}/race_sensitivity_by_question.png"
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file3}")
    plt.close()

    # ===== PLOT 4: Distribution of Deltas by Race =====
    print("[Creating plot 4: Distribution...]")
    fig, ax = plt.subplots(figsize=(10, 6))

    races = sorted(deltas['race'].unique())
    delta_by_race = [deltas[deltas['race'] == race]['delta'].values for race in races]

    bp = ax.boxplot(delta_by_race, patch_artist=True)
    ax.set_xticklabels(races)

    # Color the boxes
    colors_box = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Delta (points)', fontsize=12)
    ax.set_xlabel('Race Specification', fontsize=12)
    ax.set_title('Distribution of Response Changes by Race', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_file4 = f"{output_dir}/race_sensitivity_distribution.png"
    plt.savefig(output_file4, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file4}")
    plt.close()

 # ===== PLOT 5: Faceted by Persona & Question =====
    print("[Creating plot 5: Faceted grid...]")

    personas = sorted(deltas['persona'].unique())
    questions = sorted(deltas['question'].unique())

    fig, axes = plt.subplots(len(personas), len(questions),
                            figsize=(14, 20), sharex=True, sharey=True)

    if len(personas) == 1:
        axes = axes.reshape(1, -1)

    for i, persona in enumerate(personas):
        for j, question in enumerate(questions):
            ax = axes[i, j]

            # Get data for this persona × question
            mask = (deltas['persona'] == persona) & (deltas['question'] == question)
            data_subset = deltas[mask].sort_values('race')

            colors_bar = ['red' if x < 0 else 'green' for x in data_subset['delta']]
            bars = ax.barh(range(len(data_subset)), data_subset['delta'].values, color=colors_bar, alpha=0.7)

            # Set race labels
            ax.set_yticks(range(len(data_subset)))
            ax.set_yticklabels(data_subset['race'].values, fontsize=8)

            # Add persona details box in top left
            if j == 0:
                # Parse persona details
                parts = persona.split(',')
                party = parts[0].strip()
                education = parts[1].strip() if len(parts) > 1 else ""
                ideology = parts[2].strip() if len(parts) > 2 else ""

                # Format details for display
                details_text = f"{party}\n{education}\n{ideology}"
                ax.text(-0.98, 0.98, details_text,
                       transform=ax.transAxes,
                       fontsize=9, fontweight='bold',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Question label on top
            if i == 0:
                q_label = question.replace('_', ' ').title()
                ax.set_title(q_label, fontsize=11, fontweight='bold', pad=10)

            ax.axvline(x=0, color='black', linestyle='-', linewidth=1.0)
            ax.grid(axis='x', alpha=0.3)
            ax.set_xlim(-35, 45)

            # Add value labels aligned with bars
            for idx, (bar, delta) in enumerate(zip(bars, data_subset['delta'].values)):
                x_pos = delta + (2 if delta > 0 else -2)
                ax.text(x_pos, idx, f'{delta:.0f}', va='center',
                       ha='left' if delta > 0 else 'right', fontsize=9, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Positive delta (race increases response)'),
        Patch(facecolor='red', alpha=0.7, label='Negative delta (race decreases response)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle('Race Sensitivity by Persona\n(Each row = Persona details | Each column = Question type)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    output_file5 = f"{output_dir}/race_sensitivity_faceted.png"
    plt.savefig(output_file5, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file5}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", required=True)
    parser.add_argument("--output_dir", default="data/results")
    args = parser.parse_args()

    visualize_race_sensitivity(args.results_file, args.output_dir)
