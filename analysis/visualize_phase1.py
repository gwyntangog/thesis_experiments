#!/usr/bin/env python3
"""Visualize Phase 1: Consistency and Fidelity"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def visualize_phase1(consistency_file, fidelity_file, output_dir="data/results"):
    """Create Phase 1 visualizations"""

    with open(consistency_file) as f:
        consistency_data = json.load(f)

    with open(fidelity_file) as f:
        fidelity_data = json.load(f)

    consistency_df = pd.DataFrame(consistency_data)
    fidelity_df = pd.DataFrame(fidelity_data)

    sns.set_style("whitegrid")

    # ===== PLOT 1: Consistency Heatmap (Personas × Questions) =====
    print("[Creating plot 1: Consistency heatmap...]")
    fig, ax = plt.subplots(figsize=(10, 8))

    consistency_pivot = consistency_df.pivot_table(
        values='cv',
        index='persona_name',
        columns='q_id',
        aggfunc='mean'
    )

    # Sort by overall consistency
    consistency_pivot = consistency_pivot.reindex(
        consistency_df.groupby('persona_name')['cv'].mean().sort_values().index
    )

    sns.heatmap(consistency_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'CV (%)'}, vmin=5, vmax=80)
    ax.set_title('Consistency Heatmap (Coefficient of Variation %)\nLower = More Consistent',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Persona', fontsize=12)
    ax.set_xlabel('Question', fontsize=12)

    plt.tight_layout()
    output_file = f"{output_dir}/phase1_consistency_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file}")
    plt.close()

    # ===== PLOT 2: Consistency Ranking =====
    print("[Creating plot 2: Consistency ranking...]")
    fig, ax = plt.subplots(figsize=(11, 8))

    persona_cv = consistency_df.groupby('persona_name')['cv'].mean().sort_values()

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(persona_cv)))
    bars = ax.barh(persona_cv.index, persona_cv.values, color=colors)

    ax.set_xlabel('Average Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Consistency Ranking by Persona\n(Lower = More Consistent)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (persona, cv) in enumerate(persona_cv.items()):
        ax.text(cv + 1, i, f'{cv:.1f}%', va='center', fontweight='bold')

    plt.tight_layout()
    output_file = f"{output_dir}/phase1_consistency_ranking.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file}")
    plt.close()

    # ===== PLOT 3: Fidelity Heatmap =====
    print("[Creating plot 3: Fidelity heatmap...]")
    fig, ax = plt.subplots(figsize=(10, 8))

    fidelity_pivot = fidelity_df.pivot_table(
        values='wasserstein_distance',
        index='persona_name',
        columns='q_id',
        aggfunc='mean'
    )

    # Sort by overall fidelity
    fidelity_pivot = fidelity_pivot.reindex(
        fidelity_df.groupby('persona_name')['wasserstein_distance'].mean().sort_values().index
    )

    sns.heatmap(fidelity_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'Wasserstein Distance'}, vmin=10, vmax=50)
    ax.set_title('Fidelity Heatmap (Wasserstein Distance)\nLower = Better Match to Humans',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Persona', fontsize=12)
    ax.set_xlabel('Question', fontsize=12)

    plt.tight_layout()
    output_file = f"{output_dir}/phase1_fidelity_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file}")
    plt.close()

    # ===== PLOT 4: Fidelity Ranking =====
    print("[Creating plot 4: Fidelity ranking...]")
    fig, ax = plt.subplots(figsize=(11, 8))

    persona_wasserstein = fidelity_df.groupby('persona_name')['wasserstein_distance'].mean().sort_values()

    colors = plt.cm.RdYlGn_r(np.linspace(0.8, 0.2, len(persona_wasserstein)))
    bars = ax.barh(persona_wasserstein.index, persona_wasserstein.values, color=colors)

    ax.set_xlabel('Average Wasserstein Distance', fontsize=12, fontweight='bold')
    ax.set_title('Fidelity Ranking by Persona\n(Lower = Better Match to Humans)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (persona, w) in enumerate(persona_wasserstein.items()):
        ax.text(w + 0.5, i, f'{w:.2f}', va='center', fontweight='bold')

    plt.tight_layout()
    output_file = f"{output_dir}/phase1_fidelity_ranking.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file}")
    plt.close()

    # ===== PLOT 5: Consistency vs Fidelity Scatter =====
    print("[Creating plot 5: Consistency vs Fidelity scatter...]")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get average metrics per persona
    persona_metrics = pd.DataFrame({
        'Consistency (CV %)': consistency_df.groupby('persona_name')['cv'].mean(),
        'Fidelity (Wasserstein)': fidelity_df.groupby('persona_name')['wasserstein_distance'].mean()
    }).reset_index()

    scatter = ax.scatter(
        persona_metrics['Consistency (CV %)'],
        persona_metrics['Fidelity (Wasserstein)'],
        s=300,
        alpha=0.6,
        c=range(len(persona_metrics)),
        cmap='viridis'
    )

    # Add labels for each point
    for idx, row in persona_metrics.iterrows():
        party = row['persona_name'].split(',')[0].strip()
        ax.annotate(party,
                   (row['Consistency (CV %)'], row['Fidelity (Wasserstein)']),
                   fontsize=9, ha='center', va='center', fontweight='bold')

    ax.set_xlabel('Consistency (CV %) — Lower is Better →', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fidelity (Wasserstein) — Lower is Better →', fontsize=12, fontweight='bold')
    ax.set_title('Trade-off: Consistency vs Fidelity\n(Ideal: Lower Left)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = f"{output_dir}/phase1_consistency_fidelity_tradeoff.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file}")
    plt.close()

    # ===== PLOT 6: KL Divergence Comparison =====
    print("[Creating plot 6: KL divergence...")
    fig, ax = plt.subplots(figsize=(11, 8))

    kl_by_persona = fidelity_df.groupby('persona_name')['kl_divergence'].mean().sort_values()

    colors_kl = plt.cm.Reds(np.linspace(0.3, 0.9, len(kl_by_persona)))
    bars = ax.barh(kl_by_persona.index, kl_by_persona.values, color=colors_kl)

    ax.set_xlabel('Average KL Divergence', fontsize=12, fontweight='bold')
    ax.set_title('KL Divergence by Persona\n(Measures distribution shape similarity)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for i, (persona, kl) in enumerate(kl_by_persona.items()):
        ax.text(kl + 0.05, i, f'{kl:.2f}', va='center', fontweight='bold')

    plt.tight_layout()
    output_file = f"{output_dir}/phase1_kl_divergence.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved to: {output_file}")
    plt.close()

    print(f"\n[✓] All visualizations complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--consistency_file", required=True)
    parser.add_argument("--fidelity_file", required=True)
    parser.add_argument("--output_dir", default="data/results")
    args = parser.parse_args()

    visualize_phase1(args.consistency_file, args.fidelity_file, args.output_dir)
