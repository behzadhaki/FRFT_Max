#!/usr/bin/env python3
"""
Generate Publication-Quality Plots for AES Journal Paper
Analyzes FRFT Homomorphism and Commutativity Properties
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib.patches import Rectangle
import argparse

# Set publication-quality plot defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color scheme for consistency
COLORS = {
    'mss': '#2E86AB',  # Blue
    'mse': '#A23B72',  # Purple/Magenta
    'highlight': '#F18F01',  # Orange
    'grid': '#C73E1D',  # Red
}


def load_results_data(filename):
    """Load test results from tab-separated file."""
    try:
        data = pd.read_csv(filename, sep='\t', comment='#')
        print(f"✓ Loaded {len(data)} records from {filename}")
        return data
    except Exception as e:
        print(f"✗ Error loading file {filename}: {e}")
        return None


def wrap_alpha(alpha):
    """Wrap alpha to [-2, 2] range."""
    while alpha > 2.0:
        alpha -= 4.0
    while alpha < -2.0:
        alpha += 4.0
    return alpha


def compute_alpha_sum(data):
    """Compute wrapped alpha sum (alpha1 + alpha2)."""
    data = data.copy()
    data['AlphaSum'] = data.apply(lambda row: wrap_alpha(row['Alpha1'] + row['Alpha2']), axis=1)
    return data


# ============================================================================
# HOMOMORPHISM PLOTS
# ============================================================================

def plot_homomorphism_heatmaps_combined(data, output_dir):
    """
    Plot 1: Side-by-side heatmaps of MSS and MSE (all windows and frequencies mixed)
    """
    print("  Generating homomorphism combined heatmap...")

    # Aggregate over all frequencies and windows
    agg_data = data.groupby(['Alpha1', 'Alpha2']).agg({
        'MSS_Homomorphic': 'mean',
        'MSE_Homomorphic': 'mean'
    }).reset_index()

    # Create pivot tables for heatmap
    mss_pivot = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSS_Homomorphic')
    mse_pivot = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSE_Homomorphic')

    # Sort indices in descending order for proper display (top = +2, bottom = -2)
    mss_pivot = mss_pivot.sort_index(ascending=False)
    mse_pivot = mse_pivot.sort_index(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MSS heatmap
    im1 = axes[0].imshow(mss_pivot.values, cmap='viridis', aspect='auto',
                         vmin=0, vmax=mss_pivot.values.max())
    axes[0].set_title('Mean Structural Similarity (MSS)', fontweight='bold')
    axes[0].set_xlabel('α₁')
    axes[0].set_ylabel('α₂')

    # Set tick positions and labels
    n_ticks = 5
    tick_positions = np.linspace(0, len(mss_pivot.columns) - 1, n_ticks, dtype=int)
    tick_labels = [f"{mss_pivot.columns[i]:.1f}" for i in tick_positions]
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels)
    axes[0].set_yticks(tick_positions)
    axes[0].set_yticklabels([f"{mss_pivot.index[i]:.1f}" for i in tick_positions])

    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('MSS Loss', rotation=270, labelpad=15)

    # MSE heatmap
    im2 = axes[1].imshow(mse_pivot.values, cmap='viridis', aspect='auto',
                         vmin=0, vmax=mse_pivot.values.max())
    axes[1].set_title('Mean Squared Error (MSE)', fontweight='bold')
    axes[1].set_xlabel('α₁')
    axes[1].set_ylabel('α₂')
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels)
    axes[1].set_yticks(tick_positions)
    axes[1].set_yticklabels([f"{mse_pivot.index[i]:.1f}" for i in tick_positions])

    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('MSE', rotation=270, labelpad=15)

    plt.tight_layout()

    filename = output_dir / 'heatmap_combined_all.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_homomorphism_vs_alpha_sum_barplot(data, output_dir):
    """
    Plot 2: MSS and MSE vs alpha_sum (all windows and frequencies mixed)
    Bar plot with error bars showing spread
    """
    print("  Generating homomorphism vs alpha sum barplot...")

    data = compute_alpha_sum(data)

    # Aggregate by alpha sum
    agg_data = data.groupby('AlphaSum').agg({
        'MSS_Homomorphic': ['mean', 'std'],
        'MSE_Homomorphic': ['mean', 'std']
    }).reset_index()

    agg_data.columns = ['AlphaSum', 'MSS_mean', 'MSS_std', 'MSE_mean', 'MSE_std']
    agg_data = agg_data.sort_values('AlphaSum')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # MSS barplot
    axes[0].bar(agg_data['AlphaSum'], agg_data['MSS_mean'],
                yerr=agg_data['MSS_std'], capsize=3,
                color=COLORS['mss'], alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('α₁ + α₂ (wrapped)')
    axes[0].set_ylabel('MSS Loss')
    axes[0].set_title('Homomorphism: MSS vs Total Alpha', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_xticks(np.arange(-2, 2.1, 0.5))

    # MSE barplot
    axes[1].bar(agg_data['AlphaSum'], agg_data['MSE_mean'],
                yerr=agg_data['MSE_std'], capsize=3,
                color=COLORS['mse'], alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('α₁ + α₂ (wrapped)')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Homomorphism: MSE vs Total Alpha', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_xticks(np.arange(-2, 2.1, 0.5))

    plt.tight_layout()

    filename = output_dir / 'barplot_vs_alpha_sum_all.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_homomorphism_boxplot_by_window(data, output_dir):
    """
    Plot 3: MSS and MSE vs alpha_sum, grouped by window size
    Boxplots for each window size side-by-side for each alpha_sum
    """
    print("  Generating homomorphism boxplot grouped by window...")

    data = compute_alpha_sum(data)

    # Round alpha sum for grouping
    data['AlphaSum_rounded'] = data['AlphaSum'].round(1)

    window_sizes = sorted(data['Window'].unique())
    alpha_sums = sorted(data['AlphaSum_rounded'].unique())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Prepare data for boxplot
    positions_base = np.arange(len(alpha_sums))
    width = 0.8 / len(window_sizes)

    for metric_idx, (metric, ax) in enumerate(zip(['MSS_Homomorphic', 'MSE_Homomorphic'], axes)):
        boxplot_data = []
        positions = []
        colors = []
        window_labels = []

        cmap = plt.cm.Set3(np.linspace(0, 1, len(window_sizes)))

        for win_idx, window in enumerate(window_sizes):
            for alpha_idx, alpha_sum in enumerate(alpha_sums):
                subset = data[(data['Window'] == window) &
                              (data['AlphaSum_rounded'] == alpha_sum)]
                if len(subset) > 0:
                    boxplot_data.append(subset[metric].values)
                    pos = positions_base[alpha_idx] + (win_idx - len(window_sizes)/2 + 0.5) * width
                    positions.append(pos)
                    colors.append(cmap[win_idx])
                    if alpha_idx == 0:
                        window_labels.append(f'Win {window}')

        bp = ax.boxplot(boxplot_data, positions=positions, widths=width*0.9,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        medianprops=dict(linewidth=1.2, color='red'))

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        metric_name = 'MSS Loss' if metric == 'MSS_Homomorphic' else 'MSE'
        ax.set_ylabel(metric_name)
        ax.set_xlabel('α₁ + α₂ (wrapped)')
        ax.set_title(f'Homomorphism: {metric_name} by Window Size', fontweight='bold')
        ax.set_xticks(positions_base)
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_sums], rotation=0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Create legend
        legend_handles = [plt.Rectangle((0,0),1,1, facecolor=cmap[i], alpha=0.7, edgecolor='black')
                          for i in range(len(window_sizes))]
        ax.legend(legend_handles, [f'Window {w}' for w in window_sizes],
                  loc='upper right', ncol=len(window_sizes)//2, fontsize=8)

    plt.tight_layout()

    filename = output_dir / 'boxplot_by_window.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_homomorphism_boxplot_by_frequency(data, output_dir):
    """
    Plot 4: MSS and MSE vs alpha_sum, grouped by frequency
    Boxplots for each frequency side-by-side for each alpha_sum
    """
    print("  Generating homomorphism boxplot grouped by frequency...")

    data = compute_alpha_sum(data)

    # Round alpha sum for grouping
    data['AlphaSum_rounded'] = data['AlphaSum'].round(1)

    frequencies = sorted(data['Frequency'].unique())
    alpha_sums = sorted(data['AlphaSum_rounded'].unique())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Prepare data for boxplot
    positions_base = np.arange(len(alpha_sums))
    width = 0.8 / len(frequencies)

    for metric_idx, (metric, ax) in enumerate(zip(['MSS_Homomorphic', 'MSE_Homomorphic'], axes)):
        boxplot_data = []
        positions = []
        colors = []

        cmap = plt.cm.tab20(np.linspace(0, 1, len(frequencies)))

        for freq_idx, freq in enumerate(frequencies):
            for alpha_idx, alpha_sum in enumerate(alpha_sums):
                subset = data[(data['Frequency'] == freq) &
                              (data['AlphaSum_rounded'] == alpha_sum)]
                if len(subset) > 0:
                    boxplot_data.append(subset[metric].values)
                    pos = positions_base[alpha_idx] + (freq_idx - len(frequencies)/2 + 0.5) * width
                    positions.append(pos)
                    colors.append(cmap[freq_idx])

        bp = ax.boxplot(boxplot_data, positions=positions, widths=width*0.9,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        medianprops=dict(linewidth=1.2, color='red'))

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        metric_name = 'MSS Loss' if metric == 'MSS_Homomorphic' else 'MSE'
        ax.set_ylabel(metric_name)
        ax.set_xlabel('α₁ + α₂ (wrapped)')
        ax.set_title(f'Homomorphism: {metric_name} by Frequency', fontweight='bold')
        ax.set_xticks(positions_base)
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_sums], rotation=0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Create legend
        legend_handles = [plt.Rectangle((0,0),1,1, facecolor=cmap[i], alpha=0.7, edgecolor='black')
                          for i in range(len(frequencies))]
        ax.legend(legend_handles, [f'{int(f)} Hz' for f in frequencies],
                  loc='upper right', ncol=min(6, len(frequencies)), fontsize=7)

    plt.tight_layout()

    filename = output_dir / 'boxplot_by_frequency.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


# ============================================================================
# COMMUTATIVITY PLOTS
# ============================================================================

def plot_commutativity_heatmaps_combined(data, output_dir):
    """
    Plot 1: Side-by-side heatmaps of MSS and MSE commutativity (all windows and frequencies mixed)
    """
    print("  Generating commutativity combined heatmap...")

    # Aggregate over all frequencies and windows
    agg_data = data.groupby(['Alpha1', 'Alpha2']).agg({
        'MSS_Commutativity': 'mean',
        'MSE_Commutativity': 'mean'
    }).reset_index()

    # Create pivot tables for heatmap
    mss_pivot = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSS_Commutativity')
    mse_pivot = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSE_Commutativity')

    # Sort indices in descending order
    mss_pivot = mss_pivot.sort_index(ascending=False)
    mse_pivot = mse_pivot.sort_index(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MSS heatmap
    im1 = axes[0].imshow(mss_pivot.values, cmap='viridis', aspect='auto',
                         vmin=0, vmax=mss_pivot.values.max())
    axes[0].set_title('Mean Structural Similarity (MSS)', fontweight='bold')
    axes[0].set_xlabel('α₁')
    axes[0].set_ylabel('α₂')

    # Set tick positions and labels
    n_ticks = 5
    tick_positions = np.linspace(0, len(mss_pivot.columns) - 1, n_ticks, dtype=int)
    tick_labels = [f"{mss_pivot.columns[i]:.1f}" for i in tick_positions]
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels)
    axes[0].set_yticks(tick_positions)
    axes[0].set_yticklabels([f"{mss_pivot.index[i]:.1f}" for i in tick_positions])

    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('MSS Loss', rotation=270, labelpad=15)

    # MSE heatmap
    im2 = axes[1].imshow(mse_pivot.values, cmap='viridis', aspect='auto',
                         vmin=0, vmax=mse_pivot.values.max())
    axes[1].set_title('Mean Squared Error (MSE)', fontweight='bold')
    axes[1].set_xlabel('α₁')
    axes[1].set_ylabel('α₂')
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels)
    axes[1].set_yticks(tick_positions)
    axes[1].set_yticklabels([f"{mse_pivot.index[i]:.1f}" for i in tick_positions])

    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('MSE', rotation=270, labelpad=15)

    plt.tight_layout()

    filename = output_dir / 'heatmap_combined_all.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_commutativity_vs_alpha_sum_barplot(data, output_dir):
    """
    Plot 2: MSS and MSE commutativity vs alpha_sum (all windows and frequencies mixed)
    """
    print("  Generating commutativity vs alpha sum barplot...")

    data = compute_alpha_sum(data)

    # Aggregate by alpha sum
    agg_data = data.groupby('AlphaSum').agg({
        'MSS_Commutativity': ['mean', 'std'],
        'MSE_Commutativity': ['mean', 'std']
    }).reset_index()

    agg_data.columns = ['AlphaSum', 'MSS_mean', 'MSS_std', 'MSE_mean', 'MSE_std']
    agg_data = agg_data.sort_values('AlphaSum')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # MSS barplot
    axes[0].bar(agg_data['AlphaSum'], agg_data['MSS_mean'],
                yerr=agg_data['MSS_std'], capsize=3,
                color=COLORS['mss'], alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('α₁ + α₂ (wrapped)')
    axes[0].set_ylabel('MSS Loss')
    axes[0].set_title('Commutativity: MSS vs Total Alpha', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_xticks(np.arange(-2, 2.1, 0.5))

    # MSE barplot
    axes[1].bar(agg_data['AlphaSum'], agg_data['MSE_mean'],
                yerr=agg_data['MSE_std'], capsize=3,
                color=COLORS['mse'], alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('α₁ + α₂ (wrapped)')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Commutativity: MSE vs Total Alpha', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_xticks(np.arange(-2, 2.1, 0.5))

    plt.tight_layout()

    filename = output_dir / 'barplot_vs_alpha_sum_all.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_commutativity_boxplot_by_window(data, output_dir):
    """
    Plot 3: MSS and MSE commutativity vs alpha_sum, grouped by window size
    """
    print("  Generating commutativity boxplot grouped by window...")

    data = compute_alpha_sum(data)
    data['AlphaSum_rounded'] = data['AlphaSum'].round(1)

    window_sizes = sorted(data['Window'].unique())
    alpha_sums = sorted(data['AlphaSum_rounded'].unique())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    positions_base = np.arange(len(alpha_sums))
    width = 0.8 / len(window_sizes)

    for metric_idx, (metric, ax) in enumerate(zip(['MSS_Commutativity', 'MSE_Commutativity'], axes)):
        boxplot_data = []
        positions = []
        colors = []

        cmap = plt.cm.Set3(np.linspace(0, 1, len(window_sizes)))

        for win_idx, window in enumerate(window_sizes):
            for alpha_idx, alpha_sum in enumerate(alpha_sums):
                subset = data[(data['Window'] == window) &
                              (data['AlphaSum_rounded'] == alpha_sum)]
                if len(subset) > 0:
                    boxplot_data.append(subset[metric].values)
                    pos = positions_base[alpha_idx] + (win_idx - len(window_sizes)/2 + 0.5) * width
                    positions.append(pos)
                    colors.append(cmap[win_idx])

        bp = ax.boxplot(boxplot_data, positions=positions, widths=width*0.9,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        medianprops=dict(linewidth=1.2, color='red'))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        metric_name = 'MSS Loss' if metric == 'MSS_Commutativity' else 'MSE'
        ax.set_ylabel(metric_name)
        ax.set_xlabel('α₁ + α₂ (wrapped)')
        ax.set_title(f'Commutativity: {metric_name} by Window Size', fontweight='bold')
        ax.set_xticks(positions_base)
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_sums], rotation=0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        legend_handles = [plt.Rectangle((0,0),1,1, facecolor=cmap[i], alpha=0.7, edgecolor='black')
                          for i in range(len(window_sizes))]
        ax.legend(legend_handles, [f'Window {w}' for w in window_sizes],
                  loc='upper right', ncol=len(window_sizes)//2, fontsize=8)

    plt.tight_layout()

    filename = output_dir / 'boxplot_by_window.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_commutativity_boxplot_by_frequency(data, output_dir):
    """
    Plot 4: MSS and MSE commutativity vs alpha_sum, grouped by frequency
    """
    print("  Generating commutativity boxplot grouped by frequency...")

    data = compute_alpha_sum(data)
    data['AlphaSum_rounded'] = data['AlphaSum'].round(1)

    frequencies = sorted(data['Frequency'].unique())
    alpha_sums = sorted(data['AlphaSum_rounded'].unique())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    positions_base = np.arange(len(alpha_sums))
    width = 0.8 / len(frequencies)

    for metric_idx, (metric, ax) in enumerate(zip(['MSS_Commutativity', 'MSE_Commutativity'], axes)):
        boxplot_data = []
        positions = []
        colors = []

        cmap = plt.cm.tab20(np.linspace(0, 1, len(frequencies)))

        for freq_idx, freq in enumerate(frequencies):
            for alpha_idx, alpha_sum in enumerate(alpha_sums):
                subset = data[(data['Frequency'] == freq) &
                              (data['AlphaSum_rounded'] == alpha_sum)]
                if len(subset) > 0:
                    boxplot_data.append(subset[metric].values)
                    pos = positions_base[alpha_idx] + (freq_idx - len(frequencies)/2 + 0.5) * width
                    positions.append(pos)
                    colors.append(cmap[freq_idx])

        bp = ax.boxplot(boxplot_data, positions=positions, widths=width*0.9,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        medianprops=dict(linewidth=1.2, color='red'))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        metric_name = 'MSS Loss' if metric == 'MSS_Commutativity' else 'MSE'
        ax.set_ylabel(metric_name)
        ax.set_xlabel('α₁ + α₂ (wrapped)')
        ax.set_title(f'Commutativity: {metric_name} by Frequency', fontweight='bold')
        ax.set_xticks(positions_base)
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_sums], rotation=0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        legend_handles = [plt.Rectangle((0,0),1,1, facecolor=cmap[i], alpha=0.7, edgecolor='black')
                          for i in range(len(frequencies))]
        ax.legend(legend_handles, [f'{int(f)} Hz' for f in frequencies],
                  loc='upper right', ncol=min(6, len(frequencies)), fontsize=7)

    plt.tight_layout()

    filename = output_dir / 'boxplot_by_frequency.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality plots for AES journal paper'
    )
    parser.add_argument('--results-dir', type=str, default='./test_results',
                        help='Base directory containing test results')
    parser.add_argument('--output-dir', type=str, default='./paper_plots',
                        help='Output directory for generated plots')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print("\n" + "=" * 70)
    print("  FRFT Paper Plot Generator")
    print("=" * 70 + "\n")

    # Create output directories
    homo_output = output_dir / 'homomorphism'
    comm_output = output_dir / 'commutativity'
    homo_output.mkdir(parents=True, exist_ok=True)
    comm_output.mkdir(parents=True, exist_ok=True)

    total_plots = 0

    # ========================================================================
    # PROCESS HOMOMORPHISM RESULTS
    # ========================================================================

    print("\n[Processing Homomorphism Results]")
    homo_dir = results_dir / 'homomorphism_mss_grid_windowed'

    if not homo_dir.exists():
        print(f"⚠ Homomorphism directory not found: {homo_dir}")
        print("  Trying direct mode...")
        homo_dir = results_dir / 'homomorphism_mss_grid_direct'

    if homo_dir.exists():
        # Find all result files
        homo_files = list(homo_dir.glob('mss_grid_results_win_*.txt'))

        if not homo_files:
            # Try without window suffix (direct mode)
            homo_files = list(homo_dir.glob('mss_grid_results.txt'))

        if homo_files:
            print(f"  Found {len(homo_files)} homomorphism result file(s)")

            # Load and concatenate all files
            all_homo_data = []
            for file in homo_files:
                data = load_results_data(file)
                if data is not None:
                    all_homo_data.append(data)

            if all_homo_data:
                homo_df = pd.concat(all_homo_data, ignore_index=True)
                print(f"  Total homomorphism records: {len(homo_df)}")

                # Generate all plots
                total_plots += plot_homomorphism_heatmaps_combined(homo_df, homo_output)
                total_plots += plot_homomorphism_vs_alpha_sum_barplot(homo_df, homo_output)
                total_plots += plot_homomorphism_boxplot_by_window(homo_df, homo_output)
                total_plots += plot_homomorphism_boxplot_by_frequency(homo_df, homo_output)
        else:
            print(f"  ⚠ No homomorphism result files found in {homo_dir}")
    else:
        print(f"  ⚠ Homomorphism directory not found: {homo_dir}")

    # ========================================================================
    # PROCESS COMMUTATIVITY RESULTS
    # ========================================================================

    print("\n[Processing Commutativity Results]")

    # Try windowed mode first
    comm_files = list(homo_dir.glob('mss_commutativity_results_win_*.txt'))

    if not comm_files:
        # Try direct mode
        comm_files = list(homo_dir.glob('mss_commutativity_results.txt'))

    if comm_files:
        print(f"  Found {len(comm_files)} commutativity result file(s)")

        # Load and concatenate all files
        all_comm_data = []
        for file in comm_files:
            data = load_results_data(file)
            if data is not None:
                all_comm_data.append(data)

        if all_comm_data:
            comm_df = pd.concat(all_comm_data, ignore_index=True)
            print(f"  Total commutativity records: {len(comm_df)}")

            # Generate all plots
            total_plots += plot_commutativity_heatmaps_combined(comm_df, comm_output)
            total_plots += plot_commutativity_vs_alpha_sum_barplot(comm_df, comm_output)
            total_plots += plot_commutativity_boxplot_by_window(comm_df, comm_output)
            total_plots += plot_commutativity_boxplot_by_frequency(comm_df, comm_output)
    else:
        print(f"  ⚠ No commutativity result files found")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 70)
    print("  Plot Generation Complete!")
    print("=" * 70)
    print(f"\n✓ Generated {total_plots} publication-quality plots")
    print(f"\nOutput locations:")
    print(f"  Homomorphism plots → {homo_output}/")
    print(f"  Commutativity plots → {comm_output}/")
    print("\nAll plots saved as PDF files at 300 DPI\n")

    return 0


if __name__ == '__main__':
    exit(main())