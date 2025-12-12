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
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
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

    # Aggregate over all frequencies and windows using MEDIAN
    agg_data = data.groupby(['Alpha1', 'Alpha2']).agg({
        'MSS_Homomorphic': 'median',
        'MSE_Homomorphic': 'median'
    }).reset_index()

    # Create pivot tables for heatmap
    mss_pivot = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSS_Homomorphic')
    mse_pivot = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSE_Homomorphic')

    # Ensure full range from -2 to 2 on both axes (step 0.2)
    full_alpha_range = np.arange(-2.0, 2.1, 0.2)
    full_alpha_range = np.round(full_alpha_range, 1)  # Round to avoid floating point issues

    # Reindex to ensure full range on both axes
    mss_pivot = mss_pivot.reindex(index=full_alpha_range, columns=full_alpha_range)
    mse_pivot = mse_pivot.reindex(index=full_alpha_range, columns=full_alpha_range)

    # Sort indices in descending order for proper display (top = +2, bottom = -2)
    mss_pivot = mss_pivot.sort_index(ascending=False)
    mse_pivot = mse_pivot.sort_index(ascending=False)

    # Ensure no negative values
    mss_pivot = mss_pivot.clip(lower=0)
    mse_pivot = mse_pivot.clip(lower=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MSS heatmap
    im1 = axes[0].imshow(mss_pivot.values, cmap='inferno', aspect='auto',
                         vmin=0, vmax=np.nanmax(mss_pivot.values))
    axes[0].set_title('MSS (Median)', fontweight='bold', pad=15)
    axes[0].set_xlabel('α₁', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('α₂', fontsize=16, fontweight='bold')

    # Set tick positions and labels
    n_ticks = 5
    tick_positions = np.linspace(0, len(mss_pivot.columns) - 1, n_ticks, dtype=int)
    tick_labels = [f"{mss_pivot.columns[i]:.1f}" for i in tick_positions]
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels, fontsize=15)
    axes[0].set_yticks(tick_positions)
    axes[0].set_yticklabels([f"{mss_pivot.index[i]:.1f}" for i in tick_positions], fontsize=15)

    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=12)

    # MSE heatmap
    im2 = axes[1].imshow(mse_pivot.values, cmap='inferno', aspect='auto',
                         vmin=0, vmax=np.nanmax(mse_pivot.values))
    axes[1].set_title('MSE (Median)', fontweight='bold', pad=15)
    axes[1].set_xlabel('α₁', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('α₂', fontsize=16, fontweight='bold')
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels, fontsize=15)
    axes[1].set_yticks(tick_positions)
    axes[1].set_yticklabels([f"{mse_pivot.index[i]:.1f}" for i in tick_positions], fontsize=15)

    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=12)

    plt.tight_layout()

    filename = output_dir / 'heatmap_combined_all.png'
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

    # Round alpha sum for cleaner grouping
    data['AlphaSum_rounded'] = data['AlphaSum'].round(1)

    # Aggregate by alpha sum using MEDIAN
    agg_data = data.groupby('AlphaSum_rounded').agg({
        'MSS_Homomorphic': ['median', 'std'],
        'MSE_Homomorphic': ['median', 'std']
    }).reset_index()

    agg_data.columns = ['AlphaSum', 'MSS_median', 'MSS_std', 'MSE_median', 'MSE_std']
    agg_data = agg_data.sort_values('AlphaSum')

    # Use ALL alpha values - no filtering

    # Ensure no negative values
    agg_data['MSS_median'] = agg_data['MSS_median'].clip(lower=0)
    agg_data['MSE_median'] = agg_data['MSE_median'].clip(lower=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # MSS barplot - narrower bars
    axes[0].bar(agg_data['AlphaSum'], agg_data['MSS_median'],
                yerr=agg_data['MSS_std'], capsize=3, width=0.15,
                color=COLORS['mss'], alpha=0.7, edgecolor='black', linewidth=0.8)
    axes[0].set_xlabel('α₁ + α₂ (wrapped)', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('MSS', fontsize=16)
    axes[0].set_title('Homomorphism: MSS vs Total Alpha', fontweight='bold', pad=15)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].tick_params(labelsize=11)
    axes[0].set_ylim(bottom=0)

    # MSE barplot - narrower bars
    axes[1].bar(agg_data['AlphaSum'], agg_data['MSE_median'],
                yerr=agg_data['MSE_std'], capsize=3, width=0.15,
                color=COLORS['mse'], alpha=0.7, edgecolor='black', linewidth=0.8)
    axes[1].set_xlabel('α₁ + α₂ (wrapped)', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('MSE', fontsize=16)
    axes[1].set_title('Homomorphism: MSE vs Total Alpha', fontweight='bold', pad=15)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].tick_params(labelsize=11)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()

    filename = output_dir / 'barplot_vs_alpha_sum_all.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_homomorphism_boxplot_by_window(data, output_dir):
    """
    Plot 3: MSS and MSE vs alpha_sum, grouped by window size
    Boxplots for each window size side-by-side for each alpha_sum
    Only showing 0 to 2 due to symmetric nature
    Only showing window sizes >= 2048
    """
    print("  Generating homomorphism boxplot grouped by window...")

    data = compute_alpha_sum(data)

    # Round alpha sum for grouping
    data['AlphaSum_rounded'] = data['AlphaSum'].round(1)

    # Filter to only 0 to 2 range (symmetric nature)
    data = data[(data['AlphaSum_rounded'] >= 0) & (data['AlphaSum_rounded'] <= 2.0)]

    # Filter to only window sizes >= 2048
    data = data[data['Window'] >= 2048]

    window_sizes = sorted(data['Window'].unique())
    alpha_sums = sorted(data['AlphaSum_rounded'].unique())

    print(f"  Using window sizes: {window_sizes}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Prepare data for boxplot
    positions_base = np.arange(len(alpha_sums))
    width = 0.15  # Fixed width per box

    for metric_idx, (metric, ax) in enumerate(zip(['MSS_Homomorphic', 'MSE_Homomorphic'], axes)):
        boxplot_data = []
        positions = []
        colors = []

        cmap = plt.cm.Set3(np.linspace(0, 1, len(window_sizes)))

        for win_idx, window in enumerate(window_sizes):
            for alpha_idx, alpha_sum in enumerate(alpha_sums):
                subset = data[(data['Window'] == window) &
                              (abs(data['AlphaSum_rounded'] - alpha_sum) < 0.05)]
                if len(subset) > 0:
                    boxplot_data.append(subset[metric].values)
                    pos = positions_base[alpha_idx] + (win_idx - len(window_sizes)/2 + 0.5) * width
                    positions.append(pos)
                    colors.append(cmap[win_idx])

        bp = ax.boxplot(boxplot_data, positions=positions, widths=width*0.85,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=1.0),
                        whiskerprops=dict(linewidth=1.0),
                        capprops=dict(linewidth=1.0),
                        medianprops=dict(linewidth=1.5, color='red'))

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        metric_name = 'MSS' if metric == 'MSS_Homomorphic' else 'MSE'
        #ax.set_ylabel(metric_name, fontsize=16)
        ax.set_xlabel('α₁ + α₂ (wrapped)', fontsize=16, fontweight='bold')
        ax.set_title(f'{metric_name}', fontweight='bold', pad=15, fontsize=18)
        ax.set_xticks(positions_base)
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_sums], rotation=0, fontsize=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='y', labelsize=13)
        ax.set_ylim(bottom=0, top= ax.get_ylim()[1]*1.2)
        ax.set_xlim(-0.5, len(alpha_sums) - 0.5)

        # Create legend
        legend_handles = [plt.Rectangle((0,0),1,1, facecolor=cmap[i], alpha=0.7, edgecolor='black')
                          for i in range(len(window_sizes))]
        ax.legend(legend_handles, [f'{w}' for w in window_sizes],
                  title='Window Size', title_fontsize=15,
                  loc='upper right', ncol=len(window_sizes), fontsize=11, framealpha=0.5)

    plt.tight_layout()

    filename = output_dir / 'boxplot_by_window.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_homomorphism_boxplot_by_frequency(data, output_dir):
    """
    Plot 4: MSS and MSE vs alpha_sum, grouped by frequency
    Boxplots for each frequency side-by-side for each alpha_sum
    Only showing 0 to 2 due to symmetric nature
    """
    print("  Generating homomorphism boxplot grouped by frequency...")

    data = compute_alpha_sum(data)

    # Round alpha sum for grouping
    data['AlphaSum_rounded'] = data['AlphaSum'].round(1)

    # Filter to only 0 to 2 range (symmetric nature)
    data = data[(data['AlphaSum_rounded'] >= 0) & (data['AlphaSum_rounded'] <= 2.0)]

    frequencies = sorted(data['Frequency'].unique())
    alpha_sums = sorted(data['AlphaSum_rounded'].unique())

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Prepare data for boxplot
    positions_base = np.arange(len(alpha_sums))
    width = 0.08  # Narrower boxes since there are more frequencies

    for metric_idx, (metric, ax) in enumerate(zip(['MSS_Homomorphic', 'MSE_Homomorphic'], axes)):
        boxplot_data = []
        positions = []
        colors = []

        cmap = plt.cm.tab20(np.linspace(0, 1, len(frequencies)))

        for freq_idx, freq in enumerate(frequencies):
            for alpha_idx, alpha_sum in enumerate(alpha_sums):
                subset = data[(data['Frequency'] == freq) &
                              (abs(data['AlphaSum_rounded'] - alpha_sum) < 0.05)]
                if len(subset) > 0:
                    boxplot_data.append(subset[metric].values)
                    pos = positions_base[alpha_idx] + (freq_idx - len(frequencies)/2 + 0.5) * width
                    positions.append(pos)
                    colors.append(cmap[freq_idx])

        bp = ax.boxplot(boxplot_data, positions=positions, widths=width*0.85,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        medianprops=dict(linewidth=1.2, color='red'))

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        metric_name = 'MSS' if metric == 'MSS_Homomorphic' else 'MSE'
        ax.set_ylabel(metric_name, fontsize=16)
        ax.set_xlabel('α₁ + α₂ (wrapped)', fontsize=16, fontweight='bold')
        ax.set_title(metric_name, fontweight='bold', pad=15, fontsize=18)
        ax.set_xticks(positions_base)
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_sums], rotation=0, fontsize=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='y', labelsize=13)
        ax.set_ylim(bottom=0, top= ax.get_ylim()[1]*1.2)
        ax.set_xlim(-0.5, len(alpha_sums) - 0.5)

        # Create legend
        legend_handles = [plt.Rectangle((0,0),1,1, facecolor=cmap[i], alpha=0.7, edgecolor='black')
                          for i in range(len(frequencies))]
        ax.legend(legend_handles, [f'{int(f)} Hz' for f in frequencies],
                  title='Frequency', title_fontsize=15,
                  loc='upper right', ncol=min(7, len(frequencies)), fontsize=15, framealpha=0.5)

    plt.tight_layout()

    filename = output_dir / 'boxplot_by_frequency.png'
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
    Only plots data that actually exists - no synthetic symmetric filling
    """
    print("  Generating commutativity combined heatmap...")

    # Aggregate over all frequencies and windows using MEDIAN
    agg_data = data.groupby(['Alpha1', 'Alpha2']).agg({
        'MSS_Commutativity': 'median',
        'MSE_Commutativity': 'median'
    }).reset_index()

    # Create pivot tables for heatmap - NO symmetric filling
    mss_pivot = agg_data.pivot_table(index='Alpha2', columns='Alpha1',
                                     values='MSS_Commutativity', aggfunc='mean')
    mse_pivot = agg_data.pivot_table(index='Alpha2', columns='Alpha1',
                                     values='MSE_Commutativity', aggfunc='mean')

    # Ensure full range from -2 to 2 on both axes (step 0.2)
    full_alpha_range = np.arange(-2.0, 2.1, 0.2)
    full_alpha_range = np.round(full_alpha_range, 1)  # Round to avoid floating point issues

    # Reindex to ensure full range on both axes
    mss_pivot = mss_pivot.reindex(index=full_alpha_range, columns=full_alpha_range)
    mse_pivot = mse_pivot.reindex(index=full_alpha_range, columns=full_alpha_range)

    # Sort indices in descending order for display (top = +2, bottom = -2)
    mss_pivot = mss_pivot.sort_index(ascending=False)
    mse_pivot = mse_pivot.sort_index(ascending=False)

    # Ensure no negative values (but keep NaN for missing data)
    mss_pivot = mss_pivot.clip(lower=0)
    mse_pivot = mse_pivot.clip(lower=0)

    # Create masked arrays to handle NaN (will show as white)
    mss_masked = np.ma.masked_invalid(mss_pivot.values)
    mse_masked = np.ma.masked_invalid(mse_pivot.values)

    # Get actual min/max for colorbar (all positive now)
    mss_vmax = np.nanmax(mss_masked)
    mse_vmax = np.nanmax(mse_masked)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MSS heatmap
    im1 = axes[0].imshow(mss_masked, cmap='inferno', aspect='auto',
                         vmin=0, vmax=mss_vmax)
    axes[0].set_title('MSS (Median)', fontweight='bold', pad=15)
    axes[0].set_xlabel('α₁', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('α₂', fontsize=16, fontweight='bold')

    # Set tick positions and labels
    n_ticks = 5
    tick_positions = np.linspace(0, len(mss_pivot.columns) - 1, n_ticks, dtype=int)
    tick_labels = [f"{mss_pivot.columns[i]:.1f}" for i in tick_positions]
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels, fontsize=15)
    axes[0].set_yticks(tick_positions)
    axes[0].set_yticklabels([f"{mss_pivot.index[i]:.1f}" for i in tick_positions], fontsize=15)

    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=12)

    # MSE heatmap
    im2 = axes[1].imshow(mse_masked, cmap='inferno', aspect='auto',
                         vmin=0, vmax=mse_vmax)
    axes[1].set_title('MSE (Median)', fontweight='bold', pad=15)
    axes[1].set_xlabel('α₁', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('α₂', fontsize=16, fontweight='bold')
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels, fontsize=15)
    axes[1].set_yticks(tick_positions)
    axes[1].set_yticklabels([f"{mse_pivot.index[i]:.1f}" for i in tick_positions], fontsize=15)

    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=12)

    plt.tight_layout()

    filename = output_dir / 'heatmap_combined_all.png'
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

    # Round alpha sum for cleaner grouping
    data['AlphaSum_rounded'] = data['AlphaSum'].round(1)

    # Aggregate by alpha sum using MEDIAN
    agg_data = data.groupby('AlphaSum_rounded').agg({
        'MSS_Commutativity': ['median', 'std'],
        'MSE_Commutativity': ['median', 'std']
    }).reset_index()

    agg_data.columns = ['AlphaSum', 'MSS_median', 'MSS_std', 'MSE_median', 'MSE_std']
    agg_data = agg_data.sort_values('AlphaSum')

    # Use ALL alpha values - no filtering

    # Ensure no negative values
    agg_data['MSS_median'] = agg_data['MSS_median'].clip(lower=0)
    agg_data['MSE_median'] = agg_data['MSE_median'].clip(lower=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # MSS barplot - narrower bars
    axes[0].bar(agg_data['AlphaSum'], agg_data['MSS_median'],
                yerr=agg_data['MSS_std'], capsize=3, width=0.15,
                color=COLORS['mss'], alpha=0.7, edgecolor='black', linewidth=0.8)
    axes[0].set_xlabel('α₁ + α₂ (wrapped)', fontsize=16, fontweight='bold')
    axes[0].set_title('MSS', fontweight='bold', pad=15)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].tick_params(labelsize=11)
    axes[0].set_ylim(bottom=0)

    # MSE barplot - narrower bars
    axes[1].bar(agg_data['AlphaSum'], agg_data['MSE_median'],
                yerr=agg_data['MSE_std'], capsize=3, width=0.15,
                color=COLORS['mse'], alpha=0.7, edgecolor='black', linewidth=0.8)
    axes[1].set_xlabel('α₁ + α₂ (wrapped)', fontsize=16, fontweight='bold')
    axes[1].set_title('MSE', fontweight='bold', pad=15)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].tick_params(labelsize=11)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()

    filename = output_dir / 'barplot_vs_alpha_sum_all.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_commutativity_boxplot_by_window(data, output_dir):
    """
    Plot 3: MSS and MSE commutativity vs alpha_sum, grouped by window size
    Only showing 0 to 2 due to symmetric nature
    Only showing window sizes >= 2048
    """
    print("  Generating commutativity boxplot grouped by window...")

    data = compute_alpha_sum(data)
    data['AlphaSum_rounded'] = data['AlphaSum'].round(1)

    # Filter to only 0 to 2 range (symmetric nature)
    data = data[(data['AlphaSum_rounded'] >= 0) & (data['AlphaSum_rounded'] <= 2.0)]

    # Filter to only window sizes >= 2048
    data = data[data['Window'] >= 2048]

    window_sizes = sorted(data['Window'].unique())
    alpha_sums = sorted(data['AlphaSum_rounded'].unique())

    print(f"  Using window sizes: {window_sizes}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    positions_base = np.arange(len(alpha_sums))
    width = 0.15

    for metric_idx, (metric, ax) in enumerate(zip(['MSS_Commutativity', 'MSE_Commutativity'], axes)):
        boxplot_data = []
        positions = []
        colors = []

        cmap = plt.cm.Set3(np.linspace(0, 1, len(window_sizes)))

        for win_idx, window in enumerate(window_sizes):
            for alpha_idx, alpha_sum in enumerate(alpha_sums):
                subset = data[(data['Window'] == window) &
                              (abs(data['AlphaSum_rounded'] - alpha_sum) < 0.05)]
                if len(subset) > 0:
                    boxplot_data.append(subset[metric].values)
                    pos = positions_base[alpha_idx] + (win_idx - len(window_sizes)/2 + 0.5) * width
                    positions.append(pos)
                    colors.append(cmap[win_idx])

        bp = ax.boxplot(boxplot_data, positions=positions, widths=width*0.85,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=1.0),
                        whiskerprops=dict(linewidth=1.0),
                        capprops=dict(linewidth=1.0),
                        medianprops=dict(linewidth=1.5, color='red'))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        metric_name = 'MSS' if metric == 'MSS_Commutativity' else 'MSE'
        ax.set_xlabel('α₁ + α₂ (wrapped)', fontsize=16, fontweight='bold')
        ax.set_title(f'{metric_name}', fontweight='bold', pad=15, fontsize=18)
        ax.set_xticks(positions_base)
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_sums], rotation=0, fontsize=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='y', labelsize=13)
        ax.set_ylim(bottom=0, top= ax.get_ylim()[1]*1.2)
        ax.set_xlim(-0.5, len(alpha_sums) - 0.5)

        legend_handles = [plt.Rectangle((0,0),1,1, facecolor=cmap[i], alpha=0.7, edgecolor='black')
                          for i in range(len(window_sizes))]
        ax.legend(legend_handles, [f'{w}' for w in window_sizes],
                  title='Window Size', title_fontsize=15,
                  loc='upper right', ncol=len(window_sizes), fontsize=11, framealpha=0.5)

    plt.tight_layout()

    filename = output_dir / 'boxplot_by_window.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_commutativity_boxplot_by_frequency(data, output_dir):
    """
    Plot 4: MSS and MSE commutativity vs alpha_sum, grouped by frequency
    Only showing 0 to 2 due to symmetric nature
    """
    print("  Generating commutativity boxplot grouped by frequency...")

    data = compute_alpha_sum(data)
    data['AlphaSum_rounded'] = data['AlphaSum'].round(1)

    # Filter to only 0 to 2 range (symmetric nature)
    data = data[(data['AlphaSum_rounded'] >= 0) & (data['AlphaSum_rounded'] <= 2.0)]

    frequencies = sorted(data['Frequency'].unique())
    alpha_sums = sorted(data['AlphaSum_rounded'].unique())

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    positions_base = np.arange(len(alpha_sums))
    width = 0.08

    for metric_idx, (metric, ax) in enumerate(zip(['MSS_Commutativity', 'MSE_Commutativity'], axes)):
        boxplot_data = []
        positions = []
        colors = []

        cmap = plt.cm.tab20(np.linspace(0, 1, len(frequencies)))

        for freq_idx, freq in enumerate(frequencies):
            for alpha_idx, alpha_sum in enumerate(alpha_sums):
                subset = data[(data['Frequency'] == freq) &
                              (abs(data['AlphaSum_rounded'] - alpha_sum) < 0.05)]
                if len(subset) > 0:
                    boxplot_data.append(subset[metric].values)
                    pos = positions_base[alpha_idx] + (freq_idx - len(frequencies)/2 + 0.5) * width
                    positions.append(pos)
                    colors.append(cmap[freq_idx])

        bp = ax.boxplot(boxplot_data, positions=positions, widths=width*0.85,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        medianprops=dict(linewidth=1.2, color='red'))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')

        metric_name = 'MSS' if metric == 'MSS_Commutativity' else 'MSE'
        ax.set_xlabel('α₁ + α₂ (wrapped)', fontsize=16, fontweight='bold')
        ax.set_title(metric_name, fontweight='bold', pad=15, fontsize=18)
        ax.set_xticks(positions_base)
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_sums], rotation=0, fontsize=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='y', labelsize=13)
        ax.set_ylim(bottom=0, top= ax.get_ylim()[1]*1.2)
        ax.set_xlim(-0.5, len(alpha_sums) - 0.5)

        legend_handles = [plt.Rectangle((0,0),1,1, facecolor=cmap[i], alpha=0.7, edgecolor='black')
                          for i in range(len(frequencies))]
        ax.legend(legend_handles, [f'{int(f)} Hz' for f in frequencies],
                  title='Frequency', title_fontsize=15,
                  loc='upper right', ncol=min(7, len(frequencies)), fontsize=15, framealpha=0.5)

    plt.tight_layout()

    filename = output_dir / 'boxplot_by_frequency.png'
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
    print("\nAll plots saved as png files at 300 DPI\n")

    return 0


if __name__ == '__main__':
    exit(main())