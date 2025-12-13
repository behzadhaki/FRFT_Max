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


def load_passthrough_reversal_results(filename):
    """Load pass-through/reversal MSS analysis results, skipping text headers."""
    try:
        # Read the file line by line to find where the CSV table starts
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Find the line that contains the column headers (starts with test_type)
        header_line_idx = None
        for idx, line in enumerate(lines):
            if line.startswith('test_type'):
                header_line_idx = idx
                break

        if header_line_idx is None:
            print(f"✗ Could not find CSV header in {filename}")
            return None

        # Read from header line onwards
        data = pd.read_csv(filename, sep='\t', skiprows=header_line_idx)
        print(f"✓ Loaded {len(data)} records from {filename}")
        return data

    except Exception as e:
        print(f"✗ Error loading pass-through/reversal results: {e}")
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
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].tick_params(labelsize=11)
    axes[0].set_ylim(bottom=0)

    # MSE barplot - narrower bars
    axes[1].bar(agg_data['AlphaSum'], agg_data['MSE_median'],
                yerr=agg_data['MSE_std'], capsize=3, width=0.15,
                color=COLORS['mse'], alpha=0.7, edgecolor='black', linewidth=0.8)
    axes[1].set_xlabel('α₁ + α₂ (wrapped)', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('MSE', fontsize=16)
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
# PERFORMANCE TIMING PLOTS
# ============================================================================

def load_timing_data(filename):
    """Load timing results from tab-separated file."""
    try:
        data = pd.read_csv(filename, sep='\t')
        print(f"✓ Loaded {len(data)} records from {filename}")
        return data
    except Exception as e:
        print(f"✗ Error loading file {filename}: {e}")
        return None


def compute_complexity_trajectories(window_sizes, base_time_64):
    """
    Compute theoretical complexity trajectories starting from the time at window size 64.

    Args:
        window_sizes: Array of window sizes
        base_time_64: The measured time at window size 64

    Returns:
        Dictionary with O(N), O(N log N), and O(N²) trajectories
    """
    # Use 64 as reference point
    N_base = 64

    trajectories = {}

    for N in window_sizes:
        if N not in trajectories:
            trajectories[N] = {}

        # O(N) complexity: time scales linearly with N
        trajectories[N]['O(N)'] = base_time_64 * (N / N_base)

        # O(N log N) complexity: time scales as N * log(N)
        trajectories[N]['O(NlogN)'] = base_time_64 * (N / N_base) * (np.log2(N) / np.log2(N_base))

        # O(N²) complexity: time scales as N²
        trajectories[N]['O(N²)'] = base_time_64 * ((N / N_base) ** 2)

    return trajectories


def plot_performance_split_alpha(data, output_dir):
    """
    Plot performance in a single figure with multiple subplots:
    - One subplot for each alpha value
    - X axis: window size (log scale) - starting from 64
    - Y axis: mean inference time (ms, log scale)
    - Three complexity trajectories: O(N), O(N log N), O(N²)
    """
    print("  Generating performance plots for all alpha values...")

    # Filter data to start from window size 64
    data = data[data['window_size'] >= 64].copy()

    window_sizes = sorted(data['window_size'].unique())
    alphas = sorted(data['alpha'].unique())

    # Create subplots - 3 rows x 4 columns for 11 alpha values
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    # Define colors
    alpha_colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]

        alpha_data = data[data['alpha'] == alpha]
        alpha_data = alpha_data.sort_values('window_size')

        # Plot measured data
        ax.plot(alpha_data['window_size'], alpha_data['mean_time_ms'],
                marker='o', markersize=5, linewidth=2, alpha=0.8,
                color=alpha_colors[idx], label=f'Measured')

        # Get base time at window size 64 for complexity trajectories
        base_data = alpha_data[alpha_data['window_size'] == 64]

        if len(base_data) > 0:
            base_time_64 = base_data['mean_time_ms'].values[0]

            # Compute complexity trajectories
            trajectories = compute_complexity_trajectories(window_sizes, base_time_64)

            # Extract trajectory arrays
            O_N = [trajectories[N]['O(N)'] for N in window_sizes]
            O_NlogN = [trajectories[N]['O(NlogN)'] for N in window_sizes]
            O_N2 = [trajectories[N]['O(N²)'] for N in window_sizes]

            # Plot complexity trajectories
            ax.plot(window_sizes, O_N, '--', linewidth=2, color='green',
                    label='O(N)', alpha=0.7)
            ax.plot(window_sizes, O_NlogN, '--', linewidth=2, color='orange',
                    label='O(N log N)', alpha=0.7)
            ax.plot(window_sizes, O_N2, '--', linewidth=2, color='red',
                    label='O(N²)', alpha=0.7)

        # Formatting
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_title(f'α = {alpha:.1f}', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

        # Set x-axis ticks (starting from 64)
        ax.set_xticks([64, 256, 1024, 4096, 16384, 65536])
        ax.set_xticklabels(['64', '256', '1K', '4K', '16K', '64K'],
                           rotation=45, ha='right', fontsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # Only add axis labels to edge subplots
        if idx >= 8:  # Bottom row
            ax.set_xlabel('Window Size', fontsize=11, fontweight='bold')
        if idx % 4 == 0:  # Left column
            ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')

    # Hide unused subplots (we have 11 alphas, so 1 subplot will be unused)
    for idx in range(len(alphas), len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    fig.suptitle('FRFT Performance by Alpha Value', fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    filename = output_dir / 'performance_by_alpha.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_rtf_split_alpha(data, output_dir):
    """
    Plot Real-Time Factor in a single figure with multiple subplots:
    - One subplot for each alpha value
    - X axis: window size (log scale) - starting from 64
    - Y axis: Real-Time Factor (linear scale)
    - Horizontal line at RTF = 1.0 (real-time threshold)
    """
    print("  Generating RTF plots for all alpha values...")

    # Filter data to start from window size 64
    data = data[data['window_size'] >= 64].copy()

    window_sizes = sorted(data['window_size'].unique())
    alphas = sorted(data['alpha'].unique())

    # Create subplots - 3 rows x 4 columns for 11 alpha values
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    # Define colors
    alpha_colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

    for idx, alpha in enumerate(alphas):
        ax = axes[idx]

        alpha_data = data[data['alpha'] == alpha]
        alpha_data = alpha_data.sort_values('window_size')

        # Plot measured data
        ax.plot(alpha_data['window_size'], alpha_data['mean_rt_factor'],
                marker='o', markersize=5, linewidth=2, alpha=0.8,
                color=alpha_colors[idx], label=f'RTF')

        # Add horizontal line at RTF = 1.0
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                   label='RT Threshold', alpha=0.7)

        # Formatting
        ax.set_xscale('log', base=2)
        # NO log scale on y-axis for RTF
        ax.set_title(f'α = {alpha:.1f}', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

        # Set x-axis ticks (starting from 64)
        ax.set_xticks([64, 256, 1024, 4096, 16384, 65536])
        ax.set_xticklabels(['64', '256', '1K', '4K', '16K', '64K'],
                           rotation=45, ha='right', fontsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)

        # Only add axis labels to edge subplots
        if idx >= 8:  # Bottom row
            ax.set_xlabel('Window Size', fontsize=11, fontweight='bold')
        if idx % 4 == 0:  # Left column
            ax.set_ylabel('RTF', fontsize=11, fontweight='bold')

    # Hide unused subplots (we have 11 alphas, so 1 subplot will be unused)
    for idx in range(len(alphas), len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    # fig.suptitle('FRFT Real-Time Factor by Alpha Value', fontsize=18, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    filename = output_dir / 'rtf_by_alpha.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_performance_by_alpha_heatmap(data, output_dir):
    """
    Plot heatmap showing inference time as a function of window size and alpha.
    Starting from window size 64.
    """
    print("  Generating performance heatmap...")

    # Filter data to start from window size 64
    data = data[data['window_size'] >= 64].copy()

    # Create pivot table
    pivot_data = data.pivot_table(
        values='mean_time_ms',
        index='alpha',
        columns='window_size',
        aggfunc='mean'
    )

    # Sort by alpha (descending for proper display)
    pivot_data = pivot_data.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(pivot_data.values, cmap='inferno', aspect='auto')

    ax.set_title('Mean Inference Time Heatmap', fontweight='bold', pad=15)
    ax.set_xlabel('Window Size (samples)', fontweight='bold')
    ax.set_ylabel('Alpha (α)', fontweight='bold')

    # Set ticks
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels([str(w) for w in pivot_data.columns], rotation=45, ha='right')

    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels([f'{a:.1f}' for a in pivot_data.index])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Time (ms)', fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    filename = output_dir / 'performance_heatmap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_performance_all_alphas_aggregated(data, output_dir):
    """
    Plot performance with all alphas aggregated together for each window size.
    Shows mean, min, max with error bars/shading.
    Includes O(N), O(N log N), O(N²) complexity trajectories.
    Starting from window size 64.
    Second y-axis shows audio buffer duration in ms.
    """
    print("  Generating performance plot with all alphas aggregated...")

    # Filter data to start from window size 64
    data = data[data['window_size'] >= 64].copy()

    window_sizes = sorted(data['window_size'].unique())

    # Aggregate statistics across all alphas for each window size
    agg_stats = data.groupby('window_size').agg({
        'mean_time_ms': ['mean', 'min', 'max', 'std']
    }).reset_index()

    agg_stats.columns = ['window_size', 'mean', 'min', 'max', 'std']
    agg_stats = agg_stats.sort_values('window_size')

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot mean with shaded region for min/max
    ax.plot(agg_stats['window_size'], agg_stats['mean'],
            marker='o', markersize=7, linewidth=3,
            color=COLORS['mss'], label='Mean (all α)', alpha=0.9)

    # Add shaded region for min/max range
    ax.fill_between(agg_stats['window_size'],
                    agg_stats['min'],
                    agg_stats['max'],
                    alpha=0.3, color=COLORS['mss'],
                    label='Min-Max Range')

    # Get base time at window size 64 for complexity trajectories
    base_data = data[(data['window_size'] == 64)]
    if len(base_data) > 0:
        base_time_64 = base_data['mean_time_ms'].mean()

        # Compute complexity trajectories
        trajectories = compute_complexity_trajectories(window_sizes, base_time_64)

        # Extract trajectory arrays
        O_N = [trajectories[N]['O(N)'] for N in window_sizes]
        O_NlogN = [trajectories[N]['O(NlogN)'] for N in window_sizes]
        O_N2 = [trajectories[N]['O(N²)'] for N in window_sizes]

        # Plot complexity trajectories
        ax.plot(window_sizes, O_N, '--', linewidth=2.5, color='green',
                label='O(N)', alpha=0.7)
        ax.plot(window_sizes, O_NlogN, '--', linewidth=2.5, color='orange',
                label='O(N log N)', alpha=0.7)
        ax.plot(window_sizes, O_N2, '--', linewidth=2.5, color='red',
                label='O(N²)', alpha=0.7)

    # Formatting for primary axis
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Window Size (samples)', fontweight='bold')
    ax.set_ylabel('Mean Inference Time (ms)', fontweight='bold')
    ax.set_title('FRFT Performance (All α Aggregated)', fontweight='bold', pad=15)
    # Don't add grid here - we'll add it after setting up the second axis
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    # Set x-axis ticks to show all window sizes (starting from 64)
    ax.set_xticks(window_sizes)
    ax.set_xticklabels([str(w) for w in window_sizes], rotation=45, ha='right')

    # Create second y-axis on the right for audio buffer durations
    ax2 = ax.twinx()

    # Calculate audio buffer durations in ms for each window size
    sample_rate = 44100  # Hz

    # Include window size 32 even if not in main data (for right axis)
    all_io_sizes = [32, 64, 128, 256, 512, 1024, 2048]
    io_durations_ms = [(w / sample_rate) * 1000 for w in all_io_sizes]

    # Set the same y-limits as the primary axis (in log scale)
    ax2.set_yscale('log')
    ax2.set_ylim(ax.get_ylim())

    # Set tick positions for I/O vector sizes (32 to 2048)
    ax2.set_yticks(io_durations_ms)
    ax2.set_yticklabels([str(w) for w in all_io_sizes], fontsize=11)

    # Turn off minor ticks on the right axis
    ax2.minorticks_off()

    # Set the label for the right axis
    ax2.set_ylabel('Max/MSP Audio Buffer Size\n(Samples at 44.1kHz)',
                   fontweight='bold', fontsize=14)

    # Add horizontal gridlines ONLY at I/O vector size positions (32-2048)
    for duration_ms in io_durations_ms:
        ax.axhline(y=duration_ms, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

    plt.tight_layout()

    filename = output_dir / 'performance_all_alphas_aggregated.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_rtf_all_alphas_aggregated(data, output_dir):
    """
    Plot Real-Time Factor with all alphas aggregated together for each window size.
    Shows mean, min, max with error bars/shading.
    Includes horizontal line at RTF = 1.0 (real-time threshold).
    Starting from window size 64.
    """
    print("  Generating RTF plot with all alphas aggregated...")

    # Filter data to start from window size 64
    data = data[data['window_size'] >= 64].copy()

    window_sizes = sorted(data['window_size'].unique())

    # Aggregate statistics across all alphas for each window size
    agg_stats = data.groupby('window_size').agg({
        'mean_rt_factor': ['mean', 'min', 'max', 'std']
    }).reset_index()

    agg_stats.columns = ['window_size', 'mean', 'min', 'max', 'std']
    agg_stats = agg_stats.sort_values('window_size')

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot mean with shaded region for min/max
    ax.plot(agg_stats['window_size'], agg_stats['mean'],
            marker='o', markersize=7, linewidth=3,
            color=COLORS['mse'], label='Mean RTF (all α)', alpha=0.9)

    # Add shaded region for min/max range
    ax.fill_between(agg_stats['window_size'],
                    agg_stats['min'],
                    agg_stats['max'],
                    alpha=0.3, color=COLORS['mse'],
                    label='Min-Max Range')

    # Add horizontal line at RTF = 1.0
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5,
               label='Real-Time Threshold (RTF=1.0)', alpha=0.7)

    # Formatting
    ax.set_xscale('log', base=2)
    # NO log scale on y-axis for RTF
    ax.set_xlabel('Window Size (samples)', fontweight='bold')
    ax.set_ylabel('Real-Time Factor', fontweight='bold')
    # ax.set_title('FRFT Real-Time Performance (All α Aggregated)', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Set x-axis ticks to show all window sizes (starting from 64)
    ax.set_xticks(window_sizes)
    ax.set_xticklabels([str(w) for w in window_sizes], rotation=45, ha='right')

    plt.tight_layout()

    filename = output_dir / 'rtf_all_alphas_aggregated.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


# ============================================================================
# PASSTHROUGH AND REVERSAL PLOTS
# ============================================================================

def plot_passthrough_mss_mse_vs_window_size(data, output_dir):
    """
    Plot MSS and MSE vs window size for passthrough test (α=0).
    Publication-quality style with no title.
    """
    print("  Generating passthrough MSS/MSE vs window size plot...")

    test_data = data[data['test_type'] == 'passthrough'].copy()

    if len(test_data) == 0:
        print(f"  No data found for passthrough test")
        return 0

    # Group by window size and compute statistics
    grouped = test_data.groupby('window_size').agg({
        'mss_loss': ['mean', 'std', 'min', 'max'],
        'mse_loss': ['mean', 'std', 'min', 'max']
    }).reset_index()

    # MSS: only use window sizes >= 8096
    mss_grouped = grouped[grouped['window_size'] >= 8096].copy()

    # MSE: use all window sizes
    mse_grouped = grouped.copy()

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: MSS Loss (only window_size >= 8096)
    ax = axes[0]
    if len(mss_grouped) > 0:
        window_sizes_mss = mss_grouped['window_size'].values
        mss_mean = mss_grouped['mss_loss']['mean'].values
        mss_std = mss_grouped['mss_loss']['std'].values
        mss_min = mss_grouped['mss_loss']['min'].values
        mss_max = mss_grouped['mss_loss']['max'].values

        ax.fill_between(window_sizes_mss, mss_min, mss_max,
                        alpha=0.2, color=COLORS['mss'], label='Min/Max Range')
        ax.errorbar(window_sizes_mss, mss_mean, yerr=mss_std,
                    fmt='o-', label='Mean ± Std', linewidth=2.5, markersize=7,
                    color=COLORS['mss'], capsize=4, capthick=1.5, alpha=0.8)

    ax.set_xlabel('Window Size (samples)', fontweight='bold')
    ax.set_ylabel('MSS Loss', fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    # Plot 2: MSE Loss (all window sizes)
    ax = axes[1]
    window_sizes_mse = mse_grouped['window_size'].values
    mse_mean = mse_grouped['mse_loss']['mean'].values
    mse_std = mse_grouped['mse_loss']['std'].values
    mse_min = mse_grouped['mse_loss']['min'].values
    mse_max = mse_grouped['mse_loss']['max'].values

    ax.fill_between(window_sizes_mse, mse_min, mse_max,
                    alpha=0.2, color=COLORS['mse'], label='Min/Max Range')
    ax.errorbar(window_sizes_mse, mse_mean, yerr=mse_std,
                fmt='o-', label='Mean ± Std', linewidth=2.5, markersize=7,
                color=COLORS['mse'], capsize=4, capthick=1.5, alpha=0.8)

    ax.set_xlabel('Window Size (samples)', fontweight='bold')
    ax.set_ylabel('MSE Loss', fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    filename = output_dir / 'passthrough_mss_mse_vs_window_size.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    return 1


def plot_reversal_mss_mse_vs_window_size(data, output_dir):
    """
    Plot MSS and MSE vs window size for reversal test (α=±2).
    Aggregates both forward and backward reversal data.
    Publication-quality style with no title.
    """
    print("  Generating reversal MSS/MSE vs window size plot...")

    # Aggregate both reversal directions (α=±2)
    reversal_data = data[data['test_type'].isin(['reversal_fwd', 'reversal_bwd'])].copy()

    if len(reversal_data) == 0:
        print(f"  No data found for reversal test")
        return 0

    # Group by window size and compute statistics
    grouped = reversal_data.groupby('window_size').agg({
        'mss_loss': ['mean', 'std', 'min', 'max'],
        'mse_loss': ['mean', 'std', 'min', 'max']
    }).reset_index()

    # MSS: only use window sizes >= 8096
    mss_grouped = grouped[grouped['window_size'] >= 8096].copy()

    # MSE: use all window sizes
    mse_grouped = grouped.copy()

    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: MSS Loss (only window_size >= 8096)
    ax = axes[0]
    if len(mss_grouped) > 0:
        window_sizes_mss = mss_grouped['window_size'].values
        mss_mean = mss_grouped['mss_loss']['mean'].values
        mss_std = mss_grouped['mss_loss']['std'].values
        mss_min = mss_grouped['mss_loss']['min'].values
        mss_max = mss_grouped['mss_loss']['max'].values

        ax.fill_between(window_sizes_mss, mss_min, mss_max,
                        alpha=0.2, color=COLORS['mss'], label='Min/Max Range')
        ax.errorbar(window_sizes_mss, mss_mean, yerr=mss_std,
                    fmt='o-', label='Mean ± Std', linewidth=2.5, markersize=7,
                    color=COLORS['mss'], capsize=4, capthick=1.5, alpha=0.8)

    ax.set_xlabel('Window Size (samples)', fontweight='bold')
    ax.set_ylabel('MSS Loss', fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    # Plot 2: MSE Loss (all window sizes)
    ax = axes[1]
    window_sizes_mse = mse_grouped['window_size'].values
    mse_mean = mse_grouped['mse_loss']['mean'].values
    mse_std = mse_grouped['mse_loss']['std'].values
    mse_min = mse_grouped['mse_loss']['min'].values
    mse_max = mse_grouped['mse_loss']['max'].values

    ax.fill_between(window_sizes_mse, mse_min, mse_max,
                    alpha=0.2, color=COLORS['mse'], label='Min/Max Range')
    ax.errorbar(window_sizes_mse, mse_mean, yerr=mse_std,
                fmt='o-', label='Mean ± Std', linewidth=2.5, markersize=7,
                color=COLORS['mse'], capsize=4, capthick=1.5, alpha=0.8)

    ax.set_xlabel('Window Size (samples)', fontweight='bold')
    ax.set_ylabel('MSE Loss', fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    filename = output_dir / 'reversal_mss_mse_vs_window_size.png'
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
    # PROCESS PERFORMANCE TIMING RESULTS
    # ========================================================================

    print("\n[Processing Performance Timing Results]")

    timing_file = results_dir / 'rt_timing_performance.txt'

    if timing_file.exists():
        print(f"  Found timing results file")

        timing_data = load_timing_data(timing_file)

        if timing_data is not None:
            print(f"  Total timing records: {len(timing_data)}")

            # Create performance output directory
            perf_output = output_dir / 'performance'
            perf_output.mkdir(parents=True, exist_ok=True)

            # Generate performance plots
            total_plots += plot_performance_split_alpha(timing_data, perf_output)
            total_plots += plot_rtf_split_alpha(timing_data, perf_output)
            total_plots += plot_performance_by_alpha_heatmap(timing_data, perf_output)
            total_plots += plot_performance_all_alphas_aggregated(timing_data, perf_output)
            total_plots += plot_rtf_all_alphas_aggregated(timing_data, perf_output)
    else:
        print(f"  ⚠ Timing results file not found: {timing_file}")

    # ========================================================================
    # PROCESS PASSTHROUGH AND REVERSAL RESULTS
    # ========================================================================

    print("\n[Processing Passthrough and Reversal Results]")

    # Create passthrough/reversal output directory
    passthrough_reversal_output = output_dir / 'passthrough_reversal'
    passthrough_reversal_output.mkdir(parents=True, exist_ok=True)

    # Look for the passthrough_reversal directory (contains all test types)
    passthrough_reversal_dir = results_dir / 'passthrough_reversal'
    mss_results_file = passthrough_reversal_dir / 'mss_analysis_results.txt'

    if mss_results_file.exists():
        print(f"  Found passthrough/reversal results file: {mss_results_file}")

        data = load_passthrough_reversal_results(mss_results_file)

        if data is not None:
            print(f"  Total records loaded: {len(data)}")

            # Check what test types are available
            available_types = data['test_type'].unique()
            print(f"  Available test types: {list(available_types)}")

            # Generate passthrough plot (α=0)
            if 'passthrough' in available_types:
                passthrough_count = len(data[data['test_type'] == 'passthrough'])
                print(f"  Generating passthrough plot ({passthrough_count} records)...")
                total_plots += plot_passthrough_mss_mse_vs_window_size(
                    data, passthrough_reversal_output)
            else:
                print(f"  ⚠ No passthrough data found")

            # Generate reversal plot (α=±2, aggregating fwd and bwd)
            reversal_types = [t for t in available_types if t in ['reversal_fwd', 'reversal_bwd']]
            if reversal_types:
                reversal_count = len(data[data['test_type'].isin(['reversal_fwd', 'reversal_bwd'])])
                print(f"  Generating reversal plot ({reversal_count} records from {reversal_types})...")
                total_plots += plot_reversal_mss_mse_vs_window_size(
                    data, passthrough_reversal_output)
            else:
                print(f"  ⚠ No reversal data found")
    else:
        print(f"  ⚠ Results file not found: {mss_results_file}")

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

    # Check if performance plots were generated
    perf_output = output_dir / 'performance'
    if perf_output.exists():
        print(f"  Performance plots → {perf_output}/")

    # Check if passthrough/reversal plots were generated
    passthrough_reversal_output = output_dir / 'passthrough_reversal'
    if passthrough_reversal_output.exists():
        print(f"  Passthrough/Reversal plots → {passthrough_reversal_output}/")

    print("\nAll plots saved as png files at 300 DPI\n")

    return 0


if __name__ == '__main__':
    exit(main())