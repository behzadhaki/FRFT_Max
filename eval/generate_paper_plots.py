#!/usr/bin/env python3
"""
Generate Publication-Quality Plots for AES Journal Paper
Analyzes FRFT Homomorphism and Commutativity Properties
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib.patches import Rectangle
import argparse

# ============================================================================
# PLOT SETTINGS - CUSTOMIZE FONT SIZES HERE
# ============================================================================

# Global default settings (applied to all plots unless overridden)
GLOBAL_SETTINGS = {
    'font_family': 'serif',
    'base_font_size': 20,
    'axes_labelsize': 20,
    'axes_titlesize': 20,
    'xtick_labelsize': 20,
    'ytick_labelsize': 20,
    'legend_fontsize': 14,
    'figure_dpi': 300,
}

# Homomorphism plots (heatmaps, bar plots, box plots)
HOMOMORPHISM_SETTINGS = {
    'heatmap': {
        'title_fontsize': 22,
        'xlabel_fontsize': 22,
        'ylabel_fontsize': 22,
        'tick_labelsize': 22,
        'colorbar_labelsize': 18,
    },
    'barplot': {
        'title_fontsize': 22,
        'xlabel_fontsize': 22,
        'ylabel_fontsize': 22,
        'tick_labelsize': 22,
        'legend_fontsize': 22,
    },
    'boxplot': {
        'title_fontsize': 18,
        'xlabel_fontsize': 16,
        'ylabel_fontsize': 16,
        'tick_labelsize': 13,
        'legend_fontsize': 12,
    },
}

# Commutativity plots (heatmaps, bar plots, box plots)
COMMUTATIVITY_SETTINGS = {
    'heatmap': {
        'title_fontsize': 18,
        'xlabel_fontsize': 16,
        'ylabel_fontsize': 16,
        'tick_labelsize': 15,
        'colorbar_labelsize': 12,
    },
    'barplot': {
        'title_fontsize': 18,
        'xlabel_fontsize': 16,
        'ylabel_fontsize': 16,
        'tick_labelsize': 14,
        'legend_fontsize': 12,
    },
    'boxplot': {
        'title_fontsize': 18,
        'xlabel_fontsize': 16,
        'ylabel_fontsize': 16,
        'tick_labelsize': 13,
        'legend_fontsize': 12,
    },
}

# Performance/timing plots
PERFORMANCE_SETTINGS = {
    'split_alpha': {
        'title_fontsize': 18,
        'xlabel_fontsize': 16,
        'ylabel_fontsize': 16,
        'tick_labelsize': 13,
        'legend_fontsize': 11,
    },
    'heatmap': {
        'title_fontsize': 18,
        'xlabel_fontsize': 16,
        'ylabel_fontsize': 16,
        'tick_labelsize': 13,
        'colorbar_labelsize': 12,
    },
    'aggregated': {
        'title_fontsize': 18,
        'xlabel_fontsize': 16,
        'ylabel_fontsize': 16,
        'tick_labelsize': 13,
        'legend_fontsize': 12,
    },
}

# Passthrough/Reversal plots
PASSTHROUGH_REVERSAL_SETTINGS = {
    'title_fontsize': 34,
    'xlabel_fontsize': 34,
    'ylabel_fontsize': 34,
    'tick_labelsize': 34,
    'legend_fontsize': 34,
}

# FFT Comparison plots
FFT_COMPARISON_SETTINGS = {
    'title_fontsize': 18,
    'xlabel_fontsize': 16,
    'ylabel_fontsize': 16,
    'tick_labelsize': 13,
    'legend_fontsize': 11,
}

# ============================================================================

# Set global publication-quality plot defaults
plt.rcParams['font.family'] = GLOBAL_SETTINGS['font_family']
plt.rcParams['font.size'] = GLOBAL_SETTINGS['base_font_size']
plt.rcParams['axes.labelsize'] = GLOBAL_SETTINGS['axes_labelsize']
plt.rcParams['axes.titlesize'] = GLOBAL_SETTINGS['axes_titlesize']
plt.rcParams['xtick.labelsize'] = GLOBAL_SETTINGS['xtick_labelsize']
plt.rcParams['ytick.labelsize'] = GLOBAL_SETTINGS['ytick_labelsize']
plt.rcParams['legend.fontsize'] = GLOBAL_SETTINGS['legend_fontsize']
plt.rcParams['figure.dpi'] = GLOBAL_SETTINGS['figure_dpi']
plt.rcParams['savefig.dpi'] = GLOBAL_SETTINGS['figure_dpi']
plt.rcParams['savefig.bbox'] = 'tight'

# Color scheme for consistency
COLORS = {
    'mss': '#2E86AB',  # Blue
    'mse': '#A23B72',  # Purple/Magenta
    'highlight': '#F18F01',  # Orange
    'grid': '#C73E1D',  # Red
}


def write_sample_analysis(filepath, lines):
    """
    Write a plain-text sample-analysis report alongside a plot.

    Args:
        filepath : pathlib.Path  – destination .txt file path
        lines    : list[str]     – lines to write (newline added automatically)
    """
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"    → {filepath}")


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

    # Get font size settings for this plot type
    settings = HOMOMORPHISM_SETTINGS['heatmap']

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
    axes[0].set_title('MSS (Median)', fontweight='bold', fontsize=settings['title_fontsize'], pad=15)
    axes[0].set_xlabel('α₁', fontsize=settings['xlabel_fontsize'], fontweight='bold')
    axes[0].set_ylabel('α₂', fontsize=settings['ylabel_fontsize'], fontweight='bold')

    # Set tick positions and labels
    n_ticks = 5
    tick_positions = np.linspace(0, len(mss_pivot.columns) - 1, n_ticks, dtype=int)
    tick_labels = [f"{mss_pivot.columns[i]:.1f}" for i in tick_positions]
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels, fontsize=settings['tick_labelsize'])
    axes[0].set_yticks(tick_positions)
    axes[0].set_yticklabels([f"{mss_pivot.index[i]:.1f}" for i in tick_positions], fontsize=settings['tick_labelsize'])

    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=settings['colorbar_labelsize'])

    # MSE heatmap
    im2 = axes[1].imshow(mse_pivot.values, cmap='inferno', aspect='auto',
                         vmin=0, vmax=np.nanmax(mse_pivot.values))
    axes[1].set_title('MSE (Median)', fontweight='bold', fontsize=settings['title_fontsize'], pad=15)
    axes[1].set_xlabel('α₁', fontsize=settings['xlabel_fontsize'], fontweight='bold')
    axes[1].set_ylabel('α₂', fontsize=settings['ylabel_fontsize'], fontweight='bold')
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels, fontsize=settings['tick_labelsize'])
    axes[1].set_yticks(tick_positions)
    axes[1].set_yticklabels([f"{mse_pivot.index[i]:.1f}" for i in tick_positions], fontsize=settings['tick_labelsize'])

    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=settings['colorbar_labelsize'])

    plt.tight_layout()

    filename = output_dir / 'heatmap_combined_all.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    total_records     = len(data)
    n_alpha_pairs     = len(agg_data)
    grid_cells        = len(full_alpha_range) ** 2          # 21 × 21
    filled_cells      = int(mss_pivot.notna().values.sum()) # same for MSS and MSE
    records_per_pair  = total_records / n_alpha_pairs if n_alpha_pairs > 0 else float('nan')
    n_windows         = data['Window'].nunique()    if 'Window'    in data.columns else 'N/A'
    n_frequencies     = data['Frequency'].nunique() if 'Frequency' in data.columns else 'N/A'

    txt_lines = [
        "=" * 70,
        "SAMPLE ANALYSIS: Homomorphism Combined Heatmap",
        "File: heatmap_combined_all.png",
        "=" * 70,
        "",
        "INPUT DATA",
        f"  Total records (rows) in homo_df            : {total_records}",
        f"  Unique window sizes                        : {n_windows}",
        f"  Unique frequencies                         : {n_frequencies}",
        "",
        "AGGREGATION STEP",
        "  Data is grouped by (Alpha1, Alpha2) and the MEDIAN of",
        "  MSS_Homomorphic and MSE_Homomorphic is computed per cell.",
        f"  Unique (Alpha1, Alpha2) pairs found        : {n_alpha_pairs}",
        f"  Avg records contributing to each cell      : {records_per_pair:.1f}",
        "  (each record contributes to exactly one cell)",
        "",
        "HEATMAP GRID",
        f"  Alpha range                                : -2.0 to +2.0, step 0.2",
        f"  Grid dimensions                            : 21 x 21 = {grid_cells} cells",
        f"  Cells with data (non-NaN)                  : {filled_cells}",
        f"  Empty cells (NaN / missing alpha pairs)    : {grid_cells - filled_cells}",
        "",
        "CALCULATION SUMMARY",
        f"  {total_records} rows  →  grouped into {n_alpha_pairs} (Alpha1, Alpha2) pairs",
        f"  →  each pair yields 1 median MSS value and 1 median MSE value",
        f"  →  pivoted onto a 21×21 grid ({grid_cells} cells, {filled_cells} filled)",
        "",
        "METRICS DISPLAYED",
        "  MSS_Homomorphic (median per alpha pair) — left heatmap",
        "  MSE_Homomorphic (median per alpha pair) — right heatmap",
        ]
    write_sample_analysis(output_dir / 'heatmap_combined_all_sample_analysis.txt', txt_lines)

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
    axes[0].set_xlabel('α₁ + α₂ (wrapped)', fontsize=22, fontweight='bold')
    axes[0].set_ylabel('MSS', fontsize=22, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].tick_params(labelsize=22)
    axes[0].set_ylim(bottom=0)

    # MSE barplot - narrower bars
    axes[1].bar(agg_data['AlphaSum'], agg_data['MSE_median'],
                yerr=agg_data['MSE_std'], capsize=3, width=0.15,
                color=COLORS['mse'], alpha=0.7, edgecolor='black', linewidth=0.8)
    axes[1].set_xlabel('α₁ + α₂ (wrapped)', fontsize=22, fontweight='bold')
    axes[1].set_ylabel('MSE', fontsize=22, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].tick_params(labelsize=22)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()

    filename = output_dir / 'barplot_vs_alpha_sum_all.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    total_records  = len(data)
    n_alpha_sums   = len(agg_data)
    counts_by_sum  = data.groupby('AlphaSum_rounded').size().to_dict()
    detail_lines   = [f"    AlphaSum={k:.1f}  →  {v:4d} samples" for k, v in sorted(counts_by_sum.items())]

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Homomorphism Bar Plot vs Alpha Sum",
                    "File: barplot_vs_alpha_sum_all.png",
                    "=" * 70,
                    "",
                    "INPUT DATA",
                    f"  Total records in homo_df                   : {total_records}",
                    "",
                    "AGGREGATION",
                    "  AlphaSum = wrap(α₁+α₂) rounded to 1 decimal place.",
                    "  Grouped by AlphaSum_rounded → median and std of MSS_Homomorphic",
                    "  and MSE_Homomorphic. No window or frequency filtering applied.",
                    f"  Unique AlphaSum groups                     : {n_alpha_sums}",
                    "",
                    "SAMPLES PER BAR  (each bar = median over all rows with that AlphaSum)",
                    ] + detail_lines + [
                    "",
                    "METRICS DISPLAYED",
                    "  Left  : MSS_Homomorphic (median ± std per AlphaSum)",
                    "  Right : MSE_Homomorphic (median ± std per AlphaSum)",
                ]
    write_sample_analysis(output_dir / 'barplot_vs_alpha_sum_all_sample_analysis.txt', txt_lines)

    return 1


def plot_homomorphism_boxplot_by_window(data, output_dir):
    """
    Plot 3: MSS and MSE vs alpha_sum, grouped by window size
    Boxplots for each window size side-by-side for each alpha_sum
    Only showing 0 to 2 due to symmetric nature
    Only showing window sizes >= 64
    """
    print("  Generating homomorphism boxplot grouped by window...")

    data = compute_alpha_sum(data)

    # Round alpha sum for grouping
    data['AlphaSum_rounded'] = data['AlphaSum'].round(1)

    # Filter to only 0 to 2 range (symmetric nature)
    data = data[(data['AlphaSum_rounded'] >= 0) & (data['AlphaSum_rounded'] <= 2.0)]

    # Filter to only window sizes >= 64
    data = data[data['Window'] >= 64]

    window_sizes = sorted(data['Window'].unique())[::2]
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
        ax.set_xlabel('α₁ + α₂ (wrapped)', fontsize=22, fontweight='bold')
        ax.set_title(f'{metric_name}', fontweight='bold', pad=15, fontsize=22)
        ax.set_xticks(positions_base)
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_sums], rotation=0, fontsize=22)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='y', labelsize=22)
        ax.set_ylim(bottom=0, top= ax.get_ylim()[1]*1.2)
        ax.set_xlim(-0.5, len(alpha_sums) - 0.5)

        # Create legend
        legend_handles = [plt.Rectangle((0,0),1,1, facecolor=cmap[i], alpha=0.7, edgecolor='black')
                          for i in range(len(window_sizes))]
        ax.legend(legend_handles, [f'{w}' for w in window_sizes],
                  title='Window Size', title_fontsize=16,
                  loc='upper right', ncol=len(window_sizes), fontsize=22, framealpha=0.5)

    plt.tight_layout()

    filename = output_dir / 'boxplot_by_window.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    all_windows_ge64  = sorted(data['Window'].unique())   # already filtered >= 64
    n_frequencies     = data['Frequency'].nunique() if 'Frequency' in data.columns else 0
    total_filtered    = len(data)

    # For the math breakdown: count how many (Alpha1,Alpha2) pairs map to each AlphaSum
    alpha_pair_counts = {}
    for alpha_sum in alpha_sums:
        pairs = data[abs(data['AlphaSum_rounded'] - alpha_sum) < 0.05]
        if 'Alpha1' in data.columns and 'Alpha2' in data.columns:
            alpha_pair_counts[alpha_sum] = pairs.groupby(['Alpha1','Alpha2']).ngroups
        else:
            alpha_pair_counts[alpha_sum] = 'N/A'

    detail_lines = []
    for alpha_sum in alpha_sums:
        n_pairs = alpha_pair_counts[alpha_sum]
        detail_lines.append(f"  AlphaSum = {alpha_sum:.1f}  "
                            f"[{n_pairs} (α₁,α₂) pairs × {n_frequencies} frequencies]")
        alpha_total = 0
        for window in window_sizes:
            subset = data[(data['Window'] == window) &
                          (abs(data['AlphaSum_rounded'] - alpha_sum) < 0.05)]
            n = len(subset)
            alpha_total += n
            n_pairs_w = subset.groupby(['Alpha1','Alpha2']).ngroups if 'Alpha1' in data.columns else '?'
            n_freqs_w = subset['Frequency'].nunique() if 'Frequency' in data.columns else '?'
            detail_lines.append(
                f"    BOX  Window={window:6d} samples, AlphaSum={alpha_sum:.1f}"
                f"  →  N = {n:4d}  "
                f"({n_pairs_w} (α₁,α₂) pairs × {n_freqs_w} frequencies)"
            )
        detail_lines.append(f"    TOTAL across all windows for AlphaSum={alpha_sum:.1f}  →  {alpha_total:4d} samples")
        detail_lines.append("")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Homomorphism Boxplot by Window Size",
                    "File: boxplot_by_window.png",
                    "=" * 70,
                    "",
                    "FILTERS APPLIED",
                    "  1. AlphaSum (wrapped α₁+α₂) restricted to [0.0, 2.0]  (symmetric half)",
                    "  2. Window size restricted to >= 64",
                    "  3. Window sizes further sub-sampled with [::2] (every other value)",
                    "",
                    "DATA AFTER FILTERING",
                    f"  Total records remaining                    : {total_filtered}",
                    f"  Window sizes available (>= 64, all)        : {all_windows_ge64}",
                    f"  Window sizes actually plotted ([::2])       : {window_sizes}",
                    f"  AlphaSum values plotted                    : {[round(a,1) for a in alpha_sums]}",
                    f"  Unique frequencies in dataset              : {n_frequencies} frequencies",
                    "",
                    "HOW N IS CALCULATED PER BOX",
                    "  Each row in the dataset is one measurement for a specific",
                    "  (α₁, α₂, Window, Frequency) combination.",
                    "",
                    "  For a box at (Window = W, AlphaSum = s):",
                    "    N = #{(α₁,α₂) pairs with wrap(α₁+α₂) ≈ s}  ×  #{frequencies}",
                    "      = [α-pair count]  ×  [frequency count]",
                    "  (window size is fixed for the box, so it does not multiply in)",
                    "",
                    "SAMPLE COUNT PER BOX  (N = [α-pair count] × [frequency count])",
                    "",
                    ] + detail_lines + [
                    "CALCULATION SUMMARY",
                    f"  {total_filtered} filtered rows  →  split into",
                    f"  {len(window_sizes)} windows × {len(alpha_sums)} AlphaSum groups",
                    f"  = up to {len(window_sizes) * len(alpha_sums)} boxes (fewer if any combo has 0 samples)",
                    "  Each box shows IQR + whiskers (outliers hidden with showfliers=False).",
                    "",
                    "METRICS DISPLAYED",
                    "  Top subplot   : MSS_Homomorphic",
                    "  Bottom subplot: MSE_Homomorphic",
                ]
    write_sample_analysis(output_dir / 'boxplot_by_window_sample_analysis.txt', txt_lines)

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
        # ax.set_ylabel(metric_name, fontsize=16)
        ax.set_xlabel('α₁ + α₂ (wrapped)', fontsize=22, fontweight='bold')
        ax.set_title(metric_name, fontweight='bold', pad=15, fontsize=22)
        ax.set_xticks(positions_base)
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_sums], rotation=0, fontsize=22)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='y', labelsize=22)
        ax.set_ylim(bottom=0, top= ax.get_ylim()[1]*1.2)
        ax.set_xlim(-0.5, len(alpha_sums) - 0.5)

        # Create legend
        legend_handles = [plt.Rectangle((0,0),1,1, facecolor=cmap[i], alpha=0.7, edgecolor='black')
                          for i in range(len(frequencies))]
        ax.legend(legend_handles, [f'{int(f)} Hz' for f in frequencies],
                  title='Frequency', title_fontsize=15,
                  loc='upper right', ncol=min(7, len(frequencies)), fontsize=16, framealpha=0.5)

    plt.tight_layout()

    filename = output_dir / 'boxplot_by_frequency.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    total_filtered = len(data)
    n_windows      = data['Window'].nunique() if 'Window' in data.columns else 0

    # For the math breakdown: count how many (Alpha1,Alpha2) pairs map to each AlphaSum
    alpha_pair_counts = {}
    for alpha_sum in alpha_sums:
        pairs = data[abs(data['AlphaSum_rounded'] - alpha_sum) < 0.05]
        if 'Alpha1' in data.columns and 'Alpha2' in data.columns:
            alpha_pair_counts[alpha_sum] = pairs.groupby(['Alpha1','Alpha2']).ngroups
        else:
            alpha_pair_counts[alpha_sum] = 'N/A'

    detail_lines = []
    for alpha_sum in alpha_sums:
        n_pairs = alpha_pair_counts[alpha_sum]
        detail_lines.append(f"  AlphaSum = {alpha_sum:.1f}  "
                            f"[{n_pairs} (α₁,α₂) pairs × {n_windows} windows]")
        alpha_total = 0
        for freq in frequencies:
            subset = data[(data['Frequency'] == freq) &
                          (abs(data['AlphaSum_rounded'] - alpha_sum) < 0.05)]
            n = len(subset)
            alpha_total += n
            n_pairs_f = subset.groupby(['Alpha1','Alpha2']).ngroups if 'Alpha1' in data.columns else '?'
            n_wins_f  = subset['Window'].nunique() if 'Window' in data.columns else '?'
            detail_lines.append(
                f"    BOX  Freq={int(freq):6d} Hz, AlphaSum={alpha_sum:.1f}"
                f"  →  N = {n:4d}  "
                f"({n_pairs_f} (α₁,α₂) pairs × {n_wins_f} windows)"
            )
        detail_lines.append(f"    TOTAL across all frequencies for AlphaSum={alpha_sum:.1f}  →  {alpha_total:4d} samples")
        detail_lines.append("")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Homomorphism Boxplot by Frequency",
                    "File: boxplot_by_frequency.png",
                    "=" * 70,
                    "",
                    "FILTERS APPLIED",
                    "  1. AlphaSum (wrapped α₁+α₂) restricted to [0.0, 2.0]  (symmetric half)",
                    "  (No window size filter — all window sizes included)",
                    "",
                    "DATA AFTER FILTERING",
                    f"  Total records remaining                    : {total_filtered}",
                    f"  Unique frequencies plotted                 : {len(frequencies)}  →  {[int(f) for f in frequencies]} Hz",
                    f"  AlphaSum values plotted                    : {[round(a,1) for a in alpha_sums]}",
                    f"  Unique window sizes in dataset             : {n_windows} windows",
                    "",
                    "HOW N IS CALCULATED PER BOX",
                    "  Each row in the dataset is one measurement for a specific",
                    "  (α₁, α₂, Window, Frequency) combination.",
                    "",
                    "  For a box at (Frequency = f, AlphaSum = s):",
                    "    N = #{(α₁,α₂) pairs with wrap(α₁+α₂) ≈ s}  ×  #{windows}",
                    "      = [α-pair count]  ×  [window count]",
                    "  (frequency is fixed for the box, so it does not multiply in)",
                    "",
                    "SAMPLE COUNT PER BOX  (N = [α-pair count] × [window count])",
                    "",
                    ] + detail_lines + [
                    "CALCULATION SUMMARY",
                    f"  {total_filtered} filtered rows  →  split into",
                    f"  {len(frequencies)} frequencies × {len(alpha_sums)} AlphaSum groups",
                    f"  = up to {len(frequencies) * len(alpha_sums)} boxes (fewer if any combo has 0 samples)",
                    "  Each box shows IQR + whiskers (outliers hidden with showfliers=False).",
                    "",
                    "METRICS DISPLAYED",
                    "  Top subplot   : MSS_Homomorphic",
                    "  Bottom subplot: MSE_Homomorphic",
                ]
    write_sample_analysis(output_dir / 'boxplot_by_frequency_sample_analysis.txt', txt_lines)

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
    axes[0].set_title('MSS (Median)', fontweight='bold', pad=22)
    axes[0].set_xlabel('α₁', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('α₂', fontsize=16, fontweight='bold')

    # Set tick positions and labels
    n_ticks = 5
    tick_positions = np.linspace(0, len(mss_pivot.columns) - 1, n_ticks, dtype=int)
    tick_labels = [f"{mss_pivot.columns[i]:.1f}" for i in tick_positions]
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels, fontsize=22)
    axes[0].set_yticks(tick_positions)
    axes[0].set_yticklabels([f"{mss_pivot.index[i]:.1f}" for i in tick_positions], fontsize=22)

    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=12)

    # MSE heatmap
    im2 = axes[1].imshow(mse_masked, cmap='inferno', aspect='auto',
                         vmin=0, vmax=mse_vmax)
    axes[1].set_title('MSE (Median)', fontweight='bold', pad=15)
    axes[1].set_xlabel('α₁', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('α₂', fontsize=16, fontweight='bold')
    axes[1].set_xticks(tick_positions)
    axes[1].set_xticklabels(tick_labels, fontsize=22)
    axes[1].set_yticks(tick_positions)
    axes[1].set_yticklabels([f"{mse_pivot.index[i]:.1f}" for i in tick_positions], fontsize=22)

    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=12)

    plt.tight_layout()

    filename = output_dir / 'heatmap_combined_all.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    total_records    = len(data)
    n_alpha_pairs    = len(agg_data)
    grid_cells       = len(full_alpha_range) ** 2
    filled_cells_mss = int(mss_pivot.notna().values.sum())
    filled_cells_mse = int(mse_pivot.notna().values.sum())
    records_per_pair = total_records / n_alpha_pairs if n_alpha_pairs > 0 else float('nan')
    n_windows        = data['Window'].nunique()    if 'Window'    in data.columns else 'N/A'
    n_frequencies    = data['Frequency'].nunique() if 'Frequency' in data.columns else 'N/A'

    txt_lines = [
        "=" * 70,
        "SAMPLE ANALYSIS: Commutativity Combined Heatmap",
        "File: heatmap_combined_all.png",
        "=" * 70,
        "",
        "INPUT DATA",
        f"  Total records (rows) in comm_df            : {total_records}",
        f"  Unique window sizes                        : {n_windows}",
        f"  Unique frequencies                         : {n_frequencies}",
        "",
        "AGGREGATION STEP",
        "  Data is grouped by (Alpha1, Alpha2) and the MEDIAN of",
        "  MSS_Commutativity and MSE_Commutativity is computed per pair.",
        "  A second pivot_table with aggfunc='mean' is then applied",
        "  (effectively a mean of medians if duplicate pairs exist).",
        f"  Unique (Alpha1, Alpha2) pairs found        : {n_alpha_pairs}",
        f"  Avg records contributing to each cell      : {records_per_pair:.1f}",
        "  Note: Unlike the homomorphism heatmap, missing cells are shown",
        "        as NaN (white) — no symmetric filling is applied.",
        "",
        "HEATMAP GRID",
        f"  Alpha range                                : -2.0 to +2.0, step 0.2",
        f"  Grid dimensions                            : 21 x 21 = {grid_cells} cells",
        f"  MSS cells with data (non-NaN)              : {filled_cells_mss}",
        f"  MSS empty cells (NaN / missing)            : {grid_cells - filled_cells_mss}",
        f"  MSE cells with data (non-NaN)              : {filled_cells_mse}",
        f"  MSE empty cells (NaN / missing)            : {grid_cells - filled_cells_mse}",
        "",
        "CALCULATION SUMMARY",
        f"  {total_records} rows  →  grouped into {n_alpha_pairs} (Alpha1, Alpha2) pairs",
        f"  →  each pair yields 1 median MSS value and 1 median MSE value",
        f"  →  pivoted onto a 21×21 grid ({grid_cells} cells)",
        f"     MSS: {filled_cells_mss} filled / {grid_cells - filled_cells_mss} empty",
        f"     MSE: {filled_cells_mse} filled / {grid_cells - filled_cells_mse} empty",
        "",
        "METRICS DISPLAYED",
        "  MSS_Commutativity (median per alpha pair) — left heatmap",
        "  MSE_Commutativity (median per alpha pair) — right heatmap",
        ]
    write_sample_analysis(output_dir / 'heatmap_combined_all_sample_analysis.txt', txt_lines)

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

    # ---- sample analysis report ----
    total_records  = len(data)
    n_alpha_sums   = len(agg_data)
    counts_by_sum  = data.groupby('AlphaSum_rounded').size().to_dict()
    detail_lines   = [f"    AlphaSum={k:.1f}  →  {v:4d} samples" for k, v in sorted(counts_by_sum.items())]

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Commutativity Bar Plot vs Alpha Sum",
                    "File: barplot_vs_alpha_sum_all.png",
                    "=" * 70,
                    "",
                    "INPUT DATA",
                    f"  Total records in comm_df                   : {total_records}",
                    "",
                    "AGGREGATION",
                    "  AlphaSum = wrap(α₁+α₂) rounded to 1 decimal place.",
                    "  Grouped by AlphaSum_rounded → median and std of MSS_Commutativity",
                    "  and MSE_Commutativity. No window or frequency filtering applied.",
                    f"  Unique AlphaSum groups                     : {n_alpha_sums}",
                    "",
                    "SAMPLES PER BAR  (each bar = median over all rows with that AlphaSum)",
                    ] + detail_lines + [
                    "",
                    "METRICS DISPLAYED",
                    "  Left  : MSS_Commutativity (median ± std per AlphaSum)",
                    "  Right : MSE_Commutativity (median ± std per AlphaSum)",
                ]
    write_sample_analysis(output_dir / 'barplot_vs_alpha_sum_all_sample_analysis.txt', txt_lines)

    return 1


def plot_commutativity_boxplot_by_window(data, output_dir):
    """
    Plot 3: MSS and MSE commutativity vs alpha_sum, grouped by window size
    Only showing 0 to 2 due to symmetric nature
    Only showing window sizes >= 64
    """
    print("  Generating commutativity boxplot grouped by window...")

    data = compute_alpha_sum(data)
    data['AlphaSum_rounded'] = data['AlphaSum'].round(1)

    # Filter to only 0 to 2 range (symmetric nature)
    data = data[(data['AlphaSum_rounded'] >= 0) & (data['AlphaSum_rounded'] <= 2.0)]

    # Filter to only window sizes >= 64
    data = data[data['Window'] >= 64]

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

    # ---- sample analysis report ----
    total_filtered = len(data)
    n_frequencies  = data['Frequency'].nunique() if 'Frequency' in data.columns else 0

    # For the math breakdown: count how many (Alpha1,Alpha2) pairs map to each AlphaSum
    alpha_pair_counts = {}
    for alpha_sum in alpha_sums:
        pairs = data[abs(data['AlphaSum_rounded'] - alpha_sum) < 0.05]
        if 'Alpha1' in data.columns and 'Alpha2' in data.columns:
            alpha_pair_counts[alpha_sum] = pairs.groupby(['Alpha1','Alpha2']).ngroups
        else:
            alpha_pair_counts[alpha_sum] = 'N/A'

    detail_lines = []
    for alpha_sum in alpha_sums:
        n_pairs = alpha_pair_counts[alpha_sum]
        detail_lines.append(f"  AlphaSum = {alpha_sum:.1f}  "
                            f"[{n_pairs} (α₁,α₂) pairs × {n_frequencies} frequencies]")
        alpha_total = 0
        for window in window_sizes:
            subset = data[(data['Window'] == window) &
                          (abs(data['AlphaSum_rounded'] - alpha_sum) < 0.05)]
            n = len(subset)
            alpha_total += n
            n_pairs_w = subset.groupby(['Alpha1','Alpha2']).ngroups if 'Alpha1' in data.columns else '?'
            n_freqs_w = subset['Frequency'].nunique() if 'Frequency' in data.columns else '?'
            detail_lines.append(
                f"    BOX  Window={window:6d} samples, AlphaSum={alpha_sum:.1f}"
                f"  →  N = {n:4d}  "
                f"({n_pairs_w} (α₁,α₂) pairs × {n_freqs_w} frequencies)"
            )
        detail_lines.append(f"    TOTAL across all windows for AlphaSum={alpha_sum:.1f}  →  {alpha_total:4d} samples")
        detail_lines.append("")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Commutativity Boxplot by Window Size",
                    "File: boxplot_by_window.png",
                    "=" * 70,
                    "",
                    "FILTERS APPLIED",
                    "  1. AlphaSum (wrapped α₁+α₂) restricted to [0.0, 2.0]  (symmetric half)",
                    "  2. Window size restricted to >= 64",
                    "  NOTE: Unlike the homomorphism equivalent, NO [::2] sub-sampling is",
                    "        applied — all window sizes >= 64 are plotted.",
                    "",
                    "DATA AFTER FILTERING",
                    f"  Total records remaining                    : {total_filtered}",
                    f"  Window sizes plotted                       : {window_sizes}",
                    f"  AlphaSum values plotted                    : {[round(a,1) for a in alpha_sums]}",
                    f"  Unique frequencies in dataset              : {n_frequencies} frequencies",
                    "",
                    "HOW N IS CALCULATED PER BOX",
                    "  Each row in the dataset is one measurement for a specific",
                    "  (α₁, α₂, Window, Frequency) combination.",
                    "",
                    "  For a box at (Window = W, AlphaSum = s):",
                    "    N = #{(α₁,α₂) pairs with wrap(α₁+α₂) ≈ s}  ×  #{frequencies}",
                    "      = [α-pair count]  ×  [frequency count]",
                    "  (window size is fixed for the box, so it does not multiply in)",
                    "",
                    "SAMPLE COUNT PER BOX  (N = [α-pair count] × [frequency count])",
                    "",
                    ] + detail_lines + [
                    "CALCULATION SUMMARY",
                    f"  {total_filtered} filtered rows  →  split into",
                    f"  {len(window_sizes)} windows × {len(alpha_sums)} AlphaSum groups",
                    f"  = up to {len(window_sizes) * len(alpha_sums)} boxes (fewer if any combo has 0 samples)",
                    "  Each box shows IQR + whiskers (outliers hidden with showfliers=False).",
                    "",
                    "METRICS DISPLAYED",
                    "  Top subplot   : MSS_Commutativity",
                    "  Bottom subplot: MSE_Commutativity",
                ]
    write_sample_analysis(output_dir / 'boxplot_by_window_sample_analysis.txt', txt_lines)

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

    # ---- sample analysis report ----
    total_filtered = len(data)
    n_windows      = data['Window'].nunique() if 'Window' in data.columns else 0

    # For the math breakdown: count how many (Alpha1,Alpha2) pairs map to each AlphaSum
    alpha_pair_counts = {}
    for alpha_sum in alpha_sums:
        pairs = data[abs(data['AlphaSum_rounded'] - alpha_sum) < 0.05]
        if 'Alpha1' in data.columns and 'Alpha2' in data.columns:
            alpha_pair_counts[alpha_sum] = pairs.groupby(['Alpha1','Alpha2']).ngroups
        else:
            alpha_pair_counts[alpha_sum] = 'N/A'

    detail_lines = []
    for alpha_sum in alpha_sums:
        n_pairs = alpha_pair_counts[alpha_sum]
        detail_lines.append(f"  AlphaSum = {alpha_sum:.1f}  "
                            f"[{n_pairs} (α₁,α₂) pairs × {n_windows} windows]")
        alpha_total = 0
        for freq in frequencies:
            subset = data[(data['Frequency'] == freq) &
                          (abs(data['AlphaSum_rounded'] - alpha_sum) < 0.05)]
            n = len(subset)
            alpha_total += n
            n_pairs_f = subset.groupby(['Alpha1','Alpha2']).ngroups if 'Alpha1' in data.columns else '?'
            n_wins_f  = subset['Window'].nunique() if 'Window' in data.columns else '?'
            detail_lines.append(
                f"    BOX  Freq={int(freq):6d} Hz, AlphaSum={alpha_sum:.1f}"
                f"  →  N = {n:4d}  "
                f"({n_pairs_f} (α₁,α₂) pairs × {n_wins_f} windows)"
            )
        detail_lines.append(f"    TOTAL across all frequencies for AlphaSum={alpha_sum:.1f}  →  {alpha_total:4d} samples")
        detail_lines.append("")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Commutativity Boxplot by Frequency",
                    "File: boxplot_by_frequency.png",
                    "=" * 70,
                    "",
                    "FILTERS APPLIED",
                    "  1. AlphaSum (wrapped α₁+α₂) restricted to [0.0, 2.0]  (symmetric half)",
                    "  (No window size filter — all window sizes included)",
                    "",
                    "DATA AFTER FILTERING",
                    f"  Total records remaining                    : {total_filtered}",
                    f"  Unique frequencies plotted                 : {len(frequencies)}  →  {[int(f) for f in frequencies]} Hz",
                    f"  AlphaSum values plotted                    : {[round(a,1) for a in alpha_sums]}",
                    f"  Unique window sizes in dataset             : {n_windows} windows",
                    "",
                    "HOW N IS CALCULATED PER BOX",
                    "  Each row in the dataset is one measurement for a specific",
                    "  (α₁, α₂, Window, Frequency) combination.",
                    "",
                    "  For a box at (Frequency = f, AlphaSum = s):",
                    "    N = #{(α₁,α₂) pairs with wrap(α₁+α₂) ≈ s}  ×  #{windows}",
                    "      = [α-pair count]  ×  [window count]",
                    "  (frequency is fixed for the box, so it does not multiply in)",
                    "",
                    "SAMPLE COUNT PER BOX  (N = [α-pair count] × [window count])",
                    "",
                    ] + detail_lines + [
                    "CALCULATION SUMMARY",
                    f"  {total_filtered} filtered rows  →  split into",
                    f"  {len(frequencies)} frequencies × {len(alpha_sums)} AlphaSum groups",
                    f"  = up to {len(frequencies) * len(alpha_sums)} boxes (fewer if any combo has 0 samples)",
                    "  Each box shows IQR + whiskers (outliers hidden with showfliers=False).",
                    "",
                    "METRICS DISPLAYED",
                    "  Top subplot   : MSS_Commutativity",
                    "  Bottom subplot: MSE_Commutativity",
                ]
    write_sample_analysis(output_dir / 'boxplot_by_frequency_sample_analysis.txt', txt_lines)

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

    # ---- sample analysis report ----
    total_records = len(data)
    detail_lines  = []
    for alpha in alphas:
        ad = data[data['alpha'] == alpha].sort_values('window_size')
        detail_lines.append(f"    α={alpha:.1f}  →  {len(ad):3d} records  "
                            f"window_sizes={sorted(ad['window_size'].unique())}")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Performance by Alpha (multi-subplot line chart)",
                    "File: performance_by_alpha.png",
                    "=" * 70,
                    "",
                    "FILTERS APPLIED",
                    "  window_size >= 64",
                    "",
                    "DATA AFTER FILTERING",
                    f"  Total records                              : {total_records}",
                    f"  Alpha values (one subplot each)            : {alphas}",
                    f"  Window sizes                               : {window_sizes}",
                    "",
                    "SAMPLES PER SUBPLOT (each point = 1 timing record for that alpha × window_size)",
                    ] + detail_lines + [
                    "",
                    "METRICS DISPLAYED",
                    "  Y-axis: mean_time_ms (measured inference time in ms, log scale)",
                    "  Complexity reference lines (O(N), O(N log N), O(N²)) anchored at window_size=64.",
                ]
    write_sample_analysis(output_dir / 'performance_by_alpha_sample_analysis.txt', txt_lines)

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

    # ---- sample analysis report ----
    total_records = len(data)
    detail_lines  = []
    for alpha in alphas:
        ad = data[data['alpha'] == alpha].sort_values('window_size')
        detail_lines.append(f"    α={alpha:.1f}  →  {len(ad):3d} records  "
                            f"window_sizes={sorted(ad['window_size'].unique())}")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Real-Time Factor by Alpha (multi-subplot line chart)",
                    "File: rtf_by_alpha.png",
                    "=" * 70,
                    "",
                    "FILTERS APPLIED",
                    "  window_size >= 64",
                    "",
                    "DATA AFTER FILTERING",
                    f"  Total records                              : {total_records}",
                    f"  Alpha values (one subplot each)            : {alphas}",
                    f"  Window sizes                               : {window_sizes}",
                    "",
                    "SAMPLES PER SUBPLOT (each point = 1 timing record for that alpha × window_size)",
                    ] + detail_lines + [
                    "",
                    "METRICS DISPLAYED",
                    "  Y-axis: mean_rt_factor (Real-Time Factor = inference_time / audio_buffer_duration)",
                    "  RTF < 1.0 = faster than real-time. Reference line drawn at RTF = 1.0.",
                ]
    write_sample_analysis(output_dir / 'rtf_by_alpha_sample_analysis.txt', txt_lines)

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

    # ---- sample analysis report ----
    total_records  = len(data)
    n_alphas       = len(pivot_data.index)
    n_windows      = len(pivot_data.columns)
    grid_cells     = n_alphas * n_windows
    filled_cells   = int(pivot_data.notna().values.sum())
    records_per_cell = total_records / grid_cells if grid_cells > 0 else float('nan')

    detail_lines = []
    for alpha in sorted(data['alpha'].unique()):
        for ws in sorted(data['window_size'].unique()):
            cnt = len(data[(data['alpha'] == alpha) & (data['window_size'] == ws)])
            detail_lines.append(
                f"    alpha={alpha:.1f}, window_size={ws:6d}  →  {cnt:4d} records"
            )

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Performance Inference Time Heatmap",
                    "File: performance_heatmap.png",
                    "=" * 70,
                    "",
                    "FILTERS APPLIED",
                    "  1. window_size restricted to >= 64",
                    "",
                    "INPUT DATA AFTER FILTERING",
                    f"  Total records                              : {total_records}",
                    f"  Unique alpha values                        : {n_alphas}  →  {sorted(data['alpha'].unique())}",
                    f"  Unique window sizes                        : {n_windows}  →  {sorted(data['window_size'].unique())}",
                    "",
                    "AGGREGATION STEP",
                    "  pivot_table with aggfunc='mean' over mean_time_ms.",
                    "  Each cell = mean of all timing records with that (alpha, window_size) pair.",
                    f"  Grid dimensions                            : {n_alphas} alphas × {n_windows} windows = {grid_cells} cells",
                    f"  Cells with data (non-NaN)                  : {filled_cells}",
                    f"  Empty cells                                : {grid_cells - filled_cells}",
                    f"  Avg records per cell                       : {records_per_cell:.1f}",
                    "",
                    "RECORD COUNT PER CELL",
                    ] + detail_lines + [
                    "",
                    "METRICS DISPLAYED",
                    "  mean_time_ms (mean per alpha × window_size) — single heatmap",
                ]
    write_sample_analysis(output_dir / 'performance_heatmap_sample_analysis.txt', txt_lines)

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
    # ax.set_title('FRFT Performance (All α Aggregated)', fontweight='bold', pad=15)
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

    # ---- sample analysis report ----
    n_alphas      = data['alpha'].nunique()
    total_records = len(data)
    detail_lines  = []
    for _, row in agg_stats.iterrows():
        n = len(data[data['window_size'] == row['window_size']])
        detail_lines.append(
            f"    window_size={int(row['window_size']):6d}  n={n:3d}  "
            f"mean={row['mean']:.3e} ms  min={row['min']:.3e}  max={row['max']:.3e}  std={row['std']:.3e}")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Performance All Alphas Aggregated",
                    "File: performance_all_alphas_aggregated.png",
                    "=" * 70,
                    "",
                    "FILTERS APPLIED",
                    "  window_size >= 64",
                    "",
                    "DATA AFTER FILTERING",
                    f"  Total records                              : {total_records}",
                    f"  Alpha values pooled                        : {n_alphas}",
                    f"  Window sizes                               : {window_sizes}",
                    "",
                    "AGGREGATION",
                    "  Grouped by window_size. For each window, mean/min/max/std of",
                    "  mean_time_ms computed across all alpha values.",
                    f"  Samples per window size point              : {n_alphas} (one per alpha)",
                    "",
                    "PER-WINDOW STATISTICS  (format: window  n  mean  min  max  std  [ms])",
                    ] + detail_lines + [
                    "",
                    "METRICS DISPLAYED",
                    "  Mean line + min/max shaded band of mean_time_ms across all α.",
                    "  Right y-axis shows equivalent Max/MSP audio buffer sizes at 44.1 kHz.",
                    "  Complexity reference lines (O(N), O(N log N), O(N²)) anchored at window=64.",
                ]
    write_sample_analysis(output_dir / 'performance_all_alphas_aggregated_sample_analysis.txt', txt_lines)

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

    # ---- sample analysis report ----
    n_alphas      = data['alpha'].nunique()
    total_records = len(data)
    detail_lines  = []
    for _, row in agg_stats.iterrows():
        n = len(data[data['window_size'] == row['window_size']])
        detail_lines.append(
            f"    window_size={int(row['window_size']):6d}  n={n:3d}  "
            f"mean_RTF={row['mean']:.4f}  min={row['min']:.4f}  max={row['max']:.4f}  std={row['std']:.4f}")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: RTF All Alphas Aggregated",
                    "File: rtf_all_alphas_aggregated.png",
                    "=" * 70,
                    "",
                    "FILTERS APPLIED",
                    "  window_size >= 64",
                    "",
                    "DATA AFTER FILTERING",
                    f"  Total records                              : {total_records}",
                    f"  Alpha values pooled                        : {n_alphas}",
                    f"  Window sizes                               : {window_sizes}",
                    "",
                    "AGGREGATION",
                    "  Grouped by window_size. For each window, mean/min/max/std of",
                    "  mean_rt_factor computed across all alpha values.",
                    f"  Samples per window size point              : {n_alphas} (one per alpha)",
                    "",
                    "PER-WINDOW STATISTICS  (format: window  n  mean_RTF  min  max  std)",
                    ] + detail_lines + [
                    "",
                    "METRICS DISPLAYED",
                    "  Mean RTF line + min/max shaded band across all α.",
                    "  Reference line at RTF=1.0 (real-time boundary).",
                    "  RTF < 1 means faster than real-time.",
                ]
    write_sample_analysis(output_dir / 'rtf_all_alphas_aggregated_sample_analysis.txt', txt_lines)

    return 1


# ============================================================================
# PASSTHROUGH AND REVERSAL PLOTS
# ============================================================================

def plot_passthrough_reversal_combined(data, output_dir):
    """
    Plot MSS and MSE vs window size for both passthrough and reversal tests.
    Publication-quality style with no title.
    2x2 grid: Top row = passthrough (α=0), Bottom row = reversal (α=±2)
    Left column = MSS, Right column = MSE
    MSS plots start from window size 8192, MSE plots start from 64.
    """
    print("  Generating combined passthrough/reversal MSS/MSE plot...")

    # Prepare passthrough data
    passthrough_data = data[data['test_type'] == 'passthrough'].copy()

    if len(passthrough_data) == 0:
        print(f"  No passthrough data found")
        return 0

    # Prepare reversal data (aggregate both directions)
    reversal_data = data[data['test_type'].isin(['reversal_fwd', 'reversal_bwd'])].copy()

    if len(reversal_data) == 0:
        print(f"  No reversal data found")
        return 0

    # Create 2x2 subplot grid with reduced height (5 instead of 10)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ========================================================================
    # TOP ROW: PASSTHROUGH (α=0)
    # ========================================================================

    # Group passthrough data by window size
    passthrough_grouped = passthrough_data.groupby('window_size').agg({
        'mss_loss': ['mean', 'std'],
        'mse_loss': ['mean', 'std']
    }).reset_index()

    # MSS: Filter to window sizes >= 8192
    passthrough_mss_grouped = passthrough_grouped[passthrough_grouped['window_size'] >= 8192].copy()
    # MSE: Filter to window sizes >= 64
    passthrough_mse_grouped = passthrough_grouped[passthrough_grouped['window_size'] >= 64].copy()

    # TOP LEFT: Passthrough MSS
    ax = axes[0, 0]
    if len(passthrough_mss_grouped) > 0:
        window_sizes_mss = passthrough_mss_grouped['window_size'].values
        mss_mean = passthrough_mss_grouped['mss_loss']['mean'].values
        mss_std = passthrough_mss_grouped['mss_loss']['std'].values

        ax.errorbar(window_sizes_mss, mss_mean, yerr=mss_std,
                    fmt='o-', label='Mean ± Std', linewidth=2.5, markersize=7,
                    color=COLORS['mss'], capsize=4, capthick=1.5, alpha=0.8)

    ax.set_ylabel('MSS (α=0)', fontweight='bold', fontsize=28)
    ax.set_xscale('log')
    ax.set_xticks(window_sizes_mss)
    ax.set_xticklabels([f'$2^{{{int(np.log2(w))}}}$' for w in window_sizes_mss], rotation=0, ha='center', fontsize=30)
    # x-axis label omitted here; shown only on the α=±2 row below (same window sizes)
    ax.grid(False)
    ax.xaxis.set_minor_locator(mticker.NullLocator())  # tick marks only at 2^k positions
    ax.legend(loc='best', framealpha=0.9, fontsize=28)  # 2x the default 14pt legend

    # TOP RIGHT: Passthrough MSE
    ax = axes[0, 1]
    window_sizes_mse = passthrough_mse_grouped['window_size'].values
    mse_mean = passthrough_mse_grouped['mse_loss']['mean'].values
    mse_std = passthrough_mse_grouped['mse_loss']['std'].values
    # mse_min = passthrough_mse_grouped['mse_loss']['min'].values
    # mse_max = passthrough_mse_grouped['mse_loss']['max'].values

    ax.errorbar(window_sizes_mse, mse_mean, yerr=mse_std,
                fmt='o-', label='Mean ± Std', linewidth=2.5, markersize=7,
                color=COLORS['mse'], capsize=4, capthick=1.5, alpha=0.8)

    ax.set_ylabel('MSE (α=0)', fontweight='bold', fontsize=28)
    ax.set_xscale('log')
    ax.set_xticks(window_sizes_mse)
    ax.set_xticklabels([f'$2^{{{int(np.log2(w))}}}$' if i % 2 == 0 else ''
                        for i, w in enumerate(window_sizes_mse)], rotation=0, ha='center', fontsize=30)
    # x-axis label omitted here; shown only on the α=±2 row below (same window sizes)
    ax.grid(False)
    ax.xaxis.set_minor_locator(mticker.NullLocator())  # tick marks only at 2^k positions
    ax.legend(loc='best', framealpha=0.9, fontsize=28)  # 2x the default 14pt legend

    # ========================================================================
    # BOTTOM ROW: REVERSAL (α=±2)
    # ========================================================================

    # Group reversal data by window size
    reversal_grouped = reversal_data.groupby('window_size').agg({
        'mss_loss': ['mean', 'std'],
        'mse_loss': ['mean', 'std']
    }).reset_index()

    # MSS: Filter to window sizes >= 8192
    reversal_mss_grouped = reversal_grouped[reversal_grouped['window_size'] >= 8192].copy()
    # MSE: Filter to window sizes >= 64
    reversal_mse_grouped = reversal_grouped[reversal_grouped['window_size'] >= 64].copy()

    # BOTTOM LEFT: Reversal MSS
    ax = axes[1, 0]
    if len(reversal_mss_grouped) > 0:
        window_sizes_mss = reversal_mss_grouped['window_size'].values
        mss_mean = reversal_mss_grouped['mss_loss']['mean'].values
        mss_std = reversal_mss_grouped['mss_loss']['std'].values

        ax.errorbar(window_sizes_mss, mss_mean, yerr=mss_std,
                    fmt='o-', label='Mean ± Std', linewidth=2.5, markersize=7,
                    color=COLORS['mss'], capsize=4, capthick=1.5, alpha=0.8)

    ax.set_xlabel('Window Size (samples)', fontweight='bold', fontsize=28)
    ax.set_ylabel('MSS (α=±2)', fontweight='bold', fontsize=28)
    ax.set_xscale('log')
    ax.set_xticks(window_sizes_mss)
    ax.set_xticklabels([f'$2^{{{int(np.log2(w))}}}$' for w in window_sizes_mss], rotation=0, ha='center', fontsize=30)
    ax.grid(False)
    ax.xaxis.set_minor_locator(mticker.NullLocator())  # tick marks only at 2^k positions
    ax.legend(loc='best', framealpha=0.9, fontsize=28)  # 2x the default 14pt legend

    # BOTTOM RIGHT: Reversal MSE
    ax = axes[1, 1]
    window_sizes_mse = reversal_mse_grouped['window_size'].values
    mse_mean = reversal_mse_grouped['mse_loss']['mean'].values
    mse_std = reversal_mse_grouped['mse_loss']['std'].values
    # mse_min = reversal_mse_grouped['mse_loss']['min'].values
    # mse_max = reversal_mse_grouped['mse_loss']['max'].values
    #
    # ax.fill_between(window_sizes_mse, mse_min, mse_max,
    #                 alpha=0.2, color=COLORS['mse'], label='Min/Max Range')
    ax.errorbar(window_sizes_mse, mse_mean, yerr=mse_std,
                fmt='o-', label='Mean ± Std', linewidth=2.5, markersize=7,
                color=COLORS['mse'], capsize=4, capthick=1.5, alpha=0.8)

    ax.set_xlabel('Window Size (samples)', fontweight='bold', fontsize=28)
    ax.set_ylabel('MSE (α=±2)', fontweight='bold', fontsize=28)
    ax.set_xscale('log')
    ax.set_xticks(window_sizes_mse)
    ax.set_xticklabels([f'$2^{{{int(np.log2(w))}}}$' if i % 2 == 0 else ''
                        for i, w in enumerate(window_sizes_mse)], rotation=0, ha='center', fontsize=30)
    ax.grid(False)
    ax.xaxis.set_minor_locator(mticker.NullLocator())  # tick marks only at 2^k positions
    ax.legend(loc='best', framealpha=0.9, fontsize=28)  # 2x the default 14pt legend

    # Right column (MSE): compact y tick labels with a single shared ×10⁻³ factor,
    # and a common y-scale across both MSE panels so they are directly comparable.
    class _Sci3(mticker.ScalarFormatter):
        def _set_order_of_magnitude(self):
            self.orderOfMagnitude = -3  # force the shared factor to ×10⁻³
    mse_axes = [axes[0, 1], axes[1, 1]]
    ylo = min(a.get_ylim()[0] for a in mse_axes)
    yhi = max(a.get_ylim()[1] for a in mse_axes)
    for a in mse_axes:
        a.set_ylim(ylo, yhi)
        a.set_yticks([0.0, 0.005, 0.010])  # show only 0, 5, 10 (×10⁻³)
        fmt = _Sci3(useMathText=True)
        fmt.set_scientific(True)
        a.yaxis.set_major_formatter(fmt)

    # Match y tick-label size to the x tick labels (30 pt)
    for a in axes.flat:
        a.tick_params(axis='y', labelsize=30)
    for a in mse_axes:
        a.yaxis.get_offset_text().set_fontsize(30)  # the ×10⁻³ factor

    plt.tight_layout()
    filename = output_dir / 'passthrough_reversal_combined.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    jpeg_filename = filename.with_suffix('.jpeg')
    plt.savefig(jpeg_filename, dpi=300, bbox_inches='tight')
    print(f"    → {jpeg_filename}")
    plt.close()

    # ========================================================================
    # GENERATE LATEX TABLE
    # ========================================================================

    # Collect all data for table (using >= 64 for all metrics)
    passthrough_all = passthrough_grouped[passthrough_grouped['window_size'] >= 64].copy()
    reversal_grouped = reversal_data.groupby('window_size').agg({
        'mss_loss': ['mean', 'std'],
        'mse_loss': ['mean', 'std']
    }).reset_index()
    reversal_all = reversal_grouped[reversal_grouped['window_size'] >= 64].copy()

    # Determine MSS threshold (MSS only valid for window_size >= 8192)
    mss_threshold = 8192

    # Create LaTeX table (ROWS = window sizes, COLUMNS = metrics)
    tex_filename = output_dir / 'passthrough_reversal_table.tex'

    with open(tex_filename, 'w') as f:
        f.write("% Passthrough/Reversal Test Results\n")
        f.write("% Rows: Window Size (as power of 2)\n")
        f.write("% Columns: MSE (α=0), MSS (α=0), MSE (α=±2), MSS (α=±2)\n\n")

        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Passthrough and Reversal Test Results: Mean $\\pm$ Std. ")
        f.write(f"MSS requires window sizes $\\geq {mss_threshold}$ and is marked N/A where not applicable.}}\n")
        f.write("\\label{tab:passthrough_reversal}\n")
        f.write("\\small\n")

        # Get window sizes (union of both datasets)
        all_window_sizes = sorted(set(passthrough_all['window_size'].values) |
                                  set(reversal_all['window_size'].values))

        # Write table header (4 metrics as columns)
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\\hline\n")

        # Column headers (metrics)
        header = "Window Size & MSE ($\\alpha=0$) & MSS ($\\alpha=0$) & MSE ($\\alpha=\\pm 2$) & MSS ($\\alpha=\\pm 2$) \\\\\n"
        f.write(header)
        f.write("\\hline\n")

        # Data rows (one per window size)
        for ws in all_window_sizes:
            # Format window size as power of 2
            f.write(f"$2^{{{int(np.log2(ws))}}}$ & ")

            values = []

            # Column 1: MSE (α=0) - Passthrough
            match = passthrough_all[passthrough_all['window_size'] == ws]
            if len(match) > 0:
                mean = match['mse_loss']['mean'].values[0]
                std = match['mse_loss']['std'].values[0]
                if np.isnan(mean) or np.isnan(std):
                    values.append("N/A")
                else:
                    values.append(f"{mean:.2e} $\\pm$ {std:.2e}")
            else:
                values.append("---")

            # Column 2: MSS (α=0) - Passthrough
            match = passthrough_all[passthrough_all['window_size'] == ws]
            if len(match) > 0 and ws >= mss_threshold:
                mean = match['mss_loss']['mean'].values[0]
                std = match['mss_loss']['std'].values[0]
                if np.isnan(mean) or np.isnan(std):
                    values.append("N/A")
                else:
                    values.append(f"{mean:.2e} $\\pm$ {std:.2e}")
            else:
                values.append("N/A")

            # Column 3: MSE (α=±2) - Reversal
            match = reversal_all[reversal_all['window_size'] == ws]
            if len(match) > 0:
                mean = match['mse_loss']['mean'].values[0]
                std = match['mse_loss']['std'].values[0]
                if np.isnan(mean) or np.isnan(std):
                    values.append("N/A")
                else:
                    values.append(f"{mean:.2e} $\\pm$ {std:.2e}")
            else:
                values.append("---")

            # Column 4: MSS (α=±2) - Reversal
            match = reversal_all[reversal_all['window_size'] == ws]
            if len(match) > 0 and ws >= mss_threshold:
                mean = match['mss_loss']['mean'].values[0]
                std = match['mss_loss']['std'].values[0]
                if np.isnan(mean) or np.isnan(std):
                    values.append("N/A")
                else:
                    values.append(f"{mean:.2e} $\\pm$ {std:.2e}")
            else:
                values.append("N/A")

            f.write(" & ".join(values) + " \\\\\n")

        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"    → {tex_filename}")

    # ---- sample analysis report ----
    pt_count   = len(passthrough_data)
    rev_count  = len(reversal_data)
    pt_windows = sorted(passthrough_data['window_size'].unique())
    rev_windows= sorted(reversal_data['window_size'].unique())
    rev_types  = sorted(reversal_data['test_type'].unique()) if 'test_type' in reversal_data.columns else []

    txt_lines = [
        "=" * 70,
        "SAMPLE ANALYSIS: Passthrough / Reversal Combined Plot",
        "File: passthrough_reversal_combined.png",
        "=" * 70,
        "",
        "DATA SPLIT",
        f"  Passthrough records (test_type='passthrough') : {pt_count}",
        f"  Reversal records (test_type in {rev_types})   : {rev_count}",
        "",
        "PASSTHROUGH (α=0) — top row",
        f"  Window sizes in data                         : {pt_windows}",
        f"  MSS subplot (top-left)  : window_size >= 8192  →  {sorted(passthrough_data[passthrough_data['window_size']>=8192]['window_size'].unique())}",
        f"  MSE subplot (top-right) : window_size >= 64    →  {sorted(passthrough_data[passthrough_data['window_size']>=64]['window_size'].unique())}",
        "  Each point = mean ± std of mss_loss / mse_loss across all records at that window size.",
        "",
        "REVERSAL (α=±2) — bottom row",
        f"  Window sizes in data                         : {rev_windows}",
        f"  MSS subplot (bot-left)  : window_size >= 8192  →  {sorted(reversal_data[reversal_data['window_size']>=8192]['window_size'].unique())}",
        f"  MSE subplot (bot-right) : window_size >= 64    →  {sorted(reversal_data[reversal_data['window_size']>=64]['window_size'].unique())}",
        "  Each point = mean ± std aggregated over both reversal_fwd and reversal_bwd records.",
        "",
        "METRICS DISPLAYED",
        "  mss_loss (top/bottom left) — MSS (perceptual similarity score)",
        "  mse_loss (top/bottom right) — mean squared error",
        "",
        "ADDITIONAL OUTPUT",
        f"  LaTeX table also saved to: {tex_filename}",
        ]
    write_sample_analysis(output_dir / 'passthrough_reversal_combined_sample_analysis.txt', txt_lines)

    return 1


# ============================================================================
# FFT COMPARISON PLOTS
# ============================================================================

def plot_fft_comparison_errors(data, output_dir):
    """
    Plot FFT comparison errors vs frequency for magnitude, phase, and complex.
    Uses all available frequencies (continuous log-sampled from 100–10000 Hz).
    X-axis is log-scaled to match the log-uniform sampling of test frequencies.
    Aggregates across all window sizes: mean line + min/max shaded band.
    Publication-quality style with no title.
    """
    print("  Generating FFT comparison error plots...")

    if len(data) == 0:
        print(f"  No data found")
        return 0

    # Group by frequency and compute statistics across all window sizes
    grouped = data.groupby('Frequency').agg({
        'MSE_Magnitude': ['mean', 'min', 'max'],
        'MSE_Phase': ['mean', 'min', 'max'],
        'MSE_Complex': ['mean', 'min', 'max']
    }).reset_index()
    grouped = grouped.sort_values('Frequency')

    frequencies = grouped['Frequency'].values

    tick_freqs  = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    tick_labels = ['100', '1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k', '9k', '10k']

    # Create 3-subplot figure (vertical layout)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Magnitude MSE
    ax = axes[0]
    mse_mag_mean = grouped['MSE_Magnitude']['mean'].values
    mse_mag_min  = grouped['MSE_Magnitude']['min'].values
    mse_mag_max  = grouped['MSE_Magnitude']['max'].values

    ax.fill_between(frequencies, mse_mag_min, mse_mag_max,
                    alpha=0.2, color=COLORS['mss'], label='Min/Max Range')
    ax.plot(frequencies, mse_mag_mean, '-', label='Mean',
            linewidth=2, color=COLORS['mss'], alpha=0.9)

    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('MSE Magnitude', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xlim([100, 10000])
    ax.set_xticks(tick_freqs)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(ax.get_ylim())
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f'{10*np.log10(v):.0f} dB' if v > 0 else ''))
    ax2.set_ylabel('dB', fontweight='bold')

    # Plot 2: Phase MSE
    ax = axes[1]
    mse_phase_mean = grouped['MSE_Phase']['mean'].values
    mse_phase_min  = grouped['MSE_Phase']['min'].values
    mse_phase_max  = grouped['MSE_Phase']['max'].values

    ax.fill_between(frequencies, mse_phase_min, mse_phase_max,
                    alpha=0.2, color=COLORS['mse'], label='Min/Max Range')
    ax.plot(frequencies, mse_phase_mean, '-', label='Mean',
            linewidth=2, color=COLORS['mse'], alpha=0.9)

    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('MSE Phase', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xlim([100, 10000])
    ax.set_xticks(tick_freqs)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(ax.get_ylim())
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f'{10*np.log10(v):.0f} dB' if v > 0 else ''))
    ax2.set_ylabel('dB', fontweight='bold')

    # Plot 3: Complex MSE
    ax = axes[2]
    mse_complex_mean = grouped['MSE_Complex']['mean'].values
    mse_complex_min  = grouped['MSE_Complex']['min'].values
    mse_complex_max  = grouped['MSE_Complex']['max'].values

    ax.fill_between(frequencies, mse_complex_min, mse_complex_max,
                    alpha=0.2, color=COLORS['highlight'], label='Min/Max Range')
    ax.plot(frequencies, mse_complex_mean, '-', label='Mean',
            linewidth=2, color=COLORS['highlight'], alpha=0.9)

    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('MSE Complex', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xlim([100, 10000])
    ax.set_xticks(tick_freqs)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(ax.get_ylim())
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f'{10*np.log10(v):.0f} dB' if v > 0 else ''))
    ax2.set_ylabel('dB', fontweight='bold')

    plt.tight_layout()
    filename = output_dir / 'fft_comparison_errors_vs_frequency.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    total_records  = len(data)
    n_freqs        = len(frequencies)
    n_windows      = data['WindowSize'].nunique() if 'WindowSize' in data.columns else 'N/A'
    records_per_freq = total_records / n_freqs if n_freqs > 0 else float('nan')

    txt_lines = [
        "=" * 70,
        "SAMPLE ANALYSIS: FFT Comparison Errors vs Frequency (aggregated)",
        "File: fft_comparison_errors_vs_frequency.png",
        "=" * 70,
        "",
        "INPUT DATA",
        f"  Total records                              : {total_records}",
        f"  Unique frequencies                         : {n_freqs}  (range {frequencies.min():.1f} – {frequencies.max():.1f} Hz)",
        f"  Unique window sizes (pooled into band)     : {n_windows}",
        "",
        "AGGREGATION",
        "  Grouped by Frequency → mean/min/max of MSE_Magnitude, MSE_Phase,",
        "  MSE_Complex across ALL window sizes.",
        f"  Avg records per frequency point            : {records_per_freq:.1f}",
        "  (min/max band width reflects variation across window sizes)",
        "",
        "METRICS DISPLAYED  (3 subplots, each: mean line + min/max band)",
        "  Top    : MSE_Magnitude",
        "  Middle : MSE_Phase",
        "  Bottom : MSE_Complex",
        ]
    write_sample_analysis(output_dir / 'fft_comparison_errors_vs_frequency_sample_analysis.txt', txt_lines)

    return 1


def plot_fft_comparison_errors_by_window_size(data, output_dir):
    """
    Plot FFT comparison errors vs frequency for magnitude, phase, and complex,
    separated by window size.
    Uses all available frequencies (continuous log-sampled from 100–10000 Hz).
    X-axis is log-scaled to match the log-uniform sampling of test frequencies.
    Only shows window sizes >= 64.
    Publication-quality style with no title.
    """
    print("  Generating FFT comparison error plots by window size...")

    # Filter to only window sizes >= 64
    data_filtered = data[data['WindowSize'] >= 64].copy()

    if len(data_filtered) == 0:
        print(f"  No data found")
        return 0

    # Get unique window sizes and build a colour map
    window_sizes = sorted(data_filtered['WindowSize'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(window_sizes)))

    freq_min = data_filtered['Frequency'].min()
    freq_max = data_filtered['Frequency'].max()

    tick_freqs  = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    tick_labels = ['100', '1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k', '9k', '10k']

    # Create 3-subplot figure (vertical layout)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    for metric, ax, ylabel in [
        ('MSE_Magnitude', axes[0], 'MSE\nMagnitude'),
        ('MSE_Phase',     axes[1], 'MSE\nPhase'),
        ('MSE_Complex',   axes[2], 'MSE\nComplex'),
    ]:
        for idx, ws in enumerate(window_sizes):
            ws_data = data_filtered[data_filtered['WindowSize'] == ws]
            grouped = ws_data.groupby('Frequency').agg(
                {metric: 'mean'}
            ).reset_index().sort_values('Frequency')

            ax.plot(grouped['Frequency'].values, grouped[metric].values, '-',
                    label=str(ws), linewidth=1.5, color=colors[idx], alpha=0.85)

        ax.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xlim([100, 10000])
        ax.set_xticks(tick_freqs)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax2 = ax.twinx()
        ax2.set_yscale('log')
        ax2.set_ylim(ax.get_ylim())
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: f'{10*np.log10(v):.0f} dB' if v > 0 else ''))
        ax2.set_ylabel('dB', fontweight='bold')

    # Single shared legend on the bottom subplot
    axes[2].legend(loc='best', framealpha=0.9, title='Window Size', ncol=2)

    plt.tight_layout()
    filename = output_dir / 'fft_comparison_errors_by_window_size.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    total_records = len(data_filtered)
    detail_lines  = []
    for ws in window_sizes:
        n = len(data_filtered[data_filtered['WindowSize'] == ws])
        n_f = data_filtered[data_filtered['WindowSize'] == ws]['Frequency'].nunique()
        detail_lines.append(f"    WindowSize={ws:6d}  total_rows={n:5d}  unique_freqs={n_f}")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: FFT Comparison Errors by Window Size",
                    "File: fft_comparison_errors_by_window_size.png",
                    "=" * 70,
                    "",
                    "FILTERS APPLIED",
                    "  WindowSize >= 64",
                    "",
                    "DATA AFTER FILTERING",
                    f"  Total records                              : {total_records}",
                    f"  Window sizes plotted                       : {window_sizes}",
                    "",
                    "AGGREGATION",
                    "  For each (WindowSize, Frequency) group: mean of MSE_Magnitude,",
                    "  MSE_Phase, MSE_Complex plotted as a separate line per window size.",
                    "",
                    "RECORDS PER WINDOW SIZE LINE",
                    ] + detail_lines + [
                    "",
                    "METRICS DISPLAYED  (3 subplots, one line per window size)",
                    "  Top    : MSE_Magnitude",
                    "  Middle : MSE_Phase",
                    "  Bottom : MSE_Complex",
                ]
    write_sample_analysis(output_dir / 'fft_comparison_errors_by_window_size_sample_analysis.txt', txt_lines)

    return 1


def plot_fft_comparison_complex_mse_by_window_size(data, output_dir):
    """
    Plot complex MSE vs frequency separated by window size — standalone figure.
    Legend placed outside the plot area to the right.
    Only shows window sizes >= 64.
    Publication-quality style with no title.
    """
    print("  Generating FFT comparison complex MSE plot (standalone)...")

    data_filtered = data[data['WindowSize'] >= 64].copy()

    if len(data_filtered) == 0:
        print(f"  No data found")
        return 0

    window_sizes = sorted(data_filtered['WindowSize'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(window_sizes)))

    tick_freqs  = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    tick_labels = ['100', '1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k', '9k', '10k']

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    for idx, ws in enumerate(window_sizes):
        ws_data = data_filtered[data_filtered['WindowSize'] == ws]
        grouped = ws_data.groupby('Frequency').agg(
            {'MSE_Complex': 'mean'}
        ).reset_index().sort_values('Frequency')

        ax.plot(grouped['Frequency'].values, grouped['MSE_Complex'].values, '-',
                label=str(ws), linewidth=1.5, color=colors[idx], alpha=0.85)

    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('MSE Complex', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xlim([100, 10000])
    ax.set_xticks(tick_freqs)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    # Legend outside to the right
    ax.legend(title='Window Size', loc='upper left',
              bbox_to_anchor=(1.03, 1), borderaxespad=0,
              framealpha=0.9, ncol=1)

    plt.tight_layout()
    filename = output_dir / 'fft_comparison_complex_mse_by_window_size.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    total_records = len(data_filtered)
    detail_lines  = []
    for ws in window_sizes:
        n = len(data_filtered[data_filtered['WindowSize'] == ws])
        n_f = data_filtered[data_filtered['WindowSize'] == ws]['Frequency'].nunique()
        detail_lines.append(f"    WindowSize={ws:6d}  total_rows={n:5d}  unique_freqs={n_f}")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: FFT Comparison Complex MSE by Window Size (standalone)",
                    "File: fft_comparison_complex_mse_by_window_size.png",
                    "=" * 70,
                    "",
                    "FILTERS APPLIED",
                    "  WindowSize >= 64",
                    "",
                    "DATA AFTER FILTERING",
                    f"  Total records                              : {total_records}",
                    f"  Window sizes plotted                       : {window_sizes}",
                    "",
                    "AGGREGATION",
                    "  For each (WindowSize, Frequency) group: mean of MSE_Complex.",
                    "  One line per window size.",
                    "",
                    "RECORDS PER LINE",
                    ] + detail_lines + [
                    "",
                    "METRIC DISPLAYED",
                    "  MSE_Complex (mean per frequency, one line per window size)",
                    "  Legend placed outside right. Y-axis log scale.",
                ]
    write_sample_analysis(output_dir / 'fft_comparison_complex_mse_by_window_size_sample_analysis.txt', txt_lines)

    return 1

# ============================================================================
# EXTRA ANALYSIS PLOTS — Exact-Bin Sine Frequencies
# ============================================================================

def plot_extra_exact_bin_complex_mse(data, output_dir):
    """
    Plot complex MSE vs frequency for exact-bin sine tests.
    Aggregated across all window sizes: mean line + min/max shaded band.
    X-axis linear (Hz), y-axis log. No leakage in these results.
    """
    print("  Generating exact-bin complex MSE vs frequency (aggregated)...")

    if len(data) == 0:
        print("  No data found")
        return 0

    grouped = data.groupby('Frequency').agg(
        mse_mean=('MSE_Complex', 'mean'),
        mse_min =('MSE_Complex', 'min'),
        mse_max =('MSE_Complex', 'max'),
    ).reset_index().sort_values('Frequency')

    tick_freqs  = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    tick_labels = ['100', '1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k', '9k', '10k']

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(grouped['Frequency'], grouped['mse_min'], grouped['mse_max'],
                    alpha=0.2, color=COLORS['highlight'], label='Min/Max Range')
    ax.plot(grouped['Frequency'], grouped['mse_mean'], '-',
            linewidth=2, color=COLORS['highlight'], alpha=0.9, label='Mean')

    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('MSE Complex', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xlim([100, 10000])
    ax.set_xticks(tick_freqs)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    filename = output_dir / 'exact_bin_complex_mse_vs_frequency.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    total_records    = len(data)
    n_freqs          = len(grouped)
    n_windows        = data['WindowSize'].nunique() if 'WindowSize' in data.columns else 'N/A'
    records_per_freq = total_records / n_freqs if n_freqs > 0 else float('nan')

    txt_lines = [
        "=" * 70,
        "SAMPLE ANALYSIS: Exact-Bin Complex MSE vs Frequency (aggregated)",
        "File: exact_bin_complex_mse_vs_frequency.png",
        "=" * 70,
        "",
        "BACKGROUND",
        "  Test uses sine waves whose frequency falls exactly on an FFT bin,",
        "  eliminating spectral leakage. Any remaining MSE is FRFT numerical error.",
        "",
        "INPUT DATA",
        f"  Total records                              : {total_records}",
        f"  Unique frequencies tested                  : {n_freqs}",
        f"  Unique window sizes (pooled into band)     : {n_windows}",
        f"  Avg records per frequency point            : {records_per_freq:.1f}",
        "",
        "AGGREGATION",
        "  Grouped by Frequency → mean / min / max of MSE_Complex across",
        "  ALL window sizes. The shaded band width reflects variation across windows.",
        "",
        "METRIC DISPLAYED",
        "  MSE_Complex (mean line + min/max shaded band vs frequency)",
        ]
    write_sample_analysis(
        output_dir / 'exact_bin_complex_mse_vs_frequency_sample_analysis.txt', txt_lines)

    return 1


def plot_extra_exact_bin_complex_mse_by_window_size(data, output_dir):
    """
    Plot complex MSE vs frequency for exact-bin sine tests, one line per window size.
    Legend placed outside to the right.
    """
    print("  Generating exact-bin complex MSE vs frequency by window size...")

    data_filtered = data[data['WindowSize'] >= 64].copy()
    if len(data_filtered) == 0:
        print("  No data found")
        return 0

    window_sizes = sorted(data_filtered['WindowSize'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(window_sizes)))

    tick_freqs  = [100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    tick_labels = ['100', '1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k', '9k', '10k']

    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, ws in enumerate(window_sizes):
        ws_data = data_filtered[data_filtered['WindowSize'] == ws]
        grouped = ws_data.groupby('Frequency').agg(
            mse_mean=('MSE_Complex', 'mean')
        ).reset_index().sort_values('Frequency')

        ax.plot(grouped['Frequency'], grouped['mse_mean'], '-',
                label=str(ws), linewidth=1.5, color=colors[idx], alpha=0.85)

    ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax.set_ylabel('MSE Complex', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xlim([100, 10000])
    ax.set_xticks(tick_freqs)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(title='Window Size', loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0,
              framealpha=0.9, ncol=1)

    plt.tight_layout()
    filename = output_dir / 'exact_bin_complex_mse_by_window_size.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    total_records = len(data_filtered)
    detail_lines  = []
    for ws in window_sizes:
        n   = len(data_filtered[data_filtered['WindowSize'] == ws])
        n_f = data_filtered[data_filtered['WindowSize'] == ws]['Frequency'].nunique()
        detail_lines.append(
            f"    WindowSize={ws:6d}  total_rows={n:5d}  unique_freqs={n_f}")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Exact-Bin Complex MSE by Window Size",
                    "File: exact_bin_complex_mse_by_window_size.png",
                    "=" * 70,
                    "",
                    "BACKGROUND",
                    "  Same exact-bin sine data as the aggregated plot, but separated by",
                    "  window size so per-window accuracy trends are visible.",
                    "",
                    "FILTERS APPLIED",
                    "  WindowSize >= 64",
                    "",
                    "DATA AFTER FILTERING",
                    f"  Total records                              : {total_records}",
                    f"  Window sizes plotted (one line each)       : {window_sizes}",
                    "",
                    "AGGREGATION",
                    "  For each (WindowSize, Frequency) group: mean of MSE_Complex.",
                    "  One line per window size; no cross-window aggregation.",
                    "",
                    "RECORDS PER LINE",
                    ] + detail_lines + [
                    "",
                    "METRIC DISPLAYED",
                    "  MSE_Complex (mean per frequency, one coloured line per window size)",
                ]
    write_sample_analysis(
        output_dir / 'exact_bin_complex_mse_by_window_size_sample_analysis.txt', txt_lines)

    return 1


# ============================================================================
# EXTRA ANALYSIS PLOTS — Impulse Response
# ============================================================================

def plot_extra_impulse_complex_mse_by_window_size(data, output_dir):
    """
    Plot complex MSE (FRFT vs FFT) per window size for the impulse test.
    Single bar/line chart — one value per window size (single frame, no frequency axis).
    """
    print("  Generating impulse complex MSE vs window size...")

    if len(data) == 0:
        print("  No data found")
        return 0

    # For the impulse test each (window_size, overlap_factor) row is one measurement.
    # Focus on overlap_factor == 1 for the cleanest view.
    d = data[data['OverlapFactor'] == 1].copy() if 'OverlapFactor' in data.columns else data.copy()
    d = d.sort_values('WindowSize')

    if len(d) == 0:
        print("  No overlap=1 rows found, using all rows")
        d = data.sort_values('WindowSize')

    fig, ax = plt.subplots(figsize=(10, 5))

    ws   = d['WindowSize'].values
    mse  = d['MSE_Complex'].values
    x    = np.arange(len(ws))

    ax.bar(x, mse, color=COLORS['highlight'], alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Window Size (N)', fontweight='bold')
    ax.set_ylabel('MSE Complex\n(FRFT vs FFT)', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'$2^{{{int(np.log2(w))}}}$' for w in ws], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, which='both', linestyle='--', axis='y')

    plt.tight_layout()
    filename = output_dir / 'impulse_complex_mse_by_window_size.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    total_records = len(data)
    used_records  = len(d)
    overlap_note  = "OverlapFactor == 1 only" if 'OverlapFactor' in data.columns else "all rows (no OverlapFactor column)"

    detail_lines = []
    for w, m in zip(ws, mse):
        detail_lines.append(
            f"    WindowSize=2^{int(np.log2(w))} ({w:6d})  MSE_Complex={m:.4e}")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Impulse Complex MSE by Window Size (bar chart)",
                    "File: impulse_complex_mse_by_window_size.png",
                    "=" * 70,
                    "",
                    "BACKGROUND",
                    "  For a unit impulse at sample 0, FRFT at α=1 should equal FFT exactly.",
                    "  Any non-zero MSE_Complex is purely floating-point numerical noise.",
                    "",
                    "FILTER APPLIED",
                    f"  {overlap_note}",
                    "",
                    "DATA",
                    f"  Total records in impulse results.txt      : {total_records}",
                    f"  Records used for this plot                : {used_records}",
                    "  One bar per window size — no aggregation (single measurement each).",
                    "",
                    "VALUES PLOTTED",
                    ] + detail_lines + [
                    "",
                    "METRIC DISPLAYED",
                    "  MSE_Complex (FRFT vs FFT), log y-axis. Lower = better numerical accuracy.",
                ]
    write_sample_analysis(
        output_dir / 'impulse_complex_mse_by_window_size_sample_analysis.txt', txt_lines)

    return 1


def plot_extra_impulse_analytical_error(anal_data, output_dir):
    """
    Plot FRFT absolute error vs the analytical ground truth (1/sqrt(N)) per window size.
    Shows MSE, Max Error, and Mean Error on a single log-y axis.
    This is the cleanest measure of raw FRFT numerical accuracy.
    """
    print("  Generating impulse analytical error vs window size...")

    if anal_data is None or len(anal_data) == 0:
        print("  No analytical error data found")
        return 0

    anal_data = anal_data.sort_values('WindowSize')
    ws  = anal_data['WindowSize'].values
    x   = np.arange(len(ws))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, anal_data['MSE_Complex_vs_Analytical'].values,  'o-',
            linewidth=2, markersize=6, color=COLORS['highlight'], label='MSE Complex')
    ax.plot(x, anal_data['MaxError_vs_Analytical'].values,     's--',
            linewidth=2, markersize=6, color=COLORS['mse'],       label='Max Error')
    ax.plot(x, anal_data['MeanError_vs_Analytical'].values,    '^:',
            linewidth=2, markersize=6, color=COLORS['mss'],       label='Mean Error')

    ax.set_xlabel('Window Size (N)', fontweight='bold')
    ax.set_ylabel('Error vs Analytical\nGround Truth (1/√N)', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'$2^{{{int(np.log2(w))}}}$' for w in ws], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    # Annotate log-log slope if enough points
    if len(ws) > 2:
        log_ws  = np.log2(ws)
        log_mse = np.log10(anal_data['MSE_Complex_vs_Analytical'].values)
        valid   = np.isfinite(log_mse)
        if valid.sum() > 2:
            slope, _ = np.polyfit(log_ws[valid], log_mse[valid], 1)
            ax.annotate(f'log-log slope: {slope:.2f}',
                        xy=(0.02, 0.05), xycoords='axes fraction',
                        fontsize=11, color='dimgray',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.6))

    plt.tight_layout()
    filename = output_dir / 'impulse_analytical_error_vs_window_size.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    detail_lines = []
    for i, w in enumerate(ws):
        mse_v  = anal_data['MSE_Complex_vs_Analytical'].values[i]
        max_v  = anal_data['MaxError_vs_Analytical'].values[i]
        mean_v = anal_data['MeanError_vs_Analytical'].values[i]
        detail_lines.append(
            f"    2^{int(np.log2(w))} ({w:6d})  MSE={mse_v:.4e}  MaxErr={max_v:.4e}  MeanErr={mean_v:.4e}")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Impulse Analytical Error vs Window Size",
                    "File: impulse_analytical_error_vs_window_size.png",
                    "=" * 70,
                    "",
                    "BACKGROUND",
                    "  Compares FRFT output against the exact analytical ground truth:",
                    "  for a unit impulse at sample 0, the ideal spectrum is 1/sqrt(N)",
                    "  at every bin with zero imaginary part. This is the purest measure",
                    "  of FRFT numerical accuracy, independent of FFT error.",
                    "",
                    "DATA SOURCE",
                    f"  File: impulse/analytical_error.txt",
                    f"  Rows (one per window size, OverlapFactor=1 only): {len(anal_data)}",
                    "  Columns: WindowSize, MSE_Complex_vs_Analytical, MaxError_vs_Analytical,",
                    "           MeanError_vs_Analytical",
                    "",
                    "SAMPLES PER POINT",
                    "  Each point = single measurement (OverlapFactor=1, 1 frame).",
                    "  No aggregation — each window size has exactly 1 data point.",
                    "",
                    "VALUES PLOTTED",
                    ] + detail_lines + [
                    "",
                    "METRICS DISPLAYED",
                    "  MSE_Complex_vs_Analytical  (o- line)",
                    "  MaxError_vs_Analytical     (s-- line)",
                    "  MeanError_vs_Analytical    (^:  line)",
                    "  All on log y-axis. Log-log slope annotated if ≥3 window sizes.",
                ]
    write_sample_analysis(
        output_dir / 'impulse_analytical_error_vs_window_size_sample_analysis.txt', txt_lines)

    return 1


def plot_extra_impulse_analytical_error_comparison(fft_data, anal_data, output_dir):
    """
    Side-by-side comparison: FRFT vs FFT complex MSE (impulse) alongside
    FRFT vs analytical ground truth, both plotted vs window size.
    Lets the reader see how much of the FRFT-vs-FFT error is shared with FFT's
    own numerical error vs. how much is unique to the FRFT.
    """
    print("  Generating impulse FRFT-vs-FFT vs analytical comparison...")

    if fft_data is None or len(fft_data) == 0:
        print("  No FFT data found")
        return 0
    if anal_data is None or len(anal_data) == 0:
        print("  No analytical error data found")
        return 0

    fft_d  = fft_data[fft_data.get('OverlapFactor', fft_data.iloc[:, 1]) == 1].sort_values('WindowSize') \
        if 'OverlapFactor' in fft_data.columns else fft_data.sort_values('WindowSize')
    anal_d = anal_data.sort_values('WindowSize')

    # Merge on WindowSize so both series share the same x-positions
    merged = pd.merge(
        fft_d[['WindowSize', 'MSE_Complex']].rename(columns={'MSE_Complex': 'mse_vs_fft'}),
        anal_d[['WindowSize', 'MSE_Complex_vs_Analytical']].rename(
            columns={'MSE_Complex_vs_Analytical': 'mse_vs_analytical'}),
        on='WindowSize', how='inner'
    ).sort_values('WindowSize')

    if len(merged) == 0:
        print("  No overlapping window sizes between FFT and analytical data")
        return 0

    ws = merged['WindowSize'].values
    x  = np.arange(len(ws))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, merged['mse_vs_fft'].values,         'o-',
            linewidth=2, markersize=6, color=COLORS['mss'],
            label='FRFT vs FFT (both have FP noise)')
    ax.plot(x, merged['mse_vs_analytical'].values,  's--',
            linewidth=2, markersize=6, color=COLORS['highlight'],
            label='FRFT vs Analytical (1/√N)')

    ax.set_xlabel('Window Size (N)', fontweight='bold')
    ax.set_ylabel('MSE Complex', fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'$2^{{{int(np.log2(w))}}}$' for w in ws], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    filename = output_dir / 'impulse_frft_vs_fft_vs_analytical.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    detail_lines = []
    for i, w in enumerate(ws):
        fft_v  = merged['mse_vs_fft'].values[i]
        anal_v = merged['mse_vs_analytical'].values[i]
        detail_lines.append(
            f"    2^{int(np.log2(w))} ({w:6d})  vs_FFT={fft_v:.4e}  vs_Analytical={anal_v:.4e}")

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Impulse FRFT-vs-FFT vs Analytical Comparison",
                    "File: impulse_frft_vs_fft_vs_analytical.png",
                    "=" * 70,
                    "",
                    "BACKGROUND",
                    "  Overlays two error measures to separate sources of numerical error:",
                    "  - FRFT vs FFT: both transforms have floating-point noise; a low value",
                    "    here shows FRFT and FFT agree, but doesn't tell us how accurate either is.",
                    "  - FRFT vs Analytical (1/√N): the absolute accuracy benchmark.",
                    "  If both lines track each other, FFT and FRFT share the same error floor.",
                    "",
                    "DATA SOURCES",
                    f"  FRFT-vs-FFT   : impulse/results.txt        (OverlapFactor=1, {len(fft_d)} rows)",
                    f"  vs-Analytical : impulse/analytical_error.txt ({len(anal_d)} rows)",
                    f"  Inner-joined on WindowSize → {len(merged)} shared window sizes",
                    "",
                    "SAMPLES PER POINT",
                    "  Each point = single measurement per window size. No aggregation.",
                    "",
                    "VALUES PLOTTED",
                    ] + detail_lines + [
                    "",
                    "METRICS DISPLAYED",
                    "  o-  line : MSE_Complex between FRFT and FFT outputs",
                    "  s-- line : MSE_Complex between FRFT and analytical ground truth (1/√N)",
                    "  Both on log y-axis.",
                ]
    write_sample_analysis(
        output_dir / 'impulse_frft_vs_fft_vs_analytical_sample_analysis.txt', txt_lines)

    return 1


def plot_extra_impulse_spectrum(spectrum_data, output_dir):
    """
    Plot the actual per-bin FRFT and FFT spectrum magnitudes for the impulse test.

    For a unit impulse at sample 0, the ideal spectrum magnitude is 1/sqrt(N) flat
    at every bin.  We plot one subplot per window size, each showing:
      - Mean FRFT magnitude across overlap factors (solid line, blue)
      - Min/max FRFT band (shaded blue)
      - Mean FFT  magnitude across overlap factors (dashed line, purple)
      - Min/max FFT  band (shaded purple)
      - Analytical reference line at 1/sqrt(N) (dotted grey)

    The spread (min/max) comes from the different overlap factors tested, which
    place the impulse at different positions within the analysis frame.

    Data file: impulse/spectrum.txt
    Columns  : WindowSize, OverlapFactor, BinIndex, FRFT_Mag, FFT_Mag
    """
    print("  Generating impulse spectrum mean/min/max plot...")

    if spectrum_data is None or len(spectrum_data) == 0:
        print("  No spectrum data found")
        return 0

    window_sizes = sorted(spectrum_data['WindowSize'].unique())
    n_ws = len(window_sizes)
    if n_ws == 0:
        return 0

    # Layout: up to 4 columns
    n_cols = min(4, n_ws)
    n_rows = int(np.ceil(n_ws / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for idx, ws in enumerate(window_sizes):
        ax  = axes[idx // n_cols][idx % n_cols]
        wsd = spectrum_data[spectrum_data['WindowSize'] == ws]

        # Aggregate over OverlapFactor for each BinIndex
        agg = wsd.groupby('BinIndex').agg(
            frft_mean=('FRFT_Mag', 'mean'),
            frft_min =('FRFT_Mag', 'min'),
            frft_max =('FRFT_Mag', 'max'),
            fft_mean =('FFT_Mag',  'mean'),
            fft_min  =('FFT_Mag',  'min'),
            fft_max  =('FFT_Mag',  'max'),
        ).reset_index().sort_values('BinIndex')

        bins = agg['BinIndex'].values
        analytical = 1.0 / np.sqrt(ws)

        # FRFT
        ax.fill_between(bins,
                        agg['frft_min'].values, agg['frft_max'].values,
                        alpha=0.20, color=COLORS['mss'], label='FRFT min/max')
        ax.plot(bins, agg['frft_mean'].values,
                '-', linewidth=1.5, color=COLORS['mss'], label='FRFT mean', alpha=0.9)

        # FFT
        ax.fill_between(bins,
                        agg['fft_min'].values, agg['fft_max'].values,
                        alpha=0.20, color=COLORS['mse'], label='FFT min/max')
        ax.plot(bins, agg['fft_mean'].values,
                '--', linewidth=1.5, color=COLORS['mse'], label='FFT mean', alpha=0.9)

        # Analytical reference: 1/sqrt(N)
        ax.axhline(analytical, color='gray', linestyle=':', linewidth=1.2,
                   label=f'1/√N = {analytical:.4f}', alpha=0.8)

        ax.set_title(f'N = 2^{int(np.log2(ws))} = {ws}', fontweight='bold', fontsize=13)
        ax.set_xlabel('Bin Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Magnitude', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, framealpha=0.8, loc='upper right')

    # Hide unused subplots
    for idx in range(n_ws, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()
    filename = output_dir / 'impulse_spectrum_mean_minmax.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    → {filename}")
    plt.close()

    # ---- sample analysis report ----
    overlap_values  = sorted(spectrum_data['OverlapFactor'].unique())
    total_rows      = len(spectrum_data)

    detail_lines = []
    for ws in window_sizes:
        wsd  = spectrum_data[spectrum_data['WindowSize'] == ws]
        n_of = wsd['OverlapFactor'].nunique()
        n_bins = wsd['BinIndex'].nunique()
        detail_lines.append(
            f"    WindowSize=2^{int(np.log2(ws))} ({ws:6d})"
            f"  overlap_factors={sorted(wsd['OverlapFactor'].unique())}"
            f"  bins={n_bins}"
            f"  rows={len(wsd)}"
            f"  analytical_ref=1/√{ws}={1.0/np.sqrt(ws):.6e}"
        )

    txt_lines = [
                    "=" * 70,
                    "SAMPLE ANALYSIS: Impulse Spectrum Mean/Min/Max (FRFT vs FFT)",
                    "File: impulse_spectrum_mean_minmax.png",
                    "=" * 70,
                    "",
                    "BACKGROUND",
                    "  For a unit impulse at sample 0, the ideal spectrum magnitude is",
                    "  1/sqrt(N) at every bin (perfectly flat). Both FRFT (at α=1) and FFT",
                    "  should produce this flat spectrum. Any deviation is numerical noise.",
                    "  The spread across overlap factors reflects how impulse position within",
                    "  the frame affects per-bin magnitudes.",
                    "",
                    "DATA SOURCE",
                    f"  File            : impulse/spectrum.txt",
                    f"  Total rows      : {total_rows}",
                    f"  Columns         : WindowSize, OverlapFactor, BinIndex, FRFT_Mag, FFT_Mag",
                    f"  Window sizes    : {window_sizes}",
                    f"  Overlap factors : {overlap_values}",
                    "",
                    "AGGREGATION PER SUBPLOT",
                    "  For each WindowSize, data is grouped by BinIndex and the mean/min/max",
                    "  of FRFT_Mag and FFT_Mag are computed across all OverlapFactor rows.",
                    "  Each bin therefore has (n_overlap_factors) contributing measurements.",
                    f"  Samples per bin per window : {len(overlap_values)}  "
                    f"(one per overlap factor: {overlap_values})",
                    "",
                    "PER-WINDOW DETAIL",
                    "  Format: WindowSize  overlap_factors  n_bins  total_rows  analytical_ref",
                    ] + detail_lines + [
                    "",
                    "INTERPRETATION",
                    "  FRFT mean ≈ FFT mean ≈ 1/sqrt(N) everywhere → FRFT behaves as FFT at α=1.",
                    "  Min/max band width shows sensitivity to impulse position in the frame.",
                    "  Larger windows tend to tighter bands (more bins, same N impulse samples).",
                ]
    write_sample_analysis(
        output_dir / 'impulse_spectrum_mean_minmax_sample_analysis.txt', txt_lines)

    return 1


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

            # Check if we have both passthrough and reversal data
            has_passthrough = 'passthrough' in available_types
            has_reversal = any(t in available_types for t in ['reversal_fwd', 'reversal_bwd'])

            if has_passthrough and has_reversal:
                # Generate combined 2x2 plot
                passthrough_count = len(data[data['test_type'] == 'passthrough'])
                reversal_count = len(data[data['test_type'].isin(['reversal_fwd', 'reversal_bwd'])])
                print(f"  Generating combined plot ({passthrough_count} passthrough + {reversal_count} reversal records)...")
                total_plots += plot_passthrough_reversal_combined(data, passthrough_reversal_output)
            else:
                if not has_passthrough:
                    print(f"  ⚠ No passthrough data found")
                if not has_reversal:
                    print(f"  ⚠ No reversal data found")
    else:
        print(f"  ⚠ Results file not found: {mss_results_file}")

    # ========================================================================
    # PROCESS PASSTHROUGH AND REVERSAL RESULTS (SINE)
    # ========================================================================

    print("\n[Processing Passthrough and Reversal Results (Sine)]")

    # Create passthrough/reversal (sine) output directory
    passthrough_reversal_sine_output = output_dir / 'passthrough_reversal_with_sine'
    passthrough_reversal_sine_output.mkdir(parents=True, exist_ok=True)

    passthrough_reversal_sine_dir = results_dir / 'passthrough_reversal_with_sine'
    mss_results_sine_file = passthrough_reversal_sine_dir / 'mss_analysis_results.txt'

    if mss_results_sine_file.exists():
        print(f"  Found passthrough/reversal (sine) results file: {mss_results_sine_file}")

        data = load_passthrough_reversal_results(mss_results_sine_file)

        if data is not None:
            print(f"  Total records loaded: {len(data)}")

            available_types = data['test_type'].unique()
            print(f"  Available test types: {list(available_types)}")

            has_passthrough = 'passthrough' in available_types
            has_reversal = any(t in available_types for t in ['reversal_fwd', 'reversal_bwd'])

            if has_passthrough and has_reversal:
                passthrough_count = len(data[data['test_type'] == 'passthrough'])
                reversal_count = len(data[data['test_type'].isin(['reversal_fwd', 'reversal_bwd'])])
                print(f"  Generating combined plot ({passthrough_count} passthrough + {reversal_count} reversal records)...")
                total_plots += plot_passthrough_reversal_combined(data, passthrough_reversal_sine_output)
            else:
                if not has_passthrough:
                    print(f"  ⚠ No passthrough data found")
                if not has_reversal:
                    print(f"  ⚠ No reversal data found")
    else:
        print(f"  ⚠ Results file not found: {mss_results_sine_file}")

    # ========================================================================
    # PROCESS FFT COMPARISON RESULTS
    # ========================================================================

    print("\n[Processing FFT Comparison Results]")

    # Create FFT comparison output directory
    fft_comparison_output = output_dir / 'fft_comparison'
    fft_comparison_output.mkdir(parents=True, exist_ok=True)

    # Look for FFT comparison results
    fft_comparison_dir = results_dir / 'fft_comparison'
    fft_results_file = fft_comparison_dir / 'results.txt'

    if fft_results_file.exists():
        print(f"  Found FFT comparison results file: {fft_results_file}")

        fft_data = load_results_data(fft_results_file)

        if fft_data is not None:
            print(f"  Total FFT comparison records: {len(fft_data)}")

            # Generate error plots vs frequency (aggregated across window sizes)
            total_plots += plot_fft_comparison_errors(fft_data, fft_comparison_output)

            # Generate error plots vs frequency (separated by window size)
            total_plots += plot_fft_comparison_errors_by_window_size(fft_data, fft_comparison_output)

            # Generate standalone complex MSE plot with legend outside
            total_plots += plot_fft_comparison_complex_mse_by_window_size(fft_data, fft_comparison_output)
    else:
        print(f"  ⚠ FFT comparison results file not found: {fft_results_file}")

    # ========================================================================
    # PROCESS FFT EXTRA ANALYSIS RESULTS (exact-bin + impulse)
    # ========================================================================

    print("\n[Processing FFT Extra Analysis Results]")

    extra_analysis_dir = results_dir / 'fft_comparison_extra_analysis'

    # --- Exact-bin sine ---
    exact_bin_output = output_dir / 'fft_comparison_extra_analysis' / 'exact_bin'
    exact_bin_output.mkdir(parents=True, exist_ok=True)
    exact_bin_file = extra_analysis_dir / 'exact_bin' / 'results.txt'

    if exact_bin_file.exists():
        print(f"  Found exact-bin results: {exact_bin_file}")
        exact_bin_data = load_results_data(exact_bin_file)
        if exact_bin_data is not None:
            print(f"  Total exact-bin records: {len(exact_bin_data)}")
            total_plots += plot_extra_exact_bin_complex_mse(exact_bin_data, exact_bin_output)
            total_plots += plot_extra_exact_bin_complex_mse_by_window_size(exact_bin_data, exact_bin_output)
    else:
        print(f"  ⚠ Exact-bin results not found: {exact_bin_file}")

    # --- Impulse response ---
    impulse_output = output_dir / 'fft_comparison_extra_analysis' / 'impulse'
    impulse_output.mkdir(parents=True, exist_ok=True)
    impulse_file   = extra_analysis_dir / 'impulse' / 'results.txt'
    analytical_file = extra_analysis_dir / 'impulse' / 'analytical_error.txt'

    if impulse_file.exists():
        print(f"  Found impulse results: {impulse_file}")
        impulse_data = load_results_data(impulse_file)
        anal_data    = load_results_data(analytical_file) if analytical_file.exists() else None

        # Load per-bin spectrum data (written by updated C++ test)
        spectrum_file = extra_analysis_dir / 'impulse' / 'spectrum.txt'
        spectrum_data = load_results_data(spectrum_file) if spectrum_file.exists() else None

        if impulse_data is not None:
            print(f"  Total impulse records: {len(impulse_data)}")
            total_plots += plot_extra_impulse_complex_mse_by_window_size(impulse_data, impulse_output)

        if spectrum_data is not None:
            print(f"  Total impulse spectrum rows: {len(spectrum_data)}")
            total_plots += plot_extra_impulse_spectrum(spectrum_data, impulse_output)
        else:
            print(f"  ⚠ Impulse spectrum file not found: {spectrum_file}")
            print(f"    (Re-run the C++ test to generate it)")

        if anal_data is not None:
            print(f"  Total analytical error records: {len(anal_data)}")
            total_plots += plot_extra_impulse_analytical_error(anal_data, impulse_output)
            total_plots += plot_extra_impulse_analytical_error_comparison(impulse_data, anal_data, impulse_output)
    else:
        print(f"  ⚠ Impulse results not found: {impulse_file}")

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

    # Check if passthrough/reversal plots were generated (with Sines)
    passthrough_reversal_sine_output = output_dir / 'passthrough_reversal_with_sine'
    if passthrough_reversal_sine_output.exists():
        print(f"  Passthrough/Reversal (Sine) plots → {passthrough_reversal_sine_output}/")

    # Check if FFT comparison plots were generated
    fft_comparison_output = output_dir / 'fft_comparison'
    if fft_comparison_output.exists():
        print(f"  FFT Comparison plots → {fft_comparison_output}/")

    # Check if extra analysis plots were generated
    extra_analysis_output = output_dir / 'fft_comparison_extra_analysis'
    if extra_analysis_output.exists():
        print(f"  FFT Extra Analysis plots → {extra_analysis_output}/")

    print("\nAll plots saved as png files at 300 DPI\n")

    return 0


if __name__ == '__main__':
    exit(main())