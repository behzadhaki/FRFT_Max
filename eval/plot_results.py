#!/usr/bin/env python3
"""
FRFT Test Results Plotter - Updated for New Folder Structure

Generates plots for:
- Reconstruction test (reconstruction_test/)
- Homomorphic test (homomorphic_test/)
- FFT comparison test (fft_comparison/)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import seaborn as sns


def load_timing_data(filename):
    """Load timing benchmark data."""
    try:
        data = pd.read_csv(filename, sep='\t', comment='#')
        print(f"✓ Loaded {len(data)} timing records from {filename}")
        return data
    except Exception as e:
        print(f"✗ Error loading timing file: {e}")
        return None


def load_results_data(filename):
    """Load test results (reconstruction, homomorphic, or FFT comparison)."""
    try:
        data = pd.read_csv(filename, sep='\t', comment='#')
        print(f"✓ Loaded {len(data)} records from {filename}")
        return data
    except Exception as e:
        print(f"✗ Error loading results file: {e}")
        return None


def plot_processing_time_and_complexity(timing_data, output_dir):
    """Plot processing time with complexity reference lines."""
    fig, ax = plt.subplots(figsize=(12, 7))

    window_sizes = timing_data['WindowSize'].values
    mean_forward = timing_data['MeanForward'].values
    min_forward = timing_data['MinForward'].values
    max_forward = timing_data['MaxForward'].values
    num_samples = timing_data['NumSamples']

    n_trials = num_samples.iloc[0] if len(num_samples) > 0 else 0

    # Professional color scheme
    color_mean = '#1f77b4'    # Blue
    color_nlogn = '#ff7f0e'   # Orange
    color_n2 = '#8c564b'      # Brown
    color_n = '#9467bd'       # Purple

    # Shaded area for min/max range
    ax.fill_between(window_sizes, min_forward, max_forward,
                    alpha=0.2, color=color_mean, label='Min/Max Range', zorder=1)

    # Plot mean
    ax.plot(window_sizes, mean_forward, 'o-', label='Average Time',
            linewidth=2.5, markersize=8, color=color_mean, alpha=0.8, zorder=3)

    # Complexity reference lines
    nlogn_fit = mean_forward[0] * (window_sizes / window_sizes[0]) * \
                (np.log2(window_sizes) / np.log2(window_sizes[0]))
    ax.plot(window_sizes, nlogn_fit, ':', linewidth=2.5,
            label='O(N log N) Reference', color=color_nlogn, alpha=0.7, zorder=1)

    n2_fit = mean_forward[0] * (window_sizes / window_sizes[0])**2
    ax.plot(window_sizes, n2_fit, ':', linewidth=2.5,
            label='O(N²) Reference', color=color_n2, alpha=0.7, zorder=1)

    n_fit = mean_forward[0] * (window_sizes / window_sizes[0])
    ax.plot(window_sizes, n_fit, ':', linewidth=2.5,
            label='O(N) Reference', color=color_n, alpha=0.7, zorder=1)

    ax.set_xlabel('Window Size (samples)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Processing Time (ms/frame)', fontsize=14, fontweight='bold')
    ax.set_title('FRFT Processing Time vs Window Size', fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)

    plt.tight_layout()
    filename = output_dir / f'processing_time_complexity.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()


def plot_rtf(timing_data, output_dir):
    """Plot Real-Time Factor vs window size."""
    window_sizes = timing_data['WindowSize']
    rtf_mean = timing_data['RTF_Mean']
    rtf_best = timing_data['RTF_Best']
    rtf_worst = timing_data['RTF_Worst']

    color_mean = '#1f77b4'
    color_threshold = '#d62728'

    fig, ax = plt.subplots(figsize=(12, 7))

    # Shaded area for best/worst
    ax.fill_between(window_sizes, rtf_best, rtf_worst,
                    alpha=0.2, color=color_mean, label='Best/Worst Range')

    # Plot mean RTF
    ax.plot(window_sizes, rtf_mean, 'o-', label='Mean RTF',
            linewidth=2.5, markersize=8, color=color_mean, alpha=0.8)

    # Real-time threshold
    ax.axhline(y=1.0, color=color_threshold, linestyle='--', linewidth=2.5,
               label='Real-Time Threshold (RTF=1.0)', alpha=0.8)

    ax.set_xlabel('Window Size (samples)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Real-Time Factor (RTF)', fontsize=14, fontweight='bold')
    ax.set_title('FRFT Real-Time Performance', fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log', base=2)
    ax
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)

    plt.tight_layout()
    filename = output_dir / 'rtf_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()


def plot_reconstruction_mse_vs_alpha(data, output_dir):
    """Plot reconstruction test MSE vs alpha for each frequency."""
    # Filter alpha range
    subset = data[(data['Alpha'] >= 0.0) & (data['Alpha'] <= 2.1)]

    frequencies = sorted(subset['Frequency'].unique())
    alphas = sorted(subset['Alpha'].unique())
    alphas = [round(a, 1) for a in alphas]
    alphas = sorted(list(set(alphas)))

    n_freqs = len(frequencies)
    fig, axes = plt.subplots(n_freqs, 1, figsize=(12, max(1.2 * n_freqs, 6)))

    if n_freqs == 1:
        axes = [axes]

    # Find global range for consistent y-axis
    all_mses = []
    for freq in frequencies:
        freq_data = subset[np.abs(subset['Frequency'] - freq) < 0.01]
        for alpha in alphas:
            alpha_data = freq_data[np.abs(freq_data['Alpha'] - alpha) < 0.05]
            if len(alpha_data) > 0:
                all_mses.append(alpha_data['MSE'].mean())

    if all_mses:
        global_min = max(0, min(all_mses) * 0.5)
        global_max = max(all_mses) * 1.5
    else:
        global_min, global_max = 0, 1e-10

    bar_color = '#1f77b4'
    x_pos = np.arange(len(alphas))
    bar_width = 0.7

    for idx, freq in enumerate(frequencies):
        ax = axes[idx]
        freq_data = subset[np.abs(subset['Frequency'] - freq) < 0.01]

        means = []
        for alpha in alphas:
            alpha_data = freq_data[np.abs(freq_data['Alpha'] - alpha) < 0.05]
            if len(alpha_data) > 0:
                means.append(alpha_data['MSE'].mean())
            else:
                means.append(0)

        ax.bar(x_pos, means, bar_width, color=bar_color, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Frequency label
        ax.text(0.98, 0.95, f'{freq:.0f} Hz',
                transform=ax.transAxes, fontsize=13, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        ax.set_xticks(x_pos)

        if idx == n_freqs - 1:
            ax.set_xticklabels([f'{a:.1f}' for a in alphas], fontsize=11)
            ax.set_xlabel('Alpha (α)', fontsize=13, fontweight='bold')
        else:
            ax.set_xticklabels([])

        if idx == n_freqs // 2:
            ax.set_ylabel('Mean Squared Error (MSE)', fontsize=13, fontweight='bold')

        ax.set_yscale('log')
        ax
        ax.tick_params(axis='y', which='major', labelsize=11)
        ax.set_ylim(global_min, global_max)

    fig.suptitle('Reconstruction Test: MSE vs Alpha', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = output_dir / 'reconstruction_mse_vs_alpha.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()


def plot_homomorphic_mse_vs_alpha_kde(data, output_dir):
    """
    Plot KDE-style representation of MSE vs Alpha for each frequency.
    X-axis: Alpha values (-2 to 2), Y-axis: MSE (linear scale)
    Each subplot shows one frequency with all window sizes mixed together.
    Journal-quality formatting.
    """
    # Filter for the focus window sizes
    focus_window_sizes = [512, 1024, 2048, 4096]
    data_filtered = data[data['WindowSize'].isin(focus_window_sizes)]

    if len(data_filtered) == 0:
        print("⚠ No data found for focus window sizes. Using all available window sizes.")
        data_filtered = data

    frequencies = sorted(data_filtered['Frequency'].unique())
    n_freqs = len(frequencies)

    # First pass: determine global y-range
    global_min = float('inf')
    global_max = float('-inf')

    for freq in frequencies:
        freq_data = data_filtered[np.abs(data_filtered['Frequency'] - freq) < 0.01]
        alphas = freq_data['Alpha_Total'].values
        mses = freq_data['MSE_Homomorphic'].values
        valid_mask = mses > 0
        mses = mses[valid_mask]

        if len(mses) > 0:
            global_min = min(global_min, np.min(mses))
            global_max = max(global_max, np.percentile(mses, 99))  # Use 99th percentile to avoid outliers

    # Add some padding
    y_range = global_max - global_min
    global_min = max(0, global_min - 0.05 * y_range)
    global_max = global_max + 0.05 * y_range

    # Create subplots: one row per frequency
    fig, axes = plt.subplots(n_freqs, 1, figsize=(14, max(4 * n_freqs, 10)))

    if n_freqs == 1:
        axes = [axes]

    # Define alpha bins (0.1 width) from -2 to 2 - same as boxplots
    alpha_bins = np.arange(-2.0, 2.1, 0.1)
    alpha_bin_centers = alpha_bins[:-1] + 0.05

    for idx, freq in enumerate(frequencies):
        ax = axes[idx]
        freq_data = data_filtered[np.abs(data_filtered['Frequency'] - freq) < 0.01]

        # Mix all window sizes together
        alphas = freq_data['Alpha_Total'].values
        mses = freq_data['MSE_Homomorphic'].values

        # Filter valid MSE values
        valid_mask = mses > 0
        alphas = alphas[valid_mask]
        mses = mses[valid_mask]

        if len(alphas) > 10:
            mse_medians = []
            mse_p25 = []
            mse_p75 = []
            alpha_centers = []

            # Use explicit bins like boxplots
            for bin_idx, (bin_start, bin_end) in enumerate(zip(alpha_bins[:-1], alpha_bins[1:])):
                bin_center = alpha_bin_centers[bin_idx]

                # Get MSE values in this alpha bin
                mask = (alphas >= bin_start) & (alphas < bin_end)
                bin_mses = mses[mask]

                if len(bin_mses) > 0:
                    mse_medians.append(np.median(bin_mses))
                    mse_p25.append(np.percentile(bin_mses, 25))
                    mse_p75.append(np.percentile(bin_mses, 75))
                    alpha_centers.append(bin_center)

            # Plot smooth curve
            if len(alpha_centers) > 0:
                ax.plot(alpha_centers, mse_medians, '-', linewidth=3,
                        color='#1f77b4', alpha=0.85)

                # Add shaded region for IQR
                ax.fill_between(alpha_centers, mse_p25, mse_p75,
                                alpha=0.3, color='#1f77b4', label='IQR (25th-75th percentile)')

        # Frequency label
        ax.text(0.02, 0.95, f'{freq:.0f} Hz (n={len(alphas)} samples)',
                transform=ax.transAxes, fontsize=18, fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'))

        ax.set_xlabel('Alpha (α)' if idx == n_freqs - 1 else '', fontsize=20, fontweight='bold')
        ax.set_ylabel('MSE (median)', fontsize=20, fontweight='bold')
        ax
        ax.tick_params(axis='both', which='major', labelsize=16)
        if len(alpha_centers) > 0:
            ax.legend(fontsize=16, loc='upper right', framealpha=0.95)
        ax.set_xlim(-2, 2)
        ax.set_ylim(global_min, global_max)

    plt.tight_layout()

    filename = output_dir / 'homomorphic_mse_vs_alpha_kde.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()


def plot_homomorphic_mse_vs_alpha_boxplots(data, output_dir):
    """
    Plot box plots of MSE for alpha bins (0.1 width) for each frequency.
    X-axis: Alpha bins (-2 to 2), Y-axis: MSE (linear scale)
    Each subplot shows one frequency with all window sizes mixed together in each box.
    """
    # Filter for the focus window sizes
    focus_window_sizes = [512, 1024, 2048, 4096]
    data_filtered = data[data['WindowSize'].isin(focus_window_sizes)]

    if len(data_filtered) == 0:
        print("⚠ No data found for focus window sizes. Using all available window sizes.")
        data_filtered = data

    frequencies = sorted(data_filtered['Frequency'].unique())
    n_freqs = len(frequencies)

    # Define alpha bins (0.1 width) from -2 to 2
    alpha_bins = np.arange(-2.0, 2.1, 0.1)
    alpha_bin_centers = alpha_bins[:-1] + 0.05
    n_bins = len(alpha_bin_centers)

    # First pass: determine global y-range across all frequencies
    global_min = float('inf')
    global_max = float('-inf')

    for freq in frequencies:
        freq_data = data_filtered[np.abs(data_filtered['Frequency'] - freq) < 0.01]
        mse_values = freq_data['MSE_Homomorphic'].values
        mse_values = mse_values[mse_values > 0]

        if len(mse_values) > 0:
            global_min = min(global_min, np.min(mse_values))
            global_max = max(global_max, np.percentile(mse_values, 99))  # Use 99th percentile to avoid outliers

    # Add some padding
    y_range = global_max - global_min
    global_min = max(0, global_min - 0.05 * y_range)
    global_max = global_max + 0.05 * y_range

    # Create subplots: one row per frequency
    fig, axes = plt.subplots(n_freqs, 1, figsize=(14, max(4 * n_freqs, 10)))

    if n_freqs == 1:
        axes = [axes]

    for idx, freq in enumerate(frequencies):
        ax = axes[idx]
        freq_data = data_filtered[np.abs(data_filtered['Frequency'] - freq) < 0.01]

        # Prepare data for box plots - all window sizes mixed together
        all_boxplot_data = []
        all_positions = []

        for bin_idx, (bin_start, bin_end) in enumerate(zip(alpha_bins[:-1], alpha_bins[1:])):
            bin_center = alpha_bin_centers[bin_idx]

            # Get MSE values in this alpha bin (all window sizes combined)
            mask = (freq_data['Alpha_Total'] >= bin_start) & (freq_data['Alpha_Total'] < bin_end)
            mse_values = freq_data.loc[mask, 'MSE_Homomorphic'].values

            # Filter valid values
            mse_values = mse_values[mse_values > 0]

            if len(mse_values) > 0:
                all_boxplot_data.append(mse_values)
                all_positions.append(bin_center)

        # Create box plots
        if len(all_boxplot_data) > 0:
            bp = ax.boxplot(all_boxplot_data, positions=all_positions, widths=0.08,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='red', linewidth=2.5),
                            boxprops=dict(linewidth=2, facecolor='#1f77b4', alpha=0.6),
                            whiskerprops=dict(linewidth=2),
                            capprops=dict(linewidth=2))

        # Frequency label with sample count
        total_samples = len(freq_data[freq_data['MSE_Homomorphic'] > 0])
        ax.text(0.02, 0.95, f'{freq:.0f} Hz (n={total_samples} samples)',
                transform=ax.transAxes, fontsize=18, fontweight='bold',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'))

        ax.set_xlabel('Alpha (α)' if idx == n_freqs - 1 else '', fontsize=20, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=20, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both', axis='y')
        ax.grid(True, alpha=0.2, which='major', axis='x')
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(global_min, global_max)
        ax.set_xticks(alpha_bin_centers[::4])  # Show every 4th tick
        ax.set_xticklabels([f'{x:.1f}' for x in alpha_bin_centers[::4]], fontsize=16)
        ax.tick_params(axis='y', which='major', labelsize=16)

    plt.tight_layout()

    filename = output_dir / 'homomorphic_mse_vs_alpha_boxplots.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()


def plot_homomorphic_mse_vs_alpha_all_frequencies(data, output_dir):
    """
    Plot median MSE vs Alpha with all frequencies in one plot.
    X-axis: Alpha values (-2 to 2), Y-axis: MSE (linear scale)
    Each frequency gets a different colored line with legend.
    Only selected frequencies: 100, 220, 440, 1000, 2000, 4000, 6000, 8000, 10000 Hz
    """
    # Filter for the focus window sizes
    focus_window_sizes = [512, 1024, 2048, 4096]
    data_filtered = data[data['WindowSize'].isin(focus_window_sizes)]

    if len(data_filtered) == 0:
        print("⚠ No data found for focus window sizes. Using all available window sizes.")
        data_filtered = data

    # Filter for selected frequencies only
    selected_frequencies = [100, 220, 440, 1000, 2000, 4000, 6000, 8000, 10000]
    frequencies = sorted([f for f in data_filtered['Frequency'].unique() if f in selected_frequencies])
    n_freqs = len(frequencies)

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Color scheme for different frequencies
    colors = plt.cm.tab20(np.linspace(0, 1, n_freqs))

    # Determine global y-range
    global_min = float('inf')
    global_max = float('-inf')

    all_plot_data = []

    # Define alpha bins (0.1 width) from -2 to 2 - same as boxplots
    alpha_bins = np.arange(-2.0, 2.1, 0.1)
    alpha_bin_centers = alpha_bins[:-1] + 0.05

    for freq_idx, freq in enumerate(frequencies):
        freq_data = data_filtered[np.abs(data_filtered['Frequency'] - freq) < 0.01]

        # Mix all window sizes together
        alphas = freq_data['Alpha_Total'].values
        mses = freq_data['MSE_Homomorphic'].values

        # Filter valid MSE values
        valid_mask = mses > 0
        alphas = alphas[valid_mask]
        mses = mses[valid_mask]

        if len(alphas) > 10:
            mse_medians = []
            alpha_centers = []

            # Use explicit bins like boxplots
            for bin_idx, (bin_start, bin_end) in enumerate(zip(alpha_bins[:-1], alpha_bins[1:])):
                bin_center = alpha_bin_centers[bin_idx]

                # Get MSE values in this alpha bin
                mask = (alphas >= bin_start) & (alphas < bin_end)
                bin_mses = mses[mask]

                if len(bin_mses) > 0:
                    mse_medians.append(np.median(bin_mses))
                    alpha_centers.append(bin_center)

            # Store for plotting
            if len(alpha_centers) > 0:
                all_plot_data.append({
                    'freq': freq,
                    'alphas': alpha_centers,
                    'mses': mse_medians,
                    'color': colors[freq_idx]
                })

                # Update global range
                global_min = min(global_min, np.min(mse_medians))
                global_max = max(global_max, np.max(mse_medians))

    # Add padding to y-range
    if global_min != float('inf'):
        y_range = global_max - global_min
        global_min = max(0, global_min - 0.05 * y_range)
        global_max = global_max + 0.05 * y_range

    # Plot all frequencies
    for plot_data in all_plot_data:
        ax.plot(plot_data['alphas'], plot_data['mses'], '-',
                linewidth=3, color=plot_data['color'], alpha=0.85,
                label=f"{plot_data['freq']:.0f} Hz", markersize=8)

    ax.set_xlabel('Alpha (α)', fontsize=22, fontweight='bold')
    ax.set_ylabel('MSE (median)', fontsize=22, fontweight='bold')
    ax
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim(-2, 2)
    if global_min != float('inf'):
        ax.set_ylim(global_min, global_max)

    # Legend with multiple columns for many frequencies
    ncol = min(3, (n_freqs + 2) // 3)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.95, ncol=ncol)

    plt.tight_layout()

    filename = output_dir / 'homomorphic_mse_vs_alpha_all_frequencies.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()


def plot_homomorphic_mse_vs_alpha_all_frequencies_with_iqr(data, output_dir):
    """
    Plot median MSE vs Alpha with all frequencies in one plot, including IQR shading.
    X-axis: Alpha values (-2 to 2), Y-axis: MSE (linear scale)
    Each frequency gets a different colored line and shaded IQR region.
    Only selected frequencies: 100, 220, 440, 1000, 2000, 4000, 6000, 8000, 10000 Hz
    """
    # Filter for the focus window sizes
    focus_window_sizes = [512, 1024, 2048, 4096]
    data_filtered = data[data['WindowSize'].isin(focus_window_sizes)]

    if len(data_filtered) == 0:
        print("⚠ No data found for focus window sizes. Using all available window sizes.")
        data_filtered = data

    # Filter for selected frequencies only
    selected_frequencies = [100, 220, 440, 1000, 2000, 4000, 6000, 8000, 10000]
    frequencies = sorted([f for f in data_filtered['Frequency'].unique() if f in selected_frequencies])
    n_freqs = len(frequencies)

    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Color scheme for different frequencies
    colors = plt.cm.tab20(np.linspace(0, 1, n_freqs))

    # Determine global y-range
    global_min = float('inf')
    global_max = float('-inf')

    all_plot_data = []

    # Define alpha bins (0.1 width) from -2 to 2 - same as boxplots
    alpha_bins = np.arange(-2.0, 2.1, 0.1)
    alpha_bin_centers = alpha_bins[:-1] + 0.05

    for freq_idx, freq in enumerate(frequencies):
        freq_data = data_filtered[np.abs(data_filtered['Frequency'] - freq) < 0.01]

        # Mix all window sizes together
        alphas = freq_data['Alpha_Total'].values
        mses = freq_data['MSE_Homomorphic'].values

        # Filter valid MSE values
        valid_mask = mses > 0
        alphas = alphas[valid_mask]
        mses = mses[valid_mask]

        if len(alphas) > 10:
            mse_medians = []
            mse_p25 = []
            mse_p75 = []
            alpha_centers = []

            # Use explicit bins like boxplots
            for bin_idx, (bin_start, bin_end) in enumerate(zip(alpha_bins[:-1], alpha_bins[1:])):
                bin_center = alpha_bin_centers[bin_idx]

                # Get MSE values in this alpha bin
                mask = (alphas >= bin_start) & (alphas < bin_end)
                bin_mses = mses[mask]

                if len(bin_mses) > 0:
                    mse_medians.append(np.median(bin_mses))
                    mse_p25.append(np.percentile(bin_mses, 25))
                    mse_p75.append(np.percentile(bin_mses, 75))
                    alpha_centers.append(bin_center)

            # Store for plotting
            if len(alpha_centers) > 0:
                all_plot_data.append({
                    'freq': freq,
                    'alphas': alpha_centers,
                    'mses': mse_medians,
                    'p25': mse_p25,
                    'p75': mse_p75,
                    'color': colors[freq_idx]
                })

                # Update global range (include IQR in range calculation)
                global_min = min(global_min, np.min(mse_p25))
                global_max = max(global_max, np.max(mse_p75))

    # Add padding to y-range
    if global_min != float('inf'):
        y_range = global_max - global_min
        global_min = max(0, global_min - 0.05 * y_range)
        global_max = global_max + 0.05 * y_range

    # Plot all frequencies with IQR shading
    for plot_data in all_plot_data:
        # Plot shaded IQR region
        ax.fill_between(plot_data['alphas'], plot_data['p25'], plot_data['p75'],
                        alpha=0.2, color=plot_data['color'])

        # Plot median line
        ax.plot(plot_data['alphas'], plot_data['mses'], '-',
                linewidth=3, color=plot_data['color'], alpha=0.85,
                label=f"{plot_data['freq']:.0f} Hz", markersize=8)

    ax.set_xlabel('Alpha (α)', fontsize=22, fontweight='bold')
    ax.set_ylabel('MSE (median with IQR)', fontsize=22, fontweight='bold')
    ax
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim(-2, 2)
    if global_min != float('inf'):
        ax.set_ylim(global_min, global_max)

    # Legend with multiple columns for many frequencies
    ncol = min(3, (n_freqs + 2) // 3)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.95, ncol=ncol)

    plt.tight_layout()

    filename = output_dir / 'homomorphic_mse_vs_alpha_all_frequencies_iqr.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()


def plot_homomorphic_mse_vs_alpha(data, output_dir):
    """Plot homomorphic test MSE vs alpha for each frequency - DEPRECATED, use KDE and boxplot versions."""
    pass  # Keeping for backward compatibility but not used
    """Plot homomorphic test MSE vs alpha for each frequency."""
    frequencies = sorted(data['Frequency'].unique())
    alphas = sorted(data['Alpha_Total'].unique())
    alphas = [round(a, 1) for a in alphas]
    alphas = sorted(list(set(alphas)))

    n_freqs = len(frequencies)
    fig, axes = plt.subplots(n_freqs, 1, figsize=(12, max(1.2 * n_freqs, 6)))

    if n_freqs == 1:
        axes = [axes]

    # Find global range
    all_mses = []
    for freq in frequencies:
        freq_data = data[np.abs(data['Frequency'] - freq) < 0.01]
        for alpha in alphas:
            alpha_data = freq_data[np.abs(freq_data['Alpha_Total'] - alpha) < 0.05]
            if len(alpha_data) > 0:
                all_mses.append(alpha_data['MSE_Homomorphic'].mean())

    if all_mses:
        global_min = max(0, min(all_mses) * 0.5)
        global_max = max(all_mses) * 1.5
    else:
        global_min, global_max = 0, 1e-10

    bar_color = '#2ca02c'  # Green for homomorphic
    x_pos = np.arange(len(alphas))
    bar_width = 0.7

    for idx, freq in enumerate(frequencies):
        ax = axes[idx]
        freq_data = data[np.abs(data['Frequency'] - freq) < 0.01]

        means = []
        for alpha in alphas:
            alpha_data = freq_data[np.abs(freq_data['Alpha_Total'] - alpha) < 0.05]
            if len(alpha_data) > 0:
                means.append(alpha_data['MSE_Homomorphic'].mean())
            else:
                means.append(0)

        ax.bar(x_pos, means, bar_width, color=bar_color, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.text(0.98, 0.95, f'{freq:.0f} Hz',
                transform=ax.transAxes, fontsize=13, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        ax.set_xticks(x_pos)

        if idx == n_freqs - 1:
            ax.set_xticklabels([f'{a:.1f}' for a in alphas], fontsize=11)
            ax.set_xlabel('Alpha Total (α)', fontsize=13, fontweight='bold')
        else:
            ax.set_xticklabels([])

        if idx == n_freqs // 2:
            ax.set_ylabel('Homomorphic MSE', fontsize=13, fontweight='bold')

        ax.set_yscale('log')
        ax
        ax.tick_params(axis='y', which='major', labelsize=11)
        ax.set_ylim(global_min, global_max)

    fig.suptitle('Homomorphic Property Test: MSE vs Alpha', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    filename = output_dir / 'homomorphic_mse_vs_alpha.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()


def plot_fft_comparison_mse(data, output_dir):
    """Plot FFT comparison MSE vs frequency for each window size."""
    window_sizes = sorted(data['WindowSize'].unique())
    frequencies = sorted(data['Frequency'].unique())

    # Limit to reasonable number of window sizes for clarity
    if len(window_sizes) > 8:
        # Sample evenly
        step = len(window_sizes) // 8
        window_sizes = window_sizes[::step]

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(window_sizes)))

    for idx, ws in enumerate(window_sizes):
        ws_data = data[data['WindowSize'] == ws]

        mses = []
        for freq in frequencies:
            freq_data = ws_data[np.abs(ws_data['Frequency'] - freq) < 0.01]
            if len(freq_data) > 0:
                mses.append(freq_data['MSE_Magnitude'].mean())
            else:
                mses.append(np.nan)

        ax.plot(frequencies, mses, 'o-', label=f'WS={ws}',
                linewidth=2, markersize=6, color=colors[idx], alpha=0.8)

    ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Magnitude MSE', fontsize=14, fontweight='bold')
    ax.set_title('FFT Comparison: FRFT(α=1) vs FFT Magnitude Error', fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=10, loc='best', framealpha=0.95, ncol=2)

    plt.tight_layout()
    filename = output_dir / 'fft_comparison_mse.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()


def plot_fft_comparison_correlation(data, output_dir):
    """Plot FFT comparison correlation vs frequency."""
    window_sizes = sorted(data['WindowSize'].unique())
    frequencies = sorted(data['Frequency'].unique())

    if len(window_sizes) > 8:
        step = len(window_sizes) // 8
        window_sizes = window_sizes[::step]

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(window_sizes)))

    for idx, ws in enumerate(window_sizes):
        ws_data = data[data['WindowSize'] == ws]

        corrs = []
        for freq in frequencies:
            freq_data = ws_data[np.abs(ws_data['Frequency'] - freq) < 0.01]
            if len(freq_data) > 0:
                corrs.append(freq_data['Correlation_Mag'].mean())
            else:
                corrs.append(np.nan)

        ax.plot(frequencies, corrs, 'o-', label=f'WS={ws}',
                linewidth=2, markersize=6, color=colors[idx], alpha=0.8)

    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
               label='Perfect Correlation', alpha=0.7)

    ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Magnitude Spectrum Correlation', fontsize=14, fontweight='bold')
    ax.set_title('FFT Comparison: FRFT(α=1) vs FFT Correlation', fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=10, loc='best', framealpha=0.95, ncol=2)
    ax.set_ylim(0.95, 1.005)

    plt.tight_layout()
    filename = output_dir / 'fft_comparison_correlation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()


def plot_mss_grid_heatmap(data, freq, output_dir):
    """
    Create MSS and MSE heatmaps for grid analysis.

    Args:
        data: DataFrame with Alpha1, Alpha2, MSS_Homomorphic, MSE_Homomorphic columns
        freq: Frequency value (None for aggregated)
        output_dir: Output directory path
    """
    # Pivot data to create 2D grids for both MSS and MSE
    pivot_mss = data.pivot(index='Alpha2', columns='Alpha1', values='MSS_Homomorphic')
    pivot_mse = data.pivot(index='Alpha2', columns='Alpha1', values='MSE_Homomorphic')

    # Sort indices to ensure proper ordering (ascending for both)
    pivot_mss = pivot_mss.sort_index(ascending=True)
    pivot_mss = pivot_mss[sorted(pivot_mss.columns)]
    pivot_mse = pivot_mse.sort_index(ascending=True)
    pivot_mse = pivot_mse[sorted(pivot_mse.columns)]

    # Get alpha values for labels
    alpha1_values = pivot_mss.columns.values
    alpha2_values = pivot_mss.index.values

    print(f"  Data shape: {pivot_mss.shape}")
    print(f"  α₁ range: [{alpha1_values[0]:.1f}, {alpha1_values[-1]:.1f}]")
    print(f"  α₂ range: [{alpha2_values[0]:.1f}, {alpha2_values[-1]:.1f}]")

    n_rows, n_cols = pivot_mss.shape
    tick_every = 4

    # Prepare tick positions
    x_tick_indices = np.arange(0, n_cols, tick_every)
    if (n_cols - 1) not in x_tick_indices:
        x_tick_indices = np.append(x_tick_indices, n_cols - 1)
    x_tick_labels = [f'{alpha1_values[i]:.1f}' for i in x_tick_indices]

    y_tick_indices = np.arange(0, n_rows, tick_every)
    if (n_rows - 1) not in y_tick_indices:
        y_tick_indices = np.append(y_tick_indices, n_rows - 1)
    y_tick_labels = [f'{alpha2_values[i]:.1f}' for i in y_tick_indices]

    # === MSS HEATMAP ===
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(pivot_mss.values, cmap='viridis', aspect='auto',
                   interpolation='nearest', origin='lower')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'MSS($F_{\alpha_1+\alpha_2}$, $F_{\alpha_2} \cdot F_{\alpha_1}$)',
                   fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels(y_tick_labels)

    ax.set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
    ax.set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.05, alpha=0.1)

    plt.tight_layout(pad=1.0)

    if freq is None:
        filename = output_dir / 'heatmap_mss_all_frequencies.png'
    else:
        filename = output_dir / f'heatmap_mss_freq_{int(freq)}Hz.png'

    plt.savefig(filename, dpi=300)
    print(f"  → {filename}")
    plt.close()

    # === MSE HEATMAP ===
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(pivot_mse.values, cmap='viridis', aspect='auto',
                   interpolation='nearest', origin='lower')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'MSE($F_{\alpha_1+\alpha_2}$, $F_{\alpha_2} \cdot F_{\alpha_1}$)',
                   fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels(y_tick_labels)

    ax.set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
    ax.set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.05, alpha=0.1)

    plt.tight_layout(pad=1.0)

    if freq is None:
        filename = output_dir / 'heatmap_mse_all_frequencies.png'
    else:
        filename = output_dir / f'heatmap_mse_freq_{int(freq)}Hz.png'

    plt.savefig(filename, dpi=300)
    print(f"  → {filename}")
    plt.close()



def plot_mss_vs_alpha_wrapped(results_df, output_dir):
    """Plot MSS and MSE vs wrapped alpha for all frequencies."""

    frequencies = sorted(results_df['Frequency'].unique())

    # === MSS PLOT ===
    fig, ax = plt.subplots(figsize=(12, 6))

    for freq in frequencies:
        freq_data = results_df[results_df['Frequency'] == freq]
        # Group by alpha and compute mean MSS
        alpha_grouped = freq_data.groupby('Alpha')['MSS_Homomorphic'].mean().sort_index()
        ax.plot(alpha_grouped.index, alpha_grouped.values, 'o-',
                linewidth=2, markersize=4, alpha=0.7, label=f'{int(freq)} Hz')

    ax.set_xlabel(r'$\alpha_1 + \alpha_2$', fontsize=20, fontweight='bold')
    ax.set_ylabel(r'MSS($F_{\alpha_1+\alpha_2}$, $F_{\alpha_2} \cdot F_{\alpha_1}$)', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=14, ncol=2)
    ax
    plt.tight_layout()

    filename = output_dir / 'mss_vs_alpha_wrapped.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()

    # === MSE PLOT ===
    fig, ax = plt.subplots(figsize=(12, 6))

    for freq in frequencies:
        freq_data = results_df[results_df['Frequency'] == freq]
        # Group by alpha and compute mean MSE
        alpha_grouped = freq_data.groupby('Alpha')['MSE_Homomorphic'].mean().sort_index()
        ax.plot(alpha_grouped.index, alpha_grouped.values, 'o-',
                linewidth=2, markersize=4, alpha=0.7, label=f'{int(freq)} Hz')

    ax.set_xlabel(r'$\alpha_1 + \alpha_2$', fontsize=20, fontweight='bold')
    ax.set_ylabel(r'MSE($F_{\alpha_1+\alpha_2}$, $F_{\alpha_2} \cdot F_{\alpha_1}$)', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=14, ncol=2)
    ax
    plt.tight_layout()

    filename = output_dir / 'mse_vs_alpha_wrapped.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()


def plot_mss_grid_heatmap_with_alpha_sum_highlights(data, output_dir):
    """
    Create MSS and MSE heatmaps with special highlighting for cells where α₁ + α₂ = 0, ±1, ±2, ±3.
    Only for aggregated data across all frequencies.

    Args:
        data: DataFrame with Alpha1, Alpha2, MSS_Homomorphic, MSE_Homomorphic columns
        output_dir: Output directory path
    """
    # Pivot data to create 2D grids for both MSS and MSE
    pivot_mss = data.pivot(index='Alpha2', columns='Alpha1', values='MSS_Homomorphic')
    pivot_mse = data.pivot(index='Alpha2', columns='Alpha1', values='MSE_Homomorphic')

    # Sort indices to ensure proper ordering (ascending for both)
    pivot_mss = pivot_mss.sort_index(ascending=True)
    pivot_mss = pivot_mss[sorted(pivot_mss.columns)]
    pivot_mse = pivot_mse.sort_index(ascending=True)
    pivot_mse = pivot_mse[sorted(pivot_mse.columns)]

    # Get alpha values for labels
    alpha1_values = pivot_mss.columns.values
    alpha2_values = pivot_mss.index.values

    n_rows, n_cols = pivot_mss.shape
    tick_every = 4

    # Prepare tick positions
    x_tick_indices = np.arange(0, n_cols, tick_every)
    if (n_cols - 1) not in x_tick_indices:
        x_tick_indices = np.append(x_tick_indices, n_cols - 1)
    x_tick_labels = [f'{alpha1_values[i]:.1f}' for i in x_tick_indices]

    y_tick_indices = np.arange(0, n_rows, tick_every)
    if (n_rows - 1) not in y_tick_indices:
        y_tick_indices = np.append(y_tick_indices, n_rows - 1)
    y_tick_labels = [f'{alpha2_values[i]:.1f}' for i in y_tick_indices]

    # Create masks for each special alpha sum value with different colors
    # Use a dictionary to store which cells belong to which alpha sum
    alpha_sum_map = {}
    for i, a2 in enumerate(alpha2_values):
        for j, a1 in enumerate(alpha1_values):
            alpha_sum = a1 + a2
            # Check each target value
            for target in [0, 1, -1, 2, -2, 3, -3]:
                if abs(alpha_sum - target) < 0.05:
                    alpha_sum_map[(i, j)] = target
                    break

    # Define colors for each alpha sum value (same color for ± pairs)
    color_map = {
        0: 'red',
        1: 'blue',
        -1: 'blue',
        2: 'green',
        -2: 'green',
        3: 'orange',
        -3: 'orange'
    }

    # === MSS HEATMAP WITH HIGHLIGHTS ===
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(pivot_mss.values, cmap='viridis', aspect='auto',
                   interpolation='nearest', origin='lower')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'MSS($F_{\alpha_1+\alpha_2}$, $F_{\alpha_2} \cdot F_{\alpha_1}$)',
                   fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    # Overlay highlights on special cells with different colors
    for (i, j), target in alpha_sum_map.items():
        color = color_map[target]
        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             fill=False, edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)

    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels(y_tick_labels)

    ax.set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
    ax.set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.05, alpha=0.1)

    # Add legend with combined entries for ± pairs
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='red', linewidth=2.5, label=r'$\alpha_1 + \alpha_2 = 0$'),
        Patch(facecolor='none', edgecolor='blue', linewidth=2.5, label=r'$\alpha_1 + \alpha_2 = \pm 1$'),
        Patch(facecolor='none', edgecolor='green', linewidth=2.5, label=r'$\alpha_1 + \alpha_2 = \pm 2$'),
        Patch(facecolor='none', edgecolor='orange', linewidth=2.5, label=r'$\alpha_1 + \alpha_2 = \pm 3$'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

    plt.tight_layout(pad=1.0)

    filename = output_dir / 'heatmap_mss_all_frequencies_highlighted.png'
    plt.savefig(filename, dpi=300)
    print(f"  → {filename}")
    plt.close()

    # === MSE HEATMAP WITH HIGHLIGHTS ===
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(pivot_mse.values, cmap='viridis', aspect='auto',
                   interpolation='nearest', origin='lower')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'MSE($F_{\alpha_1+\alpha_2}$, $F_{\alpha_2} \cdot F_{\alpha_1}$)',
                   fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    # Overlay highlights on special cells with different colors
    for (i, j), target in alpha_sum_map.items():
        color = color_map[target]
        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             fill=False, edgecolor=color, linewidth=2.5)
        ax.add_patch(rect)

    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels(y_tick_labels)

    ax.set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
    ax.set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.05, alpha=0.1)

    # Add legend
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout(pad=1.0)

    filename = output_dir / 'heatmap_mse_all_frequencies_highlighted.png'
    plt.savefig(filename, dpi=300)
    print(f"  → {filename}")
    plt.close()


def plot_best_worst_case_examples(data, base_dir, output_dir):
    """
    Generate waveform comparison plots for worst, medium, and best MSS cases.

    Creates plots showing the direct FRFT vs composed FRFT for:
    - 20 worst cases (highest MSS)
    - 20 medium cases (around median MSS)
    - 20 best cases (lowest MSS, excluding zeros)

    Args:
        data: DataFrame with all grid results
        base_dir: Base directory containing WAV files
        output_dir: Output directory for comparison plots
    """
    import scipy.io.wavfile as wavfile

    print("\n  Generating worst/medium/best case comparison plots...")

    # Create subfolder for examples
    examples_dir = output_dir / 'case_examples'
    examples_dir.mkdir(exist_ok=True)

    # Sort by MSS to find worst/medium/best cases
    sorted_data = data.sort_values('MSS_Homomorphic', ascending=False)

    # Get 20 worst cases (highest MSS)
    worst_cases = sorted_data.head(20)

    # Get 20 medium cases (around median)
    median_idx = len(sorted_data) // 2
    medium_cases = sorted_data.iloc[median_idx-10:median_idx+10]

    # Get 20 best cases (lowest non-zero MSS)
    non_zero = data[data['MSS_Homomorphic'] > 1e-6].sort_values('MSS_Homomorphic', ascending=True)
    best_cases = non_zero.head(20)

    print(f"    Worst cases:  MSS range [{worst_cases['MSS_Homomorphic'].min():.6f}, {worst_cases['MSS_Homomorphic'].max():.6f}]")
    print(f"    Medium cases: MSS range [{medium_cases['MSS_Homomorphic'].min():.6f}, {medium_cases['MSS_Homomorphic'].max():.6f}]")
    print(f"    Best cases:   MSS range [{best_cases['MSS_Homomorphic'].min():.6f}, {best_cases['MSS_Homomorphic'].max():.6f}]")

    # Helper function to format alpha for filename
    def format_alpha_filename(alpha):
        return f"{alpha:.1f}".replace('-', 'm').replace('.', 'p')

    # Helper function to load and plot comparison
    def plot_comparison(case_data, case_name, case_num):
        freq = case_data['Frequency']
        a1 = case_data['Alpha1']
        a2 = case_data['Alpha2']
        mss = case_data['MSS_Homomorphic']
        mse = case_data['MSE_Homomorphic']
        alpha_wrapped = case_data['Alpha']

        # Construct file paths
        freq_str = f"freq_{int(freq)}"
        a1_str = format_alpha_filename(a1)
        a2_str = format_alpha_filename(a2)

        alpha_file = base_dir / freq_str / f"a1_{a1_str}_a2_{a2_str}_alpha.wav"
        composed_file = base_dir / freq_str / f"a1_{a1_str}_a2_{a2_str}_composed.wav"

        # Check if files exist
        if not alpha_file.exists() or not composed_file.exists():
            print(f"    ⚠ Files not found for {case_name} #{case_num}")
            return False

        # Load audio files
        sr1, alpha_signal = wavfile.read(alpha_file)
        sr2, composed_signal = wavfile.read(composed_file)

        # Normalize if needed
        if alpha_signal.dtype == np.int16:
            alpha_signal = alpha_signal.astype(np.float32) / 32768.0
            composed_signal = composed_signal.astype(np.float32) / 32768.0
        elif alpha_signal.dtype == np.int32:
            alpha_signal = alpha_signal.astype(np.float32) / 2147483648.0
            composed_signal = composed_signal.astype(np.float32) / 2147483648.0

        # Create time axis
        time = np.arange(len(alpha_signal)) / sr1

        # Compute difference
        diff = alpha_signal - composed_signal

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

        # Plot 1: Direct FRFT
        ax1.plot(time, alpha_signal, 'b-', linewidth=0.5, alpha=0.8)
        ax1.set_ylabel(r'$F_{\alpha_1+\alpha_2}$', fontsize=14, fontweight='bold')
        ax1.set_title(f'{case_name} #{case_num}: {freq:.0f} Hz, α₁={a1:.1f}, α₂={a2:.1f}, α={alpha_wrapped:.1f}',
                      fontsize=16, fontweight='bold')
        ax1
        ax1.set_xlim([time[0], time[-1]])

        # Plot 2: Composed FRFT
        ax2.plot(time, composed_signal, 'g-', linewidth=0.5, alpha=0.8)
        ax2.set_ylabel(r'$F_{\alpha_2} \cdot F_{\alpha_1}$', fontsize=14, fontweight='bold')
        ax2
        ax2.set_xlim([time[0], time[-1]])

        # Plot 3: Difference
        ax3.plot(time, diff, 'r-', linewidth=0.5, alpha=0.8)
        ax3.set_ylabel('Difference', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        ax3
        ax3.set_xlim([time[0], time[-1]])

        # Add metrics text box
        textstr = f'MSS = {mss:.6f}\nMSE = {mse:.6e}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)

        plt.tight_layout()

        # Save figure
        case_type = case_name.lower().replace(" ", "_")
        filename = examples_dir / f'{case_type}_{case_num:02d}_freq{int(freq)}_a1_{a1_str}_a2_{a2_str}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return True

    # Generate plots for all cases
    plot_count = 0

    print("    Generating worst case plots...")
    for i, (idx, case) in enumerate(worst_cases.iterrows(), 1):
        if plot_comparison(case, 'Worst Case', i):
            plot_count += 1

    print("    Generating medium case plots...")
    for i, (idx, case) in enumerate(medium_cases.iterrows(), 1):
        if plot_comparison(case, 'Medium Case', i):
            plot_count += 1

    print("    Generating best case plots...")
    for i, (idx, case) in enumerate(best_cases.iterrows(), 1):
        if plot_comparison(case, 'Best Case', i):
            plot_count += 1

    print(f"    → Generated {plot_count} case example plots in {examples_dir}/")

    return plot_count


def plot_commutativity_histograms(comm_data, output_dir, window_size=None):
    """
    Plot histograms for commutativity analysis (MSS and MSE).

    Args:
        comm_data: DataFrame with MSS_Commutativity and MSE_Commutativity columns
        output_dir: Output directory path
        window_size: Window size (for windowed mode) or None
    """
    # MSS Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(comm_data['MSS_Commutativity'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel(r'MSS($F_{\alpha_1} \cdot F_{\alpha_2}$, $F_{\alpha_2} \cdot F_{\alpha_1}$)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.axvline(comm_data['MSS_Commutativity'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean = {comm_data["MSS_Commutativity"].mean():.4f}')
    ax.axvline(comm_data['MSS_Commutativity'].median(), color='green', linestyle='--',
               linewidth=2, label=f'Median = {comm_data["MSS_Commutativity"].median():.4f}')
    ax.legend(fontsize=12)
    ax
    ax.tick_params(axis='both', which='major', labelsize=12)

    if window_size:
        ax.set_title(f'Commutativity MSS Distribution - Window {window_size}', fontsize=16, fontweight='bold')

    plt.tight_layout()

    if window_size:
        mss_hist_file = output_dir / f"commutativity_mss_histogram_win_{window_size}.png"
    else:
        mss_hist_file = output_dir / "commutativity_mss_histogram.png"
    plt.savefig(mss_hist_file, dpi=300)
    print(f"  → {mss_hist_file}")
    plt.close()

    # MSE Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(comm_data['MSE_Commutativity'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel(r'MSE($F_{\alpha_1} \cdot F_{\alpha_2}$, $F_{\alpha_2} \cdot F_{\alpha_1}$)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')
    ax.axvline(comm_data['MSE_Commutativity'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean = {comm_data["MSE_Commutativity"].mean():.4e}')
    ax.axvline(comm_data['MSE_Commutativity'].median(), color='green', linestyle='--',
               linewidth=2, label=f'Median = {comm_data["MSE_Commutativity"].median():.4e}')
    ax.legend(fontsize=12)
    ax
    ax.tick_params(axis='both', which='major', labelsize=12)

    if window_size:
        ax.set_title(f'Commutativity MSE Distribution - Window {window_size}', fontsize=16, fontweight='bold')

    plt.tight_layout()

    if window_size:
        mse_hist_file = output_dir / f"commutativity_mse_histogram_win_{window_size}.png"
    else:
        mse_hist_file = output_dir / "commutativity_mse_histogram.png"
    plt.savefig(mse_hist_file, dpi=300)
    print(f"  → {mse_hist_file}")
    plt.close()


def plot_window_size_comparison(results_df, output_dir):
    """Plot comparison of MSS loss across different window sizes."""

    if 'Window' not in results_df.columns:
        print("  Skipping window comparison (not in windowed mode)")
        return 0

    window_sizes = sorted(results_df['Window'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    plot_count = 0
    for idx, freq in enumerate(sorted(results_df['Frequency'].unique())):
        if idx >= 4:
            break

        ax = axes[idx]
        freq_data = results_df[results_df['Frequency'] == freq]

        # Plot MSS for each window size
        for window_size in window_sizes:
            window_data = freq_data[freq_data['Window'] == window_size]

            # Group by alpha (sum of alpha1 and alpha2) and compute mean
            alpha_grouped = window_data.groupby('Alpha')['MSS_Homomorphic'].mean().sort_index()

            ax.plot(alpha_grouped.index, alpha_grouped.values,
                    marker='o', markersize=4, linewidth=2,
                    label=f'Window {window_size}', alpha=0.8)

        ax.set_xlabel(r'$\alpha$ (wrapped)', fontsize=16, fontweight='bold')
        ax.set_ylabel('MSS Loss', fontsize=16, fontweight='bold')
        ax.set_title(f'{int(freq)} Hz', fontsize=18, fontweight='bold')
        ax
        ax.legend(fontsize=12, framealpha=0.9)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlim(-2, 2)
        plot_count += 1

    plt.suptitle('MSS Loss vs Alpha: Window Size Comparison',
                 fontsize=20, fontweight='bold', y=1.00)
    plt.tight_layout()

    filename = output_dir / 'window_size_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → comparisons/{filename.name}")
    plt.close()

    return 1


def add_alpha_sum_highlights(ax, alpha1_values, alpha2_values, extent):
    """
    Overlay box highlights on heatmap for cells where α₁ + α₂ = 0, ±1, ±2, ±3.

    Args:
        ax: Matplotlib axis
        alpha1_values: Array of α₁ values (columns)
        alpha2_values: Array of α₂ values (rows)
        extent: [left, right, bottom, top] extent used in imshow
    """
    # Define the target sums and their visual properties
    # Group ±1, ±2, ±3 together with same style
    target_groups = {
        0: {'color': 'white', 'linewidth': 3.0, 'linestyle': '-', 'label': r'$\alpha_1 + \alpha_2 = 0$'},
        1: {'color': 'cyan', 'linewidth': 2.5, 'linestyle': '--', 'label': r'$\alpha_1 + \alpha_2 = \pm 1$'},
        2: {'color': 'lime', 'linewidth': 2.0, 'linestyle': '-.', 'label': r'$\alpha_1 + \alpha_2 = \pm 2$'},
        3: {'color': 'magenta', 'linewidth': 1.8, 'linestyle': ':', 'label': r'$\alpha_1 + \alpha_2 = \pm 3$'},
    }

    # Map each target to its group style
    target_styles = {
        0: target_groups[0],
        1: target_groups[1],
        -1: target_groups[1],  # Same as +1
        2: target_groups[2],
        -2: target_groups[2],  # Same as +2
        3: target_groups[3],
        -3: target_groups[3],  # Same as +3
    }

    # Calculate step sizes
    alpha1_step = alpha1_values[1] - alpha1_values[0] if len(alpha1_values) > 1 else 0.1
    alpha2_step = alpha2_values[1] - alpha2_values[0] if len(alpha2_values) > 1 else 0.1

    # Track which legend entries we've added
    added_legend = {}

    # Find and highlight cells for each target sum
    for target, style in target_styles.items():
        # Find all cells where α₁ + α₂ ≈ target
        for a1 in alpha1_values:
            for a2 in alpha2_values:
                alpha_sum = a1 + a2

                # Check if this cell matches the target (within tolerance)
                # Use smaller tolerance for more precise matching
                tolerance = min(alpha1_step, alpha2_step) * 0.45
                if abs(alpha_sum - target) < tolerance:
                    # Calculate cell boundaries centered on alpha values
                    # These are the actual coordinates in data space
                    left = a1 - alpha1_step / 2
                    right = a1 + alpha1_step / 2
                    bottom = a2 - alpha2_step / 2
                    top = a2 + alpha2_step / 2

                    width = right - left
                    height = top - bottom

                    # Draw rectangle around this cell
                    rect = plt.Rectangle((left, bottom),
                                         width, height,
                                         fill=False,
                                         edgecolor=style['color'],
                                         linewidth=style['linewidth'],
                                         linestyle=style['linestyle'],
                                         alpha=0.95,
                                         zorder=10,
                                         transform=ax.transData)  # Use data coordinates
                    ax.add_patch(rect)

        # Add to legend only once per group (not for each ± pair)
        group_key = abs(target)  # Use absolute value as key
        if group_key not in added_legend:
            # Get the style for the group
            group_style = target_groups[group_key]
            added_legend[group_key] = plt.Line2D([0], [0],
                                                 color=group_style['color'],
                                                 linewidth=group_style['linewidth'],
                                                 linestyle=group_style['linestyle'],
                                                 label=group_style['label'])

    # Add legend with unique entries only - LARGER font
    if added_legend:
        legend_handles = [added_legend[k] for k in sorted(added_legend.keys())]
        ax.legend(handles=legend_handles, loc='upper right',
                  fontsize=14, framealpha=0.95, edgecolor='white', fancybox=True)


def plot_window_size_statistics(results_df, output_dir):
    """Plot statistics comparing different window sizes."""

    if 'Window' not in results_df.columns:
        print("  Skipping window statistics (not in windowed mode)")
        return 0

    window_sizes = sorted(results_df['Window'].unique())

    # Compute statistics for each window size
    stats = []
    for window_size in window_sizes:
        window_data = results_df[results_df['Window'] == window_size]
        stats.append({
            'Window': window_size,
            'Mean_MSS': window_data['MSS_Homomorphic'].mean(),
            'Median_MSS': window_data['MSS_Homomorphic'].median(),
            'Std_MSS': window_data['MSS_Homomorphic'].std(),
            'Mean_MSE': window_data['MSE_Homomorphic'].mean(),
            'Median_MSE': window_data['MSE_Homomorphic'].median(),
        })

    stats_df = pd.DataFrame(stats)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSS Mean
    axes[0, 0].bar(range(len(window_sizes)), stats_df['Mean_MSS'], color='steelblue', alpha=0.8)
    axes[0, 0].set_xlabel('Window Size', fontsize=16, fontweight='bold')
    axes[0, 0].set_ylabel('Mean MSS Loss', fontsize=16, fontweight='bold')
    axes[0, 0].set_title('Mean MSS Loss by Window Size', fontsize=18, fontweight='bold')
    axes[0, 0].set_xticks(range(len(window_sizes)))
    axes[0, 0].set_xticklabels(window_sizes)
    axes[0, 0].tick_params(axis='both', labelsize=14)
    axes[0, 0]

    # MSS Std
    axes[0, 1].bar(range(len(window_sizes)), stats_df['Std_MSS'], color='coral', alpha=0.8)
    axes[0, 1].set_xlabel('Window Size', fontsize=16, fontweight='bold')
    axes[0, 1].set_ylabel('Std MSS Loss', fontsize=16, fontweight='bold')
    axes[0, 1].set_title('Standard Deviation of MSS Loss', fontsize=18, fontweight='bold')
    axes[0, 1].set_xticks(range(len(window_sizes)))
    axes[0, 1].set_xticklabels(window_sizes)
    axes[0, 1].tick_params(axis='both', labelsize=14)
    axes[0, 1]

    # MSE Mean
    axes[1, 0].bar(range(len(window_sizes)), stats_df['Mean_MSE'], color='forestgreen', alpha=0.8)
    axes[1, 0].set_xlabel('Window Size', fontsize=16, fontweight='bold')
    axes[1, 0].set_ylabel('Mean MSE Loss', fontsize=16, fontweight='bold')
    axes[1, 0].set_title('Mean MSE Loss by Window Size', fontsize=18, fontweight='bold')
    axes[1, 0].set_xticks(range(len(window_sizes)))
    axes[1, 0].set_xticklabels(window_sizes)
    axes[1, 0].tick_params(axis='both', labelsize=14)
    axes[1, 0]

    # Box plot of MSS distribution
    mss_by_window = [results_df[results_df['Window'] == w]['MSS_Homomorphic'].values
                     for w in window_sizes]
    bp = axes[1, 1].boxplot(mss_by_window, tick_labels=window_sizes, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    axes[1, 1].set_xlabel('Window Size', fontsize=16, fontweight='bold')
    axes[1, 1].set_ylabel('MSS Loss', fontsize=16, fontweight='bold')
    axes[1, 1].set_title('MSS Loss Distribution by Window Size', fontsize=18, fontweight='bold')
    axes[1, 1].tick_params(axis='both', labelsize=14)
    axes[1, 1]

    plt.tight_layout()

    filename = output_dir / 'window_size_statistics.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → comparisons/{filename.name}")
    plt.close()

    return 1


def plot_mixed_all_windows_per_frequency(results_df, output_dir):
    """
    Generate heatmaps with all windows mixed for each frequency.
    Averages MSS/MSE across all window sizes for each (alpha1, alpha2) pair.
    """
    if 'Window' not in results_df.columns:
        return 0

    plot_count = 0
    frequencies = sorted(results_df['Frequency'].unique())

    for freq in frequencies:
        freq_data = results_df[results_df['Frequency'] == freq]

        # Average across all windows
        agg_data = freq_data.groupby(['Alpha1', 'Alpha2']).agg({
            'MSS_Homomorphic': 'mean',
            'MSE_Homomorphic': 'mean'
        }).reset_index()

        pivot_mss = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSS_Homomorphic')
        pivot_mse = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSE_Homomorphic')

        # Sort to ensure proper ordering
        pivot_mss = pivot_mss.sort_index(ascending=True)
        pivot_mss = pivot_mss[sorted(pivot_mss.columns)]
        pivot_mse = pivot_mse.sort_index(ascending=True)
        pivot_mse = pivot_mse[sorted(pivot_mse.columns)]

        # Get alpha values for proper extent
        alpha1_values = pivot_mss.columns.values
        alpha2_values = pivot_mss.index.values

        fig, axes = plt.subplots(1, 2, figsize=(20, 9))

        # Define extent for imshow
        extent = [alpha1_values[0], alpha1_values[-1],
                  alpha2_values[0], alpha2_values[-1]]

        # MSS heatmap
        im = axes[0].imshow(pivot_mss.values, cmap='viridis', aspect='auto',
                            interpolation='nearest', origin='lower',
                            extent=extent)
        cbar = plt.colorbar(im, ax=axes[0])
        cbar.set_label('MSS Loss (Avg over all windows)', fontsize=18, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
        axes[0].set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
        axes[0].set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
        axes[0].set_title(f'MSS Loss - {int(freq)} Hz - All Windows Mixed',
                          fontsize=20, fontweight='bold')
        axes[0].tick_params(axis='both', labelsize=16)
        axes[0]

        # Add alpha sum highlights
        add_alpha_sum_highlights(axes[0], alpha1_values, alpha2_values, extent)

        # MSE heatmap
        im = axes[1].imshow(pivot_mse.values, cmap='plasma', aspect='auto',
                            interpolation='nearest', origin='lower',
                            extent=extent)
        cbar = plt.colorbar(im, ax=axes[1])
        cbar.set_label('MSE Loss (Avg over all windows)', fontsize=18, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)
        axes[1].set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
        axes[1].set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
        axes[1].set_title(f'MSE Loss - {int(freq)} Hz - All Windows Mixed',
                          fontsize=20, fontweight='bold')
        axes[1].tick_params(axis='both', labelsize=16)
        axes[1]

        # Add alpha sum highlights
        add_alpha_sum_highlights(axes[1], alpha1_values, alpha2_values, extent)

        plt.tight_layout()
        filename = output_dir / f'heatmap_freq_{int(freq)}Hz_all_windows_mixed.png'
        plt.savefig(filename, dpi=300)
        print(f"    → per_frequency/{filename.name}")
        plt.close()
        plot_count += 1

    return plot_count


def plot_mixed_all_frequencies_per_window(results_df, output_dir):
    """
    Generate heatmaps with all frequencies mixed for each window size.
    Averages MSS/MSE across all frequencies for each (alpha1, alpha2) pair.
    """
    if 'Window' not in results_df.columns:
        return 0

    plot_count = 0
    window_sizes = sorted(results_df['Window'].unique())

    for window_size in window_sizes:
        window_data = results_df[results_df['Window'] == window_size]

        # Average across all frequencies
        agg_data = window_data.groupby(['Alpha1', 'Alpha2']).agg({
            'MSS_Homomorphic': 'mean',
            'MSE_Homomorphic': 'mean'
        }).reset_index()

        pivot_mss = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSS_Homomorphic')
        pivot_mse = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSE_Homomorphic')

        # Sort to ensure proper ordering
        pivot_mss = pivot_mss.sort_index(ascending=True)
        pivot_mss = pivot_mss[sorted(pivot_mss.columns)]
        pivot_mse = pivot_mse.sort_index(ascending=True)
        pivot_mse = pivot_mse[sorted(pivot_mse.columns)]

        # Get alpha values for proper extent
        alpha1_values = pivot_mss.columns.values
        alpha2_values = pivot_mss.index.values

        fig, axes = plt.subplots(1, 2, figsize=(20, 9))

        # Define extent for imshow
        extent = [alpha1_values[0], alpha1_values[-1],
                  alpha2_values[0], alpha2_values[-1]]

        # MSS heatmap
        im = axes[0].imshow(pivot_mss.values, cmap='viridis', aspect='auto',
                            interpolation='nearest', origin='lower',
                            extent=extent)
        cbar = plt.colorbar(im, ax=axes[0])
        cbar.set_label('MSS Loss (Avg over all frequencies)', fontsize=18, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)

        # Set ticks to align with cell centers
        tick_step = 4  # Show every 4th tick
        x_ticks = alpha1_values[::tick_step]
        y_ticks = alpha2_values[::tick_step]
        axes[0].set_xticks(x_ticks)
        axes[0].set_yticks(y_ticks)
        axes[0].set_xticklabels([f'{x:.1f}' for x in x_ticks])
        axes[0].set_yticklabels([f'{y:.1f}' for y in y_ticks])

        axes[0].set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
        axes[0].set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
        axes[0].set_title(f'MSS Loss - Window {window_size} - All Frequencies Mixed',
                          fontsize=20, fontweight='bold')
        axes[0].tick_params(axis='both', labelsize=16)
        axes[0]

        # MSE heatmap
        im = axes[1].imshow(pivot_mse.values, cmap='plasma', aspect='auto',
                            interpolation='nearest', origin='lower',
                            extent=extent)
        cbar = plt.colorbar(im, ax=axes[1])
        cbar.set_label('MSE Loss (Avg over all frequencies)', fontsize=18, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)

        # Set ticks to align with cell centers
        axes[1].set_xticks(x_ticks)
        axes[1].set_yticks(y_ticks)
        axes[1].set_xticklabels([f'{x:.1f}' for x in x_ticks])
        axes[1].set_yticklabels([f'{y:.1f}' for y in y_ticks])

        axes[1].set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
        axes[1].set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
        axes[1].set_title(f'MSE Loss - Window {window_size} - All Frequencies Mixed',
                          fontsize=20, fontweight='bold')
        axes[1].tick_params(axis='both', labelsize=16)
        axes[1]

        plt.tight_layout()
        filename = output_dir / f'heatmap_win_{window_size}_all_frequencies_mixed.png'
        plt.savefig(filename, dpi=300)
        print(f"    → mixed/{filename.name}")
        plt.close()
        plot_count += 1

    return plot_count


def plot_mixed_all_windows_and_frequencies(results_df, output_dir):
    """
    Generate heatmaps with everything mixed (all windows and all frequencies).
    Averages MSS/MSE across all windows and frequencies for each (alpha1, alpha2) pair.
    """
    if 'Window' not in results_df.columns:
        return 0

    # Average across ALL windows and ALL frequencies
    agg_data = results_df.groupby(['Alpha1', 'Alpha2']).agg({
        'MSS_Homomorphic': 'mean',
        'MSE_Homomorphic': 'mean'
    }).reset_index()

    pivot_mss = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSS_Homomorphic')
    pivot_mse = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSE_Homomorphic')

    # Sort to ensure proper ordering
    pivot_mss = pivot_mss.sort_index(ascending=True)
    pivot_mss = pivot_mss[sorted(pivot_mss.columns)]
    pivot_mse = pivot_mse.sort_index(ascending=True)
    pivot_mse = pivot_mse[sorted(pivot_mse.columns)]

    # Get alpha values for proper extent
    alpha1_values = pivot_mss.columns.values
    alpha2_values = pivot_mss.index.values

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Define extent for imshow
    extent = [alpha1_values[0], alpha1_values[-1],
              alpha2_values[0], alpha2_values[-1]]

    # MSS heatmap
    im = axes[0].imshow(pivot_mss.values, cmap='viridis', aspect='auto',
                        interpolation='nearest', origin='lower',
                        extent=extent)
    cbar = plt.colorbar(im, ax=axes[0])
    cbar.set_label('MSS Loss (Avg over all windows & frequencies)', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    # Set ticks to align with cell centers
    tick_step = 4  # Show every 4th tick
    x_ticks = alpha1_values[::tick_step]
    y_ticks = alpha2_values[::tick_step]
    axes[0].set_xticks(x_ticks)
    axes[0].set_yticks(y_ticks)
    axes[0].set_xticklabels([f'{x:.1f}' for x in x_ticks])
    axes[0].set_yticklabels([f'{y:.1f}' for y in y_ticks])

    axes[0].set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
    axes[0].set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
    axes[0].set_title('MSS Loss - All Windows & Frequencies Mixed',
                      fontsize=20, fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=16)
    axes[0]

    # MSE heatmap
    im = axes[1].imshow(pivot_mse.values, cmap='plasma', aspect='auto',
                        interpolation='nearest', origin='lower',
                        extent=extent)
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('MSE Loss (Avg over all windows & frequencies)', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    # Set ticks to align with cell centers
    axes[1].set_xticks(x_ticks)
    axes[1].set_yticks(y_ticks)
    axes[1].set_xticklabels([f'{x:.1f}' for x in x_ticks])
    axes[1].set_yticklabels([f'{y:.1f}' for y in y_ticks])

    axes[1].set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
    axes[1].set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
    axes[1].set_title('MSE Loss - All Windows & Frequencies Mixed',
                      fontsize=20, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=16)
    axes[1]

    plt.tight_layout()
    filename = output_dir / 'heatmap_all_windows_all_frequencies_mixed.png'
    plt.savefig(filename, dpi=300)
    print(f"    → mixed/{filename.name}")
    plt.close()

    return 1


def plot_mixed_commutativity_all_windows(comm_df, output_dir):
    """
    Generate commutativity histograms with all windows mixed.
    """
    if 'Window' not in comm_df.columns or len(comm_df) == 0:
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MSS histogram
    ax = axes[0]
    ax.hist(comm_df['MSS_Commutativity'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(comm_df['MSS_Commutativity'].mean(), color='red', linestyle='--',
               linewidth=2, label=f"Mean: {comm_df['MSS_Commutativity'].mean():.6f}")
    ax.axvline(comm_df['MSS_Commutativity'].median(), color='green', linestyle='--',
               linewidth=2, label=f"Median: {comm_df['MSS_Commutativity'].median():.6f}")
    ax.set_xlabel('MSS Commutativity Loss', fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold')
    ax.set_title('MSS Commutativity - All Windows Mixed', fontsize=18, fontweight='bold')
    ax.legend(fontsize=14, framealpha=0.9)
    ax.tick_params(axis='both', labelsize=14)
    ax

    # MSE histogram
    ax = axes[1]
    ax.hist(comm_df['MSE_Commutativity'], bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(comm_df['MSE_Commutativity'].mean(), color='red', linestyle='--',
               linewidth=2, label=f"Mean: {comm_df['MSE_Commutativity'].mean():.6e}")
    ax.axvline(comm_df['MSE_Commutativity'].median(), color='green', linestyle='--',
               linewidth=2, label=f"Median: {comm_df['MSE_Commutativity'].median():.6e}")
    ax.set_xlabel('MSE Commutativity Loss', fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold')
    ax.set_title('MSE Commutativity - All Windows Mixed', fontsize=18, fontweight='bold')
    ax.legend(fontsize=14, framealpha=0.9)
    ax.tick_params(axis='both', labelsize=14)
    ax

    plt.tight_layout()
    filename = output_dir / 'commutativity_histograms_all_windows_mixed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → commutativity/{filename.name}")
    plt.close()

    return 1


def main():
    parser = argparse.ArgumentParser(
        description='Generate FRFT test plots with new folder structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tests from current directory
  python3 plot_results.py
  
  # Process specific test directories
  python3 plot_results.py --reconstruction reconstruction_test
  python3 plot_results.py --homomorphic homomorphic_test
  python3 plot_results.py --fft fft_comparison
        """
    )

    parser.add_argument('--reconstruction', default='test_results/reconstruction_test',
                        help='Reconstruction test directory (default: reconstruction_test)')
    parser.add_argument('--homomorphic', default='test_results/homomorphic_test',
                        help='Homomorphic test directory (default: homomorphic_test)')
    parser.add_argument('--fft', default='fft_comparison',
                        help='FFT comparison test directory (default: fft_comparison)')
    parser.add_argument('--mss', default='test_results/homomorphism_mss_grid',
                        help='MSS grid analysis directory (default: homomorphism_mss_grid)')

    args = parser.parse_args()

    print("=" * 70)
    print("  FRFT Test Results Plotter - Updated Structure")
    print("=" * 70)
    print()

    total_plots = 0

    # Process Reconstruction Test
    recon_dir = Path(args.reconstruction)
    if recon_dir.exists():
        print(f"\n[Processing Reconstruction Test: {recon_dir}]")

        timing_file = recon_dir / 'timing_benchmarks.txt'
        results_file = recon_dir / 'results.txt'
        output_dir = recon_dir / 'analysis_plots'
        output_dir.mkdir(exist_ok=True)

        timing_data = load_timing_data(timing_file) if timing_file.exists() else None
        results_data = load_results_data(results_file) if results_file.exists() else None

        if timing_data is not None:
            print("  Generating timing plots...")
            plot_processing_time_and_complexity(timing_data, output_dir)
            plot_rtf(timing_data, output_dir)
            total_plots += 2

        if results_data is not None:
            print("  Generating accuracy plots...")
            plot_reconstruction_mse_vs_alpha(results_data, output_dir)
            total_plots += 1
    else:
        print(f"\n⚠ Skipping reconstruction test (directory not found: {recon_dir})")

    # Process Homomorphic Test
    homo_dir = Path(args.homomorphic)
    if homo_dir.exists():
        print(f"\n[Processing Homomorphic Test: {homo_dir}]")

        results_file = homo_dir / 'results.txt'
        output_dir = homo_dir / 'analysis_plots'
        output_dir.mkdir(exist_ok=True)

        results_data = load_results_data(results_file) if results_file.exists() else None

        if results_data is not None:
            print("  Generating homomorphic plots...")
            plot_homomorphic_mse_vs_alpha_kde(results_data, output_dir)
            plot_homomorphic_mse_vs_alpha_boxplots(results_data, output_dir)
            plot_homomorphic_mse_vs_alpha_all_frequencies(results_data, output_dir)
            plot_homomorphic_mse_vs_alpha_all_frequencies_with_iqr(results_data, output_dir)
            total_plots += 4
    else:
        print(f"\n⚠ Skipping homomorphic test (directory not found: {homo_dir})")

    # Process FFT Comparison Test
    fft_dir = Path(args.fft)
    if fft_dir.exists():
        print(f"\n[Processing FFT Comparison Test: {fft_dir}]")

        results_file = fft_dir / 'results.txt'
        output_dir = fft_dir / 'analysis_plots'
        output_dir.mkdir(exist_ok=True)

        results_data = load_results_data(results_file) if results_file.exists() else None

        if results_data is not None:
            print("  Generating FFT comparison plots...")
            plot_fft_comparison_mse(results_data, output_dir)
            plot_fft_comparison_correlation(results_data, output_dir)
            total_plots += 2
    else:
        print(f"\n⚠ Skipping FFT comparison test (directory not found: {fft_dir})")

    # Process MSS Grid Analysis
    mss_dir = Path(args.mss)
    if mss_dir.exists():
        print(f"\n[Processing MSS Grid Analysis: {mss_dir}]")

        is_windowed = "windowed" in str(mss_dir)
        output_dir = mss_dir / 'plots'
        output_dir.mkdir(exist_ok=True)

        if is_windowed:
            print("  Detected WINDOWED mode")

            # Find all window-specific result files
            result_files = sorted(mss_dir.glob("mss_grid_results_win_*.txt"))
            comm_files = sorted(mss_dir.glob("mss_commutativity_results_win_*.txt"))

            if not result_files:
                print("  ⚠ No windowed result files found")
            else:
                # Create subdirectories for organization
                per_window_dir = output_dir / 'per_window'
                per_freq_dir = output_dir / 'per_frequency'
                mixed_dir = output_dir / 'mixed'
                comparison_dir = output_dir / 'comparisons'
                commutativity_dir = output_dir / 'commutativity'

                per_window_dir.mkdir(exist_ok=True)
                per_freq_dir.mkdir(exist_ok=True)
                mixed_dir.mkdir(exist_ok=True)
                comparison_dir.mkdir(exist_ok=True)
                commutativity_dir.mkdir(exist_ok=True)

                # Load all results
                all_results = []
                for result_file in result_files:
                    data = load_results_data(result_file)
                    if data is not None:
                        all_results.append(data)

                if all_results:
                    results_df = pd.concat(all_results, ignore_index=True)
                    window_sizes = sorted(results_df['Window'].unique())

                    print(f"  Found {len(window_sizes)} window sizes: {window_sizes}")
                    print(f"  Organizing plots into subdirectories...")

                    # Generate plots for each window size
                    for window_size in window_sizes:
                        print(f"\n  Processing window size {window_size}...")
                        window_data = results_df[results_df['Window'] == window_size]

                        # Create subdirectory for this window
                        window_subdir = per_window_dir / f'win_{window_size}'
                        window_subdir.mkdir(exist_ok=True)

                        # Heatmap for each frequency
                        frequencies = sorted(window_data['Frequency'].unique())
                        for freq in frequencies:
                            freq_data = window_data[window_data['Frequency'] == freq]
                            pivot_mss = freq_data.pivot(index='Alpha2', columns='Alpha1', values='MSS_Homomorphic')
                            pivot_mse = freq_data.pivot(index='Alpha2', columns='Alpha1', values='MSE_Homomorphic')

                            # Sort to ensure proper ordering
                            pivot_mss = pivot_mss.sort_index(ascending=True)
                            pivot_mss = pivot_mss[sorted(pivot_mss.columns)]
                            pivot_mse = pivot_mse.sort_index(ascending=True)
                            pivot_mse = pivot_mse[sorted(pivot_mse.columns)]

                            # Get alpha values for proper extent
                            alpha1_values = pivot_mss.columns.values
                            alpha2_values = pivot_mss.index.values

                            # Define extent for imshow
                            extent = [alpha1_values[0], alpha1_values[-1],
                                      alpha2_values[0], alpha2_values[-1]]

                            # Create custom heatmap for windowed data
                            fig, axes = plt.subplots(1, 2, figsize=(20, 9))

                            # MSS heatmap
                            im = axes[0].imshow(pivot_mss.values, cmap='viridis', aspect='auto',
                                                interpolation='nearest', origin='lower',
                                                extent=extent)
                            cbar = plt.colorbar(im, ax=axes[0])
                            cbar.set_label('MSS Loss', fontsize=18, fontweight='bold')
                            cbar.ax.tick_params(labelsize=14)
                            axes[0].set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
                            axes[0].set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
                            axes[0].set_title(f'MSS Loss - {int(freq)} Hz - Window {window_size}',
                                              fontsize=20, fontweight='bold')
                            axes[0].tick_params(axis='both', labelsize=16)
                            axes[0]

                            # Add alpha sum highlights
                            add_alpha_sum_highlights(axes[0], alpha1_values, alpha2_values, extent)

                            # MSE heatmap
                            im = axes[1].imshow(pivot_mse.values, cmap='plasma', aspect='auto',
                                                interpolation='nearest', origin='lower',
                                                extent=extent)
                            cbar = plt.colorbar(im, ax=axes[1])
                            cbar.set_label('MSE Loss', fontsize=18, fontweight='bold')
                            cbar.ax.tick_params(labelsize=14)
                            axes[1].set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
                            axes[1].set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
                            axes[1].set_title(f'MSE Loss - {int(freq)} Hz - Window {window_size}',
                                              fontsize=20, fontweight='bold')
                            axes[1].tick_params(axis='both', labelsize=16)
                            axes[1]

                            # Add alpha sum highlights
                            add_alpha_sum_highlights(axes[1], alpha1_values, alpha2_values, extent)

                            plt.tight_layout()
                            filename = window_subdir / f'heatmap_freq_{int(freq)}Hz.png'
                            plt.savefig(filename, dpi=300)
                            print(f"    → per_window/win_{window_size}/{filename.name}")
                            plt.close()
                            total_plots += 1

                        # Aggregated heatmap for this window size (all frequencies)
                        agg_data = window_data.groupby(['Alpha1', 'Alpha2']).agg({
                            'MSS_Homomorphic': 'mean',
                            'MSE_Homomorphic': 'mean'
                        }).reset_index()

                        pivot_mss = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSS_Homomorphic')
                        pivot_mse = agg_data.pivot(index='Alpha2', columns='Alpha1', values='MSE_Homomorphic')

                        # Sort to ensure proper ordering
                        pivot_mss = pivot_mss.sort_index(ascending=True)
                        pivot_mss = pivot_mss[sorted(pivot_mss.columns)]
                        pivot_mse = pivot_mse.sort_index(ascending=True)
                        pivot_mse = pivot_mse[sorted(pivot_mse.columns)]

                        # Get alpha values for proper extent
                        alpha1_values = pivot_mss.columns.values
                        alpha2_values = pivot_mss.index.values

                        # Define extent for imshow
                        extent = [alpha1_values[0], alpha1_values[-1],
                                  alpha2_values[0], alpha2_values[-1]]

                        fig, axes = plt.subplots(1, 2, figsize=(20, 9))

                        im = axes[0].imshow(pivot_mss.values, cmap='viridis', aspect='auto',
                                            interpolation='nearest', origin='lower',
                                            extent=extent)
                        cbar = plt.colorbar(im, ax=axes[0])
                        cbar.set_label('MSS Loss', fontsize=18, fontweight='bold')
                        cbar.ax.tick_params(labelsize=14)

                        # Set ticks to align with cell centers
                        tick_step = 4
                        x_ticks = alpha1_values[::tick_step]
                        y_ticks = alpha2_values[::tick_step]
                        axes[0].set_xticks(x_ticks)
                        axes[0].set_yticks(y_ticks)
                        axes[0].set_xticklabels([f'{x:.1f}' for x in x_ticks])
                        axes[0].set_yticklabels([f'{y:.1f}' for y in y_ticks])

                        axes[0].set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
                        axes[0].set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
                        axes[0].set_title(f'MSS Loss - All Frequencies - Window {window_size}',
                                          fontsize=20, fontweight='bold')
                        axes[0].tick_params(axis='both', labelsize=16)
                        axes[0]

                        im = axes[1].imshow(pivot_mse.values, cmap='plasma', aspect='auto',
                                            interpolation='nearest', origin='lower',
                                            extent=extent)
                        cbar = plt.colorbar(im, ax=axes[1])
                        cbar.set_label('MSE Loss', fontsize=18, fontweight='bold')
                        cbar.ax.tick_params(labelsize=14)

                        # Set ticks to align with cell centers
                        axes[1].set_xticks(x_ticks)
                        axes[1].set_yticks(y_ticks)
                        axes[1].set_xticklabels([f'{x:.1f}' for x in x_ticks])
                        axes[1].set_yticklabels([f'{y:.1f}' for y in y_ticks])

                        axes[1].set_xlabel(r'$\alpha_1$', fontsize=24, fontweight='bold')
                        axes[1].set_ylabel(r'$\alpha_2$', fontsize=24, fontweight='bold')
                        axes[1].set_title(f'MSE Loss - All Frequencies - Window {window_size}',
                                          fontsize=20, fontweight='bold')
                        axes[1].tick_params(axis='both', labelsize=16)
                        axes[1]

                        plt.tight_layout()
                        filename = window_subdir / 'heatmap_all_frequencies.png'
                        plt.savefig(filename, dpi=300)
                        print(f"    → per_window/win_{window_size}/{filename.name}")
                        plt.close()
                        total_plots += 1

                    # Cross-window comparison plots
                    print("\n  Generating cross-window comparison plots...")
                    total_plots += plot_window_size_comparison(results_df, comparison_dir)
                    total_plots += plot_window_size_statistics(results_df, comparison_dir)

                    # Mixed plots - all windows per frequency
                    print("\n  Generating mixed plots (all windows per frequency)...")
                    count = plot_mixed_all_windows_per_frequency(results_df, per_freq_dir)
                    total_plots += count

                    # Mixed plots - all frequencies per window
                    print("\n  Generating mixed plots (all frequencies per window)...")
                    count = plot_mixed_all_frequencies_per_window(results_df, mixed_dir)
                    total_plots += count

                    # Mixed plots - everything
                    print("\n  Generating mixed plots (all windows and frequencies)...")
                    total_plots += plot_mixed_all_windows_and_frequencies(results_df, mixed_dir)

                # Process commutativity results
                if comm_files:
                    all_comm = []
                    for comm_file in comm_files:
                        data = load_results_data(comm_file)
                        if data is not None:
                            all_comm.append(data)

                    if all_comm:
                        comm_df = pd.concat(all_comm, ignore_index=True)

                        # Generate commutativity histograms for each window
                        for window_size in sorted(comm_df['Window'].unique()):
                            window_comm = comm_df[comm_df['Window'] == window_size]
                            # Save to per_window subdirectory
                            window_subdir = per_window_dir / f'win_{window_size}'
                            window_subdir.mkdir(exist_ok=True)
                            plot_commutativity_histograms(window_comm, window_subdir, window_size=window_size)
                            total_plots += 2

                        # Generate mixed commutativity plots (all windows)
                        print("\n  Generating mixed commutativity plots (all windows)...")
                        total_plots += plot_mixed_commutativity_all_windows(comm_df, commutativity_dir)

        else:
            # Direct mode (original behavior)
            print("  Detected DIRECT mode")

            # Create subdirectories for organization
            per_freq_dir = output_dir / 'per_frequency'
            aggregated_dir = output_dir / 'aggregated'
            analysis_dir = output_dir / 'analysis'
            commutativity_dir = output_dir / 'commutativity'
            examples_dir = output_dir / 'examples'

            per_freq_dir.mkdir(exist_ok=True)
            aggregated_dir.mkdir(exist_ok=True)
            analysis_dir.mkdir(exist_ok=True)
            commutativity_dir.mkdir(exist_ok=True)
            examples_dir.mkdir(exist_ok=True)

            results_file = mss_dir / 'mss_grid_results.txt'
            comm_results_file = mss_dir / 'mss_commutativity_results.txt'

            if results_file.exists():
                print("  Generating MSS grid heatmaps...")
                results_data = load_results_data(results_file)

                if results_data is not None:
                    frequencies = sorted(results_data['Frequency'].unique())

                    # Heatmap for each frequency (MSS + MSE)
                    print("  Saving per-frequency heatmaps...")
                    for freq in frequencies:
                        freq_data = results_data[results_data['Frequency'] == freq]
                        # Save to per_frequency subdirectory
                        plot_mss_grid_heatmap(freq_data, freq, per_freq_dir)
                        total_plots += 2  # Creates both MSS and MSE heatmaps

                    # Aggregated heatmap (average across all frequencies)
                    print("  Saving aggregated heatmaps...")
                    agg_mss = results_data.groupby(['Alpha1', 'Alpha2'])['MSS_Homomorphic'].mean().reset_index()
                    agg_mse = results_data.groupby(['Alpha1', 'Alpha2'])['MSE_Homomorphic'].mean().reset_index()
                    agg_data = agg_mss.copy()
                    agg_data['MSE_Homomorphic'] = agg_mse['MSE_Homomorphic']
                    plot_mss_grid_heatmap(agg_data, None, aggregated_dir)
                    total_plots += 2  # Creates both MSS and MSE heatmaps

                    # Aggregated heatmap WITH alpha sum highlights
                    print("  Generating highlighted heatmaps (α₁+α₂ ∈ {0,±1,±2})...")
                    plot_mss_grid_heatmap_with_alpha_sum_highlights(agg_data, aggregated_dir)
                    total_plots += 2  # Creates both MSS and MSE highlighted heatmaps

                    # MSS and MSE vs wrapped alpha plots
                    print("  Generating MSS vs alpha plots...")
                    plot_mss_vs_alpha_wrapped(results_data, analysis_dir)
                    total_plots += 2  # Creates both MSS and MSE plots

                    # Best/worst/medium case example comparisons
                    print("  Generating best/worst case examples...")
                    example_count = plot_best_worst_case_examples(results_data, mss_dir, examples_dir)
                    total_plots += example_count  # Creates 60 example plots (20 worst, 20 medium, 20 best)
            else:
                print(f"  ⚠ Results file not found: {results_file}")

            # Commutativity histograms
            if comm_results_file.exists():
                print("  Generating commutativity histograms...")
                comm_data = load_results_data(comm_results_file)

                if comm_data is not None:
                    plot_commutativity_histograms(comm_data, commutativity_dir)
                    total_plots += 2
            else:
                print(f"  ⚠ Commutativity results file not found: {comm_results_file}")
    else:
        print(f"\n⚠ Skipping MSS grid analysis (directory not found: {mss_dir})")

    # Summary
    print("\n" + "=" * 70)
    print("  Plotting Complete!")
    print("=" * 70)
    print(f"\n✓ Generated {total_plots} analysis plots")

    if recon_dir.exists():
        print(f"\nReconstruction Test plots → {recon_dir / 'analysis_plots'}/")
    if homo_dir.exists():
        print(f"Homomorphic Test plots → {homo_dir / 'analysis_plots'}/")
    if fft_dir.exists():
        print(f"FFT Comparison plots → {fft_dir / 'analysis_plots'}/")
    if mss_dir.exists():
        print(f"MSS Grid Analysis plots → {mss_dir / 'plots'}/")

    print()

    return 0


if __name__ == '__main__':
    exit(main())