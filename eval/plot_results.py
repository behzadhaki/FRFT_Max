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
    ax.grid(True, alpha=0.3, which='both')
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
    ax.grid(True, alpha=0.3, which='both')
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
        ax.grid(True, alpha=0.3, axis='y')
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
        ax.grid(True, alpha=0.3, which='both')
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
    ax.grid(True, alpha=0.3, which='both')
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
    ax.grid(True, alpha=0.3, which='both')
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
        ax.grid(True, alpha=0.3, axis='y')
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
    ax.grid(True, alpha=0.3, which='both')
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
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=10, loc='best', framealpha=0.95, ncol=2)
    ax.set_ylim(0.95, 1.005)

    plt.tight_layout()
    filename = output_dir / 'fft_comparison_correlation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  → {filename}")
    plt.close()


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

    parser.add_argument('--reconstruction', default='reconstruction_test',
                        help='Reconstruction test directory (default: reconstruction_test)')
    parser.add_argument('--homomorphic', default='homomorphic_test',
                        help='Homomorphic test directory (default: homomorphic_test)')
    parser.add_argument('--fft', default='fft_comparison',
                        help='FFT comparison test directory (default: fft_comparison)')

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

    print()

    return 0


if __name__ == '__main__':
    exit(main())