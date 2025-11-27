#!/usr/bin/env python3
"""
Simplified FRFT Test Results Plotter

Generates 4 key plots:
1. Processing time with complexity reference lines (O(N), O(N log N), O(N²))
2. RTF vs window size
3. MSE vs frequency (all alphas, aggregated)
4. MSE vs alpha (organized by frequency with bar charts)
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


def load_accuracy_data(filename):
    """Load accuracy test results."""
    try:
        data = pd.read_csv(filename, sep='\t', comment='#')
        print(f"✓ Loaded {len(data)} accuracy records from {filename}")
        return data
    except Exception as e:
        print(f"✗ Error loading accuracy file: {e}")
        return None


def plot_processing_time_and_complexity(timing_data, output_dir):
    """Plot processing time with complexity reference lines in a single plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    window_sizes = timing_data['WindowSize'].values
    mean_forward = timing_data['MeanForward'].values
    min_forward = timing_data['MinForward'].values
    max_forward = timing_data['MaxForward'].values
    num_samples = timing_data['NumSamples']

    # Get number of trials (use first entry, should be consistent)
    n_trials = num_samples.iloc[0] if len(num_samples) > 0 else 0

    # Consistent professional colors
    color_mean = '#1f77b4'    # Blue
    color_nlogn = '#ff7f0e'   # Orange
    color_n2 = '#8c564b'      # Brown
    color_n = '#9467bd'       # Purple

    # Add shaded area for min/max range first (so it's behind)
    ax.fill_between(window_sizes, min_forward, max_forward,
                    alpha=0.2, color=color_mean, label='Min/Max Range', zorder=1)

    # Plot mean with larger markers and thinner line, with transparency
    ax.plot(window_sizes, mean_forward, 'o-', label='Average',
            linewidth=2.0, markersize=10, color=color_mean, alpha=0.7, zorder=3)

    # Add complexity reference lines
    # Fit O(N log N) - expected for FFT-based algorithms
    nlogn_fit = mean_forward[0] * (window_sizes / window_sizes[0]) * \
                (np.log2(window_sizes) / np.log2(window_sizes[0]))
    ax.plot(window_sizes, nlogn_fit, ':', linewidth=3,
            label='O(N log N) Reference', color=color_nlogn, alpha=0.8, zorder=1)

    # O(N^2) for comparison
    n2_fit = mean_forward[0] * (window_sizes / window_sizes[0])**2
    ax.plot(window_sizes, n2_fit, ':', linewidth=3,
            label='O(N²) Reference', color=color_n2, alpha=0.8, zorder=1)

    # O(N) for comparison
    n_fit = mean_forward[0] * (window_sizes / window_sizes[0])
    ax.plot(window_sizes, n_fit, ':', linewidth=3,
            label='O(N) Reference', color=color_n, alpha=0.8, zorder=1)

    ax.set_xlabel('Window Size (samples)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Processing Time (ms/frame)', fontsize=18, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=16, loc='upper left', framealpha=0.95, ncol=2)

    plt.tight_layout()
    filename = f'processing_time_complexity_n{n_trials}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  → Saved: {output_dir / filename}")
    plt.close()


def plot_rtf(timing_data, output_dir):
    """Plot Real-Time Factor vs window size with sample count (linear scale)."""

    window_sizes = timing_data['WindowSize']
    rtf_mean = timing_data['RTF_Mean']
    rtf_best = timing_data['RTF_Best']
    rtf_worst = timing_data['RTF_Worst']
    num_samples = timing_data['NumSamples']

    # Get number of trials
    n_trials = num_samples.iloc[0] if len(num_samples) > 0 else 0

    # Consistent professional colors
    color_mean = '#1f77b4'    # Blue (changed from green)
    color_threshold = '#d62728'  # Red

    fig, ax = plt.subplots(figsize=(10, 6))

    # Shaded area for best/worst (plot first so it's behind)
    ax.fill_between(window_sizes, rtf_best, rtf_worst,
                    alpha=0.2, color=color_mean, label='Best/Worst Range')

    # Plot mean RTF with same style as processing time plot
    ax.plot(window_sizes, rtf_mean, 'o-', label=f'Mean RTF',
            linewidth=2.0, markersize=10, color=color_mean, alpha=0.7)

    # Real-time threshold line
    ax.axhline(y=1.0, color=color_threshold, linestyle='--', linewidth=3,
               label='Real-Time Threshold (RTF=1.0)', alpha=0.8)

    ax.set_xlabel('Window Size (samples)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Real-Time Factor (RTF)', fontsize=18, fontweight='bold')
    ax.set_xscale('log', base=2)
    # Linear y-scale
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=16, loc='upper left', bbox_to_anchor=(0, 0.92), framealpha=0.95)

    plt.tight_layout()
    filename = f'rtf_analysis_n{n_trials}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  → Saved: {output_dir / filename}")
    plt.close()


def plot_complexity(timing_data, output_dir):
    """Plot computational complexity analysis (log-log) with sample count."""
    fig, ax = plt.subplots(figsize=(10, 6))

    window_sizes = timing_data['WindowSize'].values
    times_mean = timing_data['MeanTotal'].values
    times_min = (timing_data['MinForward'] + timing_data['MinInverse']).values
    times_max = (timing_data['MaxForward'] + timing_data['MaxInverse']).values
    num_samples = timing_data['NumSamples']

    # Get number of trials
    n_trials = num_samples.iloc[0] if len(num_samples) > 0 else 0

    # Consistent professional colors
    color_measured = '#1f77b4'    # Blue
    color_nlogn = '#2ca02c'       # Green
    color_n2 = '#d62728'          # Red
    color_n = '#9467bd'           # Purple

    # Plot measured data (mean only) with larger markers and lines
    ax.loglog(window_sizes, times_mean, 'o-', linewidth=3.5, markersize=12,
              label=f'Mean Time', color=color_measured, zorder=3)

    # Shaded area for min/max range
    ax.fill_between(window_sizes, times_min, times_max,
                    alpha=0.15, color=color_measured, label='Min/Max Range', zorder=1)

    # Fit O(N log N) - expected for FFT-based algorithms
    nlogn_fit = times_mean[0] * (window_sizes / window_sizes[0]) * \
                (np.log2(window_sizes) / np.log2(window_sizes[0]))
    ax.loglog(window_sizes, nlogn_fit, '--', linewidth=3,
              label='O(N log N) Reference', color=color_nlogn, alpha=0.7)

    # O(N^2) for comparison
    n2_fit = times_mean[0] * (window_sizes / window_sizes[0])**2
    ax.loglog(window_sizes, n2_fit, '--', linewidth=3,
              label='O(N²) Reference', color=color_n2, alpha=0.7)

    # O(N) for comparison
    n_fit = times_mean[0] * (window_sizes / window_sizes[0])
    ax.loglog(window_sizes, n_fit, '--', linewidth=3,
              label='O(N) Reference', color=color_n, alpha=0.7)

    ax.set_xlabel('Window Size N (samples)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Processing Time (ms)', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=16, loc='upper left', framealpha=0.95)

    plt.tight_layout()
    filename = f'complexity_analysis_n{n_trials}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  → Saved: {output_dir / filename}")
    plt.close()


def plot_mse_vs_frequency(accuracy_data, output_dir):
    """Plot MSE vs frequency with alphas from 0 to 2 in four subplots (1 row x 4 cols), aggregated across all window sizes and overlap factors."""

    # Filter for alphas from 0 to 2 only
    subset = accuracy_data[(accuracy_data['Alpha'] >= 0.0) & (accuracy_data['Alpha'] <= 2.0)]

    # Count unique window sizes and overlap factors
    n_window_sizes = subset['WindowSize'].nunique()
    n_overlap_factors = subset['OverlapFactor'].nunique()
    n_combinations = n_window_sizes * n_overlap_factors

    # Get alpha values in 0.1 increments from 0 to 2
    alphas = sorted(subset['Alpha'].unique())
    # Round to avoid floating point issues
    alphas = [round(a, 1) for a in alphas if 0.0 <= a <= 2.0]
    alphas = sorted(list(set(alphas)))  # Remove duplicates and sort

    # Split alphas into 4 ranges
    alpha_ranges = [
        (0.0, 0.5, 'α: 0.0 - 0.5'),
        (0.5, 1.0, 'α: 0.5 - 1.0'),
        (1.0, 1.5, 'α: 1.0 - 1.5'),
        (1.5, 2.0, 'α: 1.5 - 2.0')
    ]

    # Create colormap for alphas - professional tab20 colors
    cmap = plt.cm.tab20
    colors = [cmap(i / len(alphas)) for i in range(len(alphas))]
    alpha_to_color = {alpha: colors[i] for i, alpha in enumerate(alphas)}

    # Create 1x4 subplot layout (one row, four columns)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for subplot_idx, (alpha_min, alpha_max, title) in enumerate(alpha_ranges):
        ax = axes[subplot_idx]

        # Filter alphas for this range
        range_alphas = [a for a in alphas if alpha_min <= a <= alpha_max]

        for alpha in range_alphas:
            # Get all data for this alpha (across all window sizes and overlap factors)
            alpha_data = subset[np.abs(subset['Alpha'] - alpha) < 0.05]

            if len(alpha_data) == 0:
                continue

            # Aggregate by frequency - calculate mean MSE across all window sizes and overlap factors
            freq_grouped = alpha_data.groupby('Frequency')['MSE'].mean().reset_index()
            freq_grouped = freq_grouped.sort_values('Frequency')

            # Label all alphas
            label = f'α = {alpha:.1f}'
            ax.plot(freq_grouped['Frequency'], freq_grouped['MSE'],
                    'o-', color=alpha_to_color[alpha], linewidth=2.5, markersize=6,
                    label=label, alpha=0.8)

        # Add subplot title as text annotation in upper left
        ax.text(0.02, 0.98, title, transform=ax.transAxes,
                fontsize=14, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='bold')
        if subplot_idx == 0:
            ax.set_ylabel('Mean Squared Error (MSE)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=14, loc='best', ncol=1, framealpha=0.95)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    filename = f'mse_vs_frequency_n{n_combinations}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  → Saved: {output_dir / filename}")
    plt.close()


def plot_mse_vs_alpha(accuracy_data, output_dir):
    """Plot MSE vs alpha as bar charts, organized by frequency in separate subplots (vertical layout)."""

    # Filter for alphas from 0 to 2 only
    subset = accuracy_data[(accuracy_data['Alpha'] >= 0.0) & (accuracy_data['Alpha'] <= 2.0)]

    # Count unique window sizes and overlap factors
    n_window_sizes = subset['WindowSize'].nunique()
    n_overlap_factors = subset['OverlapFactor'].nunique()
    n_combinations = n_window_sizes * n_overlap_factors

    # Get unique frequencies and alphas
    frequencies = sorted(subset['Frequency'].unique())
    alphas = sorted(subset['Alpha'].unique())
    # Round to avoid floating point issues
    alphas = [round(a, 1) for a in alphas if 0.0 <= a <= 2.0]
    alphas = sorted(list(set(alphas)))  # Remove duplicates and sort

    # Determine number of subplots
    n_freqs = len(frequencies)

    # Create vertical subplot layout - 1 inch per subplot (half of previous 2 inches)
    fig, axes = plt.subplots(n_freqs, 1, figsize=(10, 1 * n_freqs))

    # Handle single subplot case
    if n_freqs == 1:
        axes = [axes]

    # Find global min/max for consistent y-axis range
    global_min = float('inf')
    global_max = float('-inf')

    for freq in frequencies:
        freq_data = subset[np.abs(subset['Frequency'] - freq) < 0.01]
        for alpha in alphas:
            alpha_data = freq_data[np.abs(freq_data['Alpha'] - alpha) < 0.05]
            if len(alpha_data) > 0:
                mean_mse = alpha_data['MSE'].mean()
                global_min = min(global_min, mean_mse)
                global_max = max(global_max, mean_mse)

    # Add some padding to the range
    y_range = global_max - global_min
    global_min = max(0, global_min - 0.1 * y_range)
    global_max = global_max + 0.1 * y_range

    # Color scheme
    bar_color = '#1f77b4'  # Blue

    for idx, freq in enumerate(frequencies):
        ax = axes[idx]

        # Get data for this frequency
        freq_data = subset[np.abs(subset['Frequency'] - freq) < 0.01]

        # Aggregate by alpha - calculate mean MSE across all window sizes and overlap factors
        mse_values = []
        for alpha in alphas:
            alpha_data = freq_data[np.abs(freq_data['Alpha'] - alpha) < 0.05]
            if len(alpha_data) > 0:
                mean_mse = alpha_data['MSE'].mean()
                mse_values.append(mean_mse)
            else:
                mse_values.append(0)

        # Create bar chart
        x_pos = np.arange(len(alphas))
        bars = ax.bar(x_pos, mse_values, color=bar_color, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add frequency label - use kHz when >= 1000 Hz
        if freq >= 1000:
            freq_text = f'{freq/1000:.1f} kHz'
        else:
            freq_text = f'{freq:.1f} Hz'

        ax.text(0.98, 0.88, freq_text, transform=ax.transAxes,
                fontsize=16, fontweight='bold', verticalalignment='top',
                horizontalalignment='right')

        # Customize subplot
        ax.set_xticks(x_pos)

        # Only show x-ticks and labels on bottom subplot, label every other tick
        if idx == n_freqs - 1:
            # Create labels for every other tick (0.0, 0.2, 0.4, ...)
            tick_labels = [f'{a:.1f}' if i % 2 == 0 else '' for i, a in enumerate(alphas)]
            ax.set_xticklabels(tick_labels, fontsize=16)
            ax.set_xlabel('Alpha (α)', fontsize=18, fontweight='bold')
        else:
            # Hide x-tick labels for non-bottom subplots
            ax.set_xticklabels([])

        # Only show y-label on middle subplot
        if idx == n_freqs // 2:
            ax.set_ylabel('Mean Squared Error (MSE)', fontsize=18, fontweight='bold')

        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Set consistent y-axis range for all subplots
        ax.set_ylim(global_min, global_max)

        # Highlight minimum MSE with a marker (but not for alpha = 0)
        if mse_values:
            min_idx = np.argmin(mse_values)
            # Only add star if minimum is not at alpha = 0 (index 0)
            if min_idx > 0:
                ax.plot(min_idx, mse_values[min_idx], 'r*', markersize=20,
                        markeredgecolor='darkred', markeredgewidth=1.5, zorder=10)

    plt.tight_layout(pad=0.5, h_pad=0.3)
    filename = f'mse_vs_alpha_n{n_combinations}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  → Saved: {output_dir / filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate simplified FRFT test plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 plot_results.py
  python3 plot_results.py --dir /path/to/results
  python3 plot_results.py --dir ../test_data --output-dir figures
        """
    )

    parser.add_argument('--dir', '-d', default='.',
                        help='Directory containing input files (default: current directory)')
    parser.add_argument('--timing', default='frft_timing_benchmarks.txt',
                        help='Timing benchmark filename (default: frft_timing_benchmarks.txt)')
    parser.add_argument('--accuracy', default='frft_test_results.txt',
                        help='Accuracy results filename (default: frft_test_results.txt)')
    parser.add_argument('--output-dir', '-o', default='plots',
                        help='Output directory name for plots (default: plots)')

    args = parser.parse_args()

    # Convert directory paths
    input_dir = Path(args.dir)
    timing_file = input_dir / args.timing
    accuracy_file = input_dir / args.accuracy
    output_dir = input_dir / args.output_dir

    print("=" * 70)
    print("  FRFT Test Results Plotter")
    print("=" * 70)
    print()
    print(f"Input directory: {input_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print()

    # Load data
    print("Loading data files...")
    timing_data = load_timing_data(str(timing_file))
    accuracy_data = load_accuracy_data(str(accuracy_file))

    if timing_data is None and accuracy_data is None:
        print("\n✗ No data files could be loaded. Exiting.")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print()

    # Generate plots
    print("Generating plots...")

    if timing_data is not None:
        print("\n[1/3] Processing Time & Complexity Analysis")
        plot_processing_time_and_complexity(timing_data, output_dir)

        print("\n[2/3] RTF vs Window Size")
        plot_rtf(timing_data, output_dir)
    else:
        print("\n⚠ Skipping timing plots (no timing data)")

    if accuracy_data is not None:
        print("\n[3/4] MSE vs Frequency")
        plot_mse_vs_frequency(accuracy_data, output_dir)

        print("\n[4/4] MSE vs Alpha (by Frequency)")
        plot_mse_vs_alpha(accuracy_data, output_dir)
    else:
        print("\n⚠ Skipping accuracy plots (no accuracy data)")

    # Summary
    print("\n" + "=" * 70)
    print("  Plotting Complete!")
    print("=" * 70)

    plot_count = 0
    if timing_data is not None:
        plot_count += 2  # 2 timing plots (combined + RTF)
    if accuracy_data is not None:
        plot_count += 2  # 2 MSE plots (vs frequency + vs alpha)

    print(f"\n✓ Generated {plot_count} plots in: {output_dir}/")
    print()

    if timing_data is not None:
        print("Performance Plots:")
        print(f"  • {output_dir}/processing_time_complexity_n{{trials}}.png")
        print(f"  • {output_dir}/rtf_analysis_n{{trials}}.png")

    if accuracy_data is not None:
        print("\nAccuracy Plots:")
        print(f"  • {output_dir}/mse_vs_frequency_n{{combinations}}.png")
        print(f"  • {output_dir}/mse_vs_alpha_n{{combinations}}.png")

    print()

    return 0


if __name__ == '__main__':
    exit(main())