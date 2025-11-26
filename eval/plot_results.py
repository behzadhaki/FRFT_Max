#!/usr/bin/env python3
"""
Simplified FRFT Test Results Plotter

Generates 5 key plots:
1. Processing time vs window size (average with min/max range)
2. RTF vs window size
3. Complexity analysis (log-log)
4. MSE vs frequency (all alphas, log scale)
5. MSE vs frequency (all alphas, linear scale)
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


def plot_processing_time(timing_data, output_dir):
    """Plot processing time vs window size with min/max and sample count."""
    fig, ax = plt.subplots(figsize=(10, 6))

    window_sizes = timing_data['WindowSize']
    mean_forward = timing_data['MeanForward']
    min_forward = timing_data['MinForward']
    max_forward = timing_data['MaxForward']
    num_samples = timing_data['NumSamples']

    # Get number of trials (use first entry, should be consistent)
    n_trials = num_samples.iloc[0] if len(num_samples) > 0 else 0

    # Consistent professional colors
    color_mean = '#1f77b4'    # Blue
    color_min = '#2ca02c'     # Green
    color_max = '#d62728'     # Red

    # Plot mean, min, and max
    ax.plot(window_sizes, mean_forward, 'o-', label=f'Average',
            linewidth=2.5, markersize=8, color=color_mean, zorder=3)
    ax.plot(window_sizes, min_forward, 's--', label='Min',
            linewidth=1.5, markersize=6, color=color_min, alpha=0.7, zorder=2)
    ax.plot(window_sizes, max_forward, '^--', label='Max',
            linewidth=1.5, markersize=6, color=color_max, alpha=0.7, zorder=2)

    # Add shaded area for min/max range
    ax.fill_between(window_sizes, min_forward, max_forward,
                    alpha=0.15, color=color_mean, label='Min/Max Range', zorder=1)

    ax.set_xlabel('Window Size (samples)', fontsize=12)
    ax.set_ylabel('Processing Time (ms/frame)', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    filename = f'processing_time_n{n_trials}.png'
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
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
    color_mean = '#2ca02c'    # Green
    color_threshold = '#d62728'  # Red

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean RTF
    ax.plot(window_sizes, rtf_mean, 'o-', label=f'Mean RTF',
            linewidth=2.5, markersize=8, color=color_mean)

    # Shaded area for best/worst
    ax.fill_between(window_sizes, rtf_best, rtf_worst,
                    alpha=0.2, color=color_mean, label='Best/Worst Range')

    # Real-time threshold line
    ax.axhline(y=1.0, color=color_threshold, linestyle='--', linewidth=2,
               label='Real-Time Threshold (RTF=1.0)', alpha=0.8)

    # Highlight real-time capable sizes
    for idx, row in timing_data.iterrows():
        if row['RTF_Mean'] < 1.0:
            ax.plot(row['WindowSize'], row['RTF_Mean'], 'o',
                    markersize=14, color=color_mean, alpha=0.3)

    ax.set_xlabel('Window Size (samples)', fontsize=12)
    ax.set_ylabel('Real-Time Factor (RTF)', fontsize=12)
    ax.set_xscale('log', base=2)
    # Linear y-scale
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    filename = f'rtf_analysis_n{n_trials}.png'
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
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

    # Plot measured data (mean only)
    ax.loglog(window_sizes, times_mean, 'o-', linewidth=2.5, markersize=10,
              label=f'Mean Time', color=color_measured, zorder=3)

    # Shaded area for min/max range
    ax.fill_between(window_sizes, times_min, times_max,
                    alpha=0.15, color=color_measured, label='Min/Max Range', zorder=1)

    # Fit O(N log N) - expected for FFT-based algorithms
    nlogn_fit = times_mean[0] * (window_sizes / window_sizes[0]) * \
                (np.log2(window_sizes) / np.log2(window_sizes[0]))
    ax.loglog(window_sizes, nlogn_fit, '--', linewidth=2,
              label='O(N log N) Reference', color=color_nlogn, alpha=0.7)

    # O(N^2) for comparison
    n2_fit = times_mean[0] * (window_sizes / window_sizes[0])**2
    ax.loglog(window_sizes, n2_fit, '--', linewidth=2,
              label='O(N²) Reference', color=color_n2, alpha=0.7)

    # O(N) for comparison
    n_fit = times_mean[0] * (window_sizes / window_sizes[0])
    ax.loglog(window_sizes, n_fit, '--', linewidth=2,
              label='O(N) Reference', color=color_n, alpha=0.7)

    ax.set_xlabel('Window Size N (samples)', fontsize=12)
    ax.set_ylabel('Processing Time (ms)', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    filename = f'complexity_analysis_n{n_trials}.png'
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
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
                    'o-', color=alpha_to_color[alpha], linewidth=1.5, markersize=4,
                    label=label, alpha=0.8)

        # Add subplot title as text annotation in upper left
        ax.text(0.02, 0.98, title, transform=ax.transAxes,
                fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        if subplot_idx == 0:
            ax.set_ylabel('Mean Squared Error (MSE)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best', ncol=1)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    filename = f'mse_vs_frequency_n{n_combinations}.png'
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
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
        print("\n[1/4] Processing Time vs Window Size")
        plot_processing_time(timing_data, output_dir)

        print("\n[2/4] RTF vs Window Size")
        plot_rtf(timing_data, output_dir)

        print("\n[3/4] Complexity Analysis")
        plot_complexity(timing_data, output_dir)
    else:
        print("\n⚠ Skipping timing plots (no timing data)")

    if accuracy_data is not None:
        print("\n[4/4] MSE vs Frequency")
        plot_mse_vs_frequency(accuracy_data, output_dir)
    else:
        print("\n⚠ Skipping accuracy plots (no accuracy data)")

    # Summary
    print("\n" + "=" * 70)
    print("  Plotting Complete!")
    print("=" * 70)

    plot_count = 0
    if timing_data is not None:
        plot_count += 3  # 3 timing plots
    if accuracy_data is not None:
        plot_count += 1  # 1 MSE plot

    print(f"\n✓ Generated {plot_count} plots in: {output_dir}/")
    print()

    if timing_data is not None:
        print("Performance Plots:")
        print(f"  • {output_dir}/processing_time_n{{trials}}.png")
        print(f"  • {output_dir}/rtf_analysis_n{{trials}}.png")
        print(f"  • {output_dir}/complexity_analysis_n{{trials}}.png")

    if accuracy_data is not None:
        print("\nAccuracy Plots:")
        print(f"  • {output_dir}/mse_vs_frequency_n{{combinations}}.png")

    print()

    return 0


if __name__ == '__main__':
    exit(main())