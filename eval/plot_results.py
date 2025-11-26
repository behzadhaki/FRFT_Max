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
    """Plot processing time vs window size (forward only as average)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    window_sizes = timing_data['WindowSize']
    mean_forward = timing_data['MeanForward']
    min_forward = timing_data['MinForward']
    max_forward = timing_data['MaxForward']

    # Plot average (forward) with shaded min/max area
    ax.plot(window_sizes, mean_forward, 'o-', label='Average',
            linewidth=2.5, markersize=8, color='#1f77b4')

    # Add shaded area for min/max range
    ax.fill_between(window_sizes, min_forward, max_forward,
                    alpha=0.2, color='blue', label='Min/Max Range')

    ax.set_xlabel('Window Size (samples)', fontsize=12)
    ax.set_ylabel('Processing Time (ms/frame)', fontsize=12)
    ax.set_title('FRFT Processing Time vs Window Size', fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / 'processing_time.png', dpi=150, bbox_inches='tight')
    print(f"  → Saved: {output_dir / 'processing_time.png'}")
    plt.close()


def plot_rtf(timing_data, output_dir):
    """Plot Real-Time Factor vs window size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    window_sizes = timing_data['WindowSize']
    rtf_mean = timing_data['RTF_Mean']
    rtf_best = timing_data['RTF_Best']
    rtf_worst = timing_data['RTF_Worst']

    # Plot mean RTF
    ax.plot(window_sizes, rtf_mean, 'o-', label='Mean RTF',
            linewidth=2.5, markersize=8, color='#2ca02c')

    # Shaded area for best/worst
    ax.fill_between(window_sizes, rtf_best, rtf_worst,
                    alpha=0.3, color='green', label='Best/Worst Range')

    # Real-time threshold line
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
               label='Real-Time Threshold (RTF=1.0)', alpha=0.8)

    # Highlight real-time capable sizes
    for idx, row in timing_data.iterrows():
        if row['RTF_Mean'] < 1.0:
            ax.plot(row['WindowSize'], row['RTF_Mean'], 'o',
                    markersize=14, color='green', alpha=0.3)

    ax.set_xlabel('Window Size (samples)', fontsize=12)
    ax.set_ylabel('Real-Time Factor (RTF)', fontsize=12)
    ax.set_title('Real-Time Performance: RTF vs Window Size\n(RTF < 1.0 = Faster than real-time)',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / 'rtf_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  → Saved: {output_dir / 'rtf_analysis.png'}")
    plt.close()


def plot_complexity(timing_data, output_dir):
    """Plot computational complexity analysis (log-log)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    window_sizes = timing_data['WindowSize'].values
    times = timing_data['MeanTotal'].values

    # Plot measured data
    ax.loglog(window_sizes, times, 'o-', linewidth=2.5, markersize=10,
              label='Measured Time', color='#1f77b4', zorder=3)

    # Fit O(N log N) - expected for FFT-based algorithms
    nlogn_fit = times[0] * (window_sizes / window_sizes[0]) * \
                (np.log2(window_sizes) / np.log2(window_sizes[0]))
    ax.loglog(window_sizes, nlogn_fit, '--', linewidth=2,
              label='O(N log N) Reference', color='#2ca02c', alpha=0.7)

    # O(N^2) for comparison
    n2_fit = times[0] * (window_sizes / window_sizes[0])**2
    ax.loglog(window_sizes, n2_fit, '--', linewidth=2,
              label='O(N²) Reference', color='#d62728', alpha=0.7)

    # O(N) for comparison
    n_fit = times[0] * (window_sizes / window_sizes[0])
    ax.loglog(window_sizes, n_fit, '--', linewidth=2,
              label='O(N) Reference', color='#9467bd', alpha=0.7)

    ax.set_xlabel('Window Size N (samples)', fontsize=12)
    ax.set_ylabel('Processing Time (ms)', fontsize=12)
    ax.set_title('Computational Complexity Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='upper left')

    # Add text annotation about complexity
    complexity_text = 'FRFT follows O(N log N)\ndue to FFT-based implementation'
    ax.text(0.98, 0.05, complexity_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'complexity_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  → Saved: {output_dir / 'complexity_analysis.png'}")
    plt.close()


def plot_mse_vs_frequency(accuracy_data, output_dir):
    """Plot MSE vs frequency with alphas from 0 to 2, aggregated across all window sizes and overlap factors."""

    # Filter for alphas from 0 to 2 only
    subset = accuracy_data[(accuracy_data['Alpha'] >= 0.0) & (accuracy_data['Alpha'] <= 2.0)]

    # Get alpha values in 0.1 increments from 0 to 2
    alphas = sorted(subset['Alpha'].unique())
    # Round to avoid floating point issues
    alphas = [round(a, 1) for a in alphas if 0.0 <= a <= 2.0]
    alphas = sorted(list(set(alphas)))  # Remove duplicates and sort

    # Create colormap for alphas
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(alphas) - 1)) for i in range(len(alphas))]

    # === Plot 1: Log Y-axis ===
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, alpha in enumerate(alphas):
        # Get all data for this alpha (across all window sizes and overlap factors)
        alpha_data = subset[np.abs(subset['Alpha'] - alpha) < 0.05]

        if len(alpha_data) == 0:
            continue

        # Aggregate by frequency - calculate mean MSE across all window sizes and overlap factors
        freq_grouped = alpha_data.groupby('Frequency')['MSE'].mean().reset_index()
        freq_grouped = freq_grouped.sort_values('Frequency')

        # Label all alphas
        label = f'α = {alpha:.1f}'
        ax.semilogy(freq_grouped['Frequency'], freq_grouped['MSE'],
                    'o-', color=colors[i], linewidth=1.5, markersize=4,
                    label=label, alpha=0.8)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE) - Log Scale', fontsize=12)
    ax.set_title('MSE vs Frequency (α: 0.0 to 2.0)\n(Averaged across all window sizes and overlap factors)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / 'mse_vs_frequency_log.png', dpi=150, bbox_inches='tight')
    print(f"  → Saved: {output_dir / 'mse_vs_frequency_log.png'}")
    plt.close()

    # === Plot 2: Linear Y-axis ===
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, alpha in enumerate(alphas):
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
                'o-', color=colors[i], linewidth=1.5, markersize=4,
                label=label, alpha=0.8)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE) - Linear Scale', fontsize=12)
    ax.set_title('MSE vs Frequency (α: 0.0 to 2.0)\n(Averaged across all window sizes and overlap factors)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / 'mse_vs_frequency_linear.png', dpi=150, bbox_inches='tight')
    print(f"  → Saved: {output_dir / 'mse_vs_frequency_linear.png'}")
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
        print("\n[4/5] MSE vs Frequency (Log Scale)")
        print("[5/5] MSE vs Frequency (Linear Scale)")
        plot_mse_vs_frequency(accuracy_data, output_dir)
    else:
        print("\n⚠ Skipping accuracy plots (no accuracy data)")

    # Summary
    print("\n" + "=" * 70)
    print("  Plotting Complete!")
    print("=" * 70)

    plot_count = 0
    if timing_data is not None:
        plot_count += 3
    if accuracy_data is not None:
        plot_count += 2  # Now generating 2 MSE plots

    print(f"\n✓ Generated {plot_count} plots in: {output_dir}/")
    print()

    if timing_data is not None:
        print("Performance Plots:")
        print(f"  • {output_dir}/processing_time.png")
        print(f"  • {output_dir}/rtf_analysis.png")
        print(f"  • {output_dir}/complexity_analysis.png")

    if accuracy_data is not None:
        print("\nAccuracy Plots:")
        print(f"  • {output_dir}/mse_vs_frequency_log.png")
        print(f"  • {output_dir}/mse_vs_frequency_linear.png")

    print()

    return 0


if __name__ == '__main__':
    exit(main())