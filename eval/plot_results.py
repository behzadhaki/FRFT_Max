#!/usr/bin/env python3
"""
FRFT Test Results Plotter - Updated for Homomorphic Test

Generates plots for:
- Reconstruction test (original functionality)
- Homomorphic property test (new functionality)
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


def load_homomorphic_data(filename):
    """Load homomorphic property test results."""
    try:
        data = pd.read_csv(filename, sep='\t', comment='#')
        print(f"✓ Loaded {len(data)} homomorphic records from {filename}")
        return data
    except Exception as e:
        print(f"✗ Error loading homomorphic file: {e}")
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


def plot_mse_vs_frequency(accuracy_data, output_dir):
    """Plot MSE vs frequency with alphas from 0 to 2 in four subplots (1 row x 4 cols), aggregated across all window sizes and overlap factors."""

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

    # Create subplots: 1 row x 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    # Color scheme for frequency ranges
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Group frequencies into 4 buckets
    freq_buckets = [
        [f for f in frequencies if f < 500],
        [f for f in frequencies if 500 <= f < 2000],
        [f for f in frequencies if 2000 <= f < 8000],
        [f for f in frequencies if f >= 8000]
    ]
    bucket_labels = ['< 500 Hz', '500 Hz - 2 kHz', '2 kHz - 8 kHz', '≥ 8 kHz']

    for bucket_idx, (freq_bucket, label) in enumerate(zip(freq_buckets, bucket_labels)):
        ax = axes[bucket_idx]

        if not freq_bucket:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=16)
            ax.set_title(label, fontsize=16, fontweight='bold')
            continue

        # Aggregate MSE for all frequencies in this bucket
        for freq in freq_bucket:
            freq_data = subset[np.abs(subset['Frequency'] - freq) < 0.01]

            # Calculate mean MSE across all window sizes and overlap factors for each alpha
            mse_values = []
            for alpha in alphas:
                alpha_data = freq_data[np.abs(freq_data['Alpha'] - alpha) < 0.05]
                if len(alpha_data) > 0:
                    mse_values.append(alpha_data['MSE'].mean())
                else:
                    mse_values.append(np.nan)

            # Plot line for this frequency
            if freq >= 1000:
                freq_label = f'{freq/1000:.1f} kHz'
            else:
                freq_label = f'{freq:.0f} Hz'

            ax.semilogy(alphas, mse_values, 'o-', label=freq_label, alpha=0.7, linewidth=2)

        ax.set_title(label, fontsize=16, fontweight='bold')
        ax.set_xlabel('Alpha (α)', fontsize=14, fontweight='bold')
        if bucket_idx == 0:
            ax.set_ylabel('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    filename = f'mse_vs_frequency_n{n_combinations}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  → Saved: {output_dir / filename}")
    plt.close()


def plot_mse_vs_alpha(accuracy_data, output_dir, global_y_limits=None):
    """Plot MSE vs alpha for each frequency as bar charts in vertical subplots, aggregated across all window sizes and overlap factors."""

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

    # Find global min/max for consistent y-axis range (if not provided)
    if global_y_limits is None:
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
    else:
        global_min, global_max = global_y_limits

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

    plt.tight_layout(pad=0.5, h_pad=0.3)
    filename = f'mse_vs_alpha_n{n_combinations}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  → Saved: {output_dir / filename}")
    plt.close()

    # Return the y-limits for sharing with homomorphic plot
    return (global_min, global_max)


def plot_homomorphic_mse_vs_alpha(homo_data, output_dir, global_y_limits=None):
    """Plot homomorphic MSE (difference between direct and composed transforms) vs alpha."""

    # Count unique window sizes and overlap factors
    n_window_sizes = homo_data['WindowSize'].nunique()
    n_overlap_factors = homo_data['OverlapFactor'].nunique()
    n_combinations = n_window_sizes * n_overlap_factors

    # Get unique frequencies and alphas
    frequencies = sorted(homo_data['Frequency'].unique())
    alphas = sorted(homo_data['Alpha_Total'].unique())
    # Round to avoid floating point issues
    alphas = [round(a, 1) for a in alphas]
    alphas = sorted(list(set(alphas)))  # Remove duplicates and sort

    # Determine number of subplots
    n_freqs = len(frequencies)

    # Create vertical subplot layout
    fig, axes = plt.subplots(n_freqs, 1, figsize=(10, 1 * n_freqs))

    # Handle single subplot case
    if n_freqs == 1:
        axes = [axes]

    # Find global min/max for consistent y-axis range (if not provided)
    if global_y_limits is None:
        global_min = float('inf')
        global_max = float('-inf')

        for freq in frequencies:
            freq_data = homo_data[np.abs(homo_data['Frequency'] - freq) < 0.01]
            for alpha in alphas:
                alpha_data = freq_data[np.abs(freq_data['Alpha_Total'] - alpha) < 0.05]
                if len(alpha_data) > 0:
                    mean_mse = alpha_data['MSE_Homomorphic'].mean()
                    global_min = min(global_min, mean_mse)
                    global_max = max(global_max, mean_mse)

        # Add some padding to the range
        y_range = global_max - global_min
        global_min = max(0, global_min - 0.1 * y_range) if y_range > 0 else 0
        global_max = global_max + 0.1 * y_range if y_range > 0 else global_max * 1.1
    else:
        global_min, global_max = global_y_limits

    # Color scheme
    bar_color = '#2ca02c'  # Green (different from reconstruction test)

    for idx, freq in enumerate(frequencies):
        ax = axes[idx]

        # Get data for this frequency
        freq_data = homo_data[np.abs(homo_data['Frequency'] - freq) < 0.01]

        # Aggregate by alpha - calculate mean MSE across all window sizes and overlap factors
        mse_values = []
        for alpha in alphas:
            alpha_data = freq_data[np.abs(freq_data['Alpha_Total'] - alpha) < 0.05]
            if len(alpha_data) > 0:
                mean_mse = alpha_data['MSE_Homomorphic'].mean()
                mse_values.append(mean_mse)
            else:
                mse_values.append(0)

        # Create bar chart
        x_pos = np.arange(len(alphas))
        bars = ax.bar(x_pos, mse_values, color=bar_color, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add frequency label
        if freq >= 1000:
            freq_text = f'{freq/1000:.1f} kHz'
        else:
            freq_text = f'{freq:.1f} Hz'

        ax.text(0.98, 0.88, freq_text, transform=ax.transAxes,
                fontsize=16, fontweight='bold', verticalalignment='top',
                horizontalalignment='right')

        # Customize subplot
        ax.set_xticks(x_pos)

        # Only show x-ticks and labels on bottom subplot
        if idx == n_freqs - 1:
            tick_labels = [f'{a:.1f}' if i % 2 == 0 else '' for i, a in enumerate(alphas)]
            ax.set_xticklabels(tick_labels, fontsize=16)
            ax.set_xlabel('Alpha Total (α)', fontsize=18, fontweight='bold')
        else:
            ax.set_xticklabels([])

        # Only show y-label on middle subplot
        if idx == n_freqs // 2:
            ax.set_ylabel('Homomorphic MSE', fontsize=18, fontweight='bold')

        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', which='major', labelsize=16)

        # Set consistent y-axis range for all subplots
        ax.set_ylim(global_min, global_max)

    plt.tight_layout(pad=0.5, h_pad=0.3)
    filename = f'homomorphic_mse_vs_alpha_n{n_combinations}.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  → Saved: {output_dir / filename}")
    plt.close()


def plot_homomorphic_mse_heatmap(homo_data, output_dir):
    """Plot heatmap of homomorphic MSE across window sizes and alphas."""

    # Get unique values
    window_sizes = sorted(homo_data['WindowSize'].unique())
    alphas = sorted(homo_data['Alpha_Total'].unique())
    alphas = [round(a, 1) for a in alphas]
    alphas = sorted(list(set(alphas)))

    # Create matrix: rows = window sizes, cols = alphas
    mse_matrix = np.zeros((len(window_sizes), len(alphas)))

    for i, ws in enumerate(window_sizes):
        for j, alpha in enumerate(alphas):
            data_subset = homo_data[
                (homo_data['WindowSize'] == ws) &
                (np.abs(homo_data['Alpha_Total'] - alpha) < 0.05)
                ]
            if len(data_subset) > 0:
                mse_matrix[i, j] = data_subset['MSE_Homomorphic'].mean()
            else:
                mse_matrix[i, j] = np.nan

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap with log scale for better visualization
    im = ax.imshow(np.log10(mse_matrix + 1e-20), aspect='auto', cmap='viridis',
                   interpolation='nearest', origin='lower')

    # Set ticks
    ax.set_xticks(np.arange(len(alphas)))
    ax.set_yticks(np.arange(len(window_sizes)))

    # Set tick labels
    ax.set_xticklabels([f'{a:.1f}' for a in alphas], fontsize=12)
    ax.set_yticklabels([str(ws) for ws in window_sizes], fontsize=12)

    # Labels
    ax.set_xlabel('Alpha Total (α)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Window Size', fontsize=16, fontweight='bold')
    ax.set_title('Homomorphic MSE Heatmap (log₁₀ scale)', fontsize=18, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(MSE)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    filename = 'homomorphic_mse_heatmap.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"  → Saved: {output_dir / filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate FRFT test plots (reconstruction and homomorphic tests)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 plot_results.py
  python3 plot_results.py --dir /path/to/results
  python3 plot_results.py --homomorphic frft_homomorphic_results.txt
        """
    )

    parser.add_argument('--dir', '-d', default='.',
                        help='Directory containing input files (default: current directory)')
    parser.add_argument('--timing', default='frft_timing_benchmarks.txt',
                        help='Timing benchmark filename (default: frft_timing_benchmarks.txt)')
    parser.add_argument('--accuracy', default='frft_test_results.txt',
                        help='Accuracy results filename (default: frft_test_results.txt)')
    parser.add_argument('--homomorphic', default='frft_homomorphic_results.txt',
                        help='Homomorphic test results filename (default: frft_homomorphic_results.txt)')
    parser.add_argument('--output-dir', '-o', default='plots',
                        help='Output directory name for plots (default: plots)')

    args = parser.parse_args()

    # Convert directory paths
    input_dir = Path(args.dir)
    timing_file = input_dir / args.timing
    accuracy_file = input_dir / args.accuracy
    homomorphic_file = input_dir / args.homomorphic
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
    homomorphic_data = load_homomorphic_data(str(homomorphic_file))

    if timing_data is None and accuracy_data is None and homomorphic_data is None:
        print("\n✗ No data files could be loaded. Exiting.")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print()

    # Generate plots
    print("Generating plots...")

    plot_count = 0
    y_limits = None  # Will be computed from both reconstruction and homomorphic data

    # First pass: compute global y-limits if both datasets exist
    if accuracy_data is not None and homomorphic_data is not None:
        print("\n[Computing shared y-axis limits...]")

        # Get min/max from reconstruction data
        recon_subset = accuracy_data[(accuracy_data['Alpha'] >= 0.0) & (accuracy_data['Alpha'] <= 2.0)]
        recon_min = recon_subset['MSE'].min()
        recon_max = recon_subset['MSE'].max()

        # Get min/max from homomorphic data
        homo_min = homomorphic_data['MSE_Homomorphic'].min()
        homo_max = homomorphic_data['MSE_Homomorphic'].max()

        # Combine
        global_min = min(recon_min, homo_min)
        global_max = max(recon_max, homo_max)

        # Add padding
        y_range = global_max - global_min
        global_min = max(0, global_min - 0.1 * y_range)
        global_max = global_max + 0.1 * y_range

        y_limits = (global_min, global_max)
        print(f"  Shared y-limits: [{global_min:.2e}, {global_max:.2e}]")

    # Timing plots (reconstruction test)
    if timing_data is not None:
        print("\n[Reconstruction Test - Timing]")
        print("  [1] Processing Time & Complexity Analysis")
        plot_processing_time_and_complexity(timing_data, output_dir)
        plot_count += 1

        print("  [2] RTF vs Window Size")
        plot_rtf(timing_data, output_dir)
        plot_count += 1
    else:
        print("\n⚠ Skipping reconstruction timing plots (no timing data)")

    # Accuracy plots (reconstruction test)
    if accuracy_data is not None:
        print("\n[Reconstruction Test - Accuracy]")
        print("  [3] MSE vs Frequency")
        plot_mse_vs_frequency(accuracy_data, output_dir)
        plot_count += 1

        print("  [4] MSE vs Alpha (by Frequency)")
        recon_y_limits = plot_mse_vs_alpha(accuracy_data, output_dir, global_y_limits=y_limits)
        # Update y_limits if we didn't have homomorphic data
        if y_limits is None:
            y_limits = recon_y_limits
        plot_count += 1
    else:
        print("\n⚠ Skipping reconstruction accuracy plots (no accuracy data)")

    # Homomorphic plots
    if homomorphic_data is not None:
        print("\n[Homomorphic Property Test]")
        print("  [5] Homomorphic MSE vs Alpha")
        plot_homomorphic_mse_vs_alpha(homomorphic_data, output_dir, global_y_limits=y_limits)
        plot_count += 1

        print("  [6] Homomorphic MSE Heatmap")
        plot_homomorphic_mse_heatmap(homomorphic_data, output_dir)
        plot_count += 1
    else:
        print("\n⚠ Skipping homomorphic plots (no homomorphic data)")

    # Summary
    print("\n" + "=" * 70)
    print("  Plotting Complete!")
    print("=" * 70)

    print(f"\n✓ Generated {plot_count} plots in: {output_dir}/")
    print()

    if timing_data is not None:
        print("Reconstruction Test - Performance:")
        print(f"  • {output_dir}/processing_time_complexity_n{{trials}}.png")
        print(f"  • {output_dir}/rtf_analysis_n{{trials}}.png")

    if accuracy_data is not None:
        print("\nReconstruction Test - Accuracy:")
        print(f"  • {output_dir}/mse_vs_frequency_n{{combinations}}.png")
        print(f"  • {output_dir}/mse_vs_alpha_n{{combinations}}.png")

    if homomorphic_data is not None:
        print("\nHomomorphic Property Test:")
        print(f"  • {output_dir}/homomorphic_mse_vs_alpha_n{{combinations}}.png")
        print(f"  • {output_dir}/homomorphic_mse_heatmap.png")

    print()

    return 0


if __name__ == '__main__':
    exit(main())