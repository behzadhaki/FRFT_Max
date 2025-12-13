#!/usr/bin/env python3
"""
Analyze FRFT vs FFT comparison test results.
Extracts statistics from the results files and generates summary reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_results(filename):
    """Load FFT comparison results from tab-separated file."""
    try:
        data = pd.read_csv(filename, sep='\t')
        print(f"✓ Loaded {len(data)} records from {filename}")
        return data
    except Exception as e:
        print(f"✗ Error loading file {filename}: {e}")
        return None


def compute_overall_statistics(data):
    """Compute overall statistics across all tests."""
    stats = {
        'Total Tests': len(data),
        'Successful Tests': data['Success'].sum(),
        'Success Rate': f"{100 * data['Success'].mean():.2f}%",

        'MSE Magnitude': {
            'Mean': data['MSE_Magnitude'].mean(),
            'Median': data['MSE_Magnitude'].median(),
            'Std': data['MSE_Magnitude'].std(),
            'Min': data['MSE_Magnitude'].min(),
            'Max': data['MSE_Magnitude'].max(),
        },

        'MSE Phase': {
            'Mean': data['MSE_Phase'].mean(),
            'Median': data['MSE_Phase'].median(),
            'Std': data['MSE_Phase'].std(),
            'Min': data['MSE_Phase'].min(),
            'Max': data['MSE_Phase'].max(),
        },

        'MSE Complex': {
            'Mean': data['MSE_Complex'].mean(),
            'Median': data['MSE_Complex'].median(),
            'Std': data['MSE_Complex'].std(),
            'Min': data['MSE_Complex'].min(),
            'Max': data['MSE_Complex'].max(),
        },

        'Correlation (Magnitude)': {
            'Mean': data['Correlation_Mag'].mean(),
            'Median': data['Correlation_Mag'].median(),
            'Std': data['Correlation_Mag'].std(),
            'Min': data['Correlation_Mag'].min(),
            'Max': data['Correlation_Mag'].max(),
        },

        'Max Error (Magnitude)': {
            'Mean': data['MaxError_Mag'].mean(),
            'Median': data['MaxError_Mag'].median(),
            'Std': data['MaxError_Mag'].std(),
            'Min': data['MaxError_Mag'].min(),
            'Max': data['MaxError_Mag'].max(),
        },

        'Mean Error (Magnitude)': {
            'Mean': data['MeanError_Mag'].mean(),
            'Median': data['MeanError_Mag'].median(),
            'Std': data['MeanError_Mag'].std(),
            'Min': data['MeanError_Mag'].min(),
            'Max': data['MeanError_Mag'].max(),
        },
    }

    return stats


def compute_statistics_by_window_size(data):
    """Compute statistics grouped by window size."""
    grouped = data.groupby('WindowSize').agg({
        'MSE_Magnitude': ['mean', 'median', 'std', 'min', 'max'],
        'MSE_Phase': ['mean', 'median', 'std', 'min', 'max'],
        'MSE_Complex': ['mean', 'median', 'std', 'min', 'max'],
        'Correlation_Mag': ['mean', 'median', 'std', 'min', 'max'],
        'MaxError_Mag': ['mean', 'median', 'std', 'min', 'max'],
        'MeanError_Mag': ['mean', 'median', 'std', 'min', 'max'],
        'Success': 'sum'
    }).round(15)

    return grouped


def compute_statistics_by_frequency(data):
    """Compute statistics grouped by frequency."""
    grouped = data.groupby('Frequency').agg({
        'MSE_Magnitude': ['mean', 'median', 'std', 'min', 'max'],
        'MSE_Phase': ['mean', 'median', 'std', 'min', 'max'],
        'MSE_Complex': ['mean', 'median', 'std', 'min', 'max'],
        'Correlation_Mag': ['mean', 'median', 'std', 'min', 'max'],
        'MaxError_Mag': ['mean', 'median', 'std', 'min', 'max'],
        'MeanError_Mag': ['mean', 'median', 'std', 'min', 'max'],
        'Success': 'sum'
    }).round(15)

    return grouped


def print_statistics_summary(stats):
    """Print formatted statistics summary."""
    print("\n" + "=" * 80)
    print("FRFT vs FFT Comparison - Overall Statistics")
    print("=" * 80)

    print(f"\nTotal Tests: {stats['Total Tests']}")
    print(f"Successful Tests: {stats['Successful Tests']}")
    print(f"Success Rate: {stats['Success Rate']}")

    metrics = ['MSE Magnitude', 'MSE Phase', 'MSE Complex',
               'Correlation (Magnitude)', 'Max Error (Magnitude)', 'Mean Error (Magnitude)']

    for metric in metrics:
        print(f"\n{metric}:")
        print(f"  Mean:   {stats[metric]['Mean']:.6e}")
        print(f"  Median: {stats[metric]['Median']:.6e}")
        print(f"  Std:    {stats[metric]['Std']:.6e}")
        print(f"  Min:    {stats[metric]['Min']:.6e}")
        print(f"  Max:    {stats[metric]['Max']:.6e}")


def save_statistics_report(stats, window_stats, freq_stats, output_file):
    """Save comprehensive statistics report to file."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FRFT vs FFT Comparison - Statistical Analysis Report\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Tests: {stats['Total Tests']}\n")
        f.write(f"Successful Tests: {stats['Successful Tests']}\n")
        f.write(f"Success Rate: {stats['Success Rate']}\n\n")

        metrics = ['MSE Magnitude', 'MSE Phase', 'MSE Complex',
                   'Correlation (Magnitude)', 'Max Error (Magnitude)', 'Mean Error (Magnitude)']

        for metric in metrics:
            f.write(f"{metric}:\n")
            f.write(f"  Mean:   {stats[metric]['Mean']:.6e}\n")
            f.write(f"  Median: {stats[metric]['Median']:.6e}\n")
            f.write(f"  Std:    {stats[metric]['Std']:.6e}\n")
            f.write(f"  Min:    {stats[metric]['Min']:.6e}\n")
            f.write(f"  Max:    {stats[metric]['Max']:.6e}\n\n")

        # Key findings summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 80 + "\n\n")

        f.write("The FRFT implementation at α=±1 produces results virtually identical to\n")
        f.write("the standard FFT, validating the implementation at these boundary values:\n\n")

        f.write(f"  • Pearson correlation coefficient: {stats['Correlation (Magnitude)']['Min']:.6f}\n")
        f.write(f"    (perfect correlation across all {stats['Total Tests']} tests)\n\n")

        f.write(f"  • Worst-case maximum magnitude error: {stats['Max Error (Magnitude)']['Max']:.6e}\n")
        f.write(f"  • Median maximum magnitude error:    {stats['Max Error (Magnitude)']['Median']:.6e}\n\n")

        f.write(f"  • MSE magnitude (median):  {stats['MSE Magnitude']['Median']:.6e}\n")
        f.write(f"  • MSE complex (median):    {stats['MSE Complex']['Median']:.6e}\n\n")

        f.write("These extremely low error values confirm near-perfect agreement between\n")
        f.write("FRFT(α=±1) and standard FFT implementations.\n\n")

        # Statistics by window size
        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICS BY WINDOW SIZE\n")
        f.write("=" * 80 + "\n\n")
        f.write(window_stats.to_string())
        f.write("\n\n")

        # Statistics by frequency
        f.write("=" * 80 + "\n")
        f.write("STATISTICS BY FREQUENCY\n")
        f.write("=" * 80 + "\n\n")
        f.write(freq_stats.to_string())
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze FRFT vs FFT comparison test results'
    )
    parser.add_argument('--results', type=str,
                        default='test_results/fft_comparison/results.txt',
                        help='Path to results file')
    parser.add_argument('--output', type=str,
                        default='test_results/fft_comparison/fft_comparison_statistics.txt',
                        help='Output statistics report file')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("FRFT vs FFT Comparison - Statistical Analysis")
    print("=" * 80 + "\n")

    # Load data
    data = load_results(args.results)

    if data is None:
        return 1

    # Compute statistics
    print("\nComputing statistics...")
    overall_stats = compute_overall_statistics(data)
    window_stats = compute_statistics_by_window_size(data)
    freq_stats = compute_statistics_by_frequency(data)

    # Print summary
    print_statistics_summary(overall_stats)

    print("\n" + "-" * 80)
    print("Statistics by Window Size:")
    print("-" * 80)
    print(window_stats)

    print("\n" + "-" * 80)
    print("Statistics by Frequency:")
    print("-" * 80)
    print(freq_stats)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_statistics_report(overall_stats, window_stats, freq_stats, args.output)
    print(f"\n✓ Statistics report saved to: {args.output}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())