#!/usr/bin/env python3
"""
Generate AES-compatible LaTeX table for RTF timing performance
Converts rtf_all_alphas_aggregated.png plot to a table format
Three columns: α < 0.5, α ≥ 0.5, and All α
Uses detailed timing data to calculate proper mean ± std
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def load_timing_data(filename):
    """Load timing results from tab-separated file."""
    try:
        data = pd.read_csv(filename, sep='\t')
        print(f"✓ Loaded {len(data)} records from {filename}")
        return data
    except Exception as e:
        print(f"✗ Error loading file {filename}: {e}")
        return None


def generate_rtf_latex_table(data, output_filename='rtf_table.tex', use_small_font=True):
    """
    Generate AES-compatible LaTeX table with:
    - Columns: Three columns - α < 0.5, α ≥ 0.5, All α
    - Rows: Window sizes (64 to max)
    - Cells: mean ± std of RTF

    Args:
        data: DataFrame with timing data (detailed, with individual measurements)
        output_filename: Output .tex file path
        use_small_font: If True, use \\small font size for compact tables
    """

    # Filter data to start from window size 64
    data = data[data['window_size'] >= 64].copy()

    # Get unique window sizes and alphas, sorted
    window_sizes = sorted(data['window_size'].unique())
    alpha_values = sorted(data['alpha'].unique())

    print(f"Window sizes: {window_sizes}")
    print(f"Alpha values: {alpha_values}")

    # Create the table data structure
    table_data = {}

    # Process each window size
    for ws in window_sizes:
        table_data[ws] = {}

        # Group 1: α < 0.5
        subset_low = data[(data['window_size'] == ws) & (data['alpha'] < 0.5)]
        if len(subset_low) > 0:
            mean_rtf = subset_low['rt_factor'].mean()
            std_rtf = subset_low['rt_factor'].std()
            table_data[ws]['low'] = (mean_rtf, std_rtf)
        else:
            table_data[ws]['low'] = (np.nan, np.nan)

        # Group 2: α ≥ 0.5
        subset_high = data[(data['window_size'] == ws) & (data['alpha'] >= 0.5)]
        if len(subset_high) > 0:
            mean_rtf = subset_high['rt_factor'].mean()
            std_rtf = subset_high['rt_factor'].std()
            table_data[ws]['high'] = (mean_rtf, std_rtf)
        else:
            table_data[ws]['high'] = (np.nan, np.nan)

        # Group 3: All α
        subset_all = data[data['window_size'] == ws]
        if len(subset_all) > 0:
            mean_rtf = subset_all['rt_factor'].mean()
            std_rtf = subset_all['rt_factor'].std()
            table_data[ws]['all'] = (mean_rtf, std_rtf)
        else:
            table_data[ws]['all'] = (np.nan, np.nan)

    # Generate LaTeX table
    latex_lines = []

    # Table header
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    if use_small_font:
        latex_lines.append("\\small")  # Use smaller font for better fit
    latex_lines.append("\\caption{Real-Time Factor (RTF) performance across window sizes and FRFT orders. Values show mean $\\pm$ standard deviation.}")
    latex_lines.append("\\label{tab:rtf_performance}")

    # Column specification: first column for window size, then 3 columns
    col_spec = "lccc"
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\hline\\hline")  # Double line for AES style

    # Header row
    header = "Window Size"
    header += " & $\\alpha<0.5$"
    header += " & $\\alpha\\geq0.5$"
    header += " & All $\\alpha$"
    header += " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")

    # Data rows
    for ws in window_sizes:
        row = f"{ws}"

        # Add all three columns
        for group_key in ['low', 'high', 'all']:
            mean_val, std_val = table_data[ws][group_key]
            if not np.isnan(mean_val):
                row += f" & ${mean_val:.3f} \\pm {std_val:.3f}$"
            else:
                row += " & ---"

        row += " \\\\"
        latex_lines.append(row)

    # Table footer
    latex_lines.append("\\hline\\hline")  # Double line for AES style
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    # Write to file
    latex_content = '\n'.join(latex_lines)

    with open(output_filename, 'w') as f:
        f.write(latex_content)

    print(f"\n✓ LaTeX table written to {output_filename}")
    print(f"\nTable preview:")
    print("=" * 80)
    print(latex_content)
    print("=" * 80)

    return latex_content


def main():
    parser = argparse.ArgumentParser(
        description='Generate AES-compatible LaTeX table for RTF performance'
    )
    parser.add_argument('--input', type=str,
                        default='test_results/rt_timing_performance_detailed.txt',
                        help='Input timing data file (tab-separated, detailed)')
    parser.add_argument('--output', type=str,
                        default='rtf_table.tex',
                        help='Output LaTeX table file')
    parser.add_argument('--no-small-font', action='store_true',
                        help='Do not use \\small font size (use normal size)')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  RTF Performance LaTeX Table Generator")
    print("=" * 80 + "\n")

    # Load data
    data = load_timing_data(args.input)

    if data is None:
        print("\n✗ Failed to load timing data. Exiting.")
        return 1

    # Generate table
    generate_rtf_latex_table(data, args.output, use_small_font=not args.no_small_font)

    print("\n✓ Done!\n")
    return 0


if __name__ == '__main__':
    exit(main())