#!/usr/bin/env python3
"""
Analyze FRFT vs FFT extra analysis results.
Covers two test types:
  1. Exact-bin sine waves  (no spectral leakage)
  2. Impulse response      (analytical ground truth: FRFT output vs 1/sqrt(N))
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


# ============================================================================
# Loaders
# ============================================================================

def load_tsv(filename):
    try:
        data = pd.read_csv(filename, sep='\t')
        print(f"  ✓ Loaded {len(data)} records from {filename}")
        return data
    except Exception as e:
        print(f"  ✗ Error loading {filename}: {e}")
        return None


# ============================================================================
# Exact-bin analysis
# ============================================================================

def analyze_exact_bin(results_dir, output_dir):
    print("\n" + "=" * 80)
    print("EXACT-BIN SINE ANALYSIS")
    print("=" * 80)

    results_file = results_dir / 'exact_bin' / 'results.txt'
    if not results_file.exists():
        print(f"  ⚠ Not found: {results_file}")
        return

    data = load_tsv(results_file)
    if data is None:
        return

    out_dir = output_dir / 'exact_bin'
    out_dir.mkdir(parents=True, exist_ok=True)

    report_file = out_dir / 'statistics.txt'
    with open(report_file, 'w') as f:
        _write_exact_bin_report(f, data)

    # Also print to stdout
    _write_exact_bin_report(None, data)

    print(f"\n  ✓ Report saved to: {report_file}")


def _write_exact_bin_report(f, data):
    def w(line=''):
        print(line)
        if f:
            f.write(line + '\n')

    w("=" * 80)
    w("FRFT vs FFT — Exact-Bin Sine Frequencies")
    w("(Frequencies land exactly on FFT bins → zero spectral leakage)")
    w("=" * 80)

    w(f"\nTotal tests : {len(data)}")
    w(f"Successful  : {data['Success'].sum()}")
    w(f"Success rate: {100 * data['Success'].mean():.2f}%")

    # Overall statistics
    for col, label in [
        ('MSE_Magnitude',  'MSE Magnitude (peak-normalised)'),
        ('MSE_Phase',      'MSE Phase'),
        ('MSE_Complex',    'MSE Complex'),
        ('Correlation_Mag','Correlation (Magnitude)'),
        ('MaxError_Mag',   'Max Error (Magnitude, normalised)'),
        ('MeanError_Mag',  'Mean Error (Magnitude, normalised)'),
    ]:
        w(f"\n{label}:")
        w(f"  Mean   : {data[col].mean():.6e}")
        w(f"  Median : {data[col].median():.6e}")
        w(f"  Std    : {data[col].std():.6e}")
        w(f"  Min    : {data[col].min():.6e}")
        w(f"  Max    : {data[col].max():.6e}")

    # By window size
    w("\n" + "=" * 80)
    w("BY WINDOW SIZE")
    w("=" * 80)
    by_ws = data.groupby('WindowSize').agg({
        'MSE_Complex':     ['mean', 'median', 'max'],
        'MSE_Magnitude':   ['mean', 'median', 'max'],
        'Correlation_Mag': ['mean', 'min'],
        'Success':         'sum',
    }).round(15)
    w(by_ws.to_string())

    # By frequency (condensed — median complex MSE per frequency)
    w("\n" + "=" * 80)
    w("BY FREQUENCY — Median Complex MSE (across all window sizes)")
    w("=" * 80)
    by_freq = data.groupby('Frequency').agg(
        MSE_Complex_median=('MSE_Complex', 'median'),
        MSE_Complex_max=('MSE_Complex', 'max'),
        n=('MSE_Complex', 'count'),
    ).reset_index()
    w(by_freq.to_string(index=False))


# ============================================================================
# Impulse analysis
# ============================================================================

def analyze_impulse(results_dir, output_dir):
    print("\n" + "=" * 80)
    print("IMPULSE RESPONSE ANALYSIS")
    print("=" * 80)

    results_file    = results_dir / 'impulse' / 'results.txt'
    analytical_file = results_dir / 'impulse' / 'analytical_error.txt'

    if not results_file.exists():
        print(f"  ⚠ Not found: {results_file}")
        return

    data      = load_tsv(results_file)
    anal_data = load_tsv(analytical_file) if analytical_file.exists() else None

    if data is None:
        return

    out_dir = output_dir / 'impulse'
    out_dir.mkdir(parents=True, exist_ok=True)

    report_file = out_dir / 'statistics.txt'
    with open(report_file, 'w') as f:
        _write_impulse_report(f, data, anal_data)

    _write_impulse_report(None, data, anal_data)

    print(f"\n  ✓ Report saved to: {report_file}")


def _write_impulse_report(f, data, anal_data):
    def w(line=''):
        print(line)
        if f:
            f.write(line + '\n')

    w("=" * 80)
    w("FRFT vs FFT — Impulse Response (delta at sample 0)")
    w("=" * 80)
    w("\nFRFT(α=1) vs FFT comparison on a unit impulse signal.")
    w("The FFT of a unit impulse is 1/sqrt(N) everywhere (after sqrt(N) normalisation),")
    w("so this test has no spectral leakage and a known analytical ground truth.")

    w(f"\nTotal tests : {len(data)}")
    w(f"Successful  : {data['Success'].sum()}")

    w("\n--- FRFT vs FFT (both subject to the same numerical noise) ---")
    for col, label in [
        ('MSE_Complex',    'MSE Complex'),
        ('MSE_Magnitude',  'MSE Magnitude'),
        ('Correlation_Mag','Correlation (Magnitude)'),
        ('MaxError_Mag',   'Max Error (Magnitude)'),
    ]:
        w(f"\n{label}:")
        w(f"  Mean   : {data[col].mean():.6e}")
        w(f"  Median : {data[col].median():.6e}")
        w(f"  Min    : {data[col].min():.6e}")
        w(f"  Max    : {data[col].max():.6e}")

    # By window size
    w("\n" + "=" * 80)
    w("BY WINDOW SIZE — FRFT vs FFT")
    w("=" * 80)
    by_ws = data.groupby('WindowSize').agg({
        'MSE_Complex':     ['mean', 'max'],
        'MSE_Magnitude':   ['mean', 'max'],
        'Correlation_Mag': ['mean', 'min'],
    }).round(15)
    w(by_ws.to_string())

    # Analytical comparison
    if anal_data is not None:
        w("\n" + "=" * 80)
        w("BY WINDOW SIZE — FRFT vs Analytical Ground Truth (1/sqrt(N))")
        w("(Directly measures absolute FRFT numerical error, no FFT involved)")
        w("=" * 80)
        w(anal_data.to_string(index=False))

        # Trend check
        ws   = anal_data['WindowSize'].values
        mse  = anal_data['MSE_Complex_vs_Analytical'].values
        if len(ws) > 2:
            log_ws  = np.log2(ws)
            log_mse = np.log10(mse)
            slope, intercept = np.polyfit(log_ws, log_mse, 1)
            w(f"\n  Log-log slope (MSE vs log2(N)): {slope:.4f}")
            w(f"  (A positive slope confirms error grows with window size)")
            if slope > 0.05:
                w(f"  → Error grows with N: consistent with accumulated FP arithmetic.")
            else:
                w(f"  → Error does not grow significantly with N.")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze FRFT vs FFT extra analysis results (exact-bin + impulse)'
    )
    parser.add_argument('--results-dir', type=str,
                        default='test_results/fft_comparison_extra_analysis',
                        help='Directory containing exact_bin/ and impulse/ sub-dirs')
    parser.add_argument('--output-dir', type=str,
                        default='test_results/fft_comparison_extra_analysis/analysis',
                        help='Output directory for reports')
    parser.add_argument('--exact-bin-only', action='store_true')
    parser.add_argument('--impulse-only',   action='store_true')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_exact_bin = not args.impulse_only
    run_impulse   = not args.exact_bin_only

    print("\n" + "=" * 80)
    print("FRFT vs FFT — Extra Analysis")
    print("=" * 80)
    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {output_dir}")

    if run_exact_bin:
        analyze_exact_bin(results_dir, output_dir)

    if run_impulse:
        analyze_impulse(results_dir, output_dir)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80 + "\n")
    return 0


if __name__ == '__main__':
    exit(main())