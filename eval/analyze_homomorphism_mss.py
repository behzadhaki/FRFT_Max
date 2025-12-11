#!/usr/bin/env python3
"""
FRFT Homomorphism MSS Grid Analysis Script - Updated for Windowed Mode

Analyzes FRFT(α) vs FRFT(α₂) ∘ FRFT(α₁) using Multi-Scale Spectrogram Loss
on a grid of α₁ and α₂ values. Generates heatmaps showing MSS loss across the grid.

Supports both:
- Direct mode: Full signal FRFT
- Windowed mode: Overlap-add windowed FRFT with multiple window sizes

Also includes commutativity analysis: FRFT(α₂) ∘ FRFT(α₁) vs FRFT(α₁) ∘ FRFT(α₂)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import scipy.io.wavfile as wavfile
from tqdm import tqdm
import argparse

# Import the MSS loss class
from MSS import MultiscaleSpectrogramLoss


def load_wav_as_tensor(filepath: str, use_middle_80_percent: bool = False) -> torch.Tensor:
    """
    Load a WAV file and convert to PyTorch tensor.

    Args:
        filepath: Path to WAV file
        use_middle_80_percent: If True, extract only middle 80% of signal to avoid edge artifacts

    Returns:
        Tuple of (tensor, sample_rate)
    """
    sample_rate, audio = wavfile.read(filepath)

    # Convert to float and normalize
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    # Extract middle 80% if requested (for windowed processing to avoid edge artifacts)
    if use_middle_80_percent and len(audio) > 100:
        signal_length = len(audio)
        # Calculate 10% margins on each side
        margin = int(signal_length * 0.1)
        # Extract middle 80%
        audio = audio[margin:-margin]

    # Convert to torch tensor
    tensor = torch.from_numpy(audio).float()

    return tensor, sample_rate


def compute_mss_for_pair(source_path: str, alpha_path: str, composed_path: str,
                         mss_loss: MultiscaleSpectrogramLoss, use_middle_80: bool = False) -> dict:
    """
    Compute MSS and MSE loss between alpha and composed versions.

    Args:
        source_path: Path to source WAV
        alpha_path: Path to direct FRFT result
        composed_path: Path to composed FRFT result
        mss_loss: MSS loss function
        use_middle_80: If True, use only middle 80% of signal (for windowed mode)
    """

    # Load WAV files
    source, sr1 = load_wav_as_tensor(source_path, use_middle_80_percent=use_middle_80)
    alpha, sr2 = load_wav_as_tensor(alpha_path, use_middle_80_percent=use_middle_80)
    composed, sr3 = load_wav_as_tensor(composed_path, use_middle_80_percent=use_middle_80)

    # Verify sample rates match
    assert sr1 == sr2 == sr3, "Sample rates don't match!"

    # Add batch dimension if needed
    if source.dim() == 1:
        source = source.unsqueeze(0)
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(0)
    if composed.dim() == 1:
        composed = composed.unsqueeze(0)

    # Compute losses
    with torch.no_grad():
        # MSS loss between alpha and composed (homomorphic property test)
        mss_homomorphic = mss_loss(alpha, composed).item()

        # MSE loss between alpha and composed
        mse_homomorphic = torch.mean((alpha - composed) ** 2).item()

    return {
        'mss_homomorphic': mss_homomorphic,
        'mse_homomorphic': mse_homomorphic
    }


def compute_commutativity_mss(composed_12_path: str, composed_21_path: str,
                              mss_loss: MultiscaleSpectrogramLoss, use_middle_80: bool = False) -> dict:
    """
    Compute MSS and MSE loss between two compositions in different orders.

    Tests: FRFT(α₂) ∘ FRFT(α₁) vs FRFT(α₁) ∘ FRFT(α₂)

    Args:
        composed_12_path: Path to FRFT(α₂) ∘ FRFT(α₁)
        composed_21_path: Path to FRFT(α₁) ∘ FRFT(α₂)
        mss_loss: MSS loss function
        use_middle_80: If True, use only middle 80% of signal (for windowed mode)

    Returns:
        Dictionary with 'mss' and 'mse' keys
    """
    # Load WAV files
    composed_12, sr1 = load_wav_as_tensor(composed_12_path, use_middle_80_percent=use_middle_80)
    composed_21, sr2 = load_wav_as_tensor(composed_21_path, use_middle_80_percent=use_middle_80)

    # Verify sample rates match
    assert sr1 == sr2, "Sample rates don't match!"

    # Add batch dimension if needed
    if composed_12.dim() == 1:
        composed_12 = composed_12.unsqueeze(0)
    if composed_21.dim() == 1:
        composed_21 = composed_21.unsqueeze(0)

    # Compute losses
    with torch.no_grad():
        # MSS loss
        mss = mss_loss(composed_12, composed_21).item()

        # MSE loss
        mse = torch.mean((composed_12 - composed_21) ** 2).item()

    return {'mss': mss, 'mse': mse}


def main():
    parser = argparse.ArgumentParser(description='Analyze FRFT homomorphism using MSS loss')
    parser.add_argument('--dir', type=str, default=None,
                        help='Base directory (auto-detects windowed vs direct mode)')
    parser.add_argument('--windowed', action='store_true',
                        help='Force windowed mode analysis')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("FRFT Homomorphism MSS Grid Analysis")
    print("="*70 + "\n")

    # Determine base directory and mode
    if args.dir:
        base_dir = Path(args.dir)
    else:
        # Auto-detect: check for windowed directory first
        if Path("homomorphism_mss_grid_windowed").exists():
            base_dir = Path("homomorphism_mss_grid_windowed")
            print("✓ Auto-detected WINDOWED mode")
        elif Path("homomorphism_mss_grid").exists():
            base_dir = Path("homomorphism_mss_grid")
            print("✓ Auto-detected DIRECT mode")
        else:
            print("❌ Error: No grid directory found!")
            print("Please run the WAV generator first or specify --dir")
            return

    is_windowed = "windowed" in str(base_dir) or args.windowed

    print(f"Base directory: {base_dir}")
    print(f"Processing mode: {'WINDOWED' if is_windowed else 'DIRECT'}")
    if is_windowed:
        print(f"  Using middle 80% of signal to avoid edge artifacts\n")
    else:
        print()

    metadata_file = base_dir / "metadata.txt"
    plots_dir = base_dir / "plots"

    # Create plots directory
    plots_dir.mkdir(exist_ok=True)

    # Check if base directory exists
    if not base_dir.exists():
        print(f"Error: Directory '{base_dir}' not found!")
        return

    # Check if metadata file exists
    if not metadata_file.exists():
        print(f"Error: Metadata file '{metadata_file}' not found!")
        return

    # Load metadata
    print(f"Loading metadata from {metadata_file}...")
    metadata = pd.read_csv(metadata_file, sep='\t', comment='#')

    # Convert file path columns to string and drop any NaN rows
    metadata['Source_File'] = metadata['Source_File'].astype(str)
    metadata['Alpha_File'] = metadata['Alpha_File'].astype(str)
    metadata['Composed_File'] = metadata['Composed_File'].astype(str)

    # Remove any rows with 'nan' in file paths
    metadata = metadata[metadata['Source_File'] != 'nan']
    metadata = metadata[metadata['Alpha_File'] != 'nan']
    metadata = metadata[metadata['Composed_File'] != 'nan']

    print(f"✓ Loaded {len(metadata)} valid entries")
    print(f"  Frequencies: {sorted(metadata['Frequency'].unique())}")
    print(f"  Grid points: {len(metadata['Alpha1'].unique())} × {len(metadata['Alpha2'].unique())}")

    if is_windowed:
        print(f"  Window sizes: {sorted(metadata['Window'].unique())}")
    print()

    # Initialize MSS loss
    mss_loss = MultiscaleSpectrogramLoss(
        scales=[8192, 4096, 2048],
        overlap=0.75,
        verbose=False
    )
    print("✓ Initialized Multi-Scale Spectrogram Loss")
    print(f"  Scales: {mss_loss.scales}")
    print(f"  Overlap: {mss_loss.overlap}\n")

    # Prepare results storage
    results = []

    # Process each entry
    print("Processing WAV files...")
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Computing MSS"):
        source_path = base_dir / row['Source_File']
        alpha_path = base_dir / row['Alpha_File']
        composed_path = base_dir / row['Composed_File']

        # Check if files exist
        if not all([source_path.exists(), alpha_path.exists(), composed_path.exists()]):
            print(f"\n⚠ Warning: Missing files for frequency {row['Frequency']}, α₁={row['Alpha1']}, α₂={row['Alpha2']}")
            continue

        try:
            # Compute MSS losses
            losses = compute_mss_for_pair(
                str(source_path),
                str(alpha_path),
                str(composed_path),
                mss_loss,
                use_middle_80=True  # Use middle 80% for windowed mode
            )

            # Store results
            result_entry = {
                'Frequency': row['Frequency'],
                'Alpha1': row['Alpha1'],
                'Alpha2': row['Alpha2'],
                'Alpha': row['Alpha'],
                'MSS_Homomorphic': losses['mss_homomorphic'],
                'MSE_Homomorphic': losses['mse_homomorphic']
            }

            if is_windowed:
                result_entry['Window'] = row['Window']

            results.append(result_entry)

        except Exception as e:
            print(f"\n✗ Error processing frequency {row['Frequency']}, α₁={row['Alpha1']}, α₂={row['Alpha2']}: {e}")
            continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results - separate files for each window size if windowed
    if is_windowed:
        for window_size in sorted(results_df['Window'].unique()):
            window_results = results_df[results_df['Window'] == window_size]
            output_file = base_dir / f"mss_grid_results_win_{window_size}.txt"

            print(f"\nSaving results for window {window_size} to {output_file}...")
            with open(output_file, 'w') as f:
                f.write("# FRFT Homomorphism MSS Grid Analysis Results - WINDOWED MODE\n")
                f.write(f"# Window size: {window_size}\n")
                f.write("# Grid: α₁, α₂ ∈ [-2, 2] with step 0.1\n")
                f.write("# MSS_Homomorphic: MSS loss between FRFT(α) and FRFT(α₂) ∘ FRFT(α₁)\n")
                f.write("# MSE_Homomorphic: Mean Squared Error between FRFT(α) and FRFT(α₂) ∘ FRFT(α₁)\n")
                f.write("# Lower values indicate better homomorphic property\n\n")

            window_results.to_csv(output_file, sep='\t', index=False, mode='a')
            print(f"✓ Saved {len(window_results)} results for window {window_size}")
    else:
        output_file = base_dir / "mss_grid_results.txt"
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w') as f:
            f.write("# FRFT Homomorphism MSS Grid Analysis Results - DIRECT MODE\n")
            f.write("# Grid: α₁, α₂ ∈ [-2, 2] with step 0.1\n")
            f.write("# MSS_Homomorphic: MSS loss between FRFT(α) and FRFT(α₂) ∘ FRFT(α₁)\n")
            f.write("# MSE_Homomorphic: Mean Squared Error between FRFT(α) and FRFT(α₂) ∘ FRFT(α₁)\n")
            f.write("# Lower values indicate better homomorphic property\n\n")

        results_df.to_csv(output_file, sep='\t', index=False, mode='a')
        print(f"✓ Saved {len(results_df)} results")

    # =========================================================================
    # COMMUTATIVITY ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("COMMUTATIVITY ANALYSIS")
    print("="*70)
    print("\nTesting: FRFT(α₂) ∘ FRFT(α₁) vs FRFT(α₁) ∘ FRFT(α₂)")
    print("This directly compares the composed results in different orders\n")

    commutativity_results = []

    # Get unique combinations depending on mode
    if is_windowed:
        combinations = [(freq, win) for freq in sorted(metadata['Frequency'].unique())
                        for win in sorted(metadata['Window'].unique())]
    else:
        combinations = [(freq, None) for freq in sorted(metadata['Frequency'].unique())]

    # Group metadata by frequency (and window if windowed) for easier lookup
    for combo in combinations:
        if is_windowed:
            freq, window_size = combo
            freq_meta = metadata[(metadata['Frequency'] == freq) & (metadata['Window'] == window_size)]
            print(f"Processing {freq} Hz, window {window_size}...")
        else:
            freq = combo[0]
            freq_meta = metadata[metadata['Frequency'] == freq]
            print(f"Processing {freq} Hz...")

        # Create lookup dictionary
        # Key: (alpha1, alpha2) -> composed_file
        lookup = {}
        for _, row in freq_meta.iterrows():
            key = (row['Alpha1'], row['Alpha2'])
            lookup[key] = base_dir / row['Composed_File']

        # Find pairs where both (α₁, α₂) and (α₂, α₁) exist
        processed_pairs = set()
        pair_count = 0

        for (a1, a2), path_12 in lookup.items():
            # Skip if we've already processed this pair
            if (a1, a2) in processed_pairs or (a2, a1) in processed_pairs:
                continue

            # Skip diagonal (α₁ = α₂)
            if abs(a1 - a2) < 1e-9:
                continue

            # Check if reverse pair exists
            if (a2, a1) not in lookup:
                continue

            path_21 = lookup[(a2, a1)]

            # Check if both files exist
            if not path_12.exists() or not path_21.exists():
                continue

            try:
                # Compute commutativity MSS and MSE
                comm_losses = compute_commutativity_mss(
                    str(path_12),
                    str(path_21),
                    mss_loss,
                    use_middle_80=True  # Use middle 80% for windowed mode
                )

                comm_entry = {
                    'Frequency': freq,
                    'Alpha1': a1,
                    'Alpha2': a2,
                    'MSS_Commutativity': comm_losses['mss'],
                    'MSE_Commutativity': comm_losses['mse']
                }

                if is_windowed:
                    comm_entry['Window'] = window_size

                commutativity_results.append(comm_entry)

                processed_pairs.add((a1, a2))
                processed_pairs.add((a2, a1))
                pair_count += 1

            except Exception as e:
                print(f"  ✗ Error with pair ({a1:.1f}, {a2:.1f}): {e}")
                continue

        print(f"  ✓ Processed {pair_count} unique pairs")

    # Convert to DataFrame
    comm_df = pd.DataFrame(commutativity_results)

    # Save commutativity results
    if is_windowed:
        for window_size in sorted(comm_df['Window'].unique()):
            window_comm = comm_df[comm_df['Window'] == window_size]
            comm_output_file = base_dir / f"mss_commutativity_results_win_{window_size}.txt"

            print(f"\nSaving commutativity results for window {window_size} to {comm_output_file}...")
            with open(comm_output_file, 'w') as f:
                f.write("# FRFT Commutativity Analysis Results - WINDOWED MODE\n")
                f.write(f"# Window size: {window_size}\n")
                f.write("# MSS_Commutativity: MSS loss between FRFT(α₂)∘FRFT(α₁) and FRFT(α₁)∘FRFT(α₂)\n")
                f.write("# MSE_Commutativity: Mean Squared Error between the two compositions\n")
                f.write("# Lower values indicate better commutativity\n")
                f.write("# Perfect commutativity would give MSS = 0 and MSE = 0\n\n")

            window_comm.to_csv(comm_output_file, sep='\t', index=False, mode='a')
            print(f"✓ Saved {len(window_comm)} commutativity test results for window {window_size}")
    else:
        comm_output_file = base_dir / "mss_commutativity_results.txt"
        print(f"\nSaving commutativity results to {comm_output_file}...")
        with open(comm_output_file, 'w') as f:
            f.write("# FRFT Commutativity Analysis Results - DIRECT MODE\n")
            f.write("# MSS_Commutativity: MSS loss between FRFT(α₂)∘FRFT(α₁) and FRFT(α₁)∘FRFT(α₂)\n")
            f.write("# MSE_Commutativity: Mean Squared Error between the two compositions\n")
            f.write("# Lower values indicate better commutativity\n")
            f.write("# Perfect commutativity would give MSS = 0 and MSE = 0\n\n")

        comm_df.to_csv(comm_output_file, sep='\t', index=False, mode='a')
        print(f"✓ Saved {len(comm_df)} commutativity test results")

    # Print statistics
    print("\n" + "="*70)
    print("Statistics Summary")
    print("="*70)

    if is_windowed:
        # Print stats for each window size
        for window_size in sorted(results_df['Window'].unique()):
            print(f"\n{'='*70}")
            print(f"WINDOW SIZE: {window_size}")
            print(f"{'='*70}")

            window_results = results_df[results_df['Window'] == window_size]
            window_comm = comm_df[comm_df['Window'] == window_size]

            print_statistics(window_results, window_comm)
    else:
        print_statistics(results_df, comm_df)

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nResults saved to: {base_dir}/")
    print(f"\nTo generate visualizations, run:")
    print(f"  python3 plot_results.py --mss {base_dir}")
    print()


def print_statistics(results_df, comm_df):
    """Print statistics for homomorphic and commutativity tests."""

    print(f"\n1. Homomorphic - MSS Loss (FRFT(α) vs FRFT(α₂)∘FRFT(α₁)):")
    print(f"  Mean:   {results_df['MSS_Homomorphic'].mean():.6f}")
    print(f"  Median: {results_df['MSS_Homomorphic'].median():.6f}")
    print(f"  Std:    {results_df['MSS_Homomorphic'].std():.6f}")
    print(f"  Min:    {results_df['MSS_Homomorphic'].min():.6f}")
    print(f"  Max:    {results_df['MSS_Homomorphic'].max():.6f}")

    print(f"\n  By Frequency:")
    freq_stats = results_df.groupby('Frequency')['MSS_Homomorphic'].agg(['mean', 'std', 'min', 'max'])
    print("  " + freq_stats.to_string().replace('\n', '\n  '))

    print(f"\n2. Homomorphic - MSE Loss:")
    print(f"  Mean:   {results_df['MSE_Homomorphic'].mean():.6e}")
    print(f"  Median: {results_df['MSE_Homomorphic'].median():.6e}")
    print(f"  Std:    {results_df['MSE_Homomorphic'].std():.6e}")
    print(f"  Min:    {results_df['MSE_Homomorphic'].min():.6e}")
    print(f"  Max:    {results_df['MSE_Homomorphic'].max():.6e}")

    print(f"\n  By Frequency:")
    mse_freq_stats = results_df.groupby('Frequency')['MSE_Homomorphic'].agg(['mean', 'std', 'min', 'max'])
    print("  " + mse_freq_stats.to_string().replace('\n', '\n  '))

    if len(comm_df) > 0:
        print(f"\n3. Commutativity - MSS Loss (FRFT(α₂)∘FRFT(α₁) vs FRFT(α₁)∘FRFT(α₂)):")
        print(f"  Mean:   {comm_df['MSS_Commutativity'].mean():.6f}")
        print(f"  Median: {comm_df['MSS_Commutativity'].median():.6f}")
        print(f"  Std:    {comm_df['MSS_Commutativity'].std():.6f}")
        print(f"  Min:    {comm_df['MSS_Commutativity'].min():.6f}")
        print(f"  Max:    {comm_df['MSS_Commutativity'].max():.6f}")

        print(f"\n  By Frequency:")
        comm_freq_stats = comm_df.groupby('Frequency')['MSS_Commutativity'].agg(['mean', 'std', 'min', 'max'])
        print("  " + comm_freq_stats.to_string().replace('\n', '\n  '))

        print(f"\n4. Commutativity - MSE Loss:")
        print(f"  Mean:   {comm_df['MSE_Commutativity'].mean():.6e}")
        print(f"  Median: {comm_df['MSE_Commutativity'].median():.6e}")
        print(f"  Std:    {comm_df['MSE_Commutativity'].std():.6e}")
        print(f"  Min:    {comm_df['MSE_Commutativity'].min():.6e}")
        print(f"  Max:    {comm_df['MSE_Commutativity'].max():.6e}")

        print(f"\n  By Frequency:")
        mse_comm_freq_stats = comm_df.groupby('Frequency')['MSE_Commutativity'].agg(['mean', 'std', 'min', 'max'])
        print("  " + mse_comm_freq_stats.to_string().replace('\n', '\n  '))

        # Commutativity assessment
        print(f"\n5. Commutativity Assessment (MSS-based):")
        perfect_comm = (comm_df['MSS_Commutativity'] < 0.001).sum()
        good_comm = (comm_df['MSS_Commutativity'] < 0.01).sum()
        poor_comm = (comm_df['MSS_Commutativity'] > 0.1).sum()
        total_pairs = len(comm_df)

        print(f"  Perfect (MSS < 0.001):  {perfect_comm:5d} / {total_pairs} ({100*perfect_comm/total_pairs:.1f}%)")
        print(f"  Good    (MSS < 0.01):   {good_comm:5d} / {total_pairs} ({100*good_comm/total_pairs:.1f}%)")
        print(f"  Poor    (MSS > 0.1):    {poor_comm:5d} / {total_pairs} ({100*poor_comm/total_pairs:.1f}%)")


if __name__ == "__main__":
    main()