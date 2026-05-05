#!/usr/bin/env python3
"""
FRFT Comprehensive Test Suite with MSS Loss and Statistical Analysis
Python equivalent of Frft_test_reconstruction.cpp
"""

import numpy as np
import torch
import frft_cpp
import time
import json
import math # Added for math.pi in signal generation
from dataclasses import dataclass, asdict
from typing import List, Dict
import sys
from pathlib import Path
# Import tqdm for progress bar
from tqdm import tqdm
import itertools

# --- Import Multi-scale Spectrogram Loss (assuming it's in MSS.py) ---
try:
    from MSS import MultiscaleSpectrogramLoss
except ImportError:
    print("Error: Could not import MultiscaleSpectrogramLoss. Ensure MSS.py is available.")
    sys.exit(1)


@dataclass
class TestConfig:
    """Test configuration parameters"""
    window_sizes: List[int] = None
    overlap_factors: List[int] = None  # 1=no overlap, 2=50%, 4=75%
    test_frequencies: List[float] = None
    sample_rate: float = 44100.0
    n_analysis: int = 20  # Number of frames to analyze
    alpha_start: float = 0.0
    alpha_end: float = 2.0
    alpha_step: float = 0.1
    output_filename: str = "frft_test_results_python.txt"
    timing_filename: str = "frft_timing_benchmarks_python.txt"
    stats_filename: str = "frft_statistics_python.json"

    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        if self.overlap_factors is None:
            # Only test for overlap factor 4 (75% overlap) as requested
            self.overlap_factors = [4]
        if self.test_frequencies is None:
            self.test_frequencies = [440.0, 1000.0, 2000.0, 5000.0]

    def get_alpha_steps(self) -> List[float]:
        """Generates the list of alpha values to test"""
        alphas = []
        alpha = self.alpha_start
        while alpha <= self.alpha_end + self.alpha_step / 2: # Add tolerance for float comparison
            alphas.append(round(alpha, 5))
            alpha += self.alpha_step
        return alphas


def generate_input_signal(size: int, frequency: float, sample_rate: float) -> torch.Tensor:
    """Generates a pure sine wave signal (currently unused in favor of complex signal)"""
    t = torch.arange(size) / sample_rate
    # Normalize to -1 to 1 and ensure it's a float tensor
    signal = torch.sin(2.0 * math.pi * frequency * t).double()
    return signal

def generate_complex_signal(size: int, freq_real: float, freq_imag: float, sample_rate: float) -> (np.ndarray, np.ndarray):
    """Generates complex signal with two components"""
    t = np.arange(size) / sample_rate
    # Real component: cosine wave
    real = np.cos(2.0 * np.pi * freq_real * t)
    # Imaginary component: sine wave
    imag = np.sin(2.0 * np.pi * freq_imag * t)
    return real.astype(np.float64), imag.astype(np.float64)


def test_frft_reconstruction(config: TestConfig):
    """
    Runs the comprehensive FRFT test suite and benchmarks.
    """
    frft_engine = frft_cpp.Frft()
    alpha_steps = config.get_alpha_steps()

    # Pre-define all possible MSS scales based on the max window size
    max_possible_scale = max(config.window_sizes)
    # Define a list of scales that are powers of two, descending from the maximum
    all_possible_mss_scales = []
    scale = max_possible_scale
    while scale >= 128: # Use 128 as a reasonable minimum scale for spectrogram analysis
        all_possible_mss_scales.append(scale)
        scale //= 2

    print(f"--- Running FRFT Tests (C++ module: frft_cpp) ---")
    print(f"Total analysis steps (alphas) per window: {len(alpha_steps)}")
    print(f"Max Possible MSS Scales: {all_possible_mss_scales}")

    # Prepare output files
    Path(config.output_filename).write_text("Window,OverlapFactor,Frequency,Alpha,ReconLoss(L),ReconLoss(L+Log)\n")
    Path(config.timing_filename).write_text("Window,OverlapFactor,Frequency,Alpha,Time_ms\n")
    all_stats = {}

    # Total number of iterations for the progress bar
    # The progress bar will track the number of (window, overlap, frequency) combinations
    outer_loop_params = list(itertools.product(
        config.window_sizes,
        config.overlap_factors,
        config.test_frequencies
    ))

    # --- Main Loop with TQDM Progress Bar ---
    # Wrap the iteration over the main test parameters with tqdm
    for window_size, overlap_factor, freq in tqdm(
            outer_loop_params,
            desc="Overall FRFT Testing",
            unit="test_config"
    ):

        # --- CRITICAL FIX: Dynamically set MSS scales based on current window size ---
        # Only use MSS scales that are less than or equal to the current window size (N)
        N = window_size
        current_mss_scales = [
            s for s in all_possible_mss_scales
            if s <= N
        ]

        if not current_mss_scales:
            # Skip if N is too small for any defined MSS scale (e.g., N=16)
            continue

            # Initialize MSS loss function with the dynamically determined scales
        mss_loss_fn = MultiscaleSpectrogramLoss(scales=current_mss_scales, overlap=0.75, verbose=False)

        # --- Prepare Buffers and Input Signals ---

        # Generate input complex signal
        input_real_np, input_imag_np = generate_complex_signal(N, freq, freq, config.sample_rate)

        # Prepare buffers for C++ call (Numpy arrays must be created once)
        output_real_np = np.zeros(N, dtype=np.float64)
        output_imag_np = np.zeros(N, dtype=np.float64)

        # Buffers for Inverse FRFT
        inverse_real_np = np.zeros(N, dtype=np.float64)
        inverse_imag_np = np.zeros(N, dtype=np.float64)

        # Buffers for PyTorch/MSS analysis
        input_signal_torch = torch.from_numpy(input_real_np).double()

        # Store statistics for this combination
        stats_key = f"{N}_{overlap_factor}_{int(freq)}"
        stats_data = {
            'N': N,
            'OverlapFactor': overlap_factor,
            'Frequency': freq,
            'losses_linear': [],
            'losses_combined': [],
            'times_ms': []
        }

        # --- Alpha Loop (Inner Loop) ---
        for alpha in alpha_steps:
            # 1. Forward FRFT (C++ module)
            start_time = time.perf_counter()
            frft_engine.compute(
                input_real_np, input_imag_np,
                output_real_np, output_imag_np,
                alpha
            )
            # 2. Inverse FRFT (C++ module) using the property IFRFT(alpha) = FRFT(-alpha)
            frft_engine.compute(
                output_real_np, output_imag_np,
                inverse_real_np, inverse_imag_np,
                -alpha
            )
            end_time = time.perf_counter()
            time_ms = (end_time - start_time) * 1000.0

            # --- Reconstruction Analysis (PyTorch/MSS) ---
            # Get only the real part of the reconstructed signal for comparison
            # Convert reconstructed numpy array to torch tensor
            reconstructed_real_torch = torch.from_numpy(inverse_real_np).double()

            # The MSS loss calculates loss between two real signals
            # total_loss is unused, we just capture the components
            total_loss, linear_loss, combined_loss = mss_loss_fn(
                input_signal_torch,
                reconstructed_real_torch
            )

            # Store results
            stats_data['losses_linear'].append(linear_loss.item())
            stats_data['losses_combined'].append(combined_loss.item())
            stats_data['times_ms'].append(time_ms)

            # Write individual test results to output file
            with Path(config.output_filename).open('a') as f:
                f.write(f"{N},{overlap_factor},{freq},{alpha:.4f},{linear_loss.item():.8f},{combined_loss.item():.8f}\n")

            # Write timing results
            with Path(config.timing_filename).open('a') as f:
                f.write(f"{N},{overlap_factor},{freq},{alpha:.4f},{time_ms:.8f}\n")

        # After finishing all alpha steps for this configuration, save the aggregated stats
        all_stats[stats_key] = stats_data


    # --- Final Step: Save Aggregated Statistics ---
    print("\nSaving aggregated statistics...")
    # Calculate summary stats (min/max/avg loss and time) for each configuration
    final_stats = {}
    for key, data in all_stats.items():
        if not data['losses_linear']: continue

        final_stats[key] = {
            'N': data['N'],
            'OverlapFactor': data['OverlapFactor'],
            'Frequency': data['Frequency'],
            'Loss_L_Min': np.min(data['losses_linear']),
            'Loss_L_Max': np.max(data['losses_linear']),
            'Loss_L_Avg': np.mean(data['losses_linear']),
            'Loss_Combined_Min': np.min(data['losses_combined']),
            'Loss_Combined_Max': np.max(data['losses_combined']),
            'Loss_Combined_Avg': np.mean(data['losses_combined']),
            'Time_Min_ms': np.min(data['times_ms']),
            'Time_Max_ms': np.max(data['times_ms']),
            'Time_Avg_ms': np.mean(data['times_ms']),
        }

    with Path(config.stats_filename).open('w') as f:
        json.dump(final_stats, f, indent=4)

    print(f"FRFT Test Completed. Results saved to:")
    print(f"  {config.output_filename}")
    print(f"  {config.timing_filename}")
    print(f"  {config.stats_filename}")


def main():
    config = TestConfig()

    # --- Command Line Argument Parsing (Simplified) ---
    def parse_args(config):
        # A simple mechanism to allow for a 'quick' test mode or custom paths
        import argparse
        parser = argparse.ArgumentParser(description="FRFT Test Suite and Benchmarking.")
        parser.add_argument('--output', type=str, default=config.output_filename,
                            help="Output file for detailed loss values.")
        parser.add_argument('--timing', type=str, default=config.timing_filename,
                            help="Output file for detailed timing values.")
        parser.add_argument('--stats', type=str, default=config.stats_filename,
                            help="Output file for aggregated statistics (JSON).")
        parser.add_argument('--sample-rate', type=float, default=config.sample_rate,
                            help="Sample rate for signal generation.")
        parser.add_argument('--n-analysis', type=int, default=config.n_analysis,
                            help="Number of frames to analyze (currently unused in this simplified version).")
        parser.add_argument('--quick', action='store_true',
                            help="Run a quick test with fewer sizes and frequencies.")

        args = parser.parse_args()

        config.output_filename = args.output
        config.timing_filename = args.timing
        config.stats_filename = args.stats
        config.sample_rate = args.sample_rate
        config.n_analysis = args.n_analysis

        if args.quick:
            config.window_sizes = [64, 256, 1024, 4096]
            config.test_frequencies = [440.0, 1000.0, 5000.0]
            # Set to [4] for quick mode as well
            config.overlap_factors = [4]
            # Reduce alpha steps for quicker test
            config.alpha_step = 0.5

        return config


    # Note: The original argument parsing was complex. We use a simplified
    # check here to prevent errors in certain execution environments.
    try:
        # Check if running directly in a shell environment where full args are available
        if '__file__' in locals() or len(sys.argv) > 1:
            # We attempt to use the robust argparse function, which is now defined.
            config = parse_args(config)
    except SystemExit:
        # Argparse prints help and exits, which is fine, but we stop execution here.
        return


    test_frft_reconstruction(config)

if __name__ == "__main__":
    main()