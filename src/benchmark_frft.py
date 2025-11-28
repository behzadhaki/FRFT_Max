#!/usr/bin/env python3
"""
FRFT Comprehensive Test Suite with MSS Loss and Statistical Analysis
Python equivalent of Frft_test_reconstruction.cpp
"""

import numpy as np
import torch
import frft_cpp
from MSS import MultiscaleSpectrogramLoss
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict
import sys
from pathlib import Path

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
            self.overlap_factors = [1, 2, 4]
        if self.test_frequencies is None:
            self.test_frequencies = [100.0, 220.0, 440.0, 1000.0, 2000.0, 5000.0, 10000.0]


@dataclass
class TestResult:
    """Result for a single test"""
    window_size: int
    overlap_factor: int
    hop_size: int
    frequency: float
    alpha: float
    mse: float
    mss_loss: float
    max_error: float
    mean_error: float
    std_error: float  # New: standard deviation of error
    num_frames: int
    forward_time_ms: float
    inverse_time_ms: float
    total_time_ms: float
    success: bool


@dataclass
class Statistics:
    """Statistical summary for a group of tests"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    count: int


class FRFTTestSuite:
    """Comprehensive FRFT testing suite"""

    def __init__(self, config: TestConfig):
        self.config = config
        self.mss_loss = MultiscaleSpectrogramLoss(
            scales=[2048, 1024, 512],
            overlap=0.75,
            verbose=False
        )
        self.results: List[TestResult] = []

    def generate_sine_wave(self, size: int, frequency: float, phase: float = 0.0) -> np.ndarray:
        """Generate a sinusoidal signal"""
        t = np.arange(size) / self.config.sample_rate
        return np.sin(2.0 * np.pi * frequency * t + phase)

    def calculate_statistics(self, values: List[float]) -> Statistics:
        """Calculate statistical measures"""
        if not values:
            return Statistics(0, 0, 0, 0, 0, 0)

        arr = np.array(values)
        return Statistics(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            median=float(np.median(arr)),
            count=len(arr)
        )

    def test_frft_roundtrip(
            self,
            window_size: int,
            overlap_factor: int,
            frequency: float,
            alpha: float
    ) -> TestResult:
        """Perform a single round-trip FRFT test"""

        # Calculate hop size
        hop_size = window_size // overlap_factor
        if hop_size < 1:
            hop_size = 1

        # Generate signal for n_analysis frames
        signal_length = window_size + (self.config.n_analysis - 1) * hop_size
        original_signal = self.generate_sine_wave(signal_length, frequency)

        # Prepare output
        reconstructed_signal = np.zeros(signal_length)

        # Process frames
        num_frames = 0
        total_forward_time = 0.0
        total_inverse_time = 0.0

        success = True

        for pos in range(0, signal_length - window_size + 1, hop_size):
            # Extract frame
            frame = original_signal[pos:pos + window_size]

            try:
                # Time forward FRFT
                start_time = time.perf_counter()
                real_frft, imag_frft = frft_cpp.frft_compute_real(frame, alpha)
                forward_time = (time.perf_counter() - start_time) * 1000  # ms
                total_forward_time += forward_time

                # Time inverse FRFT
                start_time = time.perf_counter()
                reconstructed_frame, _ = frft_cpp.frft_compute(real_frft, imag_frft, -alpha)
                inverse_time = (time.perf_counter() - start_time) * 1000  # ms
                total_inverse_time += inverse_time

                # Store reconstructed frame (simple copy, no overlap-add)
                reconstructed_signal[pos:pos + window_size] = reconstructed_frame

                num_frames += 1

            except Exception as e:
                print(f"Error in FRFT computation: {e}")
                success = False
                break

        if not success or num_frames == 0:
            return TestResult(
                window_size=window_size,
                overlap_factor=overlap_factor,
                hop_size=hop_size,
                frequency=frequency,
                alpha=alpha,
                mse=-1.0,
                mss_loss=-1.0,
                max_error=-1.0,
                mean_error=-1.0,
                std_error=-1.0,
                num_frames=num_frames,
                forward_time_ms=total_forward_time,
                inverse_time_ms=total_inverse_time,
                total_time_ms=total_forward_time + total_inverse_time,
                success=False
            )

        # Calculate errors
        error = original_signal - reconstructed_signal
        mse = float(np.mean(error ** 2))
        max_error = float(np.max(np.abs(error)))
        mean_error = float(np.mean(np.abs(error)))
        std_error = float(np.std(error))

        # Calculate MSS loss
        try:
            original_torch = torch.from_numpy(original_signal).float().unsqueeze(0)
            reconstructed_torch = torch.from_numpy(reconstructed_signal).float().unsqueeze(0)
            mss_loss_value = self.mss_loss(original_torch, reconstructed_torch).item()
        except Exception as e:
            print(f"Warning: MSS calculation failed: {e}")
            mss_loss_value = -1.0

        return TestResult(
            window_size=window_size,
            overlap_factor=overlap_factor,
            hop_size=hop_size,
            frequency=frequency,
            alpha=alpha,
            mse=mse,
            mss_loss=mss_loss_value,
            max_error=max_error,
            mean_error=mean_error,
            std_error=std_error,
            num_frames=num_frames,
            forward_time_ms=total_forward_time,
            inverse_time_ms=total_inverse_time,
            total_time_ms=total_forward_time + total_inverse_time,
            success=True
        )

    def run_test_suite(self):
        """Run the complete test suite"""

        print("=" * 70)
        print("FRFT Comprehensive Test Suite")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Sample Rate: {self.config.sample_rate} Hz")
        print(f"  Analysis Frames: {self.config.n_analysis}")
        print(f"  Alpha Range: {self.config.alpha_start} to {self.config.alpha_end} (step {self.config.alpha_step})")
        print(f"  Window Sizes: {len(self.config.window_sizes)} sizes")
        print(f"  Frequencies: {len(self.config.test_frequencies)} frequencies")
        print(f"  Overlap Factors: {self.config.overlap_factors}")
        print(f"\n  Output File: {self.config.output_filename}")
        print(f"  Timing File: {self.config.timing_filename}")
        print(f"  Stats File: {self.config.stats_filename}\n")

        # Calculate total tests
        num_alphas = int((self.config.alpha_end - self.config.alpha_start) / self.config.alpha_step) + 1
        total_tests = (len(self.config.window_sizes) *
                       len(self.config.overlap_factors) *
                       len(self.config.test_frequencies) *
                       num_alphas)
        print(f"Total tests to run: {total_tests}\n")

        # Open output file
        with open(self.config.output_filename, 'w') as out_file:
            # Write header
            out_file.write("# FRFT Round-Trip Accuracy Test Results (Python)\n")
            out_file.write(f"# Sample Rate: {self.config.sample_rate} Hz\n")
            out_file.write(f"# Number of Analysis Frames: {self.config.n_analysis}\n")
            out_file.write("#\n")
            out_file.write("# Columns:\n")
            out_file.write("# 1. Window Size\n")
            out_file.write("# 2. Overlap Factor\n")
            out_file.write("# 3. Hop Size\n")
            out_file.write("# 4. Number of Frames\n")
            out_file.write("# 5. Frequency (Hz)\n")
            out_file.write("# 6. Alpha Parameter\n")
            out_file.write("# 7. Forward Time (ms total)\n")
            out_file.write("# 8. Inverse Time (ms total)\n")
            out_file.write("# 9. Total Time (ms)\n")
            out_file.write("# 10. MSE\n")
            out_file.write("# 11. MSS Loss\n")
            out_file.write("# 12. Max Error\n")
            out_file.write("# 13. Mean Error\n")
            out_file.write("# 14. Std Error\n")
            out_file.write("# 15. Success\n")
            out_file.write("#\n")
            out_file.write("WindowSize\tOverlapFactor\tHopSize\tNumFrames\tFrequency\tAlpha\t"
                           "ForwardTime\tInverseTime\tTotalTime\tMSE\tMSS_Loss\tMaxError\t"
                           "MeanError\tStdError\tSuccess\n")

            # Progress tracking
            test_count = 0
            failed_count = 0
            start_time = time.time()

            # Pre-allocate for faster execution
            for window_size in self.config.window_sizes:
                frft_cpp.prepare(window_size)

            # Run tests
            for window_size in self.config.window_sizes:
                print(f"\nTesting window size: {window_size}")
                print("─" * 50)

                for overlap_factor in self.config.overlap_factors:
                    hop_size = window_size // overlap_factor

                    for frequency in self.config.test_frequencies:
                        print(f"  Freq: {frequency:8.1f} Hz, Overlap: {overlap_factor}x (hop={hop_size:4d}) ", end='', flush=True)

                        alpha_results = []
                        alpha_success = 0

                        # Test all alpha values
                        alpha = self.config.alpha_start
                        while alpha <= self.config.alpha_end + 1e-9:  # Small epsilon for float comparison
                            result = self.test_frft_roundtrip(
                                window_size, overlap_factor, frequency, alpha
                            )

                            self.results.append(result)
                            alpha_results.append(result)

                            # Write to file
                            out_file.write(
                                f"{result.window_size}\t{result.overlap_factor}\t{result.hop_size}\t"
                                f"{result.num_frames}\t{result.frequency:.2f}\t{result.alpha:.2f}\t"
                                f"{result.forward_time_ms:.6f}\t{result.inverse_time_ms:.6f}\t"
                                f"{result.total_time_ms:.6f}\t{result.mse:.10e}\t{result.mss_loss:.10e}\t"
                                f"{result.max_error:.10e}\t{result.mean_error:.10e}\t{result.std_error:.10e}\t"
                                f"{1 if result.success else 0}\n"
                            )

                            test_count += 1
                            if not result.success:
                                failed_count += 1
                            else:
                                alpha_success += 1

                            alpha += self.config.alpha_step

                        # Print summary for this frequency/overlap combination
                        if alpha_success > 0:
                            successful_results = [r for r in alpha_results if r.success]
                            avg_mse = np.mean([r.mse for r in successful_results])
                            avg_mss = np.mean([r.mss_loss for r in successful_results])
                            print(f"→ Avg MSE: {avg_mse:.3e}, Avg MSS: {avg_mss:.3f}")
                        else:
                            print("→ ALL FAILED")

        end_time = time.time()
        duration = end_time - start_time

        # Generate statistics and timing benchmarks
        print("\nGenerating statistics and benchmarks...")
        self.write_statistics()
        self.write_timing_benchmarks()

        # Print summary
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 22 + "Test Suite Complete" + " " * 27 + "║")
        print("╚" + "═" * 68 + "╝")
        print(f"\nSummary:")
        print(f"  Total tests: {test_count}")
        print(f"  Successful: {test_count - failed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Success rate: {100.0 * (test_count - failed_count) / test_count:.2f}%")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"\n  Results saved to: {self.config.output_filename}")
        print(f"  Benchmarks saved to: {self.config.timing_filename}")
        print(f"  Statistics saved to: {self.config.stats_filename}\n")

    def write_statistics(self):
        """Write comprehensive statistics to JSON file"""

        successful_results = [r for r in self.results if r.success]

        if not successful_results:
            print("Warning: No successful results to analyze")
            return

        # Overall statistics
        stats = {
            "overall": {
                "mse": asdict(self.calculate_statistics([r.mse for r in successful_results])),
                "mss_loss": asdict(self.calculate_statistics([r.mss_loss for r in successful_results])),
                "max_error": asdict(self.calculate_statistics([r.max_error for r in successful_results])),
                "mean_error": asdict(self.calculate_statistics([r.mean_error for r in successful_results])),
                "std_error": asdict(self.calculate_statistics([r.std_error for r in successful_results])),
                "forward_time_ms": asdict(self.calculate_statistics([r.forward_time_ms for r in successful_results])),
                "inverse_time_ms": asdict(self.calculate_statistics([r.inverse_time_ms for r in successful_results])),
                "total_time_ms": asdict(self.calculate_statistics([r.total_time_ms for r in successful_results])),
            },
            "by_window_size": {},
            "by_alpha": {},
            "by_frequency": {}
        }

        # Statistics by window size
        for window_size in self.config.window_sizes:
            ws_results = [r for r in successful_results if r.window_size == window_size]
            if ws_results:
                stats["by_window_size"][str(window_size)] = {
                    "mse": asdict(self.calculate_statistics([r.mse for r in ws_results])),
                    "mss_loss": asdict(self.calculate_statistics([r.mss_loss for r in ws_results])),
                    "max_error": asdict(self.calculate_statistics([r.max_error for r in ws_results])),
                    "std_error": asdict(self.calculate_statistics([r.std_error for r in ws_results])),
                    "total_time_ms": asdict(self.calculate_statistics([r.total_time_ms for r in ws_results])),
                }

        # Statistics by alpha
        alpha = self.config.alpha_start
        while alpha <= self.config.alpha_end + 1e-9:
            alpha_results = [r for r in successful_results if abs(r.alpha - alpha) < 0.01]
            if alpha_results:
                stats["by_alpha"][f"{alpha:.2f}"] = {
                    "mse": asdict(self.calculate_statistics([r.mse for r in alpha_results])),
                    "mss_loss": asdict(self.calculate_statistics([r.mss_loss for r in alpha_results])),
                    "max_error": asdict(self.calculate_statistics([r.max_error for r in alpha_results])),
                    "std_error": asdict(self.calculate_statistics([r.std_error for r in alpha_results])),
                }
            alpha += self.config.alpha_step

        # Statistics by frequency
        for frequency in self.config.test_frequencies:
            freq_results = [r for r in successful_results if abs(r.frequency - frequency) < 0.1]
            if freq_results:
                stats["by_frequency"][f"{frequency:.1f}"] = {
                    "mse": asdict(self.calculate_statistics([r.mse for r in freq_results])),
                    "mss_loss": asdict(self.calculate_statistics([r.mss_loss for r in freq_results])),
                    "max_error": asdict(self.calculate_statistics([r.max_error for r in freq_results])),
                    "std_error": asdict(self.calculate_statistics([r.std_error for r in freq_results])),
                }

        # Write to JSON file
        with open(self.config.stats_filename, 'w') as f:
            json.dump(stats, f, indent=2)

    def write_timing_benchmarks(self):
        """Write timing benchmarks similar to C++ version"""

        with open(self.config.timing_filename, 'w') as f:
            f.write("# FRFT Performance Benchmarks (Python)\n")
            f.write(f"# Sample Rate: {self.config.sample_rate} Hz\n")
            f.write("#\n")
            f.write("# Columns:\n")
            f.write("# 1. Window Size\n")
            f.write("# 2. Number of Samples\n")
            f.write("# 3. Mean Forward Time (ms)\n")
            f.write("# 4. Std Forward Time (ms)\n")
            f.write("# 5. Min Forward Time (ms)\n")
            f.write("# 6. Max Forward Time (ms)\n")
            f.write("# 7. Mean Inverse Time (ms)\n")
            f.write("# 8. Std Inverse Time (ms)\n")
            f.write("# 9. Min Inverse Time (ms)\n")
            f.write("# 10. Max Inverse Time (ms)\n")
            f.write("# 11. Mean Total Time (ms)\n")
            f.write("# 12. RTF Mean (Real-Time Factor)\n")
            f.write("# 13. RTF Best\n")
            f.write("# 14. RTF Worst\n")
            f.write("#\n")
            f.write("WindowSize\tNumSamples\tMeanForward\tStdForward\tMinForward\tMaxForward\t"
                    "MeanInverse\tStdInverse\tMinInverse\tMaxInverse\tMeanTotal\t"
                    "RTF_Mean\tRTF_Best\tRTF_Worst\n")

            successful_results = [r for r in self.results if r.success]

            for window_size in self.config.window_sizes:
                ws_results = [r for r in successful_results if r.window_size == window_size]

                if not ws_results:
                    continue

                forward_times = [r.forward_time_ms / r.num_frames for r in ws_results if r.num_frames > 0]
                inverse_times = [r.inverse_time_ms / r.num_frames for r in ws_results if r.num_frames > 0]
                total_times = [r.total_time_ms / r.num_frames for r in ws_results if r.num_frames > 0]

                if not total_times:
                    continue

                # Calculate Real-Time Factor
                window_duration_ms = (window_size / self.config.sample_rate) * 1000
                rtf_values = [t / window_duration_ms for t in total_times]

                f.write(f"{window_size}\t{len(ws_results)}\t"
                        f"{np.mean(forward_times):.6f}\t{np.std(forward_times):.6f}\t"
                        f"{np.min(forward_times):.6f}\t{np.max(forward_times):.6f}\t"
                        f"{np.mean(inverse_times):.6f}\t{np.std(inverse_times):.6f}\t"
                        f"{np.min(inverse_times):.6f}\t{np.max(inverse_times):.6f}\t"
                        f"{np.mean(total_times):.6f}\t"
                        f"{np.mean(rtf_values):.6f}\t{np.min(rtf_values):.6f}\t{np.max(rtf_values):.6f}\n")


def main():
    """Main entry point"""

    # Parse command line arguments
    config = TestConfig()

    if len(sys.argv) > 1:
        if '--help' in sys.argv or '-h' in sys.argv:
            print("FRFT Test Suite (Python)")
            print("\nUsage: python frft_test_suite.py [options]")
            print("\nOptions:")
            print("  --output FILE       Output filename")
            print("  --timing FILE       Timing benchmark filename")
            print("  --stats FILE        Statistics JSON filename")
            print("  --sample-rate SR    Sample rate in Hz")
            print("  --n-analysis N      Number of frames to analyze")
            print("  --quick             Run quick test (fewer sizes and frequencies)")
            print("  --help, -h          Show this help message")
            return

        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]

            if arg == '--output' and i + 1 < len(sys.argv):
                config.output_filename = sys.argv[i + 1]
                i += 2
            elif arg == '--timing' and i + 1 < len(sys.argv):
                config.timing_filename = sys.argv[i + 1]
                i += 2
            elif arg == '--stats' and i + 1 < len(sys.argv):
                config.stats_filename = sys.argv[i + 1]
                i += 2
            elif arg == '--sample-rate' and i + 1 < len(sys.argv):
                config.sample_rate = float(sys.argv[i + 1])
                i += 2
            elif arg == '--n-analysis' and i + 1 < len(sys.argv):
                config.n_analysis = int(sys.argv[i + 1])
                i += 2
            elif arg == '--quick':
                config.window_sizes = [64, 256, 1024, 4096]
                config.test_frequencies = [440.0, 1000.0, 5000.0]
                config.overlap_factors = [1, 2, 4]
                i += 1
            else:
                i += 1

    # Run test suite
    try:
        suite = FRFTTestSuite(config)
        suite.run_test_suite()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())