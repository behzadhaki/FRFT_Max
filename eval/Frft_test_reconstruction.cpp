#include "frft_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <string>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

// Generate Hamming window coefficients
void generate_hamming_window(std::vector<double>& window, int size) {
    window.resize(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (size - 1));
    }
}

// Apply window to signal
void apply_window(std::vector<double>& signal, const std::vector<double>& window) {
    for (size_t i = 0; i < signal.size(); ++i) {
        signal[i] *= window[i];
    }
}


// Test configuration
struct TestConfig {
    std::vector<int> window_sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    std::vector<int> overlap_factors = {1};  // 1=no overlap, 2=50%, 4=75%, etc.
    std::vector<double> test_frequencies = {20.0, 100.0, 220.0, 440.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000., 9000., 10000.0, 12000.0, 15000.0, 18000.0, 20000.0};
    double sample_rate = 44100.0;
    int n_analysis = 10;  // Number of frames to analyze
    double alpha_start = 0.0;
    double alpha_end = 2.1;
    double alpha_step = 0.1;
    std::string output_filename = "reconstruction_test/results.txt";
    std::string timing_filename = "reconstruction_test/timing_benchmarks.txt";
    std::string detailed_log_filename = "reconstruction_test/detailed_log.txt";
    std::string figures_dir = "reconstruction_test/figures";
};

// Result for a single test
struct TestResult {
    int window_size;
    int overlap_factor;
    double frequency;
    double alpha;
    double mse;
    double max_error;
    double mean_error;
    int num_frames;
    double forward_time_ms;   // Time for FRFT(-alpha)
    double inverse_time_ms;   // Time for FRFT(+alpha)
    double total_time_ms;     // Total time for round-trip
    bool success;
};

// Timing statistics per window size
struct TimingStats {
    int window_size;
    int num_samples;          // Total samples in timing
    double mean_forward_ms;
    double min_forward_ms;
    double max_forward_ms;
    double mean_inverse_ms;
    double min_inverse_ms;
    double max_inverse_ms;
    double mean_total_ms;
    double rtf_mean;          // Real-Time Factor (mean)
    double rtf_best;          // Best case RTF
    double rtf_worst;         // Worst case RTF
};

// Create directory recursively
bool create_directories(const std::string& path) {
    std::string current_path;
    for (size_t i = 0; i < path.length(); ++i) {
        if (path[i] == '/' || path[i] == '\\' || i == path.length() - 1) {
            if (i == path.length() - 1 && path[i] != '/' && path[i] != '\\') {
                current_path += path[i];
            } else {
                current_path += path[i];
            }

            #ifdef _WIN32
            _mkdir(current_path.c_str());
            #else
            mkdir(current_path.c_str(), 0755);
            #endif
        } else {
            current_path += path[i];
        }
    }
    return true;
}

// Write signal comparison to CSV
void write_reconstruction_to_csv(const std::string& filename,
                                const std::vector<double>& original,
                                const std::vector<double>& reconstructed,
                                double sample_rate) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << "\n";
        return;
    }

    file << "sample,time,original,reconstructed,error\n";
    for (size_t i = 0; i < original.size(); ++i) {
        double time = static_cast<double>(i) / sample_rate;
        double error = original[i] - reconstructed[i];
        file << i << ","
             << time << ","
             << original[i] << ","
             << reconstructed[i] << ","
             << error << "\n";
    }
    file.close();
}

// Generate Python plotting script for reconstruction
void generate_reconstruction_plot_script(const std::string& csv_filename,
                                        const std::string& output_png,
                                        int window_size,
                                        double test_frequency,
                                        double alpha,
                                        double sample_rate,
                                        double mse,
                                        double max_error) {
    std::ofstream script("plot_reconstruction.py");
    if (!script.is_open()) {
        std::cerr << "Error: Cannot create plotting script\n";
        return;
    }

    script << "import matplotlib.pyplot as plt\n";
    script << "import pandas as pd\n";
    script << "import numpy as np\n\n";

    script << "# Read data\n";
    script << "data = pd.read_csv('" << csv_filename << "')\n\n";

    script << "# Create figure with 3 subplots\n";
    script << "fig, axes = plt.subplots(3, 1, figsize=(14, 12))\n";
    script << "fig.suptitle(f'FRFT Round-Trip Reconstruction Test (Single Frame)\\n";
    script << "Window Size: " << window_size
           << " samples, Frequency: " << test_frequency << " Hz, Alpha: " << alpha << "\\n";
    script << "Sample Rate: " << sample_rate << " Hz\\n";
    script << "Showing: Hamming-windowed signals (what FRFT actually processes)', fontsize=14, fontweight='bold')\n\n";

    script << "# Plot 1: Full frame comparison\n";
    script << "axes[0].plot(data['time'], data['original'], 'b-', linewidth=1, alpha=0.7, label='Windowed Original', marker='o', markersize=2)\n";
    script << "axes[0].plot(data['time'], data['reconstructed'], 'r--', linewidth=1, alpha=0.7, label='Reconstructed', marker='s', markersize=2)\n";
    script << "axes[0].set_ylabel('Amplitude', fontsize=11)\n";
    script << "axes[0].set_title(f'Single Frame ({" << window_size << "} samples): Windowed Original vs Reconstructed', fontsize=12, fontweight='bold')\n";
    script << "axes[0].grid(True, alpha=0.3)\n";
    script << "axes[0].legend(loc='upper right', fontsize=10)\n";
    script << "axes[0].set_xlabel('Time (s)', fontsize=10)\n\n";

    script << "# Plot 2: Same as plot 1 but with sample indices on x-axis for clarity\n";
    script << "axes[1].plot(data['sample'], data['original'], 'b-', linewidth=1, alpha=0.7, label='Windowed Original', marker='o', markersize=2)\n";
    script << "axes[1].plot(data['sample'], data['reconstructed'], 'r--', linewidth=1, alpha=0.7, label='Reconstructed', marker='s', markersize=2)\n";
    script << "axes[1].set_ylabel('Amplitude', fontsize=11)\n";
    script << "axes[1].set_title('Sample-by-Sample Comparison', fontsize=12, fontweight='bold')\n";
    script << "axes[1].grid(True, alpha=0.3)\n";
    script << "axes[1].legend(loc='upper right', fontsize=10)\n";
    script << "axes[1].set_xlabel('Sample Index', fontsize=10)\n\n";

    script << "# Plot 3: Error signal\n";
    script << "axes[2].plot(data['time'], data['error'], 'g-', linewidth=1.5)\n";
    script << "axes[2].set_ylabel('Error (Original - Reconstructed)', fontsize=11)\n";
    script << "axes[2].set_xlabel('Time (s)', fontsize=10)\n";
    script << "axes[2].set_title(f'Reconstruction Error (MSE: {" << mse
           << ":.2e}, Max Error: {" << max_error << ":.2e})', fontsize=12, fontweight='bold')\n";
    script << "axes[2].grid(True, alpha=0.3)\n";
    script << "axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)\n\n";

    script << "# Adjust layout and save\n";
    script << "plt.tight_layout()\n";
    script << "plt.savefig('" << output_png << "', dpi=150, bbox_inches='tight')\n";
    script << "plt.close()\n";
    script << "print('Plot saved to: " << output_png << "')\n";

    script.close();
}

// Generate a sinusoidal signal
void generate_sine_wave(std::vector<double>& signal, int size, double frequency, double sample_rate, double phase = 0.0) {
    signal.resize(size);
    for (int i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        signal[i] = std::sin(2.0 * M_PI * frequency * t + phase);
    }
}

// Calculate Mean Squared Error between two signals
double calculate_mse(const std::vector<double>& original, const std::vector<double>& reconstructed) {
    if (original.size() != reconstructed.size()) {
        return -1.0;  // Error indicator
    }

    double sum_squared_error = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        double error = original[i] - reconstructed[i];
        sum_squared_error += error * error;
    }

    return sum_squared_error / original.size();
}

// Calculate maximum absolute error
double calculate_max_error(const std::vector<double>& original, const std::vector<double>& reconstructed) {
    if (original.size() != reconstructed.size()) {
        return -1.0;
    }

    double max_err = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        double error = std::abs(original[i] - reconstructed[i]);
        if (error > max_err) {
            max_err = error;
        }
    }

    return max_err;
}

// Calculate mean absolute error
double calculate_mean_error(const std::vector<double>& original, const std::vector<double>& reconstructed) {
    if (original.size() != reconstructed.size()) {
        return -1.0;
    }

    double sum_error = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        sum_error += std::abs(original[i] - reconstructed[i]);
    }

    return sum_error / original.size();
}

// Perform a single round-trip FRFT test with timing
TestResult test_frft_roundtrip(FRFTEngine& engine,
                               int window_size,
                               int overlap_factor,
                               double frequency,
                               double alpha,
                               double sample_rate,
                               int n_analysis,
                               std::vector<double>& original_signal_out,
                               std::vector<double>& reconstructed_signal_out) {
    TestResult result;
    result.window_size = window_size;
    result.overlap_factor = overlap_factor;
    result.frequency = frequency;
    result.alpha = alpha;
    result.success = false;
    result.num_frames = 0;
    result.forward_time_ms = 0.0;
    result.inverse_time_ms = 0.0;
    result.total_time_ms = 0.0;

    // Calculate hop size from overlap factor
    int hop_size = window_size / overlap_factor;
    if (hop_size < 1) hop_size = 1;

    // Generate enough signal for n_analysis frames
    int signal_length = window_size + (n_analysis - 1) * hop_size;
    std::vector<double> original_signal;
    generate_sine_wave(original_signal, signal_length, frequency, sample_rate);

    // Prepare output signal
    std::vector<double> reconstructed_signal(signal_length, 0.0);

    // Allocate buffers for FRFT processing
    std::vector<double> real_in(window_size);
    std::vector<double> imag_in(window_size, 0.0);
    std::vector<double> real_temp(window_size);
    std::vector<double> imag_temp(window_size);
    std::vector<double> real_out(window_size);
    std::vector<double> imag_out(window_size);

    // Generate Hamming window for this window size
    std::vector<double> hamming_window;
    generate_hamming_window(hamming_window, window_size);

    // Process frames
    int num_frames = 0;
    double total_forward_time = 0.0;
    double total_inverse_time = 0.0;

    for (int pos = 0; pos + window_size <= signal_length; pos += hop_size) {
        // Extract frame
        std::copy(original_signal.begin() + pos,
                  original_signal.begin() + pos + window_size,
                  real_in.begin());
        std::fill(imag_in.begin(), imag_in.end(), 0.0);

        // Apply Hamming window
        apply_window(real_in, hamming_window);

        // Time forward FRFT (-alpha)
        auto forward_start = std::chrono::high_resolution_clock::now();
        bool forward_success = engine.compute(real_in.data(), imag_in.data(),
                                              real_temp.data(), imag_temp.data(),
                                              window_size, -alpha);
        auto forward_end = std::chrono::high_resolution_clock::now();

        if (!forward_success) {
            result.mse = -1.0;
            result.max_error = -1.0;
            result.mean_error = -1.0;
            return result;
        }

        double forward_ms = std::chrono::duration<double, std::milli>(forward_end - forward_start).count();
        total_forward_time += forward_ms;

        // Time inverse FRFT (+alpha)
        auto inverse_start = std::chrono::high_resolution_clock::now();
        bool inverse_success = engine.compute(real_temp.data(), imag_temp.data(),
                                              real_out.data(), imag_out.data(),
                                              window_size, alpha);
        auto inverse_end = std::chrono::high_resolution_clock::now();

        if (!inverse_success) {
            result.mse = -1.0;
            result.max_error = -1.0;
            result.mean_error = -1.0;
            return result;
        }

        double inverse_ms = std::chrono::duration<double, std::milli>(inverse_end - inverse_start).count();
        total_inverse_time += inverse_ms;

        // Copy reconstructed frame (simple copy, no overlap-add)
        std::copy(real_out.begin(), real_out.end(),
                  reconstructed_signal.begin() + pos);

        num_frames++;
    }

    // Calculate timing statistics
    result.num_frames = num_frames;
    result.forward_time_ms = total_forward_time;
    result.inverse_time_ms = total_inverse_time;
    result.total_time_ms = total_forward_time + total_inverse_time;

    // Calculate reconstruction quality metrics
    // Only measure on complete frames
    int valid_length = num_frames * hop_size;
    std::vector<double> original_valid(original_signal.begin(),
                                       original_signal.begin() + valid_length);
    std::vector<double> reconstructed_valid(reconstructed_signal.begin(),
                                            reconstructed_signal.begin() + valid_length);

    result.mse = calculate_mse(original_valid, reconstructed_valid);
    result.max_error = calculate_max_error(original_valid, reconstructed_valid);
    result.mean_error = calculate_mean_error(original_valid, reconstructed_valid);
    result.success = true;

    // Copy signals for plotting - ONLY FIRST FRAME (single window)
    // Store the WINDOWED signal (what actually went into FRFT)
    original_signal_out.resize(window_size);
    reconstructed_signal_out.resize(window_size);

    // Copy first frame before windowing
    std::copy(original_signal.begin(), original_signal.begin() + window_size,
              original_signal_out.begin());

    // Apply window to the copied original (to show what FRFT actually processed)
    std::vector<double> hamming_window_plot;
    generate_hamming_window(hamming_window_plot, window_size);
    apply_window(original_signal_out, hamming_window_plot);

    // Copy reconstructed (already from first frame processing)
    std::copy(reconstructed_signal.begin(), reconstructed_signal.begin() + window_size,
              reconstructed_signal_out.begin());

    // IMPORTANT: Recalculate metrics on the WINDOWED first frame (what we're plotting)
    // The previous MSE was calculated on unwindowed vs reconstructed (misleading!)
    result.mse = calculate_mse(original_signal_out, reconstructed_signal_out);
    result.max_error = calculate_max_error(original_signal_out, reconstructed_signal_out);
    result.mean_error = calculate_mean_error(original_signal_out, reconstructed_signal_out);

    return result;
}

// Calculate timing statistics for a given window size
TimingStats calculate_timing_stats(const std::vector<TestResult>& results,
                                   int window_size,
                                   double sample_rate) {
    TimingStats stats;
    stats.window_size = window_size;
    stats.num_samples = 0;

    // Filter results for this window size
    std::vector<double> forward_times;
    std::vector<double> inverse_times;
    std::vector<double> total_times;

    for (const auto& result : results) {
        if (result.window_size == window_size && result.success) {
            forward_times.push_back(result.forward_time_ms / result.num_frames);  // Per frame
            inverse_times.push_back(result.inverse_time_ms / result.num_frames);
            total_times.push_back(result.total_time_ms / result.num_frames);
            stats.num_samples++;
        }
    }

    if (forward_times.empty()) {
        return stats;  // No valid data
    }

    // Calculate statistics
    auto minmax_forward = std::minmax_element(forward_times.begin(), forward_times.end());
    auto minmax_inverse = std::minmax_element(inverse_times.begin(), inverse_times.end());
    auto minmax_total = std::minmax_element(total_times.begin(), total_times.end());

    stats.min_forward_ms = *minmax_forward.first;
    stats.max_forward_ms = *minmax_forward.second;
    stats.mean_forward_ms = std::accumulate(forward_times.begin(), forward_times.end(), 0.0) / forward_times.size();

    stats.min_inverse_ms = *minmax_inverse.first;
    stats.max_inverse_ms = *minmax_inverse.second;
    stats.mean_inverse_ms = std::accumulate(inverse_times.begin(), inverse_times.end(), 0.0) / inverse_times.size();

    stats.mean_total_ms = std::accumulate(total_times.begin(), total_times.end(), 0.0) / total_times.size();

    // Calculate Real-Time Factor (RTF)
    // RTF = time_to_process / duration_of_audio
    double audio_duration_ms = (window_size * 1000.0) / sample_rate;

    stats.rtf_mean = stats.mean_total_ms / audio_duration_ms;
    stats.rtf_best = (*minmax_total.first) / audio_duration_ms;
    stats.rtf_worst = (*minmax_total.second) / audio_duration_ms;

    return stats;
}

// Write timing benchmarks to file
void write_timing_benchmarks(const std::vector<TestResult>& all_results,
                             const TestConfig& config,
                             const std::string& filename) {
    std::ofstream out_file(filename);
    if (!out_file.is_open()) {
        std::cerr << "Error: Cannot open timing file: " << filename << "\n";
        return;
    }

    // Write header
    out_file << "# FRFT Performance Benchmarks\n";
    out_file << "# Sample Rate: " << config.sample_rate << " Hz\n";
    out_file << "# Number of Analysis Frames: " << config.n_analysis << "\n";
    out_file << "# All times are per-frame averages across all test conditions\n";
    out_file << "#\n";
    out_file << "# Real-Time Factor (RTF):\n";
    out_file << "#   RTF < 1.0 = Faster than real-time (good for real-time processing)\n";
    out_file << "#   RTF = 1.0 = Processes at exactly real-time speed\n";
    out_file << "#   RTF > 1.0 = Slower than real-time (may cause dropouts)\n";
    out_file << "#\n";
    out_file << "# Columns:\n";
    out_file << "# 1.  Window Size (samples)\n";
    out_file << "# 2.  Number of Samples (test runs)\n";
    out_file << "# 3.  Mean Forward Time (ms/frame)\n";
    out_file << "# 4.  Min Forward Time (ms/frame)\n";
    out_file << "# 5.  Max Forward Time (ms/frame)\n";
    out_file << "# 6.  Mean Inverse Time (ms/frame)\n";
    out_file << "# 7.  Min Inverse Time (ms/frame)\n";
    out_file << "# 8.  Max Inverse Time (ms/frame)\n";
    out_file << "# 9.  Mean Total Time (ms/frame)\n";
    out_file << "# 10. RTF Mean\n";
    out_file << "# 11. RTF Best\n";
    out_file << "# 12. RTF Worst\n";
    out_file << "#\n";
    out_file << "WindowSize\tNumSamples\tMeanForward\tMinForward\tMaxForward\t"
             << "MeanInverse\tMinInverse\tMaxInverse\tMeanTotal\tRTF_Mean\tRTF_Best\tRTF_Worst\n";

    // Calculate and write statistics for each window size
    for (int window_size : config.window_sizes) {
        TimingStats stats = calculate_timing_stats(all_results, window_size, config.sample_rate);

        if (stats.num_samples > 0) {
            out_file << stats.window_size << "\t"
                     << stats.num_samples << "\t"
                     << std::fixed << std::setprecision(6)
                     << stats.mean_forward_ms << "\t"
                     << stats.min_forward_ms << "\t"
                     << stats.max_forward_ms << "\t"
                     << stats.mean_inverse_ms << "\t"
                     << stats.min_inverse_ms << "\t"
                     << stats.max_inverse_ms << "\t"
                     << stats.mean_total_ms << "\t"
                     << std::setprecision(4)
                     << stats.rtf_mean << "\t"
                     << stats.rtf_best << "\t"
                     << stats.rtf_worst << "\n";
        }
    }

    out_file.close();
    std::cout << "  Timing benchmarks saved to: " << filename << "\n";
}

// Run comprehensive test suite
void run_test_suite(const TestConfig& config) {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         FRFT Round-Trip Accuracy & Performance Test           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Sample Rate: " << config.sample_rate << " Hz\n";
    std::cout << "  Number of Analysis Frames: " << config.n_analysis << "\n";
    std::cout << "  Window Sizes: ";
    for (size_t i = 0; i < config.window_sizes.size(); ++i) {
        std::cout << config.window_sizes[i];
        if (i < config.window_sizes.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";

    std::cout << "  Overlap Factors: ";
    for (size_t i = 0; i < config.overlap_factors.size(); ++i) {
        std::cout << config.overlap_factors[i];
        if (i < config.overlap_factors.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";

    std::cout << "  Test Frequencies: ";
    for (size_t i = 0; i < config.test_frequencies.size(); ++i) {
        std::cout << config.test_frequencies[i] << " Hz";
        if (i < config.test_frequencies.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";

    std::cout << "  Alpha Range: " << config.alpha_start << " to " << config.alpha_end
              << " (step: " << config.alpha_step << ")\n";
    std::cout << "  Output File: " << config.output_filename << "\n";
    std::cout << "  Timing File: " << config.timing_filename << "\n";
    std::cout << "  Figures Directory: " << config.figures_dir << "\n\n";

    // Create directories
    std::cout << "Creating output directories...\n";
    create_directories("reconstruction_test");
    create_directories(config.figures_dir);

    // Calculate total number of tests
    int num_alphas = static_cast<int>((config.alpha_end - config.alpha_start) / config.alpha_step) + 1;
    int total_tests = config.window_sizes.size() * config.overlap_factors.size() *
                      config.test_frequencies.size() * num_alphas;

    std::cout << "Total tests to run: " << total_tests << "\n\n";

    // Open output file
    std::ofstream out_file(config.output_filename);
    if (!out_file.is_open()) {
        std::cerr << "Error: Cannot open output file: " << config.output_filename << "\n";
        return;
    }

    // Write header
    out_file << "# FRFT Round-Trip Accuracy Test Results\n";
    out_file << "# Generated: " << std::chrono::system_clock::now().time_since_epoch().count() << "\n";
    out_file << "# Sample Rate: " << config.sample_rate << " Hz\n";
    out_file << "# Number of Analysis Frames: " << config.n_analysis << "\n";
    out_file << "# Processing: Frame-by-frame with Hamming window applied to each frame\n";
    out_file << "#\n";
    out_file << "# Columns:\n";
    out_file << "# 1. Window Size\n";
    out_file << "# 2. Overlap Factor\n";
    out_file << "# 3. Hop Size\n";
    out_file << "# 4. Number of Frames\n";
    out_file << "# 5. Frequency (Hz)\n";
    out_file << "# 6. Alpha Parameter\n";
    out_file << "# 7. Forward Time (ms total)\n";
    out_file << "# 8. Inverse Time (ms total)\n";
    out_file << "# 9. Total Time (ms)\n";
    out_file << "# 10. MSE\n";
    out_file << "# 11. Max Error\n";
    out_file << "# 12. Mean Error\n";
    out_file << "# 13. Success\n";
    out_file << "#\n";
    out_file << "WindowSize\tOverlapFactor\tHopSize\tNumFrames\tFrequency\tAlpha\t"
             << "ForwardTime\tInverseTime\tTotalTime\tMSE\tMaxError\tMeanError\tSuccess\n";

    // Open detailed log file
    std::ofstream detail_file(config.detailed_log_filename);
    if (!detail_file.is_open()) {
        std::cerr << "Warning: Cannot open detailed log file: " << config.detailed_log_filename << "\n";
        std::cerr << "Continuing without detailed logging...\n";
    } else {
        detail_file << "# FRFT Round-Trip Reconstruction Test - Detailed Log\n";
        detail_file << "# Every individual test result is logged here\n";
        detail_file << "# Sample Rate: " << config.sample_rate << " Hz\n";
        detail_file << "# StartSample and EndSample refer to the first frame (frame 0) position\n";
        detail_file << "#\n";
        detail_file << "TestID\tWindowSize\tOverlapFactor\tHopSize\tNumFrames\tStartSample\tEndSample\tFrequency\tAlpha\t"
                   << "ForwardTime_ms\tInverseTime_ms\tTotalTime_ms\tMSE\tMaxError\tMeanError\tSuccess\n";
    }

    // Create FRFT engine
    FRFTEngine engine;

    // Progress tracking
    int test_count = 0;
    int failed_count = 0;
    std::vector<TestResult> all_results;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run tests
    for (int window_size : config.window_sizes) {
        std::cout << "\nTesting window size: " << window_size << "\n";
        std::cout << "─────────────────────────────────────────────────\n";

        // Pre-allocate buffers for this window size
        engine.prepare(window_size);

        for (int overlap_factor : config.overlap_factors) {
            int hop_size = window_size / overlap_factor;
            if (hop_size < 1) hop_size = 1;

            for (double frequency : config.test_frequencies) {
                std::cout << "  Freq: " << std::setw(8) << frequency << " Hz, "
                          << "Overlap: " << overlap_factor << "x (hop=" << hop_size << ") ";
                std::cout.flush();

                int alpha_count = 0;
                double alpha_sum_mse = 0.0;
                int alpha_success = 0;
                std::set<int> plots_generated;  // Track which alphas we've plotted

                for (double alpha = config.alpha_start; alpha <= config.alpha_end; alpha += config.alpha_step) {
                    std::vector<double> original_signal, reconstructed_signal;

                    TestResult result = test_frft_roundtrip(engine, window_size, overlap_factor,
                                                            frequency, alpha,
                                                            config.sample_rate, config.n_analysis,
                                                            original_signal, reconstructed_signal);

                    // Store for timing analysis
                    all_results.push_back(result);

                    // Write result to file
                    out_file << result.window_size << "\t"
                             << result.overlap_factor << "\t"
                             << hop_size << "\t"
                             << result.num_frames << "\t"
                             << result.frequency << "\t"
                             << std::fixed << std::setprecision(2) << result.alpha << "\t"
                             << std::setprecision(6) << result.forward_time_ms << "\t"
                             << result.inverse_time_ms << "\t"
                             << result.total_time_ms << "\t"
                             << std::scientific << std::setprecision(10) << result.mse << "\t"
                             << result.max_error << "\t"
                             << result.mean_error << "\t"
                             << (result.success ? 1 : 0) << "\n";

                    // Write to detailed log
                    if (detail_file.is_open()) {
                        int start_sample = 0;  // First frame always starts at 0
                        int end_sample = window_size - 1;  // First frame ends at window_size-1

                        detail_file << test_count << "\t"
                                   << result.window_size << "\t"
                                   << result.overlap_factor << "\t"
                                   << hop_size << "\t"
                                   << result.num_frames << "\t"
                                   << start_sample << "\t"
                                   << end_sample << "\t"
                                   << result.frequency << "\t"
                                   << std::fixed << std::setprecision(2) << result.alpha << "\t"
                                   << std::setprecision(6) << result.forward_time_ms << "\t"
                                   << result.inverse_time_ms << "\t"
                                   << result.total_time_ms << "\t"
                                   << std::scientific << std::setprecision(10) << result.mse << "\t"
                                   << result.max_error << "\t"
                                   << result.mean_error << "\t"
                                   << (result.success ? 1 : 0) << "\n";
                    }

                    // Generate plots for alpha = 0.5, 1.0, and 1.5
                    int alpha_int = static_cast<int>(std::round(alpha * 10));
                    if (result.success && plots_generated.find(alpha_int) == plots_generated.end() &&
                        (std::abs(alpha - 0.5) < 0.05 || std::abs(alpha - 1.0) < 0.05 || std::abs(alpha - 1.5) < 0.05)) {

                        std::string csv_filename = "temp_reconstruction.csv";
                        std::string png_filename = config.figures_dir + "/freq" +
                                                  std::to_string(static_cast<int>(frequency)) +
                                                  "_ws" + std::to_string(window_size) +
                                                  "_alpha" + std::to_string(alpha_int) + ".png";

                        write_reconstruction_to_csv(csv_filename, original_signal, reconstructed_signal,
                                                   config.sample_rate);
                        generate_reconstruction_plot_script(csv_filename, png_filename, window_size, frequency,
                                                          alpha, config.sample_rate, result.mse, result.max_error);

                        int ret = system("python3 plot_reconstruction.py 2>/dev/null");
                        if (ret == 0) {
                            plots_generated.insert(alpha_int);
                        }

                        // Clean up temporary files
                        remove(csv_filename.c_str());
                        remove("plot_reconstruction.py");
                    }

                    test_count++;
                    if (!result.success) {
                        failed_count++;
                    } else {
                        alpha_sum_mse += result.mse;
                        alpha_success++;
                    }
                    alpha_count++;
                }

                // Print summary for this frequency/overlap combination
                if (alpha_success > 0) {
                    double avg_mse = alpha_sum_mse / alpha_success;
                    std::cout << "→ Avg MSE: " << std::scientific << std::setprecision(3) << avg_mse << "\n";
                } else {
                    std::cout << "→ ALL FAILED\n";
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    out_file.close();
    if (detail_file.is_open()) {
        detail_file.close();
    }

    // Generate timing benchmarks
    std::cout << "\nGenerating performance benchmarks...\n";
    write_timing_benchmarks(all_results, config, config.timing_filename);

    // Print summary
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Test Suite Complete                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nSummary:\n";
    std::cout << "  Total tests: " << test_count << "\n";
    std::cout << "  Successful: " << (test_count - failed_count) << "\n";
    std::cout << "  Failed: " << failed_count << "\n";
    std::cout << "  Success rate: " << std::fixed << std::setprecision(2)
              << (100.0 * (test_count - failed_count) / test_count) << "%\n";
    std::cout << "  Duration: " << duration.count() / 1000.0 << " seconds\n";
    std::cout << "  Results saved to: " << config.output_filename << "\n";
    std::cout << "  Detailed log saved to: " << config.detailed_log_filename << "\n";
    std::cout << "  Timing benchmarks saved to: " << config.timing_filename << "\n";
    std::cout << "  Plots saved to: " << config.figures_dir << "/\n";
    std::cout << "  Expected plots: " << (config.window_sizes.size() * config.test_frequencies.size() * 3)
              << " (α=0.5, 1.0, 1.5 for each window/freq)\n\n";
}

// Parse command line arguments
bool parse_arguments(int argc, char* argv[], TestConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            std::cout << "FRFT Test Suite\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --output FILE       Output filename (default: frft_test_results.txt)\n";
            std::cout << "  --timing FILE       Timing benchmark filename (default: frft_timing_benchmarks.txt)\n";
            std::cout << "  --sample-rate SR    Sample rate in Hz (default: 44100)\n";
            std::cout << "  --n-analysis N      Number of frames to analyze (default: 100)\n";
            std::cout << "  --quick            Run quick test (fewer window sizes and frequencies)\n";
            std::cout << "  --help             Show this help message\n";
            return false;
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.output_filename = argv[++i];
        }
        else if (arg == "--timing" && i + 1 < argc) {
            config.timing_filename = argv[++i];
        }
        else if (arg == "--sample-rate" && i + 1 < argc) {
            config.sample_rate = std::stod(argv[++i]);
        }
        else if (arg == "--n-analysis" && i + 1 < argc) {
            config.n_analysis = std::stoi(argv[++i]);
        }
        else if (arg == "--quick") {
            config.window_sizes = {64, 256, 1024};
            config.test_frequencies = {440.0, 1000.0};
            config.overlap_factors = {1, 2, 4};
        }
    }

    return true;
}

int main(int argc, char* argv[]) {


#ifdef _WIN32
    // Set console to UTF-8 mode on Windows
    SetConsoleOutputCP(CP_UTF8);
#endif

    TestConfig config;

    if (!parse_arguments(argc, argv, config)) {
        return 0;  // Help was shown
    }

    try {
        run_test_suite(config);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}