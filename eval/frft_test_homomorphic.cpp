#include "frft_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>
#include <random>

#ifdef _WIN32
#include <windows.h>
#endif

// Test configuration
struct TestConfig {
    std::vector<int> window_sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    std::vector<int> overlap_factors = {4};  // 1=no overlap, 2=50%, 4=75%, etc.
    std::vector<double> test_frequencies = {100.0, 220.0, 440.0, 1000.0, 2000.0, 4000.0, 5000.0, 8000., 10000.0, 15000.0};
    double sample_rate = 44100.0;
    int n_analysis = 20;  // Number of frames to analyze
    double alpha_start = 0.0;
    double alpha_end = 2.0;
    double alpha_step = 0.1;
    std::string output_filename = "frft_homomorphic_results.txt";
};

// Result for a single homomorphic test
struct HomomorphicTestResult {
    int window_size;
    int overlap_factor;
    double frequency;
    double alpha_total;      // The total alpha value
    double alpha1;           // First component (α₁)
    double alpha2;           // Second component (α₂)
    double mse_direct;       // MSE between original and FRFT(α) result
    double mse_composed;     // MSE between original and FRFT(α₂) ∘ FRFT(α₁) result
    double mse_homomorphic;  // MSE between FRFT(α) and FRFT(α₂) ∘ FRFT(α₁)
    double max_error_homomorphic;
    double mean_error_homomorphic;
    int num_frames;
    bool success;
};

// Generate a sinusoidal signal
void generate_sine_wave(std::vector<double>& signal, int size, double frequency, double sample_rate, double phase = 0.0) {
    signal.resize(size);
    for (int i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        signal[i] = std::sin(2.0 * M_PI * frequency * t + phase);
    }
}

// Calculate Mean Squared Error between two signals
double calculate_mse(const std::vector<double>& signal1, const std::vector<double>& signal2) {
    if (signal1.size() != signal2.size()) {
        return -1.0;  // Error indicator
    }

    double sum_squared_error = 0.0;
    for (size_t i = 0; i < signal1.size(); ++i) {
        double error = signal1[i] - signal2[i];
        sum_squared_error += error * error;
    }

    return sum_squared_error / signal1.size();
}

// Calculate maximum absolute error
double calculate_max_error(const std::vector<double>& signal1, const std::vector<double>& signal2) {
    if (signal1.size() != signal2.size()) {
        return -1.0;
    }

    double max_err = 0.0;
    for (size_t i = 0; i < signal1.size(); ++i) {
        double error = std::abs(signal1[i] - signal2[i]);
        if (error > max_err) {
            max_err = error;
        }
    }

    return max_err;
}

// Calculate mean absolute error
double calculate_mean_error(const std::vector<double>& signal1, const std::vector<double>& signal2) {
    if (signal1.size() != signal2.size()) {
        return -1.0;
    }

    double sum_error = 0.0;
    for (size_t i = 0; i < signal1.size(); ++i) {
        sum_error += std::abs(signal1[i] - signal2[i]);
    }

    return sum_error / signal1.size();
}

// Generate alpha1 and alpha2 that sum to alpha_total
// Constraint: both must be in range [0, 2.0]
void generate_alpha_decomposition(double alpha_total, double& alpha1, double& alpha2, std::mt19937& rng) {
    // We need: alpha1 + alpha2 = alpha_total
    // Constraints: 0 <= alpha1 <= 2.0 and 0 <= alpha2 <= 2.0
    
    double min_alpha1 = std::max(0.0, alpha_total - 2.0);  // alpha1 >= alpha_total - 2
    double max_alpha1 = std::min(2.0, alpha_total);        // alpha1 <= 2 and alpha1 <= alpha_total
    
    // If alpha_total > 4.0, there's no valid decomposition
    if (alpha_total > 4.0 || min_alpha1 > max_alpha1) {
        alpha1 = 0.0;
        alpha2 = 0.0;
        return;
    }
    
    // Generate random alpha1 in valid range
    std::uniform_real_distribution<double> dist(min_alpha1, max_alpha1);
    alpha1 = dist(rng);
    alpha2 = alpha_total - alpha1;
    
    // Ensure alpha2 is also within bounds (should be guaranteed by construction)
    alpha2 = std::max(0.0, std::min(2.0, alpha2));
}

// Perform homomorphic property test
HomomorphicTestResult test_homomorphic_property(FRFTEngine& engine,
                                                int window_size,
                                                int overlap_factor,
                                                double frequency,
                                                double alpha_total,
                                                double alpha1,
                                                double alpha2,
                                                double sample_rate,
                                                int n_analysis) {
    HomomorphicTestResult result;
    result.window_size = window_size;
    result.overlap_factor = overlap_factor;
    result.frequency = frequency;
    result.alpha_total = alpha_total;
    result.alpha1 = alpha1;
    result.alpha2 = alpha2;
    result.success = false;
    result.num_frames = 0;

    // Calculate hop size from overlap factor
    int hop_size = window_size / overlap_factor;
    if (hop_size < 1) hop_size = 1;

    // Generate enough signal for n_analysis frames
    int signal_length = window_size + (n_analysis - 1) * hop_size;
    std::vector<double> original_signal;
    generate_sine_wave(original_signal, signal_length, frequency, sample_rate);

    // Prepare output signals
    std::vector<double> direct_transform(signal_length, 0.0);      // FRFT(α)
    std::vector<double> composed_transform(signal_length, 0.0);    // FRFT(α₂) ∘ FRFT(α₁)

    // Allocate buffers for FRFT processing
    std::vector<double> real_in(window_size);
    std::vector<double> imag_in(window_size, 0.0);
    std::vector<double> real_direct(window_size);
    std::vector<double> imag_direct(window_size);
    std::vector<double> real_temp(window_size);      // After first transform
    std::vector<double> imag_temp(window_size);
    std::vector<double> real_composed(window_size);  // After second transform
    std::vector<double> imag_composed(window_size);

    // Process frames
    int num_frames = 0;

    for (int pos = 0; pos + window_size <= signal_length; pos += hop_size) {
        // Extract frame
        std::copy(original_signal.begin() + pos,
                  original_signal.begin() + pos + window_size,
                  real_in.begin());
        std::fill(imag_in.begin(), imag_in.end(), 0.0);

        // Path 1: Direct FRFT with alpha_total
        bool direct_success = engine.compute(real_in.data(), imag_in.data(),
                                            real_direct.data(), imag_direct.data(),
                                            window_size, alpha_total);

        if (!direct_success) {
            result.mse_direct = -1.0;
            result.mse_composed = -1.0;
            result.mse_homomorphic = -1.0;
            result.max_error_homomorphic = -1.0;
            result.mean_error_homomorphic = -1.0;
            return result;
        }

        // Path 2: Composed FRFT with alpha1, then alpha2
        // First transform: FRFT(α₁)
        bool first_success = engine.compute(real_in.data(), imag_in.data(),
                                           real_temp.data(), imag_temp.data(),
                                           window_size, alpha1);

        if (!first_success) {
            result.mse_direct = -1.0;
            result.mse_composed = -1.0;
            result.mse_homomorphic = -1.0;
            result.max_error_homomorphic = -1.0;
            result.mean_error_homomorphic = -1.0;
            return result;
        }

        // Second transform: FRFT(α₂) on result of first
        bool second_success = engine.compute(real_temp.data(), imag_temp.data(),
                                            real_composed.data(), imag_composed.data(),
                                            window_size, alpha2);

        if (!second_success) {
            result.mse_direct = -1.0;
            result.mse_composed = -1.0;
            result.mse_homomorphic = -1.0;
            result.max_error_homomorphic = -1.0;
            result.mean_error_homomorphic = -1.0;
            return result;
        }

        // Store results (simple copy, no overlap-add)
        std::copy(real_direct.begin(), real_direct.end(),
                  direct_transform.begin() + pos);
        std::copy(real_composed.begin(), real_composed.end(),
                  composed_transform.begin() + pos);

        num_frames++;
    }

    // Calculate MSE metrics
    result.mse_direct = calculate_mse(original_signal, direct_transform);
    result.mse_composed = calculate_mse(original_signal, composed_transform);
    result.mse_homomorphic = calculate_mse(direct_transform, composed_transform);
    result.max_error_homomorphic = calculate_max_error(direct_transform, composed_transform);
    result.mean_error_homomorphic = calculate_mean_error(direct_transform, composed_transform);
    result.num_frames = num_frames;
    result.success = true;

    return result;
}

// Run the complete test suite
void run_test_suite(const TestConfig& config) {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       FRFT Homomorphic Property Test Suite                    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Testing Property: FRFT(α) ≈ FRFT(α₂) ∘ FRFT(α₁) where α = α₁ + α₂\n\n";

    std::cout << "Configuration:\n";
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
    std::cout << "  Sample Rate: " << config.sample_rate << " Hz\n";
    std::cout << "  Alpha Range: " << config.alpha_start << " to " << config.alpha_end 
              << " (step " << config.alpha_step << ")\n";
    std::cout << "  Frames per Test: " << config.n_analysis << "\n";
    std::cout << "  Output File: " << config.output_filename << "\n\n";

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
    out_file << "# FRFT Homomorphic Property Test Results\n";
    out_file << "# Testing: FRFT(α) ≈ FRFT(α₂) ∘ FRFT(α₁) where α = α₁ + α₂\n";
    out_file << "# Sample Rate: " << config.sample_rate << " Hz\n";
    out_file << "# Number of Analysis Frames: " << config.n_analysis << "\n";
    out_file << "#\n";
    out_file << "# Columns:\n";
    out_file << "# 1. Window Size\n";
    out_file << "# 2. Overlap Factor\n";
    out_file << "# 3. Hop Size\n";
    out_file << "# 4. Number of Frames\n";
    out_file << "# 5. Frequency (Hz)\n";
    out_file << "# 6. Alpha Total (α)\n";
    out_file << "# 7. Alpha 1 (α₁)\n";
    out_file << "# 8. Alpha 2 (α₂)\n";
    out_file << "# 9. MSE Direct (original vs FRFT(α))\n";
    out_file << "# 10. MSE Composed (original vs FRFT(α₂)∘FRFT(α₁))\n";
    out_file << "# 11. MSE Homomorphic (FRFT(α) vs FRFT(α₂)∘FRFT(α₁))\n";
    out_file << "# 12. Max Error Homomorphic\n";
    out_file << "# 13. Mean Error Homomorphic\n";
    out_file << "# 14. Success\n";
    out_file << "#\n";
    out_file << "WindowSize\tOverlapFactor\tHopSize\tNumFrames\tFrequency\t"
             << "Alpha_Total\tAlpha1\tAlpha2\t"
             << "MSE_Direct\tMSE_Composed\tMSE_Homomorphic\t"
             << "MaxError_Homomorphic\tMeanError_Homomorphic\tSuccess\n";

    // Create FRFT engine
    FRFTEngine engine;

    // Random number generator for alpha decomposition
    std::random_device rd;
    std::mt19937 rng(rd());

    // Progress tracking
    int test_count = 0;
    int failed_count = 0;
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
                double alpha_sum_mse_homomorphic = 0.0;
                int alpha_success = 0;

                for (double alpha = config.alpha_start; alpha <= config.alpha_end + 1e-9; alpha += config.alpha_step) {
                    // Generate decomposition: alpha = alpha1 + alpha2
                    double alpha1, alpha2;
                    generate_alpha_decomposition(alpha, alpha1, alpha2, rng);

                    // Skip if invalid decomposition
                    if (alpha1 == 0.0 && alpha2 == 0.0 && alpha > 0.0) {
                        continue;
                    }

                    HomomorphicTestResult result = test_homomorphic_property(
                        engine, window_size, overlap_factor,
                        frequency, alpha, alpha1, alpha2,
                        config.sample_rate, config.n_analysis);

                    // Write result to file
                    out_file << result.window_size << "\t"
                             << result.overlap_factor << "\t"
                             << hop_size << "\t"
                             << result.num_frames << "\t"
                             << result.frequency << "\t"
                             << std::fixed << std::setprecision(2) << result.alpha_total << "\t"
                             << std::setprecision(4) << result.alpha1 << "\t"
                             << result.alpha2 << "\t"
                             << std::scientific << std::setprecision(10) 
                             << result.mse_direct << "\t"
                             << result.mse_composed << "\t"
                             << result.mse_homomorphic << "\t"
                             << result.max_error_homomorphic << "\t"
                             << result.mean_error_homomorphic << "\t"
                             << (result.success ? 1 : 0) << "\n";

                    test_count++;
                    if (!result.success) {
                        failed_count++;
                    } else {
                        alpha_sum_mse_homomorphic += result.mse_homomorphic;
                        alpha_success++;
                    }
                    alpha_count++;
                }

                // Print summary for this frequency/overlap combination
                if (alpha_success > 0) {
                    double avg_mse = alpha_sum_mse_homomorphic / alpha_success;
                    std::cout << "→ Avg Homomorphic MSE: " << std::scientific 
                              << std::setprecision(3) << avg_mse << "\n";
                } else {
                    std::cout << "→ ALL FAILED\n";
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    out_file.close();

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
    std::cout << "  Results saved to: " << config.output_filename << "\n\n";
}

// Parse command line arguments
bool parse_arguments(int argc, char* argv[], TestConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            std::cout << "FRFT Homomorphic Property Test Suite\n\n";
            std::cout << "Tests: FRFT(α) ≈ FRFT(α₂) ∘ FRFT(α₁) where α = α₁ + α₂\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --output FILE       Output filename (default: frft_homomorphic_results.txt)\n";
            std::cout << "  --sample-rate SR    Sample rate in Hz (default: 44100)\n";
            std::cout << "  --n-analysis N      Number of frames to analyze (default: 20)\n";
            std::cout << "  --quick            Run quick test (fewer window sizes and frequencies)\n";
            std::cout << "  --help             Show this help message\n";
            return false;
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.output_filename = argv[++i];
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