#include "frft_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>
#include <complex>
#include <fftw3.h>

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
    std::string output_filename = "frft_vs_fft_results.txt";
};

// Result for a single FRFT vs FFT comparison test
struct FFTComparisonResult {
    int window_size;
    int overlap_factor;
    double frequency;
    double mse_magnitude;      // MSE between FRFT and FFT magnitude spectra
    double mse_phase;          // MSE between FRFT and FFT phase spectra
    double mse_complex;        // MSE between FRFT and FFT complex values
    double correlation_mag;    // Correlation between magnitude spectra
    double max_error_mag;      // Maximum magnitude difference
    double mean_error_mag;     // Mean magnitude difference
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

// Calculate Mean Squared Error between two real signals
double calculate_mse(const std::vector<double>& signal1, const std::vector<double>& signal2) {
    if (signal1.size() != signal2.size()) {
        return -1.0;
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

// Calculate Pearson correlation coefficient
double calculate_correlation(const std::vector<double>& signal1, const std::vector<double>& signal2) {
    if (signal1.size() != signal2.size() || signal1.empty()) {
        return -1.0;
    }

    double mean1 = 0.0, mean2 = 0.0;
    for (size_t i = 0; i < signal1.size(); ++i) {
        mean1 += signal1[i];
        mean2 += signal2[i];
    }
    mean1 /= signal1.size();
    mean2 /= signal2.size();

    double numerator = 0.0;
    double denom1 = 0.0;
    double denom2 = 0.0;

    for (size_t i = 0; i < signal1.size(); ++i) {
        double diff1 = signal1[i] - mean1;
        double diff2 = signal2[i] - mean2;
        numerator += diff1 * diff2;
        denom1 += diff1 * diff1;
        denom2 += diff2 * diff2;
    }

    if (denom1 == 0.0 || denom2 == 0.0) {
        return 0.0;
    }

    return numerator / std::sqrt(denom1 * denom2);
}

// Compute standard FFT using FFTW
void compute_fft(const double* real_in, const double* imag_in, int size,
                 double* real_out, double* imag_out) {
    // Allocate FFTW arrays
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);

    // Copy input data
    for (int i = 0; i < size; ++i) {
        in[i][0] = real_in[i];
        in[i][1] = imag_in[i];
    }

    // Create plan and execute
    fftw_plan plan = fftw_plan_dft_1d(size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Copy output data
    for (int i = 0; i < size; ++i) {
        real_out[i] = out[i][0];
        imag_out[i] = out[i][1];
    }

    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

// Perform FRFT vs FFT comparison test
FFTComparisonResult test_frft_vs_fft(FRFTEngine& engine,
                                      int window_size,
                                      int overlap_factor,
                                      double frequency,
                                      double sample_rate,
                                      int n_analysis) {
    FFTComparisonResult result;
    result.window_size = window_size;
    result.overlap_factor = overlap_factor;
    result.frequency = frequency;
    result.success = false;
    result.num_frames = 0;

    // Calculate hop size from overlap factor
    int hop_size = window_size / overlap_factor;
    if (hop_size < 1) hop_size = 1;

    // Generate enough signal for n_analysis frames
    int signal_length = window_size + (n_analysis - 1) * hop_size;
    std::vector<double> original_signal;
    generate_sine_wave(original_signal, signal_length, frequency, sample_rate);

    // Allocate buffers
    std::vector<double> real_in(window_size);
    std::vector<double> imag_in(window_size, 0.0);
    std::vector<double> real_frft(window_size);
    std::vector<double> imag_frft(window_size);
    std::vector<double> real_fft(window_size);
    std::vector<double> imag_fft(window_size);

    // Accumulators for statistics across all frames
    std::vector<double> frft_magnitude_accum(window_size, 0.0);
    std::vector<double> fft_magnitude_accum(window_size, 0.0);
    std::vector<double> frft_phase_accum(window_size, 0.0);
    std::vector<double> fft_phase_accum(window_size, 0.0);
    
    double complex_error_accum = 0.0;
    double mag_error_accum = 0.0;
    double phase_error_accum = 0.0;
    double max_mag_error = 0.0;
    int num_frames = 0;

    // Process frames
    for (int pos = 0; pos + window_size <= signal_length; pos += hop_size) {
        // Extract frame
        std::copy(original_signal.begin() + pos,
                  original_signal.begin() + pos + window_size,
                  real_in.begin());
        std::fill(imag_in.begin(), imag_in.end(), 0.0);

        // Compute FRFT with alpha = 1.0
        bool frft_success = engine.compute(real_in.data(), imag_in.data(),
                                          real_frft.data(), imag_frft.data(),
                                          window_size, 1.0);

        if (!frft_success) {
            result.mse_magnitude = -1.0;
            result.mse_phase = -1.0;
            result.mse_complex = -1.0;
            result.correlation_mag = -1.0;
            result.max_error_mag = -1.0;
            result.mean_error_mag = -1.0;
            return result;
        }

        // Compute standard FFT
        compute_fft(real_in.data(), imag_in.data(), window_size,
                   real_fft.data(), imag_fft.data());

        // Calculate magnitude and phase for both transforms
        for (int i = 0; i < window_size; ++i) {
            // FRFT
            double frft_mag = std::sqrt(real_frft[i] * real_frft[i] + 
                                       imag_frft[i] * imag_frft[i]);
            double frft_phase = std::atan2(imag_frft[i], real_frft[i]);
            
            // FFT
            double fft_mag = std::sqrt(real_fft[i] * real_fft[i] + 
                                      imag_fft[i] * imag_fft[i]);
            double fft_phase = std::atan2(imag_fft[i], real_fft[i]);

            // Accumulate for averaging
            frft_magnitude_accum[i] += frft_mag;
            fft_magnitude_accum[i] += fft_mag;
            frft_phase_accum[i] += frft_phase;
            fft_phase_accum[i] += fft_phase;

            // Calculate errors
            double mag_error = frft_mag - fft_mag;
            mag_error_accum += mag_error * mag_error;
            
            // Phase error (handle wraparound)
            double phase_diff = frft_phase - fft_phase;
            while (phase_diff > M_PI) phase_diff -= 2.0 * M_PI;
            while (phase_diff < -M_PI) phase_diff += 2.0 * M_PI;
            phase_error_accum += phase_diff * phase_diff;

            // Complex error
            double real_diff = real_frft[i] - real_fft[i];
            double imag_diff = imag_frft[i] - imag_fft[i];
            complex_error_accum += real_diff * real_diff + imag_diff * imag_diff;

            // Track max magnitude error
            double abs_mag_error = std::abs(mag_error);
            if (abs_mag_error > max_mag_error) {
                max_mag_error = abs_mag_error;
            }
        }

        num_frames++;
    }

    if (num_frames == 0) {
        result.mse_magnitude = -1.0;
        result.mse_phase = -1.0;
        result.mse_complex = -1.0;
        result.correlation_mag = -1.0;
        result.max_error_mag = -1.0;
        result.mean_error_mag = -1.0;
        return result;
    }

    // Calculate average magnitude and phase spectra
    std::vector<double> avg_frft_magnitude(window_size);
    std::vector<double> avg_fft_magnitude(window_size);
    for (int i = 0; i < window_size; ++i) {
        avg_frft_magnitude[i] = frft_magnitude_accum[i] / num_frames;
        avg_fft_magnitude[i] = fft_magnitude_accum[i] / num_frames;
    }

    // Calculate final metrics
    result.mse_magnitude = mag_error_accum / (window_size * num_frames);
    result.mse_phase = phase_error_accum / (window_size * num_frames);
    result.mse_complex = complex_error_accum / (window_size * num_frames);
    result.correlation_mag = calculate_correlation(avg_frft_magnitude, avg_fft_magnitude);
    result.max_error_mag = max_mag_error;
    result.mean_error_mag = std::sqrt(mag_error_accum / (window_size * num_frames));
    result.num_frames = num_frames;
    result.success = true;

    return result;
}

// Run the complete test suite
void run_test_suite(const TestConfig& config) {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          FRFT vs FFT Comparison Test Suite (α=1)             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Testing Property: FRFT(α=1) ≈ FFT\n\n";

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
    std::cout << "  Frames per Test: " << config.n_analysis << "\n";
    std::cout << "  Output File: " << config.output_filename << "\n\n";

    // Calculate total number of tests
    int total_tests = config.window_sizes.size() * config.overlap_factors.size() *
                      config.test_frequencies.size();

    std::cout << "Total tests to run: " << total_tests << "\n\n";

    // Open output file
    std::ofstream out_file(config.output_filename);
    if (!out_file.is_open()) {
        std::cerr << "Error: Cannot open output file: " << config.output_filename << "\n";
        return;
    }

    // Write header
    out_file << "# FRFT vs FFT Comparison Test Results (α=1)\n";
    out_file << "# Testing: FRFT(α=1) ≈ FFT\n";
    out_file << "# Sample Rate: " << config.sample_rate << " Hz\n";
    out_file << "# Number of Analysis Frames: " << config.n_analysis << "\n";
    out_file << "#\n";
    out_file << "# Columns:\n";
    out_file << "# 1. Window Size\n";
    out_file << "# 2. Overlap Factor\n";
    out_file << "# 3. Hop Size\n";
    out_file << "# 4. Number of Frames\n";
    out_file << "# 5. Frequency (Hz)\n";
    out_file << "# 6. MSE Magnitude (between FRFT and FFT magnitude spectra)\n";
    out_file << "# 7. MSE Phase (between FRFT and FFT phase spectra)\n";
    out_file << "# 8. MSE Complex (between FRFT and FFT complex values)\n";
    out_file << "# 9. Correlation Magnitude (correlation between magnitude spectra)\n";
    out_file << "# 10. Max Error Magnitude\n";
    out_file << "# 11. Mean Error Magnitude\n";
    out_file << "# 12. Success\n";
    out_file << "#\n";
    out_file << "WindowSize\tOverlapFactor\tHopSize\tNumFrames\tFrequency\t"
             << "MSE_Magnitude\tMSE_Phase\tMSE_Complex\tCorrelation_Mag\t"
             << "MaxError_Mag\tMeanError_Mag\tSuccess\n";

    // Create FRFT engine
    FRFTEngine engine;

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

                FFTComparisonResult result = test_frft_vs_fft(
                    engine, window_size, overlap_factor,
                    frequency, config.sample_rate, config.n_analysis);

                // Write result to file
                out_file << result.window_size << "\t"
                         << result.overlap_factor << "\t"
                         << hop_size << "\t"
                         << result.num_frames << "\t"
                         << result.frequency << "\t"
                         << std::scientific << std::setprecision(10) 
                         << result.mse_magnitude << "\t"
                         << result.mse_phase << "\t"
                         << result.mse_complex << "\t"
                         << std::fixed << std::setprecision(6)
                         << result.correlation_mag << "\t"
                         << std::scientific << std::setprecision(10)
                         << result.max_error_mag << "\t"
                         << result.mean_error_mag << "\t"
                         << (result.success ? 1 : 0) << "\n";

                test_count++;
                if (!result.success) {
                    failed_count++;
                    std::cout << "→ FAILED\n";
                } else {
                    std::cout << "→ Mag MSE: " << std::scientific 
                              << std::setprecision(3) << result.mse_magnitude
                              << ", Corr: " << std::fixed << std::setprecision(4)
                              << result.correlation_mag << "\n";
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
            std::cout << "FRFT vs FFT Comparison Test Suite\n\n";
            std::cout << "Tests: FRFT(α=1) ≈ FFT\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --output FILE       Output filename (default: frft_vs_fft_results.txt)\n";
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