#include "frft_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

// Signal types for testing
enum class SignalType {
    IMPULSE_QUARTER,      // Impulse at N/4
    IMPULSE_THREE_QUARTER, // Impulse at 3*N/4
    DECAYING_SINE,        // Exponentially decaying sine
    GROWING_SINE,         // Exponentially growing sine
    LINEAR_CHIRP,         // Linear chirp 200->800 Hz
    STEP_TONE,           // Step function: silence then tone
    MULTI_TONE           // Multiple tones with varying amplitudes
};

// Get signal type name
std::string get_signal_name(SignalType type) {
    switch(type) {
        case SignalType::IMPULSE_QUARTER: return "Impulse_Quarter";
        case SignalType::IMPULSE_THREE_QUARTER: return "Impulse_ThreeQuarter";
        case SignalType::DECAYING_SINE: return "Decaying_Sine";
        case SignalType::GROWING_SINE: return "Growing_Sine";
        case SignalType::LINEAR_CHIRP: return "Linear_Chirp";
        case SignalType::STEP_TONE: return "Step_Tone";
        case SignalType::MULTI_TONE: return "Multi_Tone";
        default: return "Unknown";
    }
}

// Test configuration
struct ReversalTestConfig {
    std::vector<int> window_sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    std::vector<SignalType> signal_types = {
        SignalType::IMPULSE_QUARTER,
        SignalType::IMPULSE_THREE_QUARTER,
        SignalType::DECAYING_SINE,
        SignalType::GROWING_SINE,
        SignalType::LINEAR_CHIRP,
        SignalType::STEP_TONE,
        SignalType::MULTI_TONE
    };
    double sample_rate = 44100.0;
    double alpha = -2.0;            // FRFT alpha for reversal
    std::string figures_dir = "reversal_test/figures";
    std::string results_file = "reversal_test/results.txt";
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

// Generate test signal based on type
void generate_test_signal(std::vector<double>& signal, int size, SignalType type, double sample_rate) {
    signal.resize(size);

    switch(type) {
        case SignalType::IMPULSE_QUARTER: {
            // Impulse at N/4
            std::fill(signal.begin(), signal.end(), 0.0);
            int impulse_pos = size / 4;
            if (impulse_pos < size) {
                signal[impulse_pos] = 1.0;
            }
            break;
        }

        case SignalType::IMPULSE_THREE_QUARTER: {
            // Impulse at 3*N/4
            std::fill(signal.begin(), signal.end(), 0.0);
            int impulse_pos = (3 * size) / 4;
            if (impulse_pos < size) {
                signal[impulse_pos] = 1.0;
            }
            break;
        }

        case SignalType::DECAYING_SINE: {
            // Exponentially decaying sine wave at 200 Hz
            double frequency = 200.0;
            double tau = static_cast<double>(size) / (sample_rate * 3.0); // Decay to ~5% by end
            for (int i = 0; i < size; ++i) {
                double t = static_cast<double>(i) / sample_rate;
                signal[i] = std::exp(-t / tau) * std::sin(2.0 * M_PI * frequency * t);
            }
            break;
        }

        case SignalType::GROWING_SINE: {
            // Exponentially growing sine wave at 200 Hz
            double frequency = 200.0;
            double duration = static_cast<double>(size) / sample_rate;
            double tau = duration / 3.0;
            for (int i = 0; i < size; ++i) {
                double t = static_cast<double>(i) / sample_rate;
                signal[i] = std::exp(t / tau - duration / tau) * std::sin(2.0 * M_PI * frequency * t);
            }
            break;
        }

        case SignalType::LINEAR_CHIRP: {
            // Linear chirp from 200 Hz to 800 Hz
            double f_start = 200.0;
            double f_end = 800.0;
            double duration = static_cast<double>(size) / sample_rate;
            double k = (f_end - f_start) / duration;
            for (int i = 0; i < size; ++i) {
                double t = static_cast<double>(i) / sample_rate;
                double phase = 2.0 * M_PI * (f_start * t + 0.5 * k * t * t);
                signal[i] = std::sin(phase);
            }
            break;
        }

        case SignalType::STEP_TONE: {
            // First half silence, second half 440 Hz tone
            double frequency = 440.0;
            int half_size = size / 2;
            for (int i = 0; i < size; ++i) {
                if (i < half_size) {
                    signal[i] = 0.0;
                } else {
                    double t = static_cast<double>(i) / sample_rate;
                    signal[i] = std::sin(2.0 * M_PI * frequency * t);
                }
            }
            break;
        }

        case SignalType::MULTI_TONE: {
            // Multiple tones with time-varying amplitudes
            double f1 = 200.0;
            double f2 = 400.0;
            double f3 = 600.0;
            double duration = static_cast<double>(size) / sample_rate;

            for (int i = 0; i < size; ++i) {
                double t = static_cast<double>(i) / sample_rate;
                double t_norm = t / duration;

                // Varying amplitude envelopes
                double a1 = 1.0 - t_norm;           // Decreasing
                double a2 = t_norm;                  // Increasing
                double a3 = std::sin(M_PI * t_norm); // Bell curve

                signal[i] = a1 * std::sin(2.0 * M_PI * f1 * t) +
                           a2 * std::sin(2.0 * M_PI * f2 * t) +
                           a3 * std::sin(2.0 * M_PI * f3 * t);
            }
            break;
        }
    }
}

// Generate Hamming window
void generate_hamming_window(std::vector<double>& window, int size) {
    window.resize(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (size - 1));
    }
}

// Calculate Mean Squared Error
double calculate_mse(const std::vector<double>& signal1, const std::vector<double>& signal2) {
    if (signal1.size() != signal2.size()) return -1.0;

    double sum_squared_error = 0.0;
    for (size_t i = 0; i < signal1.size(); ++i) {
        double error = signal1[i] - signal2[i];
        sum_squared_error += error * error;
    }
    return sum_squared_error / signal1.size();
}

// Calculate maximum absolute error
double calculate_max_error(const std::vector<double>& signal1, const std::vector<double>& signal2) {
    if (signal1.size() != signal2.size()) return -1.0;

    double max_err = 0.0;
    for (size_t i = 0; i < signal1.size(); ++i) {
        double error = std::abs(signal1[i] - signal2[i]);
        if (error > max_err) max_err = error;
    }
    return max_err;
}

// Calculate correlation coefficient
double calculate_correlation(const std::vector<double>& signal1, const std::vector<double>& signal2) {
    if (signal1.size() != signal2.size()) return -999.0;

    int n = signal1.size();

    // Calculate means
    double mean1 = 0.0, mean2 = 0.0;
    for (int i = 0; i < n; ++i) {
        mean1 += signal1[i];
        mean2 += signal2[i];
    }
    mean1 /= n;
    mean2 /= n;

    // Calculate correlation
    double numerator = 0.0;
    double sum_sq1 = 0.0;
    double sum_sq2 = 0.0;

    for (int i = 0; i < n; ++i) {
        double diff1 = signal1[i] - mean1;
        double diff2 = signal2[i] - mean2;
        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }

    double denominator = std::sqrt(sum_sq1 * sum_sq2);
    if (denominator < 1e-10) return 0.0;

    return numerator / denominator;
}

// Manually reverse a signal (shifted reversal - what actually works with FRFT)
// This uses output[i] = input[N-i] with wraparound instead of input[N-1-i]
void reverse_signal(const std::vector<double>& input, std::vector<double>& output) {
    output.resize(input.size());
    int N = input.size();
    for (int i = 0; i < N; ++i) {
        int rev_idx = N - i;
        if (rev_idx >= N) rev_idx = 0; // Wrap around for i=0 case
        output[i] = input[rev_idx];
    }
}

// Test a single frame with FRFT
void test_single_frame(FRFTEngine& engine,
                      int window_size,
                      SignalType signal_type,
                      const ReversalTestConfig& config,
                      std::vector<double>& original_frame,
                      std::vector<double>& frft_output,
                      std::vector<double>& reversed_frft) {

    // Generate test signal
    generate_test_signal(original_frame, window_size, signal_type, config.sample_rate);

    // Apply Hamming window to reduce edge effects (except for impulses)
    if (signal_type != SignalType::IMPULSE_QUARTER &&
        signal_type != SignalType::IMPULSE_THREE_QUARTER) {
        std::vector<double> window;
        generate_hamming_window(window, window_size);
        for (int i = 0; i < window_size; ++i) {
            original_frame[i] *= window[i];
        }
    }

    // Prepare FRFT engine
    engine.prepare(window_size);

    // Allocate buffers
    std::vector<double> real_in(window_size);
    std::vector<double> imag_in(window_size, 0.0);
    std::vector<double> real_out(window_size);
    std::vector<double> imag_out(window_size);

    // Copy to FRFT input buffers
    std::copy(original_frame.begin(), original_frame.end(), real_in.begin());

    // Apply FRFT with alpha = -2
    bool success = engine.compute(real_in.data(), imag_in.data(),
                                 real_out.data(), imag_out.data(),
                                 window_size, config.alpha);

    if (!success) {
        std::cerr << "Error: FRFT computation failed\n";
        return;
    }

    // Copy FRFT output
    frft_output.resize(window_size);
    std::copy(real_out.begin(), real_out.end(), frft_output.begin());

    // Reverse the FRFT output (equivalent to Python's [::-1])
    reverse_signal(frft_output, reversed_frft);
}

// Write signal data to CSV for plotting
void write_signals_to_csv(const std::string& filename,
                         const std::vector<double>& original,
                         const std::vector<double>& frft_output,
                         const std::vector<double>& reversed_frft) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << "\n";
        return;
    }

    file << "sample,original,frft_output,reversed_frft\n";
    for (size_t i = 0; i < original.size(); ++i) {
        file << i << ","
             << original[i] << ","
             << frft_output[i] << ","
             << reversed_frft[i] << "\n";
    }
    file.close();
}

// Generate Python plotting script for frame analysis
void generate_plot_script(const std::string& csv_filename,
                         const std::string& output_png,
                         int window_size,
                         const std::string& signal_name,
                         double sample_rate,
                         double mse,
                         double correlation) {
    std::ofstream script("plot_signals.py");
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
    script << "fig, axes = plt.subplots(3, 1, figsize=(12, 10))\n";
    script << "fig.suptitle(f'FRFT Reversal Test - " << signal_name
           << "\\nWindow Size: " << window_size
           << ", Sample Rate: " << sample_rate << " Hz', fontsize=14, fontweight='bold')\n\n";

    script << "# Plot 1: Original Signal\n";
    script << "axes[0].plot(data['sample'], data['original'], 'b-', linewidth=2, label='Original: " << signal_name << "')\n";
    script << "axes[0].set_ylabel('Amplitude', fontsize=11)\n";
    script << "axes[0].set_title('Original Signal', fontsize=12, fontweight='bold')\n";
    script << "axes[0].grid(True, alpha=0.3)\n";
    script << "axes[0].legend(loc='upper right', fontsize=10)\n";
    script << "axes[0].set_xlim([0, len(data)-1])\n\n";

    script << "# Plot 2: FRFT Output (alpha = -2)\n";
    script << "axes[1].plot(data['sample'], data['frft_output'], 'r-', linewidth=2, label='FRFT Output (α=-2)')\n";
    script << "axes[1].set_ylabel('Amplitude', fontsize=11)\n";
    script << "axes[1].set_title('FRFT Output (α = -2) - Should be Time-Reversed', fontsize=12, fontweight='bold')\n";
    script << "axes[1].grid(True, alpha=0.3)\n";
    script << "axes[1].legend(loc='upper right', fontsize=10)\n";
    script << "axes[1].set_xlim([0, len(data)-1])\n\n";

    script << "# Plot 3: Overlay Comparison\n";
    script << "axes[2].plot(data['sample'], data['original'], 'b-', linewidth=2, alpha=0.7, label='Original')\n";
    script << "axes[2].plot(data['sample'], data['reversed_frft'], 'g--', linewidth=2, alpha=0.7, label='Reversed FRFT')\n";
    script << "axes[2].set_xlabel('Sample Index', fontsize=11)\n";
    script << "axes[2].set_ylabel('Amplitude', fontsize=11)\n";
    script << "axes[2].set_title(f'Overlay Comparison\\nMSE: {" << mse
           << ":.2e}, Correlation: {" << correlation << ":.6f}', fontsize=12, fontweight='bold')\n";
    script << "axes[2].grid(True, alpha=0.3)\n";
    script << "axes[2].legend(loc='upper right', fontsize=10)\n";
    script << "axes[2].set_xlim([0, len(data)-1])\n\n";

    script << "# Adjust layout and save\n";
    script << "plt.tight_layout()\n";
    script << "plt.savefig('" << output_png << "', dpi=150, bbox_inches='tight')\n";
    script << "plt.close()\n";
    script << "print('Plot saved to: " << output_png << "')\n";

    script.close();
}

// Run the reversal test with comprehensive signal suite
void run_reversal_test_suite(const ReversalTestConfig& config) {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║      FRFT Reversal Test - Comprehensive Signal Suite         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Test Configuration:\n";
    std::cout << "  Alpha: " << config.alpha << " (expected to reverse signal)\n";
    std::cout << "  Sample Rate: " << config.sample_rate << " Hz\n";
    std::cout << "  Signal Types: " << config.signal_types.size() << "\n";
    for (const auto& type : config.signal_types) {
        std::cout << "    - " << get_signal_name(type) << "\n";
    }
    std::cout << "  Window Sizes: ";
    for (size_t i = 0; i < config.window_sizes.size(); ++i) {
        std::cout << config.window_sizes[i];
        if (i < config.window_sizes.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";
    std::cout << "  Figures Directory: " << config.figures_dir << "\n";
    std::cout << "  Results File: " << config.results_file << "\n\n";

    // Create directories
    std::cout << "Creating output directories...\n";
    create_directories(config.figures_dir);
    create_directories("reversal_test");

    // Open results file
    std::ofstream results_file(config.results_file);
    if (!results_file.is_open()) {
        std::cerr << "Error: Cannot open results file: " << config.results_file << "\n";
        return;
    }

    // Write header to results file
    results_file << "# FRFT Reversal Test Results (Alpha = " << config.alpha << ")\n";
    results_file << "# Sample Rate: " << config.sample_rate << " Hz\n";
    results_file << "# Test Date: " << __DATE__ << " " << __TIME__ << "\n";
    results_file << "#\n";
    results_file << "# Reversal method: output[i] = input[N-i] with wraparound\n";
    results_file << "# Note: This differs from standard [::-1] by one sample shift\n";
    results_file << "# This is the correct reversal convention for this FRFT implementation\n";
    results_file << "#\n";
    results_file << "# Columns:\n";
    results_file << "# 1. Window Size\n";
    results_file << "# 2. Signal Type\n";
    results_file << "# 3. MSE (Mean Squared Error)\n";
    results_file << "# 4. Max Error\n";
    results_file << "# 5. Correlation Coefficient\n";
    results_file << "# 6. Reversal Status (CONFIRMED/PARTIAL/FAILED)\n";
    results_file << "#\n";
    results_file << "WindowSize\tSignalType\tMSE\tMaxError\tCorrelation\tStatus\n";

    // Create FRFT engine
    FRFTEngine engine;

    // Statistics tracking
    int total_tests = config.window_sizes.size() * config.signal_types.size();
    int tests_passed = 0;
    int tests_partial = 0;
    int tests_failed = 0;

    // Process for each window size
    for (int window_size : config.window_sizes) {
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "Testing window size: " << window_size << "\n";
        std::cout << std::string(70, '=') << "\n";

        double frame_duration = static_cast<double>(window_size) / config.sample_rate;
        std::cout << "Frame duration: " << (frame_duration * 1000.0) << " ms\n";
        std::cout << "Samples per frame: " << window_size << "\n\n";

        // Test each signal type
        for (SignalType signal_type : config.signal_types) {
            std::string signal_name = get_signal_name(signal_type);
            std::cout << "  Testing " << signal_name << "... ";
            std::cout.flush();

            // Generate and test signal
            std::vector<double> original_frame;
            std::vector<double> frft_output;
            std::vector<double> reversed_frft;

            test_single_frame(engine, window_size, signal_type, config,
                            original_frame, frft_output, reversed_frft);

            // Calculate metrics
            double mse = calculate_mse(original_frame, reversed_frft);
            double max_error = calculate_max_error(original_frame, reversed_frft);
            double correlation = calculate_correlation(original_frame, reversed_frft);

            // Determine status
            std::string status;
            if (correlation > 0.99) {
                status = "CONFIRMED";
                tests_passed++;
                std::cout << "✓ PASS";
            } else if (correlation > 0.95) {
                status = "PARTIAL";
                tests_partial++;
                std::cout << "⚠ PARTIAL";
            } else {
                status = "FAILED";
                tests_failed++;
                std::cout << "✗ FAIL";
            }

            std::cout << " (MSE=" << std::scientific << std::setprecision(2) << mse
                     << ", Corr=" << std::fixed << std::setprecision(6) << correlation << ")\n";

            // Write to results file
            results_file << window_size << "\t"
                        << signal_name << "\t"
                        << std::scientific << std::setprecision(6) << mse << "\t"
                        << max_error << "\t"
                        << std::fixed << std::setprecision(6) << correlation << "\t"
                        << status << "\n";

            // Generate ONE plot per window size and signal type
            std::string csv_filename = "temp_signal_data.csv";
            std::string png_filename = config.figures_dir + "/" + signal_name +
                                      "_ws" + std::to_string(window_size) + ".png";

            write_signals_to_csv(csv_filename, original_frame, frft_output, reversed_frft);

            generate_plot_script(csv_filename, png_filename, window_size, signal_name,
                               config.sample_rate, mse, correlation);

            int ret = system("python3 plot_signals.py 2>/dev/null");
            if (ret != 0) {
                std::cerr << "    Warning: Failed to generate plot\n";
            }

            // Clean up temporary files
            remove(csv_filename.c_str());
            remove("plot_signals.py");
        }
    }

    results_file.close();

    // Print summary
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                  Test Suite Complete                          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nSummary:\n";
    std::cout << "  Total tests: " << total_tests << "\n";
    std::cout << "  Passed (correlation > 0.99): " << tests_passed
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * tests_passed / total_tests) << "%)\n";
    std::cout << "  Partial (correlation > 0.95): " << tests_partial
              << " (" << (100.0 * tests_partial / total_tests) << "%)\n";
    std::cout << "  Failed: " << tests_failed
              << " (" << (100.0 * tests_failed / total_tests) << "%)\n";
    std::cout << "\n  Results saved to: " << config.results_file << "\n";
    std::cout << "  Plots saved to: " << config.figures_dir << "/\n";
    std::cout << "  Total plots generated: " << total_tests << "\n\n";
}

// Parse command line arguments
bool parse_arguments(int argc, char* argv[], ReversalTestConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            std::cout << "FRFT Reversal Test - Comprehensive Signal Suite (Alpha = -2)\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Tests FRFT reversal property with multiple signal types:\n";
            std::cout << "  - Impulse at N/4 position\n";
            std::cout << "  - Impulse at 3*N/4 position\n";
            std::cout << "  - Exponentially decaying sine\n";
            std::cout << "  - Exponentially growing sine\n";
            std::cout << "  - Linear chirp (200-800 Hz)\n";
            std::cout << "  - Step function (silence then tone)\n";
            std::cout << "  - Multi-tone with varying amplitudes\n\n";
            std::cout << "Options:\n";
            std::cout << "  --sample-rate SR    Sample rate in Hz (default: 44100)\n";
            std::cout << "  --figures-dir DIR   Directory for saving plots (default: reversal_test/figures)\n";
            std::cout << "  --help             Show this help message\n\n";
            std::cout << "Output:\n";
            std::cout << "  - Results text file: reversal_test/results.txt\n";
            std::cout << "  - PNG plots: reversal_test/figures/[SignalType]_ws[WindowSize].png\n";
            return false;
        }
        else if (arg == "--sample-rate" && i + 1 < argc) {
            config.sample_rate = std::stod(argv[++i]);
        }
        else if (arg == "--figures-dir" && i + 1 < argc) {
            config.figures_dir = argv[++i];
        }
    }
    
    return true;
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    // Set console to UTF-8 mode on Windows
    SetConsoleOutputCP(CP_UTF8);
#endif
    
    ReversalTestConfig config;
    
    if (!parse_arguments(argc, argv, config)) {
        return 0;  // Help was shown
    }
    
    try {
        run_reversal_test_suite(config);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}