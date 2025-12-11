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

// Test configuration
struct ReversalTestConfig {
    std::vector<int> window_sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    double sample_rate = 44100.0;
    double chirp_duration = 0.05;  // 50ms per chirp for easy viewing
    double chirp_f_start = 200.0;   // Start frequency
    double chirp_f_end = 800.0;     // End frequency (slow chirp)
    double alpha = -2.0;            // FRFT alpha for reversal
    int num_test_frames = 5;        // Test 5 frames
    std::string figures_dir = "test_results/reversal_test/figures";
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

// Generate a chirp signal
void generate_chirp(std::vector<double>& signal, int size, double f_start,
                   double f_end, double sample_rate) {
    signal.resize(size);
    double duration = static_cast<double>(size) / sample_rate;
    double k = (f_end - f_start) / duration;  // Chirp rate

    for (int i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        double phase = 2.0 * M_PI * (f_start * t + 0.5 * k * t * t);
        signal[i] = std::sin(phase);
    }
}

// Generate Hamming window
void generate_hamming_window(std::vector<double>& window, int size) {
    window.resize(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (size - 1));
    }
}

// Manually reverse a signal
void reverse_signal(const std::vector<double>& input, std::vector<double>& output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[input.size() - 1 - i];
    }
}

// Test a single frame with FRFT
void test_single_frame(FRFTEngine& engine,
                      int window_size,
                      int frame_num,
                      const ReversalTestConfig& config,
                      std::vector<double>& original_frame,
                      std::vector<double>& frft_output,
                      std::vector<double>& reversed_frft) {

    // Generate a slow-moving chirp for this frame
    // The chirp duration should match the window size
    int chirp_samples = window_size;
    generate_chirp(original_frame, chirp_samples, config.chirp_f_start,
                  config.chirp_f_end, config.sample_rate);

    // Apply Hamming window to reduce edge effects
    std::vector<double> window;
    generate_hamming_window(window, window_size);
    for (int i = 0; i < window_size; ++i) {
        original_frame[i] *= window[i];
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
        std::cerr << "Error: FRFT computation failed for frame " << frame_num << "\n";
        return;
    }

    // Copy FRFT output
    frft_output.resize(window_size);
    std::copy(real_out.begin(), real_out.end(), frft_output.begin());

    // Reverse the FRFT output
    reverse_signal(frft_output, reversed_frft);
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
                         int frame_num,
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
    script << "fig.suptitle(f'FRFT Reversal Test - Frame " << frame_num
           << "\\nWindow Size: " << window_size
           << ", Sample Rate: " << sample_rate << " Hz', fontsize=14, fontweight='bold')\n\n";

    script << "# Plot 1: Original Chirp\n";
    script << "axes[0].plot(data['sample'], data['original'], 'b-', linewidth=2, label='Original Chirp')\n";
    script << "axes[0].set_ylabel('Amplitude', fontsize=11)\n";
    script << "axes[0].set_title('Original Chirp Signal', fontsize=12, fontweight='bold')\n";
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

// Run the reversal test with frame-by-frame analysis
void run_reversal_test_suite(const ReversalTestConfig& config) {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║      FRFT Reversal Test - Frame by Frame Analysis            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Test Configuration:\n";
    std::cout << "  Alpha: " << config.alpha << " (expected to reverse signal)\n";
    std::cout << "  Sample Rate: " << config.sample_rate << " Hz\n";
    std::cout << "  Chirp per frame: " << config.chirp_f_start << " Hz → "
              << config.chirp_f_end << " Hz\n";
    std::cout << "  Number of test frames: " << config.num_test_frames << "\n";
    std::cout << "  Window Sizes: ";
    for (size_t i = 0; i < config.window_sizes.size(); ++i) {
        std::cout << config.window_sizes[i];
        if (i < config.window_sizes.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";
    std::cout << "  Figures Directory: " << config.figures_dir << "\n\n";

    // Create figures directory
    std::cout << "Creating figures directory: " << config.figures_dir << "\n\n";
    create_directories(config.figures_dir);

    // Open results file in the main reversal_test directory
    std::string results_filename = "test_results/reversal_test/reversal_test_results.txt";
    std::ofstream results_file(results_filename);
    if (!results_file.is_open()) {
        std::cerr << "Warning: Cannot create results file: " << results_filename << "\n";
    } else {
        // Write header to results file
        results_file << "FRFT Reversal Test Results\n";
        results_file << "==========================\n\n";
        results_file << "Test Configuration:\n";
        results_file << "  Alpha: " << config.alpha << " (expected to reverse signal)\n";
        results_file << "  Sample Rate: " << config.sample_rate << " Hz\n";
        results_file << "  Chirp per frame: " << config.chirp_f_start << " Hz → "
                     << config.chirp_f_end << " Hz\n";
        results_file << "  Number of test frames: " << config.num_test_frames << "\n";
        results_file << "  Window Sizes: ";
        for (size_t i = 0; i < config.window_sizes.size(); ++i) {
            results_file << config.window_sizes[i];
            if (i < config.window_sizes.size() - 1) results_file << ", ";
        }
        results_file << "\n\n";
        results_file << std::string(70, '=') << "\n\n";
    }

    // Create FRFT engine
    FRFTEngine engine;

    // Process for each window size
    for (int window_size : config.window_sizes) {
        std::cout << "\n" << std::string(70, '-') << "\n";
        std::cout << "Testing window size: " << window_size << "\n";
        std::cout << std::string(70, '-') << "\n";

        double frame_duration = static_cast<double>(window_size) / config.sample_rate;
        std::cout << "Frame duration: " << (frame_duration * 1000.0) << " ms\n";
        std::cout << "Samples per frame: " << window_size << "\n\n";

        // Write to results file
        if (results_file.is_open()) {
            results_file << "Window Size: " << window_size << "\n";
            results_file << "Frame duration: " << (frame_duration * 1000.0) << " ms\n";
            results_file << "Samples per frame: " << window_size << "\n";
            results_file << std::string(70, '-') << "\n";
        }

        // Test multiple frames
        for (int frame_num = 1; frame_num <= config.num_test_frames; ++frame_num) {
            std::cout << "  Frame " << frame_num << "/" << config.num_test_frames << ": ";
            std::cout.flush();

            // Generate and test frame
            std::vector<double> original_frame;
            std::vector<double> frft_output;
            std::vector<double> reversed_frft;

            test_single_frame(engine, window_size, frame_num, config,
                            original_frame, frft_output, reversed_frft);

            // Calculate metrics
            double mse = calculate_mse(original_frame, reversed_frft);
            double correlation = calculate_correlation(original_frame, reversed_frft);

            // Print results
            std::cout << "MSE=" << std::scientific << std::setprecision(2) << mse
                     << ", Corr=" << std::fixed << std::setprecision(6) << correlation;

            std::string status;
            if (correlation > 0.99) {
                std::cout << " ✓ REVERSAL CONFIRMED";
                status = "REVERSAL CONFIRMED";
            } else if (correlation > 0.95) {
                std::cout << " ⚠ PARTIAL REVERSAL";
                status = "PARTIAL REVERSAL";
            } else {
                std::cout << " ✗ NOT REVERSED";
                status = "NOT REVERSED";
            }
            std::cout << "\n";

            // Write to results file
            if (results_file.is_open()) {
                results_file << "  Frame " << frame_num << ": "
                           << "MSE=" << std::scientific << std::setprecision(2) << mse
                           << ", Corr=" << std::fixed << std::setprecision(6) << correlation
                           << " - " << status << "\n";
            }

            // Generate plot for this frame
            std::string csv_filename = "temp_signal_data.csv";
            std::string png_filename = config.figures_dir + "/frame" +
                                      std::to_string(frame_num) +
                                      "_ws" + std::to_string(window_size) + ".png";

            write_signals_to_csv(csv_filename, original_frame, frft_output, reversed_frft);

            generate_plot_script(csv_filename, png_filename, window_size, frame_num,
                               config.sample_rate, mse, correlation);

            int ret = system("python3 plot_signals.py");
            if (ret != 0) {
                std::cerr << "    Warning: Failed to generate plot\n";
            }

            // Clean up temporary files
            remove(csv_filename.c_str());
            remove("plot_signals.py");
        }

        if (results_file.is_open()) {
            results_file << "\n";
        }
    }

    // Close results file
    if (results_file.is_open()) {
        results_file << std::string(70, '=') << "\n";
        results_file << "Test Complete\n";
        results_file.close();
        std::cout << "\nResults saved to: " << results_filename << "\n";
    }

    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                  Reversal Test Complete                       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nAll plots saved to: " << config.figures_dir << "/\n";
    std::cout << "Plot naming: frame[1-5]_ws[window_size].png\n\n";
}

// Parse command line arguments
bool parse_arguments(int argc, char* argv[], ReversalTestConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            std::cout << "FRFT Reversal Test - Frame by Frame Analysis (Alpha = -2)\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --sample-rate SR    Sample rate in Hz (default: 44100)\n";
            std::cout << "  --num-frames N      Number of test frames (default: 5)\n";
            std::cout << "  --figures-dir DIR   Directory for saving plots (default: reversal_test/figures)\n";
            std::cout << "  --help             Show this help message\n";
            return false;
        }
        else if (arg == "--sample-rate" && i + 1 < argc) {
            config.sample_rate = std::stod(argv[++i]);
        }
        else if (arg == "--num-frames" && i + 1 < argc) {
            config.num_test_frames = std::stoi(argv[++i]);
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