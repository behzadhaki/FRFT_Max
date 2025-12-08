#include "frft_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>
#include <random>
#include <set>
#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

// Test configuration
struct TestConfig {
    std::vector<int> window_sizes = {512, 1024, 2048, 4096};  // Focus on these sizes
    std::vector<int> overlap_factors = {1};  // 1=no overlap
    std::vector<double> test_frequencies = {100.0, 220.0, 440.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000., 9000., 10000.0};
    double sample_rate = 44100.0;
    int n_analysis = 20;  // Number of frames to analyze
    int n_random_alphas = 1000;  // Generate 1000 random alphas per frequency
    int max_figures_to_save = 40;  // Maximum number of figures to save
    bool apply_window = false;  // Apply Hamming window to frames
    std::string output_filename = "homomorphic_test/results.txt";
    std::string figures_dir = "homomorphic_test/figures";
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

// Write homomorphic comparison to CSV (single frame)
void write_homomorphic_to_csv(const std::string& filename,
                              const std::vector<double>& direct_result,
                              const std::vector<double>& composed_result,
                              double sample_rate) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << "\n";
        return;
    }

    file << "sample,frft_direct,frft_composed,error\n";
    for (size_t i = 0; i < direct_result.size(); ++i) {
        double error = direct_result[i] - composed_result[i];
        file << i << ","
             << direct_result[i] << ","
             << composed_result[i] << ","
             << error << "\n";
    }
    file.close();
}

// Generate Python plotting script for homomorphic property (single frame)
void generate_homomorphic_plot_script(const std::string& csv_filename,
                                     const std::string& output_png,
                                     int window_size,
                                     double test_frequency,
                                     double alpha_total,
                                     double alpha1,
                                     double alpha2,
                                     double sample_rate,
                                     double mse,
                                     double max_error) {
    std::ofstream script("plot_homomorphic.py");
    if (!script.is_open()) {
        std::cerr << "Error: Cannot create plotting script\n";
        return;
    }

    script << "import matplotlib.pyplot as plt\n";
    script << "import pandas as pd\n";
    script << "import numpy as np\n\n";

    script << "# Read data\n";
    script << "data = pd.read_csv('" << csv_filename << "')\n\n";

    script << "# Create figure with 2 subplots\n";
    script << "fig, axes = plt.subplots(2, 1, figsize=(14, 10))\n";
    script << "fig.suptitle(f'FRFT Homomorphic Property Test (Single Frame)\\n";
    script << "Window Size: " << window_size
           << ", Frequency: " << test_frequency << " Hz\\n";
    script << "α_total = " << alpha_total << " = α₁(" << alpha1 << ") + α₂(" << alpha2 << ")\\n";
    script << "Sample Rate: " << sample_rate << " Hz', fontsize=14, fontweight='bold')\n\n";

    script << "# Plot 1: Frame comparison\n";
    script << "axes[0].plot(data['sample'], data['frft_direct'], 'b-', linewidth=1.5, alpha=0.8, label='FRFT(α)')\n";
    script << "axes[0].plot(data['sample'], data['frft_composed'], 'r--', linewidth=1.5, alpha=0.8, label='FRFT(α₂) ∘ FRFT(α₁)')\n";
    script << "axes[0].set_ylabel('Amplitude', fontsize=12, fontweight='bold')\n";
    script << "axes[0].set_title('Single Frame: FRFT(α) vs FRFT(α₂) ∘ FRFT(α₁)', fontsize=13, fontweight='bold')\n";
    script << "axes[0].grid(True, alpha=0.3)\n";
    script << "axes[0].legend(loc='upper right', fontsize=11)\n";
    script << "axes[0].set_xlabel('Sample Index', fontsize=11)\n\n";

    script << "# Plot 2: Error signal\n";
    script << "axes[1].plot(data['sample'], data['error'], 'g-', linewidth=1.5)\n";
    script << "axes[1].set_ylabel('Error (Direct - Composed)', fontsize=12, fontweight='bold')\n";
    script << "axes[1].set_xlabel('Sample Index', fontsize=11)\n";
    script << "axes[1].set_title(f'Frame Error (MSE: {" << mse
           << ":.2e}, Max Error: {" << max_error << ":.2e})', fontsize=13, fontweight='bold')\n";
    script << "axes[1].grid(True, alpha=0.3)\n";
    script << "axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.7)\n\n";

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

// Generate Hamming window
void generate_hamming_window(std::vector<double>& window, int size) {
    window.resize(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (size - 1));
    }
}

// Apply window to signal (in-place)
void apply_window(std::vector<double>& signal, const std::vector<double>& window) {
    for (size_t i = 0; i < signal.size() && i < window.size(); ++i) {
        signal[i] *= window[i];
    }
}

// Calculate MSE
double calculate_mse(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        return -1.0;
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum / a.size();
}

// Calculate max absolute error
double calculate_max_error(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        return -1.0;
    }
    double max_err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double err = std::abs(a[i] - b[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

// Calculate mean absolute error
double calculate_mean_error(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        return -1.0;
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum / a.size();
}

// Generate random alpha decomposition: alpha = alpha1 + alpha2
// Both alpha1 and alpha2 can be in range [-2, 2]
void generate_alpha_decomposition(double alpha_total, double& alpha1, double& alpha2, std::mt19937& rng) {
    // We need: alpha1 + alpha2 = alpha_total
    // Constraints: -2 <= alpha1 <= 2 and -2 <= alpha2 <= 2

    // From alpha2 = alpha_total - alpha1, we get:
    // -2 <= alpha_total - alpha1 <= 2
    // -2 - alpha_total <= -alpha1 <= 2 - alpha_total
    // alpha_total - 2 <= alpha1 <= alpha_total + 2

    // Combine with -2 <= alpha1 <= 2:
    double min_alpha1 = std::max(-2.0, alpha_total - 2.0);
    double max_alpha1 = std::min(2.0, alpha_total + 2.0);

    // If no valid range exists (shouldn't happen for alpha_total in [-4, 4])
    if (min_alpha1 > max_alpha1) {
        alpha1 = 0.0;
        alpha2 = alpha_total;
        return;
    }

    // Generate random alpha1 in valid range
    std::uniform_real_distribution<double> dist(min_alpha1, max_alpha1);
    alpha1 = dist(rng);
    alpha2 = alpha_total - alpha1;

    // Clamp to ensure bounds (should be guaranteed by construction, but for safety)
    alpha1 = std::max(-2.0, std::min(2.0, alpha1));
    alpha2 = std::max(-2.0, std::min(2.0, alpha2));
}

// Perform homomorphic property test with frame-by-frame comparison
HomomorphicTestResult test_homomorphic_property(FRFTEngine& engine,
                                                int window_size,
                                                int overlap_factor,
                                                double frequency,
                                                double alpha_total,
                                                double alpha1,
                                                double alpha2,
                                                double sample_rate,
                                                int n_analysis,
                                                bool apply_windowing,
                                                std::vector<double>& single_frame_direct_out,
                                                std::vector<double>& single_frame_composed_out) {
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

    // Generate Hamming window if needed
    std::vector<double> hamming_window;
    if (apply_windowing) {
        generate_hamming_window(hamming_window, window_size);
    }

    // Allocate buffers for FRFT processing
    std::vector<double> real_in(window_size);
    std::vector<double> imag_in(window_size, 0.0);
    std::vector<double> real_direct(window_size);
    std::vector<double> imag_direct(window_size);
    std::vector<double> real_temp(window_size);      // After first transform
    std::vector<double> imag_temp(window_size);
    std::vector<double> real_composed(window_size);  // After second transform
    std::vector<double> imag_composed(window_size);

    // Storage for frame-by-frame errors
    std::vector<double> frame_mse_homomorphic;
    std::vector<double> frame_max_error;
    std::vector<double> frame_mean_error;

    // Storage for a single representative frame (for plotting)
    bool frame_stored_for_plot = false;

    // Process frames
    int num_frames = 0;

    for (int pos = 0; pos + window_size <= signal_length; pos += hop_size) {
        // Extract frame
        std::copy(original_signal.begin() + pos,
                  original_signal.begin() + pos + window_size,
                  real_in.begin());

        // Apply windowing if enabled
        if (apply_windowing) {
            apply_window(real_in, hamming_window);
        }

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

        // Calculate frame-by-frame error metrics
        std::vector<double> direct_frame(real_direct.begin(), real_direct.end());
        std::vector<double> composed_frame(real_composed.begin(), real_composed.end());

        double frame_mse = calculate_mse(direct_frame, composed_frame);
        double frame_max = calculate_max_error(direct_frame, composed_frame);
        double frame_mean = calculate_mean_error(direct_frame, composed_frame);

        frame_mse_homomorphic.push_back(frame_mse);
        frame_max_error.push_back(frame_max);
        frame_mean_error.push_back(frame_mean);

        // Store the first frame for plotting (representative example)
        if (!frame_stored_for_plot) {
            single_frame_direct_out = direct_frame;
            single_frame_composed_out = composed_frame;
            frame_stored_for_plot = true;
        }

        num_frames++;
        if (num_frames >= n_analysis) {
            break;
        }
    }

    // Aggregate error metrics across all frames
    if (num_frames > 0) {
        // Average MSE across frames
        double sum_mse = 0.0;
        double max_of_max = 0.0;
        double sum_mean = 0.0;

        for (int i = 0; i < num_frames; ++i) {
            sum_mse += frame_mse_homomorphic[i];
            if (frame_max_error[i] > max_of_max) {
                max_of_max = frame_max_error[i];
            }
            sum_mean += frame_mean_error[i];
        }

        result.mse_homomorphic = sum_mse / num_frames;
        result.max_error_homomorphic = max_of_max;
        result.mean_error_homomorphic = sum_mean / num_frames;

        // For mse_direct and mse_composed, compare with original (less meaningful with windowing)
        result.mse_direct = -1.0;  // Not computed in frame-by-frame mode
        result.mse_composed = -1.0;  // Not computed in frame-by-frame mode
    } else {
        result.mse_direct = -1.0;
        result.mse_composed = -1.0;
        result.mse_homomorphic = -1.0;
        result.max_error_homomorphic = -1.0;
        result.mean_error_homomorphic = -1.0;
    }

    result.num_frames = num_frames;
    result.success = true;

    return result;
}

// Run the complete test suite
void run_test_suite(const TestConfig& config) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create output directory
    create_directories(config.figures_dir);

    // Open output file
    std::ofstream out_file(config.output_filename);
    if (!out_file.is_open()) {
        throw std::runtime_error("Cannot open output file: " + config.output_filename);
    }

    // Write header
    out_file << "# FRFT Homomorphic Property Test Results\n";
    out_file << "# Modified: 1000 random alphas per frequency/window size combination\n";
    out_file << "# Frame-by-frame comparison (no overlap-add synthesis)\n";
    out_file << "# Windowing: " << (config.apply_window ? "Hamming" : "None") << "\n";
    out_file << "# Alpha decomposition: Alpha1, Alpha2 can be in range [-2, 2] where Alpha1 + Alpha2 = Alpha_Total\n";
    out_file << "# Columns: WindowSize OverlapFactor HopSize NumFrames Frequency ";
    out_file << "Alpha_Total Alpha1 Alpha2 MSE_Direct MSE_Composed MSE_Homomorphic ";
    out_file << "MaxError_Homomorphic MeanError_Homomorphic Success\n";
    out_file << "# Note: MSE_Direct and MSE_Composed are -1 in frame-by-frame mode\n";
    out_file << "WindowSize\tOverlapFactor\tHopSize\tNumFrames\tFrequency\t";
    out_file << "Alpha_Total\tAlpha1\tAlpha2\tMSE_Direct\tMSE_Composed\t";
    out_file << "MSE_Homomorphic\tMaxError_Homomorphic\tMeanError_Homomorphic\tSuccess\n";

    FRFTEngine engine;
    std::random_device rd;
    std::mt19937 rng(rd());

    int test_count = 0;
    int failed_count = 0;
    int total_figures_saved = 0;

    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        FRFT Homomorphic Property Test Suite (Modified)       ║\n";
    std::cout << "║             1000 Random Alphas per Configuration              ║\n";
    std::cout << "║          Frame-by-frame comparison (no overlap-add)           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nWindowing: " << (config.apply_window ? "Hamming" : "None (rectangular)") << "\n\n";

    // Determine which tests to save figures for (up to max_figures_to_save)
    int total_configurations = config.window_sizes.size() * config.test_frequencies.size();
    std::vector<int> save_figure_indices;

    if (total_configurations <= config.max_figures_to_save) {
        // Save all
        for (int i = 0; i < total_configurations; ++i) {
            save_figure_indices.push_back(i);
        }
    } else {
        // Randomly select which configurations to save
        std::uniform_int_distribution<int> dist(0, total_configurations - 1);
        std::set<int> selected;
        while (selected.size() < static_cast<size_t>(config.max_figures_to_save)) {
            selected.insert(dist(rng));
        }
        save_figure_indices.assign(selected.begin(), selected.end());
        std::sort(save_figure_indices.begin(), save_figure_indices.end());
    }

    int config_index = 0;

    for (int window_size : config.window_sizes) {
        std::cout << "\nTesting window size: " << window_size << "\n";
        std::cout << "─────────────────────────────────────────────────\n";

        // Pre-allocate buffers for this window size
        engine.prepare(window_size);

        for (int overlap_factor : config.overlap_factors) {
            int hop_size = window_size / overlap_factor;
            if (hop_size < 1) hop_size = 1;

            for (double frequency : config.test_frequencies) {
                std::cout << "  Freq: " << std::setw(8) << frequency << " Hz ";
                std::cout.flush();

                double alpha_sum_mse_homomorphic = 0.0;
                int alpha_success = 0;
                bool should_save_figure = std::find(save_figure_indices.begin(),
                                                   save_figure_indices.end(),
                                                   config_index) != save_figure_indices.end();
                bool figure_saved = false;

                // Generate 1000 random alphas
                std::uniform_real_distribution<double> alpha_dist(0.0, 2.0);

                for (int alpha_idx = 0; alpha_idx < config.n_random_alphas; ++alpha_idx) {
                    // Generate random alpha_total in range [0, 2]
                    double alpha_total = alpha_dist(rng);

                    // Generate decomposition: alpha = alpha1 + alpha2
                    double alpha1, alpha2;
                    generate_alpha_decomposition(alpha_total, alpha1, alpha2, rng);

                    std::vector<double> direct_result, composed_result;

                    HomomorphicTestResult result = test_homomorphic_property(
                        engine, window_size, overlap_factor,
                        frequency, alpha_total, alpha1, alpha2,
                        config.sample_rate, config.n_analysis,
                        config.apply_window,
                        direct_result, composed_result);

                    // Write result to file
                    out_file << result.window_size << "\t"
                             << result.overlap_factor << "\t"
                             << hop_size << "\t"
                             << result.num_frames << "\t"
                             << result.frequency << "\t"
                             << std::fixed << std::setprecision(6) << result.alpha_total << "\t"
                             << std::setprecision(6) << result.alpha1 << "\t"
                             << result.alpha2 << "\t"
                             << std::scientific << std::setprecision(10)
                             << result.mse_direct << "\t"
                             << result.mse_composed << "\t"
                             << result.mse_homomorphic << "\t"
                             << result.max_error_homomorphic << "\t"
                             << result.mean_error_homomorphic << "\t"
                             << (result.success ? 1 : 0) << "\n";

                    // Save one figure per configuration (if selected)
                    if (should_save_figure && !figure_saved && result.success &&
                        total_figures_saved < config.max_figures_to_save) {
                        std::string csv_filename = "temp_homomorphic.csv";
                        std::string png_filename = config.figures_dir + "/freq" +
                                                  std::to_string(static_cast<int>(frequency)) +
                                                  "_ws" + std::to_string(window_size) +
                                                  "_alpha" + std::to_string(static_cast<int>(alpha_total * 100)) + ".png";

                        write_homomorphic_to_csv(csv_filename, direct_result, composed_result,
                                                config.sample_rate);
                        generate_homomorphic_plot_script(csv_filename, png_filename, window_size, frequency,
                                                        alpha_total, alpha1, alpha2, config.sample_rate,
                                                        result.mse_homomorphic, result.max_error_homomorphic);

                        int ret = system("python3 plot_homomorphic.py 2>/dev/null");
                        if (ret == 0) {
                            figure_saved = true;
                            total_figures_saved++;
                        }

                        // Clean up temporary files
                        remove(csv_filename.c_str());
                        remove("plot_homomorphic.py");
                    }

                    test_count++;
                    if (!result.success) {
                        failed_count++;
                    } else {
                        alpha_sum_mse_homomorphic += result.mse_homomorphic;
                        alpha_success++;
                    }
                }

                // Print summary for this frequency/overlap combination
                if (alpha_success > 0) {
                    double avg_mse = alpha_sum_mse_homomorphic / alpha_success;
                    std::cout << "→ Avg MSE: " << std::scientific
                              << std::setprecision(3) << avg_mse
                              << " (" << alpha_success << "/" << config.n_random_alphas << " successful)\n";
                } else {
                    std::cout << "→ ALL FAILED\n";
                }

                config_index++;
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
    std::cout << "  Results saved to: " << config.output_filename << "\n";
    std::cout << "  Figures saved: " << total_figures_saved << " (max: " << config.max_figures_to_save << ")\n";
    std::cout << "  Figures directory: " << config.figures_dir << "/\n\n";
}

// Parse command line arguments
bool parse_arguments(int argc, char* argv[], TestConfig& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            std::cout << "FRFT Homomorphic Property Test Suite (Modified)\n\n";
            std::cout << "Tests: FRFT(α) ≈ FRFT(α₂) ∘ FRFT(α₁) where α = α₁ + α₂\n";
            std::cout << "Generates 1000 random alphas per frequency/window configuration\n";
            std::cout << "Uses frame-by-frame comparison (no overlap-add)\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --output FILE       Output filename (default: homomorphic_test/results.txt)\n";
            std::cout << "  --sample-rate SR    Sample rate in Hz (default: 44100)\n";
            std::cout << "  --n-analysis N      Number of frames to analyze (default: 20)\n";
            std::cout << "  --n-alphas N        Number of random alphas per config (default: 1000)\n";
            std::cout << "  --max-figures N     Maximum figures to save (default: 40)\n";
            std::cout << "  --window            Apply Hamming window to frames (default: off)\n";
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
        else if (arg == "--n-alphas" && i + 1 < argc) {
            config.n_random_alphas = std::stoi(argv[++i]);
        }
        else if (arg == "--max-figures" && i + 1 < argc) {
            config.max_figures_to_save = std::stoi(argv[++i]);
        }
        else if (arg == "--window") {
            config.apply_window = true;
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