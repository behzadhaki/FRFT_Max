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
    int n_random_samples = 10000;  // Generate 1000 random beta/gamma pairs per frequency
    int max_figures_to_save = 40;  // Maximum number of figures to save
    bool apply_window = true;  // Apply Hamming window to frames (default: ON)
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

// Generate random alpha and decompose it into beta + gamma
// Both beta and gamma will be in [-2, 2] and alpha will be uniformly distributed in [-2, 2]
void generate_beta_gamma_pair(double& beta, double& gamma, double& alpha, std::mt19937& rng) {
    // First generate random alpha in [-2, 2] - this ensures uniform distribution
    std::uniform_real_distribution<double> alpha_dist(-2.0, 2.0);
    alpha = alpha_dist(rng);

    // Now decompose alpha = beta + gamma where both beta, gamma ∈ [-2, 2]
    // From gamma = alpha - beta and -2 ≤ gamma ≤ 2:
    // -2 ≤ alpha - beta ≤ 2
    // alpha - 2 ≤ beta ≤ alpha + 2
    // Combined with -2 ≤ beta ≤ 2:
    double min_beta = std::max(-2.0, alpha - 2.0);
    double max_beta = std::min(2.0, alpha + 2.0);

    // Generate random beta in valid range
    std::uniform_real_distribution<double> beta_dist(min_beta, max_beta);
    beta = beta_dist(rng);
    gamma = alpha - beta;

    // Safety clamps (should be guaranteed by construction)
    beta = std::max(-2.0, std::min(2.0, beta));
    gamma = std::max(-2.0, std::min(2.0, gamma));
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
                                                std::mt19937& rng,
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

    // Generate a longer signal to allow for random window selection
    // Make it 10x longer than needed to have good randomness
    int signal_length = window_size * 10;
    std::vector<double> full_signal;
    generate_sine_wave(full_signal, signal_length, frequency, sample_rate);

    // Select a random starting position for the window
    std::uniform_int_distribution<int> pos_dist(0, signal_length - window_size);
    int random_start = pos_dist(rng);

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

    // Process single frame from random position
    int pos = random_start;

    // Extract frame
    std::copy(full_signal.begin() + pos,
              full_signal.begin() + pos + window_size,
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

    // Calculate frame error metrics
    std::vector<double> direct_frame(real_direct.begin(), real_direct.end());
    std::vector<double> composed_frame(real_composed.begin(), real_composed.end());

    double frame_mse = calculate_mse(direct_frame, composed_frame);
    double frame_max = calculate_max_error(direct_frame, composed_frame);
    double frame_mean = calculate_mean_error(direct_frame, composed_frame);

    result.mse_homomorphic = frame_mse;
    result.max_error_homomorphic = frame_max;
    result.mean_error_homomorphic = frame_mean;
    result.mse_direct = -1.0;  // Not computed in single frame mode
    result.mse_composed = -1.0;  // Not computed in single frame mode
    result.num_frames = 1;
    result.success = true;

    // Store frame for plotting
    single_frame_direct_out = direct_frame;
    single_frame_composed_out = composed_frame;

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
    out_file << "# Modified: 1000 random beta/gamma pairs per frequency/window size combination\n";
    out_file << "# Beta, Gamma in [-2, 2]; Alpha = Beta + Gamma also in [-2, 2]\n";
    out_file << "# Single frame test with random window position per sample\n";
    out_file << "# Windowing: " << (config.apply_window ? "Hamming" : "None") << "\n";
    out_file << "# Columns: WindowSize OverlapFactor HopSize NumFrames Frequency ";
    out_file << "Alpha_Total Alpha1 Alpha2 MSE_Direct MSE_Composed MSE_Homomorphic ";
    out_file << "MaxError_Homomorphic MeanError_Homomorphic Success\n";
    out_file << "# Note: MSE_Direct and MSE_Composed are -1 (not computed in single frame mode)\n";
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
    std::cout << "║         1000 Random Beta/Gamma Pairs per Frequency           ║\n";
    std::cout << "║    Beta, Gamma ∈ [-2,2]; Alpha = Beta+Gamma ∈ [-2,2]        ║\n";
    std::cout << "║           Single frame with random window position            ║\n";
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

                // Generate 1000 random beta/gamma pairs
                for (int sample_idx = 0; sample_idx < config.n_random_samples; ++sample_idx) {
                    // Generate random beta and gamma (which determines alpha)
                    double beta, gamma, alpha;
                    generate_beta_gamma_pair(beta, gamma, alpha, rng);

                    std::vector<double> direct_result, composed_result;

                    HomomorphicTestResult result = test_homomorphic_property(
                        engine, window_size, overlap_factor,
                        frequency, alpha, beta, gamma,
                        config.sample_rate, config.n_analysis,
                        config.apply_window, rng,
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
                                                  "_alpha" + std::to_string(static_cast<int>(alpha * 100)) + ".png";

                        write_homomorphic_to_csv(csv_filename, direct_result, composed_result,
                                                config.sample_rate);
                        generate_homomorphic_plot_script(csv_filename, png_filename, window_size, frequency,
                                                        alpha, beta, gamma, config.sample_rate,
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
                              << " (" << alpha_success << "/" << config.n_random_samples << " successful)\n";
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
            std::cout << "Tests: FRFT(α) ≈ FRFT(γ) ∘ FRFT(β) where α = β + γ\n";
            std::cout << "Generates 1000 random β/γ pairs per frequency (β,γ ∈ [-2,2], α ∈ [-2,2])\n";
            std::cout << "Single frame test with random window position\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --output FILE       Output filename (default: homomorphic_test/results.txt)\n";
            std::cout << "  --sample-rate SR    Sample rate in Hz (default: 44100)\n";
            std::cout << "  --n-analysis N      Number of frames to analyze (default: 20, not used in single frame mode)\n";
            std::cout << "  --n-samples N       Number of random beta/gamma pairs per config (default: 1000)\n";
            std::cout << "  --max-figures N     Maximum figures to save (default: 40)\n";
            std::cout << "  --window            Apply Hamming window to frames (default: ON)\n";
            std::cout << "  --no-window         Disable Hamming window (use rectangular)\n";
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
        else if (arg == "--n-samples" && i + 1 < argc) {
            config.n_random_samples = std::stoi(argv[++i]);
        }
        else if (arg == "--n-alphas" && i + 1 < argc) {
            // Keep for backward compatibility, but use n_random_samples
            config.n_random_samples = std::stoi(argv[++i]);
        }
        else if (arg == "--max-figures" && i + 1 < argc) {
            config.max_figures_to_save = std::stoi(argv[++i]);
        }
        else if (arg == "--window") {
            config.apply_window = true;
        }
        else if (arg == "--no-window") {
            config.apply_window = false;
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