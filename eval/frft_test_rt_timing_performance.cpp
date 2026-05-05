#include "../src/frft_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <string>
#include <random>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

// Test configuration
struct RTTimingConfig {
    std::vector<int> window_sizes;  // Powers of 2 from 16 to 131072
    std::vector<double> alpha_values;  // 0.0 to 1.0 in steps of 0.1
    int num_signals = 100;  // Number of random signals to test per window/alpha pair
    double sample_rate = 44100.0;
    std::string output_filename = "test_results/rt_timing_performance.txt";

    RTTimingConfig() {
        // Generate all power-of-2 window sizes from 16 to 131072
        for (int size = 16; size <= 131072; size *= 2) {
            window_sizes.push_back(size);
        }

        // Generate alpha values from 0.0 to 1.0 in steps of 0.1
        for (double alpha = 0.0; alpha <= 1.0; alpha += 0.1) {
            alpha_values.push_back(alpha);
        }
    }
};

// Result for a single timing test
struct RTTimingResult {
    int window_size;
    double alpha;
    int signal_index;
    double inference_time_ms;
    double rt_factor;
};

// Statistics for window/alpha combination
struct RTTimingStats {
    int window_size;
    double alpha;
    int num_samples;
    double mean_time_ms;
    double min_time_ms;
    double max_time_ms;
    double std_dev_ms;
    double mean_rt_factor;
    double best_rt_factor;
    double worst_rt_factor;
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

// Generate random signal
std::vector<double> generate_random_signal(int size, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> signal(size);
    for (int i = 0; i < size; ++i) {
        signal[i] = dist(rng);
    }
    return signal;
}

// Calculate statistics from timing results
RTTimingStats calculate_statistics(const std::vector<RTTimingResult>& results,
                                   int window_size,
                                   double alpha,
                                   double sample_rate) {
    RTTimingStats stats;
    stats.window_size = window_size;
    stats.alpha = alpha;
    stats.num_samples = results.size();

    if (results.empty()) {
        stats.mean_time_ms = 0.0;
        stats.min_time_ms = 0.0;
        stats.max_time_ms = 0.0;
        stats.std_dev_ms = 0.0;
        stats.mean_rt_factor = 0.0;
        stats.best_rt_factor = 0.0;
        stats.worst_rt_factor = 0.0;
        return stats;
    }

    // Calculate mean
    double sum_time = 0.0;
    double sum_rtf = 0.0;
    stats.min_time_ms = results[0].inference_time_ms;
    stats.max_time_ms = results[0].inference_time_ms;
    stats.best_rt_factor = results[0].rt_factor;
    stats.worst_rt_factor = results[0].rt_factor;

    for (const auto& result : results) {
        sum_time += result.inference_time_ms;
        sum_rtf += result.rt_factor;

        if (result.inference_time_ms < stats.min_time_ms) {
            stats.min_time_ms = result.inference_time_ms;
        }
        if (result.inference_time_ms > stats.max_time_ms) {
            stats.max_time_ms = result.inference_time_ms;
        }
        if (result.rt_factor < stats.best_rt_factor) {
            stats.best_rt_factor = result.rt_factor;
        }
        if (result.rt_factor > stats.worst_rt_factor) {
            stats.worst_rt_factor = result.rt_factor;
        }
    }

    stats.mean_time_ms = sum_time / results.size();
    stats.mean_rt_factor = sum_rtf / results.size();

    // Calculate standard deviation
    double sum_sq_diff = 0.0;
    for (const auto& result : results) {
        double diff = result.inference_time_ms - stats.mean_time_ms;
        sum_sq_diff += diff * diff;
    }
    stats.std_dev_ms = std::sqrt(sum_sq_diff / results.size());

    return stats;
}

// Write detailed results to file
void write_detailed_results(const std::vector<RTTimingResult>& all_results,
                           const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << "\n";
        return;
    }

    // Write header
    file << "window_size\talpha\tsignal_index\tinference_time_ms\trt_factor\n";

    // Write all results
    for (const auto& result : all_results) {
        file << result.window_size << "\t"
             << std::fixed << std::setprecision(2) << result.alpha << "\t"
             << result.signal_index << "\t"
             << std::setprecision(6) << result.inference_time_ms << "\t"
             << std::setprecision(4) << result.rt_factor << "\n";
    }

    file.close();
}

// Write summary statistics to file
void write_summary_statistics(const std::vector<RTTimingStats>& stats,
                             const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open summary file for writing: " << filename << "\n";
        return;
    }

    // Write header
    file << "window_size\talpha\tnum_samples\t"
         << "mean_time_ms\tmin_time_ms\tmax_time_ms\tstd_dev_ms\t"
         << "mean_rt_factor\tbest_rt_factor\tworst_rt_factor\n";

    // Write statistics
    for (const auto& stat : stats) {
        file << stat.window_size << "\t"
             << std::fixed << std::setprecision(2) << stat.alpha << "\t"
             << stat.num_samples << "\t"
             << std::setprecision(6) << stat.mean_time_ms << "\t"
             << stat.min_time_ms << "\t"
             << stat.max_time_ms << "\t"
             << stat.std_dev_ms << "\t"
             << std::setprecision(4) << stat.mean_rt_factor << "\t"
             << stat.best_rt_factor << "\t"
             << stat.worst_rt_factor << "\n";
    }

    file.close();
}

// Run the timing performance test suite
void run_rt_timing_test(const RTTimingConfig& config) {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     FRFT Real-Time Performance Timing Test Suite              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    // Create output directory
    create_directories("test_results");

    std::cout << "Configuration:\n";
    std::cout << "  Window sizes: " << config.window_sizes.size()
              << " (from " << config.window_sizes.front()
              << " to " << config.window_sizes.back() << ")\n";
    std::cout << "  Alpha values: " << config.alpha_values.size()
              << " (from " << config.alpha_values.front()
              << " to " << config.alpha_values.back() << ")\n";
    std::cout << "  Signals per test: " << config.num_signals << "\n";
    std::cout << "  Sample rate: " << config.sample_rate << " Hz\n";
    std::cout << "  Output file: " << config.output_filename << "\n";
    std::cout << "  Total tests: " << (config.window_sizes.size() * config.alpha_values.size() * config.num_signals) << "\n\n";

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 rng(rd());

    // Create FRFT engine
    FRFTEngine engine;

    std::vector<RTTimingResult> all_results;
    std::vector<RTTimingStats> all_stats;

    auto overall_start_time = std::chrono::high_resolution_clock::now();

    int total_tests = 0;
    int completed_combinations = 0;
    int total_combinations = config.window_sizes.size() * config.alpha_values.size();

    // Loop through all window sizes
    for (int window_size : config.window_sizes) {
        std::cout << "Testing window size: " << window_size << " samples\n";

        // Generate random signals once per window size (reused for all alphas)
        std::vector<std::vector<double>> random_signals;
        for (int i = 0; i < config.num_signals; ++i) {
            random_signals.push_back(generate_random_signal(window_size, rng));
        }

        // Loop through all alpha values
        for (double alpha : config.alpha_values) {
            std::vector<RTTimingResult> current_results;

            // Test each random signal
            for (int sig_idx = 0; sig_idx < config.num_signals; ++sig_idx) {
                const auto& signal = random_signals[sig_idx];
                std::vector<double> real_in = signal;  // Copy input signal
                std::vector<double> imag_in(window_size, 0.0);  // Zero imaginary part
                std::vector<double> real_out(window_size);
                std::vector<double> imag_out(window_size);

                // Time the inference (FRFT forward transform)
                auto start = std::chrono::high_resolution_clock::now();

                bool success = engine.compute(real_in.data(), imag_in.data(),
                                             real_out.data(), imag_out.data(),
                                             window_size, alpha);

                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double time_ms = duration.count() / 1000.0;

                if (!success) {
                    std::cerr << "Warning: FRFT computation failed for window_size="
                              << window_size << ", alpha=" << alpha << "\n";
                    continue;
                }

                // Calculate real-time factor
                // RTF = processing_time / audio_duration
                // audio_duration = window_size / sample_rate (in seconds)
                // processing_time = time_ms / 1000.0 (in seconds)
                double audio_duration_sec = static_cast<double>(window_size) / config.sample_rate;
                double processing_time_sec = time_ms / 1000.0;
                double rt_factor = processing_time_sec / audio_duration_sec;

                RTTimingResult result;
                result.window_size = window_size;
                result.alpha = alpha;
                result.signal_index = sig_idx;
                result.inference_time_ms = time_ms;
                result.rt_factor = rt_factor;

                current_results.push_back(result);
                all_results.push_back(result);
                total_tests++;
            }

            // Calculate statistics for this window/alpha combination
            RTTimingStats stats = calculate_statistics(current_results, window_size, alpha, config.sample_rate);
            all_stats.push_back(stats);

            completed_combinations++;

            // Print progress
            std::cout << "  α=" << std::fixed << std::setprecision(1) << alpha
                      << " → Mean: " << std::setprecision(4) << stats.mean_time_ms << " ms, "
                      << "RTF: " << std::setprecision(4) << stats.mean_rt_factor
                      << " [" << completed_combinations << "/" << total_combinations << "]\n";
        }
        std::cout << "\n";
    }

    auto overall_end_time = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::seconds>(overall_end_time - overall_start_time);

    // Write results to files
    std::cout << "Writing results to disk...\n";

    // Detailed results
    std::string detailed_filename = "test_results/rt_timing_performance_detailed.txt";
    write_detailed_results(all_results, detailed_filename);

    // Summary statistics (main output file)
    write_summary_statistics(all_stats, config.output_filename);

    // Print summary
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Test Suite Complete                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nSummary:\n";
    std::cout << "  Total tests: " << total_tests << "\n";
    std::cout << "  Window sizes tested: " << config.window_sizes.size() << "\n";
    std::cout << "  Alpha values tested: " << config.alpha_values.size() << "\n";
    std::cout << "  Signals per combination: " << config.num_signals << "\n";
    std::cout << "  Duration: " << overall_duration.count() << " seconds\n";
    std::cout << "  Summary results saved to: " << config.output_filename << "\n";
    std::cout << "  Detailed results saved to: " << detailed_filename << "\n";

    // Find and display best/worst performance
    if (!all_stats.empty()) {
        auto best_rt = std::min_element(all_stats.begin(), all_stats.end(),
            [](const RTTimingStats& a, const RTTimingStats& b) {
                return a.mean_rt_factor < b.mean_rt_factor;
            });
        auto worst_rt = std::max_element(all_stats.begin(), all_stats.end(),
            [](const RTTimingStats& a, const RTTimingStats& b) {
                return a.mean_rt_factor < b.mean_rt_factor;
            });

        std::cout << "\nPerformance Highlights:\n";
        std::cout << "  Best RTF: " << std::fixed << std::setprecision(4) << best_rt->mean_rt_factor
                  << " (window=" << best_rt->window_size << ", α=" << std::setprecision(1) << best_rt->alpha << ")\n";
        std::cout << "  Worst RTF: " << std::setprecision(4) << worst_rt->mean_rt_factor
                  << " (window=" << worst_rt->window_size << ", α=" << std::setprecision(1) << worst_rt->alpha << ")\n";

        // Display real-time capability summary
        int rt_capable = 0;
        for (const auto& stat : all_stats) {
            if (stat.mean_rt_factor < 1.0) {
                rt_capable++;
            }
        }
        std::cout << "  Real-time capable (RTF < 1.0): " << rt_capable
                  << " out of " << all_stats.size() << " combinations ("
                  << std::setprecision(1) << (100.0 * rt_capable / all_stats.size()) << "%)\n";
    }

    std::cout << "\n";
}

int main(int argc, char* argv[]) {

#ifdef _WIN32
    // Set console to UTF-8 mode on Windows
    SetConsoleOutputCP(CP_UTF8);
#endif

    RTTimingConfig config;

    // Parse optional command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            std::cout << "FRFT Real-Time Timing Performance Test\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --num-signals N    Number of random signals per test (default: 100)\n";
            std::cout << "  --sample-rate SR   Sample rate in Hz (default: 44100)\n";
            std::cout << "  --output FILE      Output filename (default: test_results/rt_timing_performance.txt)\n";
            std::cout << "  --help             Show this help message\n\n";
            std::cout << "Test Configuration:\n";
            std::cout << "  Window sizes: Powers of 2 from 16 to 131072\n";
            std::cout << "  Alpha values: 0.0 to 1.0 in steps of 0.1\n";
            return 0;
        }
        else if (arg == "--num-signals" && i + 1 < argc) {
            config.num_signals = std::stoi(argv[++i]);
        }
        else if (arg == "--sample-rate" && i + 1 < argc) {
            config.sample_rate = std::stod(argv[++i]);
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.output_filename = argv[++i];
        }
    }
    
    try {
        run_rt_timing_test(config);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}