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

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

// Test configuration
struct TestConfig {
    std::vector<int> window_sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    double sample_rate = 44100.0;
    // Sawtooth frequency range for random generation
    double sawtooth_f_min = 100.0;   // Minimum frequency (Hz)
    double sawtooth_f_max = 10000.0;  // Maximum frequency (Hz)
    int num_test_frames = 1000;        // Number of frames to test
    int num_saved_pngs = 0;          // Only save 3 random PNGs
    std::string base_dir = "test_results/passthrough_reversal";
};

// Test types
enum TestType {
    PASSTHROUGH = 0,    // alpha = 0
    REVERSAL_FORWARD,   // alpha = 2
    REVERSAL_BACKWARD   // alpha = -2
};

std::string test_type_to_string(TestType type) {
    switch(type) {
        case PASSTHROUGH: return "passthrough";
        case REVERSAL_FORWARD: return "reversal_fwd";
        case REVERSAL_BACKWARD: return "reversal_bwd";
        default: return "unknown";
    }
}

double get_alpha(TestType type) {
    switch(type) {
        case PASSTHROUGH: return 0.0;
        case REVERSAL_FORWARD: return 2.0;
        case REVERSAL_BACKWARD: return -2.0;
        default: return 0.0;
    }
}

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

// Generate a sawtooth signal
void generate_sawtooth(std::vector<double>& signal, int size, double frequency,
                      double sample_rate) {
    signal.resize(size);

    for (int i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        double phase = std::fmod(frequency * t, 1.0);  // Phase from 0 to 1
        signal[i] = 2.0 * phase - 1.0;  // Sawtooth from -1 to +1
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

// Rotate signal by moving first sample to the end
void rotate_signal_first_to_end(std::vector<double>& signal) {
    if (signal.size() <= 1) return;
    double first_sample = signal[0];
    for (size_t i = 0; i < signal.size() - 1; ++i) {
        signal[i] = signal[i + 1];
    }
    signal[signal.size() - 1] = first_sample;
}

// Write WAV file (16-bit PCM)
bool write_wav_file(const std::string& filename, const std::vector<double>& signal,
                    double sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << "\n";
        return false;
    }

    int num_samples = signal.size();
    int byte_rate = sample_rate * 2;  // 16-bit mono
    int data_size = num_samples * 2;  // 16-bit samples

    // Write WAV header
    file.write("RIFF", 4);
    int chunk_size = 36 + data_size;
    file.write(reinterpret_cast<char*>(&chunk_size), 4);
    file.write("WAVE", 4);

    // Format chunk
    file.write("fmt ", 4);
    int fmt_size = 16;
    file.write(reinterpret_cast<char*>(&fmt_size), 4);
    short audio_format = 1;  // PCM
    file.write(reinterpret_cast<char*>(&audio_format), 2);
    short num_channels = 1;  // Mono
    file.write(reinterpret_cast<char*>(&num_channels), 2);
    int sr_int = static_cast<int>(sample_rate);
    file.write(reinterpret_cast<char*>(&sr_int), 4);
    file.write(reinterpret_cast<char*>(&byte_rate), 4);
    short block_align = 2;  // 16-bit mono
    file.write(reinterpret_cast<char*>(&block_align), 2);
    short bits_per_sample = 16;
    file.write(reinterpret_cast<char*>(&bits_per_sample), 2);

    // Data chunk
    file.write("data", 4);
    file.write(reinterpret_cast<char*>(&data_size), 4);

    // Write samples (convert to 16-bit PCM)
    for (double sample : signal) {
        // Clamp to [-1, 1] and convert to int16
        double clamped = std::max(-1.0, std::min(1.0, sample));
        short pcm_sample = static_cast<short>(clamped * 32767.0);
        file.write(reinterpret_cast<char*>(&pcm_sample), 2);
    }

    file.close();
    return true;
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

    double mean1 = 0.0, mean2 = 0.0;
    for (int i = 0; i < n; ++i) {
        mean1 += signal1[i];
        mean2 += signal2[i];
    }
    mean1 /= n;
    mean2 /= n;

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
                         const std::vector<double>& processed,
                         const std::vector<double>& expected) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << "\n";
        return;
    }

    file << "sample,original,processed,expected\n";
    for (size_t i = 0; i < original.size(); ++i) {
        file << i << ","
             << original[i] << ","
             << processed[i] << ","
             << expected[i] << "\n";
    }
    file.close();
}

// Generate Python plotting script
void generate_plot_script(const std::string& csv_filename,
                         const std::string& output_png,
                         int window_size,
                         int frame_num,
                         TestType test_type,
                         double sample_rate,
                         double mse,
                         double correlation) {
    std::ofstream script("plot_signals.py");
    if (!script.is_open()) {
        std::cerr << "Error: Cannot create plotting script\n";
        return;
    }

    std::string test_name = test_type_to_string(test_type);
    double alpha = get_alpha(test_type);

    script << "#!/usr/bin/env python3\n";
    script << "import pandas as pd\n";
    script << "import matplotlib.pyplot as plt\n";
    script << "import numpy as np\n\n";

    script << "# Load data\n";
    script << "data = pd.read_csv('" << csv_filename << "')\n\n";

    script << "# Create figure\n";
    script << "fig, axes = plt.subplots(3, 1, figsize=(12, 10))\n\n";

    script << "# Plot 1: Original signal\n";
    script << "axes[0].plot(data['sample'], data['original'], 'b-', linewidth=0.8)\n";
    script << "axes[0].set_title('Original Signal (Chirp)', fontsize=12, fontweight='bold')\n";
    script << "axes[0].set_ylabel('Amplitude')\n";
    script << "axes[0].grid(True, alpha=0.3)\n";
    script << "axes[0].set_xlim([0, len(data)-1])\n\n";

    script << "# Plot 2: FRFT processed signal\n";
    script << "axes[1].plot(data['sample'], data['processed'], 'r-', linewidth=0.8)\n";
    script << "axes[1].set_title('FRFT(α=" << alpha << ") Output - " << test_name
           << "', fontsize=12, fontweight='bold')\n";
    script << "axes[1].set_ylabel('Amplitude')\n";
    script << "axes[1].grid(True, alpha=0.3)\n";
    script << "axes[1].set_xlim([0, len(data)-1])\n\n";

    script << "# Plot 3: Comparison (processed vs expected)\n";
    script << "axes[2].plot(data['sample'], data['expected'], 'g-', linewidth=1.2, "
           << "label='Expected', alpha=0.7)\n";
    script << "axes[2].plot(data['sample'], data['processed'], 'r--', linewidth=0.8, "
           << "label='Processed', alpha=0.7)\n";
    script << "axes[2].set_title('Comparison: Processed vs Expected (MSE="
           << std::scientific << std::setprecision(2) << mse
           << ", Corr=" << std::fixed << std::setprecision(4) << correlation
           << ")', fontsize=12, fontweight='bold')\n";
    script << "axes[2].set_xlabel('Sample')\n";
    script << "axes[2].set_ylabel('Amplitude')\n";
    script << "axes[2].legend()\n";
    script << "axes[2].grid(True, alpha=0.3)\n";
    script << "axes[2].set_xlim([0, len(data)-1])\n\n";

    script << "# Adjust layout and save\n";
    script << "plt.tight_layout()\n";
    script << "plt.savefig('" << output_png << "', dpi=150, bbox_inches='tight')\n";
    script << "plt.close()\n";
    script << "print('Plot saved to: " << output_png << "')\n";

    script.close();
}

// Test a single frame
void test_single_frame(FRFTEngine& engine,
                      int window_size,
                      int frame_num,
                      TestType test_type,
                      const TestConfig& config,
                      std::ofstream& results_file,
                      const std::string& wavs_dir,
                      const std::string& figures_dir,
                      bool save_png = true) {

    double alpha = get_alpha(test_type);
    std::string test_name = test_type_to_string(test_type);

    // Generate random sawtooth frequency for this frame
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> freq_dist(config.sawtooth_f_min, config.sawtooth_f_max);

    double sawtooth_freq = freq_dist(gen);

    // Generate sawtooth signal with random frequency
    std::vector<double> original_frame;
    generate_sawtooth(original_frame, window_size, sawtooth_freq, config.sample_rate);

    // Apply Hamming window
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

    // Copy to FRFT input
    std::copy(original_frame.begin(), original_frame.end(), real_in.begin());

    // Apply FRFT
    bool success = engine.compute(real_in.data(), imag_in.data(),
                                 real_out.data(), imag_out.data(),
                                 window_size, alpha);

    if (!success) {
        std::cerr << "Error: FRFT computation failed for frame " << frame_num << "\n";
        return;
    }

    // Get processed output
    std::vector<double> processed_signal(window_size);
    std::copy(real_out.begin(), real_out.end(), processed_signal.begin());

    // For alpha = ±2 (reversal cases), rotate the processed signal
    // by moving the first sample to the end before comparison
    if (test_type == REVERSAL_FORWARD || test_type == REVERSAL_BACKWARD) {
        rotate_signal_first_to_end(processed_signal);
    }

    // Determine expected output based on test type
    std::vector<double> expected_signal;
    std::string comparison_desc;

    if (test_type == PASSTHROUGH) {
        // Expected: same as original (pass-through)
        expected_signal = original_frame;
        comparison_desc = "Original vs Processed";
    } else {
        // Expected: reversed signal (for both alpha=2 and alpha=-2)
        reverse_signal(original_frame, expected_signal);
        comparison_desc = "Reversed vs Processed";
    }

    // Calculate metrics
    double mse = calculate_mse(expected_signal, processed_signal);
    double correlation = calculate_correlation(expected_signal, processed_signal);

    // Print results with sawtooth frequency information
    std::cout << "  Frame " << frame_num << "/" << config.num_test_frames
              << " [" << test_name << ", α=" << alpha
              << ", saw:" << std::fixed << std::setprecision(0) << sawtooth_freq
              << "Hz]: ";
    std::cout << "MSE=" << std::scientific << std::setprecision(2) << mse
              << ", Corr=" << std::fixed << std::setprecision(6) << correlation;

    std::string status;
    if (test_type == PASSTHROUGH) {
        if (correlation > 0.99) {
            std::cout << " ✓ PASS-THROUGH OK";
            status = "PASS-THROUGH OK";
        } else {
            std::cout << " ✗ PASS-THROUGH FAILED";
            status = "PASS-THROUGH FAILED";
        }
    } else {
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
    }
    std::cout << "\n";

    // Write to results file
    if (results_file.is_open()) {
        results_file << "  Frame " << frame_num << ": "
                     << "MSE=" << std::scientific << std::setprecision(2) << mse
                     << ", Corr=" << std::fixed << std::setprecision(6) << correlation
                     << " - " << status << "\n";
    }

    // Save WAV files with descriptive names
    std::string base_filename = wavs_dir + "/" + test_name +
                               "_ws" + std::to_string(window_size) +
                               "_f" + std::to_string(frame_num);

    write_wav_file(base_filename + "_source.wav", original_frame, config.sample_rate);
    write_wav_file(base_filename + "_processed.wav", processed_signal, config.sample_rate);
    write_wav_file(base_filename + "_expected.wav", expected_signal, config.sample_rate);

    // Generate plot only if save_png is true
    if (save_png) {
        std::string csv_filename = "temp_signal_data.csv";
        std::string png_filename = figures_dir + "/" + test_name +
                                  "_ws" + std::to_string(window_size) +
                                  "_f" + std::to_string(frame_num) + ".png";

        write_signals_to_csv(csv_filename, original_frame, processed_signal, expected_signal);
        generate_plot_script(csv_filename, png_filename, window_size, frame_num,
                            test_type, config.sample_rate, mse, correlation);

        // Execute Python script to generate plot
        int ret = system("python3 plot_signals.py");
        if (ret != 0) {
            std::cerr << "    Warning: Failed to generate plot\n";
        }

        // Clean up temporary files
        remove(csv_filename.c_str());
        remove("plot_signals.py");
    }
}

// Run the complete test suite
void run_test_suite(const TestConfig& config) {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        FRFT Pass-Through and Reversal Test Suite             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Test Configuration:\n";
    std::cout << "  Sample Rate: " << config.sample_rate << " Hz\n";
    std::cout << "  Sawtooth Range: " << config.sawtooth_f_min << " Hz → " << config.sawtooth_f_max << " Hz (random per frame)\n";
    std::cout << "  Frames per window: " << config.num_test_frames << "\n";
    std::cout << "  PNGs saved per window: " << config.num_saved_pngs << " (randomly selected)\n";
    std::cout << "  Window Sizes: ";
    for (size_t i = 0; i < config.window_sizes.size(); ++i) {
        std::cout << config.window_sizes[i];
        if (i < config.window_sizes.size() - 1) std::cout << ", ";
    }
    std::cout << "\n\n";

    // Create directories
    std::string wavs_dir = config.base_dir + "/waveforms";
    std::string figures_dir = config.base_dir + "/figures";

    create_directories(wavs_dir);
    create_directories(figures_dir);

    // Test types to run
    std::vector<TestType> test_types = {PASSTHROUGH, REVERSAL_FORWARD, REVERSAL_BACKWARD};

    // Create FRFT engine
    FRFTEngine engine;

    // Process each test type
    for (TestType test_type : test_types) {
        double alpha = get_alpha(test_type);
        std::string test_name = test_type_to_string(test_type);

        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "TEST TYPE: " << test_name << " (α = " << alpha << ")\n";
        std::cout << std::string(70, '=') << "\n\n";

        // Open results file for this test type
        std::string results_filename = config.base_dir + "/" + test_name + "_results.txt";
        std::ofstream results_file(results_filename);
        if (results_file.is_open()) {
            results_file << "FRFT " << test_name << " Test Results (α = " << alpha << ")\n";
            results_file << std::string(70, '=') << "\n\n";
            results_file << "Test Configuration:\n";
            results_file << "  Sample Rate: " << config.sample_rate << " Hz\n";
            results_file << "  Sawtooth Range: " << config.sawtooth_f_min << " Hz → "
                        << config.sawtooth_f_max << " Hz (random per frame)\n";
            results_file << "  Frames per window: " << config.num_test_frames << "\n\n";
        }

        // Process each window size
        for (int window_size : config.window_sizes) {
            std::cout << std::string(70, '-') << "\n";
            std::cout << "Window size: " << window_size << "\n";
            std::cout << std::string(70, '-') << "\n";

            double frame_duration = static_cast<double>(window_size) / config.sample_rate;
            std::cout << "Frame duration: " << (frame_duration * 1000.0) << " ms\n\n";

            if (results_file.is_open()) {
                results_file << "Window Size: " << window_size << " (duration: "
                            << (frame_duration * 1000.0) << " ms)\n";
                results_file << std::string(70, '-') << "\n";
            }

            // Randomly select frames to save as PNG (3 out of 30)
            std::set<int> frames_to_save_png;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1, config.num_test_frames);

            while (frames_to_save_png.size() < config.num_saved_pngs) {
                frames_to_save_png.insert(dis(gen));
            }

            std::cout << "  Saving PNGs for frames: ";
            for (int frame : frames_to_save_png) {
                std::cout << frame << " ";
            }
            std::cout << "\n\n";

            // Test multiple frames
            for (int frame_num = 1; frame_num <= config.num_test_frames; ++frame_num) {
                bool save_png = frames_to_save_png.count(frame_num) > 0;
                test_single_frame(engine, window_size, frame_num, test_type, config,
                                results_file, wavs_dir, figures_dir, save_png);
            }

            if (results_file.is_open()) {
                results_file << "\n";
            }
            std::cout << "\n";
        }

        // Close results file
        if (results_file.is_open()) {
            results_file << std::string(70, '=') << "\n";
            results_file << "Test Complete\n";
            results_file.close();
            std::cout << "Results saved to: " << results_filename << "\n";
        }
    }

    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                All Tests Complete                             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    std::cout << "Test configuration:\n";
    std::cout << "  Total frames per window: " << config.num_test_frames << "\n";
    std::cout << "  PNGs saved per window: " << config.num_saved_pngs << " (randomly selected)\n\n";
    std::cout << "Waveforms saved to: " << wavs_dir << "/\n";
    std::cout << "  Naming: {test_type}_ws{size}_f{frame}_{source|processed|expected}.wav\n";
    std::cout << "  Note: All " << config.num_test_frames << " frames saved as WAV\n\n";
    std::cout << "Figures saved to: " << figures_dir << "/\n";
    std::cout << "  Naming: {test_type}_ws{size}_f{frame}.png\n";
    std::cout << "  Note: Only " << config.num_saved_pngs << " random frames saved as PNG per window\n\n";
    std::cout << "Results saved to: " << config.base_dir << "/\n";
    std::cout << "  Files: passthrough_results.txt, reversal_fwd_results.txt, reversal_bwd_results.txt\n\n";
}

int main() {
    TestConfig config;

    try {
        run_test_suite(config);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}