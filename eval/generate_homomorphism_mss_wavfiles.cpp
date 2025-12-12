#include "frft_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

// Default values
const int DEFAULT_SAMPLE_RATE = 44100;
const int DEFAULT_OVERLAPS_PER_FRAME = 4;

// Configurable parameters (will be set from command line)
double DURATION_SECONDS = 1.0;
int DURATION_SAMPLES = DEFAULT_SAMPLE_RATE;
int SAMPLE_RATE = DEFAULT_SAMPLE_RATE;
int OVERLAPS_PER_FRAME = DEFAULT_OVERLAPS_PER_FRAME;
std::vector<int> WINDOW_SIZES = {512, 1024, 2048, 4096};
std::vector<double> frequencies = {100, 200, 300, 440, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000};
double ALPHA_MIN = -2.0;
double ALPHA_MAX = 2.0;
double ALPHA_STEP = 0.1;

// Global flag for windowing mode
bool USE_WINDOWING = false;

// Simple WAV file writer
struct WavHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t file_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t sample_rate = 44100;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

bool write_wav_file(const std::string& filename, const std::vector<double>& samples, uint32_t sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << filename << "\n";
        return false;
    }

    WavHeader header;
    header.sample_rate = sample_rate;
    header.byte_rate = sample_rate * header.num_channels * header.bits_per_sample / 8;
    header.block_align = header.num_channels * header.bits_per_sample / 8;
    header.data_size = samples.size() * sizeof(int16_t);
    header.file_size = 36 + header.data_size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(WavHeader));

    // Convert double samples to int16
    for (double sample : samples) {
        int16_t sample_int = static_cast<int16_t>(std::max(-32768.0, std::min(32767.0, sample * 32767.0)));
        file.write(reinterpret_cast<const char*>(&sample_int), sizeof(int16_t));
    }

    file.close();
    return true;
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

// Generate a sinusoidal signal
void generate_sine_wave(std::vector<double>& signal, int size, double frequency, double sample_rate) {
    signal.resize(size);
    for (int i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        signal[i] = std::sin(2.0 * M_PI * frequency * t);
    }
}

// Generate Hann window
void generate_hann_window(std::vector<double>& window, int size) {
    window.resize(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (size - 1)));
    }
}

// Apply windowed FRFT with overlap-add
bool apply_windowed_frft(
    FRFTEngine& engine,
    const std::vector<double>& input_signal,
    std::vector<double>& output_signal,
    int window_size,
    double alpha,
    int sample_rate)
{
    int hop_size = window_size / OVERLAPS_PER_FRAME;
    int signal_length = input_signal.size();

    // Initialize output with zeros
    output_signal.assign(signal_length, 0.0);

    // Generate Hann window
    std::vector<double> hann_window;
    generate_hann_window(hann_window, window_size);

    // Normalization factor for overlap-add
    std::vector<double> norm_factor(signal_length, 0.0);
    for (int frame_start = 0; frame_start <= signal_length - window_size; frame_start += hop_size) {
        for (int i = 0; i < window_size; ++i) {
            norm_factor[frame_start + i] += hann_window[i] * hann_window[i];
        }
    }

    // Prepare engine for this window size
    engine.prepare(window_size);

    // Buffers for FRFT
    std::vector<double> real_in(window_size);
    std::vector<double> imag_in(window_size, 0.0);
    std::vector<double> real_out(window_size);
    std::vector<double> imag_out(window_size);

    // Process each frame
    for (int frame_start = 0; frame_start <= signal_length - window_size; frame_start += hop_size) {
        // Extract and window the frame
        for (int i = 0; i < window_size; ++i) {
            real_in[i] = input_signal[frame_start + i] * hann_window[i];
        }
        std::fill(imag_in.begin(), imag_in.end(), 0.0);

        // Apply FRFT
        bool success = engine.compute(
            real_in.data(), imag_in.data(),
            real_out.data(), imag_out.data(),
            window_size, alpha
        );

        if (!success) {
            return false;
        }

        // Overlap-add with window
        for (int i = 0; i < window_size; ++i) {
            output_signal[frame_start + i] += real_out[i] * hann_window[i];
        }
    }

    // Normalize by window overlap
    for (int i = 0; i < signal_length; ++i) {
        if (norm_factor[i] > 1e-10) {
            output_signal[i] /= norm_factor[i];
        }
    }

    return true;
}

// Apply composed windowed FRFT: alpha1 then alpha2 within each frame before overlap-add
bool apply_composed_windowed_frft(
    FRFTEngine& engine,
    const std::vector<double>& input_signal,
    std::vector<double>& output_signal,
    int window_size,
    double alpha1,
    double alpha2,
    int sample_rate)
{
    int hop_size = window_size / OVERLAPS_PER_FRAME;
    int signal_length = input_signal.size();

    // Initialize output with zeros
    output_signal.assign(signal_length, 0.0);

    // Generate Hann window
    std::vector<double> hann_window;
    generate_hann_window(hann_window, window_size);

    // Normalization factor for overlap-add
    std::vector<double> norm_factor(signal_length, 0.0);
    for (int frame_start = 0; frame_start <= signal_length - window_size; frame_start += hop_size) {
        for (int i = 0; i < window_size; ++i) {
            norm_factor[frame_start + i] += hann_window[i] * hann_window[i];
        }
    }

    // Prepare engine for this window size
    engine.prepare(window_size);

    // Buffers for FRFT
    std::vector<double> real_in(window_size);
    std::vector<double> imag_in(window_size, 0.0);
    std::vector<double> real_temp(window_size);
    std::vector<double> imag_temp(window_size);
    std::vector<double> real_out(window_size);
    std::vector<double> imag_out(window_size);

    // Process each frame
    for (int frame_start = 0; frame_start <= signal_length - window_size; frame_start += hop_size) {
        // Extract and window the frame
        for (int i = 0; i < window_size; ++i) {
            real_in[i] = input_signal[frame_start + i] * hann_window[i];
        }
        std::fill(imag_in.begin(), imag_in.end(), 0.0);

        // Apply first FRFT
        bool success = engine.compute(
            real_in.data(), imag_in.data(),
            real_temp.data(), imag_temp.data(),
            window_size, alpha1
        );

        if (!success) {
            return false;
        }

        // Apply second FRFT
        success = engine.compute(
            real_temp.data(), imag_temp.data(),
            real_out.data(), imag_out.data(),
            window_size, alpha2
        );

        if (!success) {
            return false;
        }

        // Overlap-add with window
        for (int i = 0; i < window_size; ++i) {
            output_signal[frame_start + i] += real_out[i] * hann_window[i];
        }
    }

    // Normalize by window overlap
    for (int i = 0; i < signal_length; ++i) {
        if (norm_factor[i] > 1e-10) {
            output_signal[i] /= norm_factor[i];
        }
    }

    return true;
}

// Wrap alpha to [-2, 2] range using modulo 4
double wrap_alpha(double alpha) {
    while (alpha > 2.0) alpha -= 4.0;
    while (alpha < -2.0) alpha += 4.0;
    return alpha;
}

// Parse comma-separated list of integers
std::vector<int> parse_int_list(const std::string& str) {
    std::vector<int> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stoi(item));
    }
    return result;
}

// Parse comma-separated list of doubles
std::vector<double> parse_double_list(const std::string& str) {
    std::vector<double> result;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stod(item));
    }
    return result;
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --dur <seconds>              Duration in seconds (default: 1.0)\n";
    std::cout << "  --winsizes <sizes>           Comma-separated window sizes or 'single' for direct mode\n";
    std::cout << "                               Examples: '512,1024,2048,4096' or 'single'\n";
    std::cout << "                               Default: 512,1024,2048,4096 (windowed)\n";
    std::cout << "  --freqs <frequencies>        Comma-separated frequencies in Hz\n";
    std::cout << "                               Default: 100,200,300,440,500,1000,1500,2000,3000,4000,6000,8000,10000\n";
    std::cout << "  --alpha-min <value>          Minimum alpha value for grid (default: -2.0)\n";
    std::cout << "  --alpha-max <value>          Maximum alpha value for grid (default: 2.0)\n";
    std::cout << "  --alpha-step <value>         Step size for alpha grid (default: 0.1)\n";
    std::cout << "  -h, --help                   Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  Windowed mode (0.5 seconds, custom window sizes and frequencies):\n";
    std::cout << "    " << prog_name << " --dur 0.5 --winsizes 512,1024,2048,4096 --freqs 100,440,1000,2000,4000,8000\n\n";
    std::cout << "  Direct mode (non-windowed, 0.5 seconds):\n";
    std::cout << "    " << prog_name << " --dur 0.5 --winsizes single --freqs 100,440,1000,2000,4000,8000\n\n";
    std::cout << "  Custom alpha grid (coarser grid for faster generation):\n";
    std::cout << "    " << prog_name << " --dur 0.5 --winsizes single --freqs 440,1000 --alpha-min -1 --alpha-max 1 --alpha-step 0.5\n\n";
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--dur" && i + 1 < argc) {
            DURATION_SECONDS = std::stod(argv[++i]);
            DURATION_SAMPLES = static_cast<int>(DURATION_SECONDS * SAMPLE_RATE);
        }
        else if (arg == "--winsizes" && i + 1 < argc) {
            std::string winsizes_arg = argv[++i];
            if (winsizes_arg == "single") {
                USE_WINDOWING = false;
            } else {
                USE_WINDOWING = true;
                WINDOW_SIZES = parse_int_list(winsizes_arg);
            }
        }
        else if (arg == "--freqs" && i + 1 < argc) {
            frequencies = parse_double_list(argv[++i]);
        }
        else if (arg == "--alpha-min" && i + 1 < argc) {
            ALPHA_MIN = std::stod(argv[++i]);
        }
        else if (arg == "--alpha-max" && i + 1 < argc) {
            ALPHA_MAX = std::stod(argv[++i]);
        }
        else if (arg == "--alpha-step" && i + 1 < argc) {
            ALPHA_STEP = std::stod(argv[++i]);
        }
        else if (arg == "--windowed" || arg == "-w") {
            USE_WINDOWING = true;
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Calculate total combinations
    int steps = static_cast<int>((ALPHA_MAX - ALPHA_MIN) / ALPHA_STEP + 0.5) + 1;
    int total_combinations = steps * steps;

    // Create base directory structure
    std::string test_results_dir = "./test_results";
    create_directories(test_results_dir);

    // Create subdirectory based on mode
    std::string mode_str = USE_WINDOWING ? "windowed" : "direct";
    std::string base_dir = test_results_dir + "/homomorphism_mss_grid_" + mode_str;

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       FRFT Homomorphism MSS Grid Test - WAV Generator        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Duration: " << DURATION_SECONDS << " seconds (" << DURATION_SAMPLES << " samples)\n";
    std::cout << "  Sample rate: " << SAMPLE_RATE << " Hz\n";
    std::cout << "  Frequencies: ";
    for (size_t i = 0; i < frequencies.size(); ++i) {
        std::cout << static_cast<int>(frequencies[i]);
        if (i < frequencies.size() - 1) std::cout << ", ";
    }
    std::cout << " Hz\n";
    std::cout << "  Alpha grid: [" << ALPHA_MIN << ", " << ALPHA_MAX << "] step " << ALPHA_STEP
              << " (" << total_combinations << " combinations)\n";

    if (USE_WINDOWING) {
        std::cout << "  Processing mode: WINDOWED (overlap-add)\n";
        std::cout << "  Window sizes: ";
        for (size_t i = 0; i < WINDOW_SIZES.size(); ++i) {
            std::cout << WINDOW_SIZES[i];
            if (i < WINDOW_SIZES.size() - 1) std::cout << ", ";
        }
        std::cout << "\n";
        std::cout << "  Overlaps per frame: " << OVERLAPS_PER_FRAME << "\n";
        std::cout << "  Total WAV files per freq: " << WINDOW_SIZES.size() * (1 + total_combinations * 2) << "\n\n";
    } else {
        std::cout << "  Processing mode: DIRECT (full transform)\n";
        std::cout << "  Total WAV files per freq: " << (1 + total_combinations * 2) << "\n\n";
    }

    create_directories(base_dir);

    FRFTEngine engine;
    if (!USE_WINDOWING) {
        engine.prepare(DURATION_SAMPLES);
    }

    std::ofstream metadata(base_dir + "/metadata.txt");
    metadata << "# FRFT Homomorphism MSS Grid Test Metadata\n";
    metadata << "# Duration: " << DURATION_SECONDS << " seconds (" << DURATION_SAMPLES << " samples)\n";
    metadata << "# Sample rate: " << SAMPLE_RATE << " Hz\n";
    metadata << "# Mode: " << (USE_WINDOWING ? "WINDOWED" : "DIRECT") << "\n";
    if (USE_WINDOWING) {
        metadata << "# Window sizes: ";
        for (size_t i = 0; i < WINDOW_SIZES.size(); ++i) {
            metadata << WINDOW_SIZES[i];
            if (i < WINDOW_SIZES.size() - 1) metadata << ", ";
        }
        metadata << "\n";
        metadata << "# Overlaps per frame: " << OVERLAPS_PER_FRAME << "\n";
    }
    metadata << "# Grid: α₁, α₂ ∈ [-2, 2] with step 0.1\n";
    metadata << "# α = α₁ + α₂ wrapped to [-2, 2] using modulo 4\n";
    if (USE_WINDOWING) {
        metadata << "Frequency\tWindow\tAlpha1\tAlpha2\tAlpha\tSource_File\tAlpha_File\tComposed_File\n";
    } else {
        metadata << "Frequency\tAlpha1\tAlpha2\tAlpha\tSource_File\tAlpha_File\tComposed_File\n";
    }

    int total_files = 0;
    int failed_count = 0;

    // Process each frequency
    for (double freq : frequencies) {
        std::cout << "Processing frequency: " << freq << " Hz\n";
        std::cout << "─────────────────────────────────────────────────\n";

        std::string freq_dir = base_dir + "/freq_" + std::to_string(static_cast<int>(freq));
        create_directories(freq_dir);

        // Generate source sine wave
        std::vector<double> source_signal;
        generate_sine_wave(source_signal, DURATION_SAMPLES, freq, SAMPLE_RATE);

        // Process for each window size (or once for direct mode)
        std::vector<int> windows_to_process = USE_WINDOWING ? WINDOW_SIZES : std::vector<int>{DURATION_SAMPLES};

        for (int window_size : windows_to_process) {
            std::string window_dir = freq_dir;
            if (USE_WINDOWING) {
                window_dir += "/win_" + std::to_string(window_size);
                create_directories(window_dir);
                std::cout << "  Window size: " << window_size << "\n";
            }

            // Save source file
            std::string source_filename = window_dir + "/source.wav";
            if (!write_wav_file(source_filename, source_signal, SAMPLE_RATE)) {
                std::cerr << "  ✗ Failed to write source file\n";
                failed_count++;
                continue;
            }
            total_files++;

            // Buffers for processing
            std::vector<double> alpha_result(DURATION_SAMPLES);
            std::vector<double> composed_result(DURATION_SAMPLES);
            std::vector<double> temp_result(DURATION_SAMPLES);

            int combo_count = 0;
            int combo_success = 0;

            // Iterate over alpha grid
            for (double alpha1 = ALPHA_MIN; alpha1 <= ALPHA_MAX + 1e-9; alpha1 += ALPHA_STEP) {
                for (double alpha2 = ALPHA_MIN; alpha2 <= ALPHA_MAX + 1e-9; alpha2 += ALPHA_STEP) {
                    combo_count++;

                    double alpha_sum = alpha1 + alpha2;
                    double alpha = wrap_alpha(alpha_sum);

                    bool success = false;

                    if (USE_WINDOWING) {
                        // Windowed processing
                        // Direct path: single FRFT with alpha
                        success = apply_windowed_frft(engine, source_signal, alpha_result, window_size, alpha, SAMPLE_RATE);
                        if (!success) {
                            failed_count++;
                            continue;
                        }

                        // Composed path: alpha1 then alpha2 within each frame
                        success = apply_composed_windowed_frft(engine, source_signal, composed_result, window_size, alpha1, alpha2, SAMPLE_RATE);
                        if (!success) {
                            failed_count++;
                            continue;
                        }
                    } else {
                        // Direct processing (full signal)
                        std::vector<double> real_in(DURATION_SAMPLES);
                        std::vector<double> imag_in(DURATION_SAMPLES, 0.0);
                        std::vector<double> real_direct(DURATION_SAMPLES);
                        std::vector<double> imag_direct(DURATION_SAMPLES);
                        std::vector<double> real_temp(DURATION_SAMPLES);
                        std::vector<double> imag_temp(DURATION_SAMPLES);
                        std::vector<double> real_composed(DURATION_SAMPLES);
                        std::vector<double> imag_composed(DURATION_SAMPLES);

                        std::copy(source_signal.begin(), source_signal.end(), real_in.begin());

                        success = engine.compute(real_in.data(), imag_in.data(), real_direct.data(), imag_direct.data(),
                                               DURATION_SAMPLES, alpha);
                        if (!success) {
                            failed_count++;
                            continue;
                        }

                        success = engine.compute(real_in.data(), imag_in.data(), real_temp.data(), imag_temp.data(),
                                               DURATION_SAMPLES, alpha1);
                        if (!success) {
                            failed_count++;
                            continue;
                        }

                        success = engine.compute(real_temp.data(), imag_temp.data(), real_composed.data(), imag_composed.data(),
                                               DURATION_SAMPLES, alpha2);
                        if (!success) {
                            failed_count++;
                            continue;
                        }

                        std::copy(real_direct.begin(), real_direct.end(), alpha_result.begin());
                        std::copy(real_composed.begin(), real_composed.end(), composed_result.begin());
                    }

                    // Create filenames
                    char alpha1_str[32], alpha2_str[32];
                    snprintf(alpha1_str, sizeof(alpha1_str), "%.1f", alpha1);
                    snprintf(alpha2_str, sizeof(alpha2_str), "%.1f", alpha2);

                    std::string a1_str(alpha1_str);
                    std::string a2_str(alpha2_str);
                    for (auto& c : a1_str) { if (c == '-') c = 'm'; if (c == '.') c = 'p'; }
                    for (auto& c : a2_str) { if (c == '-') c = 'm'; if (c == '.') c = 'p'; }

                    // Save files
                    std::string alpha_filename = window_dir + "/a1_" + a1_str + "_a2_" + a2_str + "_alpha.wav";
                    std::string composed_filename = window_dir + "/a1_" + a1_str + "_a2_" + a2_str + "_composed.wav";

                    if (!write_wav_file(alpha_filename, alpha_result, SAMPLE_RATE)) {
                        failed_count++;
                        continue;
                    }
                    total_files++;

                    if (!write_wav_file(composed_filename, composed_result, SAMPLE_RATE)) {
                        failed_count++;
                        continue;
                    }
                    total_files++;

                    // Write metadata
                    std::string rel_path = "freq_" + std::to_string(static_cast<int>(freq));
                    if (USE_WINDOWING) {
                        rel_path += "/win_" + std::to_string(window_size);
                        metadata << std::fixed << std::setprecision(1) << freq << "\t" << window_size << "\t";
                    } else {
                        metadata << std::fixed << std::setprecision(1) << freq << "\t";
                    }
                    metadata << alpha1 << "\t" << alpha2 << "\t" << std::setprecision(6) << alpha << "\t"
                            << rel_path << "/source.wav\t"
                            << rel_path << "/a1_" << a1_str << "_a2_" << a2_str << "_alpha.wav\t"
                            << rel_path << "/a1_" << a1_str << "_a2_" << a2_str << "_composed.wav\n";

                    combo_success++;

                    if (combo_count % 100 == 0) {
                        std::cout << "  Progress: " << combo_count << "/" << total_combinations << "\r";
                        std::cout.flush();
                    }
                }
            }

            std::cout << "  ✓ Processed " << combo_success << "/" << total_combinations << " combinations";
            if (USE_WINDOWING) {
                std::cout << " for window " << window_size;
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    
    metadata.close();
    
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Generation Complete                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    std::cout << "Summary:\n";
    std::cout << "  Total files generated: " << total_files << "\n";
    std::cout << "  Failed operations: " << failed_count << "\n";
    std::cout << "  Output directory: " << base_dir << "/\n";
    std::cout << "  Metadata file: " << base_dir << "/metadata.txt\n\n";
    
    return 0;
}