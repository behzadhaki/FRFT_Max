#include "frft_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <random>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

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

// Generate random alpha and decompose it into beta + gamma
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

int main() {
    const int SAMPLE_RATE = 44100;
    const int DURATION_SAMPLES = 44100; // 1 second
    const int N_SAMPLES = 1000;
    
    const std::vector<double> frequencies = {
        100.0, 220.0, 440.0, 1000.0, 2000.0, 3000.0, 
        4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0
    };
    
    const std::string base_dir = "homomorphism_mss_sources";
    
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║    FRFT Homomorphism MSS WAV File Generator                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    // Create base directory
    create_directories(base_dir);
    
    // Initialize FRFT engine
    FRFTEngine engine;
    engine.prepare(DURATION_SAMPLES);
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    
    // Generate 1000 random alpha/beta/gamma triplets
    std::vector<double> alphas(N_SAMPLES);
    std::vector<double> betas(N_SAMPLES);
    std::vector<double> gammas(N_SAMPLES);
    
    std::cout << "Generating " << N_SAMPLES << " random α/β/γ triplets...\n";
    for (int i = 0; i < N_SAMPLES; ++i) {
        generate_beta_gamma_pair(betas[i], gammas[i], alphas[i], rng);
    }
    std::cout << "✓ Generated triplets\n\n";
    
    // Open metadata file
    std::ofstream metadata(base_dir + "/metadata.txt");
    metadata << "# FRFT Homomorphism MSS Test Metadata\n";
    metadata << "# Generated WAV files for Multi-Scale Spectrogram loss comparison\n";
    metadata << "# Format: Frequency Sample_ID Alpha Beta Gamma Source_File Alpha_File Composed_File\n";
    metadata << "Frequency\tSample_ID\tAlpha\tBeta\tGamma\tSource_File\tAlpha_File\tComposed_File\n";
    
    int total_files = 0;
    int failed_count = 0;
    
    // Process each frequency
    for (double freq : frequencies) {
        std::cout << "Processing frequency: " << freq << " Hz\n";
        std::cout << "─────────────────────────────────────────────────\n";
        
        // Create subdirectory for this frequency
        std::string freq_dir = base_dir + "/freq_" + std::to_string(static_cast<int>(freq));
        create_directories(freq_dir);
        
        // Generate source sine wave (1 second at 44100 Hz)
        std::vector<double> source_signal;
        generate_sine_wave(source_signal, DURATION_SAMPLES, freq, SAMPLE_RATE);
        
        // Save source file
        std::string source_filename = freq_dir + "/source.wav";
        if (!write_wav_file(source_filename, source_signal, SAMPLE_RATE)) {
            std::cerr << "  ✗ Failed to write source file\n";
            failed_count++;
            continue;
        }
        total_files++;
        
        // Prepare buffers for FRFT
        std::vector<double> real_in(DURATION_SAMPLES);
        std::vector<double> imag_in(DURATION_SAMPLES, 0.0);
        std::vector<double> real_direct(DURATION_SAMPLES);
        std::vector<double> imag_direct(DURATION_SAMPLES);
        std::vector<double> real_temp(DURATION_SAMPLES);
        std::vector<double> imag_temp(DURATION_SAMPLES);
        std::vector<double> real_composed(DURATION_SAMPLES);
        std::vector<double> imag_composed(DURATION_SAMPLES);
        
        // Process each sample
        int sample_success = 0;
        for (int sample_id = 0; sample_id < N_SAMPLES; ++sample_id) {
            double alpha = alphas[sample_id];
            double beta = betas[sample_id];
            double gamma = gammas[sample_id];
            
            // Copy source to input buffer
            std::copy(source_signal.begin(), source_signal.end(), real_in.begin());
            std::fill(imag_in.begin(), imag_in.end(), 0.0);
            
            // Path 1: Direct FRFT with alpha
            bool success_direct = engine.compute(
                real_in.data(), imag_in.data(),
                real_direct.data(), imag_direct.data(),
                DURATION_SAMPLES, alpha
            );
            
            if (!success_direct) {
                failed_count++;
                continue;
            }
            
            // Path 2: Composed FRFT (beta then gamma)
            // First: FRFT(β)
            bool success_beta = engine.compute(
                real_in.data(), imag_in.data(),
                real_temp.data(), imag_temp.data(),
                DURATION_SAMPLES, beta
            );
            
            if (!success_beta) {
                failed_count++;
                continue;
            }
            
            // Second: FRFT(γ) on result of FRFT(β)
            bool success_gamma = engine.compute(
                real_temp.data(), imag_temp.data(),
                real_composed.data(), imag_composed.data(),
                DURATION_SAMPLES, gamma
            );
            
            if (!success_gamma) {
                failed_count++;
                continue;
            }
            
            // Save alpha result
            std::string alpha_filename = freq_dir + "/sample_" + 
                                        std::to_string(sample_id + 1) + "_alpha.wav";
            std::vector<double> alpha_signal(real_direct.begin(), real_direct.end());
            if (!write_wav_file(alpha_filename, alpha_signal, SAMPLE_RATE)) {
                failed_count++;
                continue;
            }
            total_files++;
            
            // Save composed result
            std::string composed_filename = freq_dir + "/sample_" + 
                                           std::to_string(sample_id + 1) + "_composed.wav";
            std::vector<double> composed_signal(real_composed.begin(), real_composed.end());
            if (!write_wav_file(composed_filename, composed_signal, SAMPLE_RATE)) {
                failed_count++;
                continue;
            }
            total_files++;
            
            // Write metadata entry
            metadata << std::fixed << std::setprecision(6)
                    << freq << "\t"
                    << (sample_id + 1) << "\t"
                    << alpha << "\t"
                    << beta << "\t"
                    << gamma << "\t"
                    << "freq_" << static_cast<int>(freq) << "/source.wav\t"
                    << "freq_" << static_cast<int>(freq) << "/sample_" << (sample_id + 1) << "_alpha.wav\t"
                    << "freq_" << static_cast<int>(freq) << "/sample_" << (sample_id + 1) << "_composed.wav\n";
            
            sample_success++;
        }
        
        std::cout << "  ✓ Processed " << sample_success << "/" << N_SAMPLES << " samples\n\n";
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