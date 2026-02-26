#include "frft_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>
#include <complex>
#include <chrono>
#include <fftw3.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

// ============================================================================
// Signal generation
// ============================================================================

void generate_sine_wave(std::vector<double>& signal, int size,
                        double frequency, double sample_rate, double phase = 0.0) {
    signal.resize(size);
    for (int i = 0; i < size; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        signal[i] = std::sin(2.0 * M_PI * frequency * t + phase);
    }
}

// Generate a unit impulse (delta) at sample 0
void generate_impulse(std::vector<double>& signal, int size) {
    signal.assign(size, 0.0);
    signal[0] = 1.0;
}

// ============================================================================
// Exact-bin frequency selection
//
// For a window of size N at sample rate fs, bin k corresponds to frequency
//   f_k = k * fs / N,  k = 0, 1, ..., N/2
//
// Strategy:
//   - Collect all bin frequencies in [100, 10000] Hz.
//   - If there are <= 1000, use all of them.
//   - Otherwise sample 1000 of them log-uniformly (snap each target to the
//     nearest bin so the result is still exactly on-bin).
// ============================================================================

static std::vector<double> make_bin_frequencies(int window_size,
                                                 double sample_rate,
                                                 int max_count = 1000) {
    // Build list of all bins in [100, 10000] Hz
    std::vector<double> all_bins;
    for (int k = 0; k <= window_size / 2; ++k) {
        double f = static_cast<double>(k) * sample_rate / window_size;
        if (f >= 100.0 - 1e-9 && f <= 10000.0 + 1e-9) {
            all_bins.push_back(f);
        }
    }

    if (all_bins.empty()) return all_bins;

    // If we have fewer than max_count, use them all
    if (static_cast<int>(all_bins.size()) <= max_count) return all_bins;

    // Sample max_count of them log-uniformly, snapping to nearest bin
    std::vector<double> selected;
    selected.reserve(max_count);

    double log_min = std::log(all_bins.front());
    double log_max = std::log(all_bins.back());

    for (int i = 0; i < max_count; ++i) {
        double target = std::exp(log_min + i * (log_max - log_min) / (max_count - 1));
        // Snap to nearest bin in all_bins
        auto it = std::lower_bound(all_bins.begin(), all_bins.end(), target);
        double best = all_bins.back();
        if (it != all_bins.end()) {
            best = *it;
            if (it != all_bins.begin()) {
                double prev = *std::prev(it);
                if (std::abs(prev - target) < std::abs(best - target))
                    best = prev;
            }
        }
        // Avoid duplicates
        if (selected.empty() || std::abs(selected.back() - best) > 1e-9)
            selected.push_back(best);
    }

    return selected;
}

// ============================================================================
// Directory creation
// ============================================================================

bool create_directories(const std::string& path) {
    std::string current_path;
    for (size_t i = 0; i < path.length(); ++i) {
        if (path[i] == '/' || path[i] == '\\' || i == path.length() - 1) {
            if (i == path.length() - 1 && path[i] != '/' && path[i] != '\\')
                current_path += path[i];
            else
                current_path += path[i];
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

// ============================================================================
// FFT (FFTW, sqrt(N) normalised)
// ============================================================================

void compute_fft(const double* real_in, const double* imag_in, int size,
                 double* real_out, double* imag_out) {
    fftw_complex* in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);
    for (int i = 0; i < size; ++i) { in[i][0] = real_in[i]; in[i][1] = imag_in[i]; }
    fftw_plan plan = fftw_plan_dft_1d(size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    double norm = 1.0 / std::sqrt(static_cast<double>(size));
    for (int i = 0; i < size; ++i) { real_out[i] = out[i][0] * norm; imag_out[i] = out[i][1] * norm; }
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

// ============================================================================
// Error metrics
// ============================================================================

double calculate_mse(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { double e = a[i]-b[i]; s += e*e; }
    return s / a.size();
}

double calculate_max_error(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i]-b[i]));
    return m;
}

double calculate_mean_error(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += std::abs(a[i]-b[i]);
    return s / a.size();
}

double calculate_correlation(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.empty()) return 0.0;
    double ma = 0.0, mb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { ma += a[i]; mb += b[i]; }
    ma /= a.size(); mb /= b.size();
    double num = 0.0, d1 = 0.0, d2 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double da = a[i]-ma, db = b[i]-mb;
        num += da*db; d1 += da*da; d2 += db*db;
    }
    if (d1 == 0.0 || d2 == 0.0) return 0.0;
    return num / std::sqrt(d1 * d2);
}

// ============================================================================
// Result struct (shared by both test types)
// ============================================================================

struct FFTComparisonResult {
    int    window_size;
    int    overlap_factor;
    double frequency;      // -1.0 for impulse tests
    double mse_magnitude;
    double mse_phase;
    double mse_complex;
    double correlation_mag;
    double max_error_mag;
    double mean_error_mag;
    int    num_frames;
    bool   success;
};

// ============================================================================
// Core comparison kernel (shared by sine and impulse paths)
// ============================================================================

FFTComparisonResult run_comparison(FRFTEngine& engine,
                                   int window_size,
                                   int overlap_factor,
                                   const std::vector<double>& full_signal,
                                   double frequency_label,
                                   int n_analysis) {
    FFTComparisonResult result;
    result.window_size    = window_size;
    result.overlap_factor = overlap_factor;
    result.frequency      = frequency_label;
    result.success        = false;
    result.num_frames     = 0;

    int hop_size    = window_size / overlap_factor;
    if (hop_size < 1) hop_size = 1;
    int signal_length = static_cast<int>(full_signal.size());

    std::vector<double> real_in(window_size), imag_in(window_size, 0.0);
    std::vector<double> real_frft(window_size), imag_frft(window_size);
    std::vector<double> real_fft(window_size),  imag_fft(window_size);

    std::vector<double> frft_mag_accum(window_size, 0.0);
    std::vector<double> fft_mag_accum(window_size,  0.0);

    double complex_error_accum = 0.0;
    double mag_error_accum     = 0.0;
    double phase_error_accum   = 0.0;
    double max_mag_error       = 0.0;
    int    num_frames          = 0;

    for (int pos = 0; pos + window_size <= signal_length && num_frames < n_analysis;
         pos += hop_size) {

        std::copy(full_signal.begin() + pos,
                  full_signal.begin() + pos + window_size,
                  real_in.begin());
        std::fill(imag_in.begin(), imag_in.end(), 0.0);

        bool ok = engine.compute(real_in.data(), imag_in.data(),
                                 real_frft.data(), imag_frft.data(),
                                 window_size, 1.0);
        if (!ok) {
            result.mse_magnitude = result.mse_phase = result.mse_complex =
            result.correlation_mag = result.max_error_mag = result.mean_error_mag = -1.0;
            return result;
        }

        compute_fft(real_in.data(), imag_in.data(), window_size,
                    real_fft.data(), imag_fft.data());

        for (int i = 0; i < window_size; ++i) {
            double fm = std::sqrt(real_frft[i]*real_frft[i] + imag_frft[i]*imag_frft[i]);
            double fp = std::atan2(imag_frft[i], real_frft[i]);
            double dm = std::sqrt(real_fft[i]*real_fft[i]  + imag_fft[i]*imag_fft[i]);
            double dp = std::atan2(imag_fft[i],  real_fft[i]);

            frft_mag_accum[i] += fm;
            fft_mag_accum[i]  += dm;

            double me = fm - dm;
            mag_error_accum += me * me;

            double pd = fp - dp;
            while (pd >  M_PI) pd -= 2.0 * M_PI;
            while (pd < -M_PI) pd += 2.0 * M_PI;
            phase_error_accum += pd * pd;

            double re = real_frft[i] - real_fft[i];
            double ie = imag_frft[i] - imag_fft[i];
            complex_error_accum += re*re + ie*ie;

            max_mag_error = std::max(max_mag_error, std::abs(me));
        }
        ++num_frames;
    }

    if (num_frames == 0) {
        result.mse_magnitude = result.mse_phase = result.mse_complex =
        result.correlation_mag = result.max_error_mag = result.mean_error_mag = -1.0;
        return result;
    }

    // Average magnitudes and peak-normalise for correlation / magnitude metrics
    std::vector<double> avg_frft(window_size), avg_fft(window_size);
    for (int i = 0; i < window_size; ++i) {
        avg_frft[i] = frft_mag_accum[i] / num_frames;
        avg_fft[i]  = fft_mag_accum[i]  / num_frames;
    }

    double frft_max = *std::max_element(avg_frft.begin(), avg_frft.end());
    double fft_max  = *std::max_element(avg_fft.begin(),  avg_fft.end());
    if (frft_max < 1e-10) frft_max = 1.0;
    if (fft_max  < 1e-10) fft_max  = 1.0;

    std::vector<double> nf(window_size), nd(window_size);
    for (int i = 0; i < window_size; ++i) {
        nf[i] = avg_frft[i] / frft_max;
        nd[i] = avg_fft[i]  / fft_max;
    }

    result.mse_magnitude   = calculate_mse(nf, nd);
    result.mse_phase       = phase_error_accum   / (window_size * num_frames);
    result.mse_complex     = complex_error_accum / (window_size * num_frames);
    result.correlation_mag = calculate_correlation(nf, nd);
    result.max_error_mag   = calculate_max_error(nf, nd);
    result.mean_error_mag  = calculate_mean_error(nf, nd);
    result.num_frames      = num_frames;
    result.success         = true;
    return result;
}

// ============================================================================
// Config
// ============================================================================

struct TestConfig {
    std::vector<int> window_sizes    = {64, 128, 256, 512, 1024, 2048, 4096,
                                        8192, 16384, 32768, 65536, 131072};
    std::vector<int> overlap_factors = {1};
    double sample_rate               = 44100.0;
    int    n_analysis                = 20;
    bool   ignore_plots              = true;
    // Output dirs — overridden per test type in main
    std::string output_dir           = "test_results/fft_comparison_extra_analysis";
};

// ============================================================================
// Write TSV header
// ============================================================================

void write_header(std::ofstream& f, bool include_frequency_col) {
    if (include_frequency_col)
        f << "WindowSize\tOverlapFactor\tHopSize\tNumFrames\tFrequency\t";
    else
        f << "WindowSize\tOverlapFactor\tHopSize\tNumFrames\t";
    f << "MSE_Magnitude\tMSE_Phase\tMSE_Complex\t"
      << "Correlation_Mag\tMaxError_Mag\tMeanError_Mag\tSuccess\n";
}

void write_result(std::ofstream& f, const FFTComparisonResult& r,
                  bool include_frequency_col) {
    int hop = r.window_size / r.overlap_factor;
    if (include_frequency_col)
        f << r.window_size << "\t" << r.overlap_factor << "\t" << hop << "\t"
          << r.num_frames  << "\t" << r.frequency      << "\t";
    else
        f << r.window_size << "\t" << r.overlap_factor << "\t" << hop << "\t"
          << r.num_frames  << "\t";
    f << std::scientific << std::setprecision(10)
      << r.mse_magnitude << "\t" << r.mse_phase << "\t" << r.mse_complex << "\t"
      << std::fixed      << std::setprecision(6) << r.correlation_mag    << "\t"
      << std::scientific << std::setprecision(10)
      << r.max_error_mag << "\t" << r.mean_error_mag << "\t"
      << (r.success ? 1 : 0) << "\n";
}

// ============================================================================
// TEST 1: Exact-bin sine frequencies
// ============================================================================

void run_exact_bin_tests(const TestConfig& config) {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Extra Analysis: Exact-Bin Sine Frequencies            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    std::cout << "For each window size, test frequencies that fall exactly on FFT bins\n";
    std::cout << "(100–10000 Hz range, up to 1000 frequencies per window size).\n";
    std::cout << "This eliminates spectral leakage asymmetry between FRFT and FFT.\n\n";

    std::string out_dir = config.output_dir + "/exact_bin";
    create_directories(out_dir);
    std::string results_file = out_dir + "/results.txt";

    std::ofstream out(results_file);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open " << results_file << "\n";
        return;
    }
    write_header(out, /*include_frequency_col=*/true);

    FRFTEngine engine;
    auto t0 = std::chrono::high_resolution_clock::now();
    int test_count = 0, failed_count = 0;

    for (int window_size : config.window_sizes) {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  Window Size: " << std::setw(6) << window_size
                  << " samples                                ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

        std::vector<double> bin_freqs =
            make_bin_frequencies(window_size, config.sample_rate, /*max_count=*/1000);

        if (bin_freqs.empty()) {
            std::cout << "  No bin frequencies in 100–10000 Hz range, skipping.\n";
            continue;
        }

        std::cout << "  Using " << bin_freqs.size() << " exact-bin frequencies "
                  << "(bin resolution: "
                  << std::fixed << std::setprecision(3)
                  << config.sample_rate / window_size << " Hz)\n\n";

        for (int overlap_factor : config.overlap_factors) {
            int hop_size = window_size / overlap_factor;
            if (hop_size < 1) hop_size = 1;

            for (double frequency : bin_freqs) {
                std::cout << "  Freq: " << std::setw(10) << std::fixed
                          << std::setprecision(3) << frequency << " Hz  ";
                std::cout.flush();

                int signal_length = window_size + (config.n_analysis - 1) * hop_size;
                std::vector<double> signal;
                generate_sine_wave(signal, signal_length, frequency, config.sample_rate);

                FFTComparisonResult result = run_comparison(
                    engine, window_size, overlap_factor, signal,
                    frequency, config.n_analysis);

                write_result(out, result, /*include_frequency_col=*/true);

                ++test_count;
                if (!result.success) {
                    ++failed_count;
                    std::cout << "→ FAILED\n";
                } else {
                    std::cout << "→ Complex MSE: " << std::scientific
                              << std::setprecision(3) << result.mse_complex
                              << "  Corr: " << std::fixed << std::setprecision(6)
                              << result.correlation_mag << "\n";
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() / 1000.0;

    out.close();
    std::cout << "\n  Exact-bin tests complete.\n";
    std::cout << "  Total: " << test_count << "  Failed: " << failed_count
              << "  Duration: " << std::fixed << std::setprecision(1) << secs << "s\n";
    std::cout << "  Results → " << results_file << "\n";
}

// ============================================================================
// TEST 2: Impulse response
// ============================================================================

void run_impulse_tests(const TestConfig& config) {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Extra Analysis: Impulse Response (delta signal)       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    std::cout << "Input: unit impulse at sample 0.  Analytical FFT = 1/sqrt(N) everywhere.\n";
    std::cout << "This measures absolute FRFT numerical error with no leakage at all.\n\n";

    std::string out_dir = config.output_dir + "/impulse";
    create_directories(out_dir);
    std::string results_file = out_dir + "/results.txt";

    std::ofstream out(results_file);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open " << results_file << "\n";
        return;
    }
    // No frequency column — impulse has no single frequency
    write_header(out, /*include_frequency_col=*/false);

    // Also write analytical error (FRFT vs known ground truth 1/sqrt(N))
    std::string analytical_file = out_dir + "/analytical_error.txt";
    std::ofstream aout(analytical_file);
    if (aout.is_open()) {
        aout << "WindowSize\tMSE_Complex_vs_Analytical\tMaxError_vs_Analytical\tMeanError_vs_Analytical\n";
    }

    FRFTEngine engine;
    auto t0 = std::chrono::high_resolution_clock::now();
    int test_count = 0, failed_count = 0;

    for (int window_size : config.window_sizes) {
        std::cout << "\n  Window Size: " << window_size << "\n";

        // For the impulse we only need a single frame (the impulse itself).
        // n_analysis > 1 would repeat the same frame — keep it meaningful
        // by using 1 frame (the impulse is deterministic, no averaging needed).
        std::vector<double> signal;
        generate_impulse(signal, window_size);

        for (int overlap_factor : config.overlap_factors) {
            std::cout << "    Overlap: " << overlap_factor << "x  ";
            std::cout.flush();

            // run_comparison will process exactly 1 frame (signal_length == window_size)
            FFTComparisonResult result = run_comparison(
                engine, window_size, overlap_factor, signal,
                /*frequency_label=*/-1.0, /*n_analysis=*/1);

            write_result(out, result, /*include_frequency_col=*/false);

            ++test_count;
            if (!result.success) {
                ++failed_count;
                std::cout << "→ FAILED\n";
            } else {
                std::cout << "→ Complex MSE: " << std::scientific
                          << std::setprecision(3) << result.mse_complex
                          << "  Corr: " << std::fixed << std::setprecision(6)
                          << result.correlation_mag << "\n";
            }

            // Compute analytical error: FRFT output vs exact ground truth (1/sqrt(N) everywhere)
            if (result.success && aout.is_open() && overlap_factor == 1) {
                std::vector<double> real_in(window_size, 0.0);
                std::vector<double> imag_in(window_size, 0.0);
                real_in[0] = 1.0;
                std::vector<double> real_frft(window_size), imag_frft(window_size);
                engine.compute(real_in.data(), imag_in.data(),
                               real_frft.data(), imag_frft.data(),
                               window_size, 1.0);

                double expected = 1.0 / std::sqrt(static_cast<double>(window_size));
                double mse_analytical = 0.0, max_err = 0.0, mean_err = 0.0;
                for (int i = 0; i < window_size; ++i) {
                    // FFT of impulse at 0: X[k] = 1 for all k (before normalisation)
                    // After sqrt(N) normalisation: X[k] = 1/sqrt(N), phase = 0
                    double re_err = real_frft[i] - expected;
                    double im_err = imag_frft[i] - 0.0;
                    double bin_err2 = re_err*re_err + im_err*im_err;
                    mse_analytical += bin_err2;
                    max_err = std::max(max_err, std::sqrt(bin_err2));
                    mean_err += std::sqrt(bin_err2);
                }
                mse_analytical /= window_size;
                mean_err       /= window_size;

                aout << window_size << "\t"
                     << std::scientific << std::setprecision(10)
                     << mse_analytical << "\t" << max_err << "\t" << mean_err << "\n";

                std::cout << "    Analytical error (vs 1/sqrt(N)): MSE="
                          << std::scientific << std::setprecision(3)
                          << mse_analytical << "  MaxErr=" << max_err << "\n";
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() / 1000.0;

    out.close();
    if (aout.is_open()) aout.close();

    std::cout << "\n  Impulse tests complete.\n";
    std::cout << "  Total: " << test_count << "  Failed: " << failed_count
              << "  Duration: " << std::fixed << std::setprecision(1) << secs << "s\n";
    std::cout << "  Results         → " << results_file << "\n";
    std::cout << "  Analytical error→ " << analytical_file << "\n";
}

// ============================================================================
// Argument parsing
// ============================================================================

bool parse_arguments(int argc, char* argv[], TestConfig& config,
                     bool& run_exact_bin, bool& run_impulse) {
    run_exact_bin = true;
    run_impulse   = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            std::cout << "FRFT vs FFT Extra Analysis Test Suite\n\n";
            std::cout << "Two targeted tests to isolate sources of numerical error:\n";
            std::cout << "  1. Exact-bin sine waves  (no spectral leakage)\n";
            std::cout << "  2. Impulse response      (absolute analytical ground truth)\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --output-dir DIR    Base output directory\n";
            std::cout << "                      (default: test_results/fft_comparison_extra_analysis)\n";
            std::cout << "  --sample-rate SR    Sample rate in Hz (default: 44100)\n";
            std::cout << "  --n-analysis N      Max frames per sine test (default: 20)\n";
            std::cout << "  --exact-bin-only    Run only the exact-bin sine test\n";
            std::cout << "  --impulse-only      Run only the impulse test\n";
            std::cout << "  --quick             Small window sizes for a fast smoke test\n";
            std::cout << "  --help              Show this help\n\n";
            std::cout << "Note: FFT is normalised by sqrt(N) before comparison with FRFT.\n";
            return false;
        }
        else if (arg == "--output-dir" && i + 1 < argc) {
            config.output_dir = argv[++i];
        }
        else if (arg == "--sample-rate" && i + 1 < argc) {
            config.sample_rate = std::stod(argv[++i]);
        }
        else if (arg == "--n-analysis" && i + 1 < argc) {
            config.n_analysis = std::stoi(argv[++i]);
        }
        else if (arg == "--exact-bin-only") {
            run_exact_bin = true;
            run_impulse   = false;
        }
        else if (arg == "--impulse-only") {
            run_exact_bin = false;
            run_impulse   = true;
        }
        else if (arg == "--quick") {
            config.window_sizes = {64, 256, 1024, 4096};
        }
    }
    return true;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    TestConfig config;
    bool run_exact_bin = true, run_impulse = true;

    if (!parse_arguments(argc, argv, config, run_exact_bin, run_impulse))
        return 0;

    create_directories(config.output_dir);

    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     FRFT vs FFT — Extra Analysis (Leakage-Free + Impulse)    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nSample rate : " << config.sample_rate << " Hz\n";
    std::cout << "Output dir  : " << config.output_dir  << "\n";

    try {
        if (run_exact_bin) run_exact_bin_tests(config);
        if (run_impulse)   run_impulse_tests(config);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    All Tests Complete                         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    return 0;
}