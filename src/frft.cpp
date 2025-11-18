// Windows-specific: Include windows.h before Max headers to avoid type conflicts
#ifdef WIN32
#include <windows.h>
#endif

#include "c74_min.h"
#include "frft_engine.h"
#include <chrono>
#include <cmath>
#include <cstdlib>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <x86intrin.h>
#endif

using namespace c74::min;

class frft : public object<frft>, public vector_operator<> {
private:
    FRFTEngine engine;
    bool initialized = false;
    int current_buffer_size = 0;

    // Pre-allocated buffers for real-time processing
    std::vector<double> real_buffer;
    std::vector<double> imag_buffer;

public:
    MIN_DESCRIPTION{"Fractional Fourier Transform using native C++ implementation"};
    MIN_TAGS{"spectral, transform, frft"};
    MIN_AUTHOR{"YourName"};

    inlet<> real_in{this, "(signal) real part input", "signal"};
    inlet<> imag_in{this, "(signal) imaginary part input", "signal"};
    inlet<> alpha_msg_in{this, "(float) alpha parameter"};

    outlet<> real_out{this, "(signal) real part output", "signal"};
    outlet<> imag_out{this, "(signal) imag part output", "signal"};

    attribute<number> alpha{this, "alpha", 0.5,
                            description{"Alpha parameter for FRFT (fractional order)"},
                            range{-10.0, 10.0}
    };

    frft() {
        initialized = true;
    }

    message<> float_input{this, "float", "Set alpha parameter",
                          MIN_FUNCTION {
                              if (args.size() > 0) {
                                  alpha = args[0];
                              }
                              return {};
                          }
    };

    message<> status{this, "status", "Print engine status", MIN_FUNCTION {
        if (initialized) {
            cout << "✅ FRFT engine is initialized and ready" << endl;
            cout << "   Current buffer size: " << current_buffer_size << endl;
            cout << "   Current alpha: " << double(alpha) << endl;
        } else {
            cout << "❌ Engine not initialized" << endl;
        }
        return {};
    }};

    message<> benchmark{this, "benchmark", "Benchmark FRFT performance", MIN_FUNCTION {
        if (!initialized) {
            cout << "❌ Engine not initialized" << endl;
            return {};
        }

        std::vector<int> sizes;
        int iterations = 1000;
        int warmup = 100;

        if (args.size() == 0) {
            sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 96000};
        } else {
            for (size_t i = 0; i < args.size() - 1; i++) {
                sizes.push_back(static_cast<int>(args[i]));
            }
            if (args.size() > 1) {
                iterations = static_cast<int>(args[args.size() - 1]);
            }
        }

        const double alpha_val = 0.5;
        const double sample_rate = 48000.0;

        cout << "\nFRFT Benchmark (alpha=" << alpha_val << ", " << iterations << " iterations, "
             << warmup << " warmup)" << endl;
        cout << "============================================================================" << endl;
        cout << "Size  | Avg (μs) | Min (μs) | Max (μs) | StdDev | Cycles    | RTF   | MS/s" << endl;
        cout << "------|----------|----------|----------|--------|-----------|-------|-------" << endl;

        for (int size : sizes) {
            if (size % 2 != 0) {
                cout << size << "    | SKIPPED (must be even)" << endl;
                continue;
            }

            ensure_buffer_size(size);

            std::vector<double> in_real(size);
            std::vector<double> in_imag(size);
            std::vector<double> out_real(size);
            std::vector<double> out_imag(size);

            for (int i = 0; i < size; i++) {
                in_real[i] = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
                in_imag[i] = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
            }

            for (int w = 0; w < warmup; w++) {
                engine.compute(in_real.data(), in_imag.data(),
                             out_real.data(), out_imag.data(), size, alpha_val);
            }

            std::vector<double> times;
            std::vector<uint64_t> cycles_vec;
            times.reserve(iterations);
            cycles_vec.reserve(iterations);

            for (int iter = 0; iter < iterations; iter++) {
#ifdef WIN32
                LARGE_INTEGER freq, start, end;
                QueryPerformanceFrequency(&freq);
                QueryPerformanceCounter(&start);
#elif defined(__APPLE__)
                uint64_t start = mach_absolute_time();
#else
                auto start = std::chrono::high_resolution_clock::now();
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                uint64_t cycles_start = __rdtsc();
#endif

                engine.compute(in_real.data(), in_imag.data(),
                             out_real.data(), out_imag.data(), size, alpha_val);

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
                uint64_t cycles_end = __rdtsc();
                cycles_vec.push_back(cycles_end - cycles_start);
#endif

#ifdef WIN32
                QueryPerformanceCounter(&end);
                double elapsed = static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart * 1e6;
                times.push_back(elapsed);
#elif defined(__APPLE__)
                uint64_t end = mach_absolute_time();
                mach_timebase_info_data_t timebase;
                mach_timebase_info(&timebase);
                double elapsed = (end - start) * timebase.numer / timebase.denom / 1e3;
                times.push_back(elapsed);
#else
                auto end = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double, std::micro>(end - start).count();
                times.push_back(elapsed);
#endif
            }

            double sum = 0.0, min_time = times[0], max_time = times[0];
            for (double t : times) {
                sum += t;
                if (t < min_time) min_time = t;
                if (t > max_time) max_time = t;
            }
            double mean = sum / iterations;

            double var_sum = 0.0;
            for (double t : times) {
                var_sum += (t - mean) * (t - mean);
            }
            double stddev = sqrt(var_sum / iterations);

            uint64_t avg_cycles = 0;
            if (!cycles_vec.empty()) {
                uint64_t cycle_sum = 0;
                for (uint64_t c : cycles_vec) {
                    cycle_sum += c;
                }
                avg_cycles = cycle_sum / cycles_vec.size();
            }

            double audio_duration = size / sample_rate * 1e6;
            double rtf = mean / audio_duration;
            double throughput = (size * 1e6 / mean) / 1e6;

            printf("%-5d | %8.2f | %8.2f | %8.2f | %6.2f | %9llu | %.4f | %.2f\n",
                   size, mean, min_time, max_time, stddev,
                   (unsigned long long)avg_cycles, rtf, throughput);
        }

        cout << "============================================================================" << endl;
        cout << "RTF < 1.0 = real-time capable | MS/s = Megasamples/second" << endl << endl;

        return {};
    }};

private:
    void ensure_buffer_size(int vs) {
        if (current_buffer_size != vs) {
            real_buffer.resize(vs);
            imag_buffer.resize(vs);

            // Pre-create FFTW plans for this buffer size
            engine.prepare(vs);

            current_buffer_size = vs;
        }
    }

public:
    void operator()(audio_bundle input, audio_bundle output) {
        auto in_real = input.samples(0);
        auto in_imag = input.samples(1);
        auto out_real = output.samples(0);
        auto out_imag = output.samples(1);

        int vs = input.frame_count();

        if (!initialized) {
            // Pass through if not initialized
            std::copy(in_real, in_real + vs, out_real);
            std::copy(in_imag, in_imag + vs, out_imag);
            return;
        }

        // Check if buffer size is even
        if (vs % 2 != 0) {
            static bool error_printed = false;
            if (!error_printed) {
                cerr << "❌ Vector size must be even, got: " << vs << endl;
                cerr << "   Passing signal through unchanged." << endl;
                error_printed = true;
            }
            std::copy(in_real, in_real + vs, out_real);
            std::copy(in_imag, in_imag + vs, out_imag);
            return;
        }

        try {
            // Ensure buffers are allocated
            ensure_buffer_size(vs);

            // Get alpha parameter
            double alpha_param = static_cast<double>(alpha);

            // Compute FRFT
            bool success = engine.compute(
                    in_real, in_imag,
                    out_real, out_imag,
                    vs, alpha_param
            );

            if (!success) {
                static bool compute_error_printed = false;
                if (!compute_error_printed) {
                    cerr << "❌ FRFT computation failed" << endl;
                    compute_error_printed = true;
                }
                // Pass through on error
                std::copy(in_real, in_real + vs, out_real);
                std::copy(in_imag, in_imag + vs, out_imag);
            }
        }
        catch (const std::exception& e) {
            static bool exception_printed = false;
            if (!exception_printed) {
                cerr << "❌ FRFT exception: " << e.what() << endl;
                exception_printed = true;
            }
            // Pass through on exception
            std::copy(in_real, in_real + vs, out_real);
            std::copy(in_imag, in_imag + vs, out_imag);
        }
    }
};

MIN_EXTERNAL(frft);