// Windows-specific: Include windows.h before Max headers to avoid type conflicts
#ifdef WIN32
#include <windows.h>
#endif

#include "c74_min.h"
#include "frft_engine.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

using namespace c74::min;

// -----------------------------------------------------------------------------
// Thread-safe queue with proper shutdown support
// -----------------------------------------------------------------------------
template <typename T>
class tsqueue {
public:
    void enqueue(T v) {
        std::lock_guard<std::mutex> lock(m_);
        if (shutdown_) return; // Don't accept new items during shutdown
        q_.emplace_back(std::move(v));
        cv_.notify_one();
    }

    bool try_dequeue(T& out) {
        std::lock_guard<std::mutex> lock(m_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    bool wait_dequeue(T& out, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(m_);
        if (cv_.wait_for(lock, timeout, [this] { return !q_.empty() || shutdown_; })) {
            if (shutdown_ && q_.empty()) return false;
            out = std::move(q_.front());
            q_.pop_front();
            return true;
        }
        return false;
    }

    void wake_all() {
        std::lock_guard<std::mutex> lock(m_);
        shutdown_ = true;
        cv_.notify_all();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_);
        q_.clear();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_);
        return q_.size();
    }

private:
    mutable std::mutex m_;
    std::deque<T> q_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
};

// -----------------------------------------------------------------------------
// Frame buffer for passing audio between threads
// -----------------------------------------------------------------------------
struct AudioFrame {
    std::vector<double> real;
    std::vector<double> imag;
    int size;
    double alpha;

    AudioFrame() : size(0), alpha(0.0) {}

    AudioFrame(int s, double a) : size(s), alpha(a) {
        real.resize(s);
        imag.resize(s);
    }
};

class frft : public object<frft>, public vector_operator<> {
private:
    FRFTEngine engine;
    bool initialized = false;
    int current_buffer_size = 0;
    int last_vector_size = -1;

    // Pre-allocated buffers for real-time processing
    std::vector<double> real_buffer;
    std::vector<double> imag_buffer;

    // Threading support
    std::unique_ptr<std::thread> worker_thread_;
    tsqueue<AudioFrame> input_queue_;
    tsqueue<AudioFrame> output_queue_;
    std::atomic<bool> thread_running_{false};

    // Stats for monitoring
    std::atomic<size_t> frames_processed_{0};
    std::atomic<size_t> frames_dropped_{0};

public:
    MIN_DESCRIPTION{"Fractional Fourier Transform using native C++ implementation"};
    MIN_TAGS{"spectral, transform, frft"};
    MIN_AUTHOR{"Behzad Haki; Esteban Guitiérrez"};

    inlet<> real_in{this, "(signal) real part input", "signal"};
    inlet<> imag_in{this, "(signal) imaginary part input", "signal"};
    inlet<> alpha_msg_in{this, "(float) alpha parameter"};

    outlet<> real_out{this, "(signal) real part output", "signal"};
    outlet<> imag_out{this, "(signal) imag part output", "signal"};

    attribute<number> alpha{this, "alpha", 0.5,
                            description{"Alpha parameter for FRFT (fractional order)"},
                            range{-10.0, 10.0}
    };

    attribute<bool> threading{this, "threading", true,
                              description{"Enable background processing (adds 1 frame latency but prevents glitches)"}
    };

    attribute<symbol> csv_path{this, "csv_path", "",
                               description{"Path to save benchmark CSV results"}
    };

    attribute<bool> debug{this, "debug", false,
                          description{"Enable debug console output"}
    };

    frft() {
        initialized = true;
        auto sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
        for (int size : sizes) {
            engine.prepare(size);
        }
        engine.set_debug(false);

        // Start worker thread
        start_worker_thread();
    }

    ~frft() {
        stop_worker_thread();
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
            cout << "   Threading mode: " << (threading ? "ON (async)" : "OFF (sync)") << endl;
            cout << "   Debug mode: " << (debug ? "ON" : "OFF") << endl;
            if (threading) {
                cout << "   Frames processed: " << frames_processed_.load() << endl;
                cout << "   Frames dropped: " << frames_dropped_.load() << endl;
                cout << "   Input queue size: " << input_queue_.size() << endl;
                cout << "   Output queue size: " << output_queue_.size() << endl;
            }
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
            sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
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
        cout << "====================================================================================" << endl;
        cout << "Size  | Buffer (ms) | Avg (ms) | Min (ms) | Max (ms) | StdDev | RTF   " << endl;
        cout << "------|-------------|----------|----------|----------|--------|-------" << endl;

        // Store results for CSV export
        std::vector<std::vector<std::string>> csv_data;
        csv_data.push_back({"Size", "Buffer_ms", "Avg_ms", "Min_ms", "Max_ms", "StdDev", "RTF"});

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
            times.reserve(iterations);

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

                engine.compute(in_real.data(), in_imag.data(),
                             out_real.data(), out_imag.data(), size, alpha_val);

#ifdef WIN32
                QueryPerformanceCounter(&end);
                double elapsed = static_cast<double>(end.QuadPart - start.QuadPart) / freq.QuadPart * 1e3;
                times.push_back(elapsed);
#elif defined(__APPLE__)
                uint64_t end = mach_absolute_time();
                mach_timebase_info_data_t timebase;
                mach_timebase_info(&timebase);
                double elapsed = (end - start) * timebase.numer / timebase.denom / 1e6;
                times.push_back(elapsed);
#else
                auto end = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
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

            // Buffer duration in milliseconds (how long this audio buffer would play)
            double buffer_duration_ms = (size / sample_rate) * 1000.0;

            // RTF = processing_time / audio_duration
            // If RTF < 1.0, we can process in real-time
            double rtf = mean / buffer_duration_ms;

            cout << size << "    | "
                 << buffer_duration_ms << " | "
                 << mean << " | "
                 << min_time << " | "
                 << max_time << " | "
                 << stddev << " | "
                 << rtf << endl;
        }

        return {};
    }};

private:
    void start_worker_thread() {
        if (thread_running_) return;

        thread_running_ = true;
        worker_thread_ = std::make_unique<std::thread>([this]() {
            worker_thread_func();
        });

        if (debug) {
            cout << "🧵 Worker thread started" << endl;
        }
    }

    void stop_worker_thread() {
        if (!thread_running_) return;

        if (debug) {
            cout << "🧵 Stopping worker thread..." << endl;
        }

        thread_running_ = false;
        input_queue_.wake_all();
        output_queue_.wake_all();

        if (worker_thread_ && worker_thread_->joinable()) {
            worker_thread_->join();
        }

        input_queue_.clear();
        output_queue_.clear();

        if (debug) {
            cout << "🧵 Worker thread stopped" << endl;
        }
    }

    void worker_thread_func() {
        while (thread_running_) {
            AudioFrame input_frame;

            // Wait for input frame with timeout to check thread_running_
            if (!input_queue_.wait_dequeue(input_frame, std::chrono::milliseconds(100))) {
                continue;
            }

            // Create output frame
            AudioFrame output_frame(input_frame.size, input_frame.alpha);

            // Process
            bool success = engine.compute(
                input_frame.real.data(), input_frame.imag.data(),
                output_frame.real.data(), output_frame.imag.data(),
                input_frame.size, input_frame.alpha
            );

            if (success) {
                output_queue_.enqueue(std::move(output_frame));
                frames_processed_++;
            } else {
                frames_dropped_++;
                if (debug) {
                    cerr << "❌ Worker thread: FRFT computation failed" << endl;
                }
            }
        }
    }

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

        // Update engine debug state
        engine.set_debug(debug);

        // Track size changes per instance
        if (vs != last_vector_size) {
            if (debug) {
                cerr << "🔄 Vector size changed: " << last_vector_size << " -> " << vs << endl;
            }
            last_vector_size = vs;
        }

        // Check if buffer size is even
        if (vs % 2 != 0) {
            static bool error_printed = false;
            if (!error_printed && debug) {
                cerr << "❌ Vector size must be even, got: " << vs << endl;
                cerr << "   Passing signal through unchanged." << endl;
                error_printed = true;
            }
            std::copy(in_real, in_real + vs, out_real);
            std::copy(in_imag, in_imag + vs, out_imag);
            return;
        }

        // THREADED MODE: Async processing with 1-frame latency
        if (threading) {
            // Create input frame and enqueue it
            AudioFrame input_frame(vs, static_cast<double>(alpha));
            std::copy(in_real, in_real + vs, input_frame.real.begin());
            std::copy(in_imag, in_imag + vs, input_frame.imag.begin());
            input_queue_.enqueue(std::move(input_frame));

            // Try to get processed frame from output queue
            AudioFrame output_frame;
            if (output_queue_.try_dequeue(output_frame)) {
                // We have a processed frame - output it
                if (output_frame.size == vs) {
                    std::copy(output_frame.real.begin(), output_frame.real.end(), out_real);
                    std::copy(output_frame.imag.begin(), output_frame.imag.end(), out_imag);
                } else {
                    // Size mismatch - output silence
                    std::fill(out_real, out_real + vs, 0.0);
                    std::fill(out_imag, out_imag + vs, 0.0);
                    if (debug) {
                        static bool size_mismatch_printed = false;
                        if (!size_mismatch_printed) {
                            cerr << "⚠️ Frame size mismatch in threaded mode" << endl;
                            size_mismatch_printed = true;
                        }
                    }
                }
            } else {
                // No processed frame available yet (startup or queue underrun)
                // Output silence and increment dropped counter
                std::fill(out_real, out_real + vs, 0.0);
                std::fill(out_imag, out_imag + vs, 0.0);
                frames_dropped_++;
            }
            return;
        }

        // SYNCHRONOUS MODE: Original behavior
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
                if (!compute_error_printed && debug) {
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
            if (!exception_printed && debug) {
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