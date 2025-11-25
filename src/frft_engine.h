#pragma once

// Define M_PI for Windows (MSVC doesn't define it by default)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <fftw3.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <thread>

// Pure C++ FRFT implementation using FFTW
// Thread-safe with per-thread engine instances

using Complex = std::complex<double>;

class FRFTEngine {
public:
    FRFTEngine();
    ~FRFTEngine();

    // Main FRFT function for audio processing
    // real_in, imag_in: input signal (must be even length)
    // real_out, imag_out: output signal (same length as input)
    // a_param: fractional order parameter
    // Returns: true on success, false on error
    bool compute(const double* real_in, const double* imag_in,
                 double* real_out, double* imag_out,
                 int size, double a_param);

    // Pre-allocate buffers for a given size (optional, for efficiency)
    void prepare(int size);

    // Enable/disable debug output
    void set_debug(bool enable) { debug_enabled_ = enable; }
    bool is_debug_enabled() const { return debug_enabled_; }

private:
    // Core algorithm functions
    void dflip(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output);
    void bizdec(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output);
    void bizinter(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output);
    void bizinter_real(const std::vector<double>& input, size_t n, std::vector<Complex>& output);
    void upsample2(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output);
    void corefrmod2(const std::vector<Complex>& signal, size_t n, double a, std::vector<Complex>& output);
    void vecmul(const std::vector<Complex>& tensor, const std::vector<Complex>& vector, size_t n, std::vector<Complex>& output);

    // FFT helpers
    void fft(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output);
    void ifft(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output);
    void fft_n(const std::vector<Complex>& input, size_t input_size, size_t n, std::vector<Complex>& output);
    void ifft_n(const std::vector<Complex>& input, size_t input_size, size_t n, std::vector<Complex>& output);

    // Utility
    int next_power_of_2(int n);
    void fftshift(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output);
    void ifftshift(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output);

    // FFTW plan cache for efficiency
    struct PlanCache {
        fftw_plan forward_plan = nullptr;
        fftw_plan backward_plan = nullptr;
        fftw_complex* in_buffer = nullptr;
        fftw_complex* out_buffer = nullptr;
        size_t size = 0;
    };

    std::vector<PlanCache> plan_cache_;
    PlanCache* get_or_create_plan(size_t size);
    void cleanup_plans();

    // Pre-allocated working buffers to avoid repeated allocations
    std::vector<Complex> work_buffer1_;
    std::vector<Complex> work_buffer2_;
    std::vector<Complex> work_buffer3_;
    std::vector<Complex> work_buffer4_;
    std::vector<Complex> work_buffer5_;
    std::vector<Complex> work_buffer6_;
    std::vector<Complex> work_buffer7_;
    std::vector<Complex> work_buffer8_;

    std::vector<double> real_work_;
    std::vector<double> imag_work_;

    // Specific operation buffers for hot paths
    std::vector<Complex> chirp_buffer_;
    std::vector<Complex> multip_buffer_;
    std::vector<Complex> hlptc_buffer_;
    std::vector<Complex> fft_pad_buffer_;
    std::vector<Complex> conv_buffer_;

    int current_prepared_size_ = 0;
    bool debug_enabled_ = false;

    // Helper to ensure buffer size without excessive reallocation
    void ensure_size(std::vector<Complex>& buffer, size_t size);
    void ensure_size(std::vector<double>& buffer, size_t size);
};

// Thread-safe manager for per-thread FRFTEngine instances
class FRFTEngineManager {
public:
    static FRFTEngineManager& instance() {
        static FRFTEngineManager inst;
        return inst;
    }

    // Get the FRFTEngine for the current thread (lock-free after first call per thread)
    FRFTEngine* get_thread_engine();

    // Get debug state (lock-free read)
    bool get_debug_enabled() const { return debug_enabled_.load(std::memory_order_relaxed); }

    // Set debug mode (affects only new engines)
    void set_debug_enabled(bool enable) { debug_enabled_.store(enable, std::memory_order_relaxed); }

private:
    FRFTEngineManager() = default;
    ~FRFTEngineManager() = default;
    FRFTEngineManager(const FRFTEngineManager&) = delete;
    FRFTEngineManager& operator=(const FRFTEngineManager&) = delete;

    std::mutex mutex_;
    std::unordered_map<std::thread::id, std::unique_ptr<FRFTEngine>> engines_;
    std::atomic<bool> debug_enabled_{false};
};