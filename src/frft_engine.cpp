#include "frft_engine.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <chrono>

// Global mutex to protect FFTW plan creation
// FFTW has global internal state that is not thread-safe, even with FFTW_ESTIMATE
std::mutex g_fftw_plan_mutex;

// ============================================================================
// FRFTEngine Implementation (per-thread instance)
// ============================================================================

FRFTEngine::FRFTEngine() {
    // Constructor
}

FRFTEngine::~FRFTEngine() {
    cleanup_plans();
}

void FRFTEngine::cleanup_plans() {
    for (auto& cache : plan_cache_) {
        if (cache.forward_plan) fftw_destroy_plan(cache.forward_plan);
        if (cache.backward_plan) fftw_destroy_plan(cache.backward_plan);
        if (cache.in_buffer) fftw_free(cache.in_buffer);
        if (cache.out_buffer) fftw_free(cache.out_buffer);
    }
    plan_cache_.clear();
}

FRFTEngine::PlanCache* FRFTEngine::get_or_create_plan(size_t size) {
    // Check if we already have a plan for this size (no lock needed for read)
    for (auto& cache : plan_cache_) {
        if (cache.size == size) {
            return &cache;
        }
    }

    // Need to create a new plan - lock the global FFTW mutex
    // CRITICAL: FFTW has global internal state that is not thread-safe
    std::lock_guard<std::mutex> fftw_lock(g_fftw_plan_mutex);

    if (debug_enabled_) {
        std::cerr << "[Thread " << std::this_thread::get_id() << "] Creating FFTW plan for size "
                  << size << " with FFTW_ESTIMATE..." << std::endl;
    }
    auto start = std::chrono::high_resolution_clock::now();

    PlanCache new_cache;
    new_cache.size = size;
    new_cache.in_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    new_cache.out_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);

    // Use FFTW_ESTIMATE - faster and safer than FFTW_MEASURE
    // But still requires global mutex due to FFTW's internal state
    new_cache.forward_plan = fftw_plan_dft_1d(size, new_cache.in_buffer, new_cache.out_buffer,
                                              FFTW_FORWARD, FFTW_ESTIMATE);
    new_cache.backward_plan = fftw_plan_dft_1d(size, new_cache.in_buffer, new_cache.out_buffer,
                                               FFTW_BACKWARD, FFTW_ESTIMATE);

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (debug_enabled_) {
        std::cerr << "[Thread " << std::this_thread::get_id() << "] Plan created in "
                  << ms << " ms" << std::endl;
    }

    plan_cache_.push_back(new_cache);
    return &plan_cache_.back();
}

void FRFTEngine::prepare(int size) {
    // Pre-create plans for common sizes used in audio
    get_or_create_plan(size);
    get_or_create_plan(size * 2);  // For upsampling
    get_or_create_plan(next_power_of_2(size * 4));  // For convolution

    current_prepared_size_ = size;

    // Pre-allocate working buffers for typical operations
    int max_size = size;  // Worst case for expanded signals
    ensure_size(work_buffer1_, max_size);
    ensure_size(work_buffer2_, max_size);
    ensure_size(work_buffer3_, max_size);
    ensure_size(work_buffer4_, max_size);
    ensure_size(work_buffer5_, max_size);
    ensure_size(work_buffer6_, max_size);
    ensure_size(work_buffer7_, max_size);
    ensure_size(work_buffer8_, max_size);
    ensure_size(real_work_, size);
    ensure_size(imag_work_, size);
    ensure_size(chirp_buffer_, max_size);
    ensure_size(multip_buffer_, max_size);
    ensure_size(hlptc_buffer_, max_size);
    ensure_size(fft_pad_buffer_, next_power_of_2(max_size));
    ensure_size(conv_buffer_, next_power_of_2(max_size));
}

void FRFTEngine::ensure_size(std::vector<Complex>& buffer, size_t size) {
    if (buffer.size() < size) {
        if (debug_enabled_) {
            std::cerr << "[Thread " << std::this_thread::get_id()
                      << "] Reallocating Complex buffer from " << buffer.size()
                      << " to " << size << std::endl;
        }
        buffer.resize(size);
    }
}

void FRFTEngine::ensure_size(std::vector<double>& buffer, size_t size) {
    if (buffer.size() < size) {
        if (debug_enabled_) {
            std::cerr << "[Thread " << std::this_thread::get_id()
                      << "] Reallocating double buffer from " << buffer.size()
                      << " to " << size << std::endl;
        }
        buffer.resize(size);
    }
}

bool FRFTEngine::compute(const double* real_in, const double* imag_in,
                         double* real_out, double* imag_out,
                         int size, double a_param) {
    // Validate inputs
    if (size % 2 != 0) {
        return false;  // Signal size must be even
    }

    // Ensure working buffers are large enough
    size_t max_size = size;
    ensure_size(work_buffer1_, max_size);
    ensure_size(work_buffer2_, max_size);
    ensure_size(work_buffer3_, max_size);

    // Construct complex signal in work_buffer1
    for (int i = 0; i < size; ++i) {
        work_buffer1_[i] = Complex(real_in[i], imag_in[i]);
    }

    // Apply fftshift to convert from FFT ordering [0, pos, neg] to centered [-N/2, ..., N/2]
    fftshift(work_buffer1_, size, work_buffer2_);  // work_buffer1 -> work_buffer2
    std::copy(work_buffer2_.begin(), work_buffer2_.begin() + size, work_buffer1_.begin());

    // 4-modulation and shifting to [-2, 2] interval
    double a = std::fmod(a_param, 4.0);
    if (a > 2.0) {
        a -= 4.0;
    } else if (a < -2.0) {
        a += 4.0;
    }

    // Special integer cases
    if (std::abs(a) < 1e-10) {
        // result = fc (already in work_buffer1)
    } else if (std::abs(a - 2.0) < 1e-10 || std::abs(a + 2.0) < 1e-10) {
        // result = dflip(fc)
        dflip(work_buffer1_, size, work_buffer2_);  // work_buffer1 -> work_buffer2
        std::copy(work_buffer2_.begin(), work_buffer2_.begin() + size, work_buffer1_.begin());
    } else {
        // General case
        // biz = bizinter(fc)
        bizinter(work_buffer1_, size, work_buffer2_);  // work_buffer1 -> work_buffer2
        size_t biz_size = size * 2;  // Fixed: bizinter returns 2*N, not 2*N-1

        // Create fc_expanded: zeros + biz + zeros
        // work_buffer3 will hold fc_expanded
        ensure_size(work_buffer3_, size + biz_size + size);

        // Fill with zeros at start
        std::fill(work_buffer3_.begin(), work_buffer3_.begin() + size, Complex(0.0, 0.0));
        // Copy biz
        std::copy(work_buffer2_.begin(), work_buffer2_.begin() + biz_size,
                  work_buffer3_.begin() + size);
        // Fill with zeros at end
        std::fill(work_buffer3_.begin() + size + biz_size,
                  work_buffer3_.begin() + size + biz_size + size, Complex(0.0, 0.0));

        size_t fc_expanded_size = size + biz_size + size;

        // Conditional transformations based on a value
        if ((0 < a && a < 0.5) || (1.5 < a && a < 2.0)) {
            corefrmod2(work_buffer3_, fc_expanded_size, 1.0, work_buffer4_);  // work_buffer3 -> work_buffer4
            std::copy(work_buffer4_.begin(), work_buffer4_.begin() + fc_expanded_size,
                      work_buffer3_.begin());
            a -= 1.0;
        }

        if ((-0.5 < a && a < 0) || (-2.0 < a && a < -1.5)) {
            corefrmod2(work_buffer3_, fc_expanded_size, -1.0, work_buffer4_);  // work_buffer3 -> work_buffer4
            std::copy(work_buffer4_.begin(), work_buffer4_.begin() + fc_expanded_size,
                      work_buffer3_.begin());
            a += 1.0;
        }

        corefrmod2(work_buffer3_, fc_expanded_size, a, work_buffer4_);  // work_buffer3 -> work_buffer4

        // Extract elements from index N to 3*N into work_buffer5
        ensure_size(work_buffer5_, size * 2);
        std::copy(work_buffer4_.begin() + size, work_buffer4_.begin() + 3 * size,
                  work_buffer5_.begin());

        // Decimate into work_buffer1
        bizdec(work_buffer5_, size * 2, work_buffer1_);  // work_buffer5 -> work_buffer1

        // CRITICAL: Double the first entry (matches Python implementation)
        // This is required for proper round-trip reconstruction
        work_buffer1_[0] *= 2.0;
    }

    // Apply ifftshift to convert back from centered to FFT ordering
    ifftshift(work_buffer1_, size, work_buffer2_);  // work_buffer1 -> work_buffer2

    // Extract real and imaginary parts
    for (int i = 0; i < size; ++i) {
        real_out[i] = work_buffer2_[i].real();
        imag_out[i] = work_buffer2_[i].imag();
    }

    return true;
}

void FRFTEngine::dflip(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    if (n == 0) {
        return;
    }

    ensure_size(output, n);

    for (size_t i = 0; i < n; ++i) {
        output[i] = input[n - i - 1];
    }
}

void FRFTEngine::bizdec(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    size_t out_size = n / 2 + 1;
    ensure_size(output, out_size);

    size_t j = 0;
    for (size_t i = 0; i < n; i += 2) {
        output[j++] = input[i];
    }
}

void FRFTEngine::bizinter(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    // Separate real and imaginary parts using pre-allocated buffers
    ensure_size(real_work_, n);
    ensure_size(imag_work_, n);

    for (size_t i = 0; i < n; ++i) {
        real_work_[i] = input[i].real();
        imag_work_[i] = input[i].imag();
    }

    // Process real and imaginary separately - outputs are 2*n elements each
    size_t result_size = 2 * n;
    ensure_size(work_buffer6_, result_size);
    ensure_size(work_buffer7_, result_size);

    bizinter_real(real_work_, n, work_buffer6_);
    bizinter_real(imag_work_, n, work_buffer7_);

    // Combine results
    ensure_size(output, result_size);

    for (size_t i = 0; i < result_size; ++i) {
        output[i] = Complex(work_buffer6_[i].real(), work_buffer7_[i].real());
    }
}

void FRFTEngine::bizinter_real(const std::vector<double>& input, size_t n, std::vector<Complex>& output) {
    size_t N1 = n / 2 + (n % 2);
    size_t N2 = 2 * n - (n / 2);

    // Convert to complex
    ensure_size(work_buffer8_, n * 2);
    for (size_t i = 0; i < n; ++i) {
        work_buffer8_[i] = Complex(input[i], 0.0);
    }

    // Upsample into fft_pad_buffer_
    upsample2(work_buffer8_, n, fft_pad_buffer_);

    // FFT into conv_buffer_
    fft(fft_pad_buffer_, n * 2, conv_buffer_);

    // Zero out middle frequencies
    for (size_t i = N1; i < N2 && i < n * 2; ++i) {
        conv_buffer_[i] = Complex(0.0, 0.0);
    }

    // IFFT back into output
    ifft(conv_buffer_, n * 2, output);

    // Scale by 2 to compensate for upsampling - CRITICAL: scale ALL 2*n elements
    for (size_t i = 0; i < 2 * n; ++i) {
        output[i] *= 2.0;
    }
}

void FRFTEngine::upsample2(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    size_t out_size = n * 2;
    ensure_size(output, out_size);

    for (size_t i = 0; i < n; ++i) {
        output[2 * i] = input[i];
        output[2 * i + 1] = Complex(0.0, 0.0);
    }
}

void FRFTEngine::corefrmod2(const std::vector<Complex>& signal, size_t N, double a, std::vector<Complex>& output) {
    int Nend = N / 2;
    int Nstart = -(static_cast<int>(N % 2) + Nend);
    double deltax = std::sqrt(static_cast<double>(N));

    double phi = a * M_PI / 2.0;
    Complex alpha = Complex(0.0, -M_PI * std::tan(phi / 2.0));
    Complex beta = Complex(0.0, M_PI / std::sin(phi));

    Complex Aphi_num = std::exp(Complex(0.0, -(M_PI * (std::sin(phi) >= 0 ? 1.0 : -1.0) / 4.0 - phi / 2.0)));
    double Aphi_denum = std::sqrt(std::abs(std::sin(phi)));
    Complex Aphi = Aphi_num / Aphi_denum;

    // Use pre-allocated buffers
    ensure_size(chirp_buffer_, N);
    ensure_size(multip_buffer_, N);

    // Compute chirp and multip
    for (int i = 0; i < static_cast<int>(N); ++i) {
        double x = static_cast<double>(Nstart + i) / deltax;
        chirp_buffer_[i] = std::exp(alpha * x * x);
        multip_buffer_[i] = signal[i] * chirp_buffer_[i];
    }

    // Compute hlptc
    size_t t_size = 2 * N - 1;
    ensure_size(hlptc_buffer_, t_size);

    for (int i = 0; i < static_cast<int>(t_size); ++i) {
        double t = static_cast<double>(-static_cast<int>(N) + 1 + i) / deltax;
        hlptc_buffer_[i] = std::exp(beta * t * t);
    }

    // Convolution via FFT
    int next_pow2 = next_power_of_2(t_size + N - 1);

    ensure_size(work_buffer5_, next_pow2);
    ensure_size(work_buffer6_, next_pow2);
    ensure_size(conv_buffer_, next_pow2);

    fft_n(multip_buffer_, N, next_pow2, work_buffer5_);
    fft_n(hlptc_buffer_, t_size, next_pow2, work_buffer6_);
    vecmul(work_buffer5_, work_buffer6_, next_pow2, conv_buffer_);

    ensure_size(work_buffer7_, next_pow2);
    ifft_n(conv_buffer_, next_pow2, next_pow2, work_buffer7_);

    // Extract relevant portion
    ensure_size(output, N);
    for (size_t i = 0; i < N; ++i) {
        output[i] = work_buffer7_[N - 1 + i] * Aphi * chirp_buffer_[i] / deltax;
    }

    // Rotate if odd N
    if (N % 2 == 1) {
        Complex temp = output[0];
        for (size_t i = 0; i < N - 1; ++i) {
            output[i] = output[i + 1];
        }
        output[N - 1] = temp;
    }
}

void FRFTEngine::vecmul(const std::vector<Complex>& tensor, const std::vector<Complex>& vector, size_t n, std::vector<Complex>& output) {
    ensure_size(output, n);

    for (size_t i = 0; i < n; ++i) {
        output[i] = tensor[i] * vector[i];
    }
}

void FRFTEngine::fft(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    PlanCache* cache = get_or_create_plan(n);

    for (size_t i = 0; i < n; ++i) {
        cache->in_buffer[i][0] = input[i].real();
        cache->in_buffer[i][1] = input[i].imag();
    }

    fftw_execute(cache->forward_plan);

    ensure_size(output, n);
    for (size_t i = 0; i < n; ++i) {
        output[i] = Complex(cache->out_buffer[i][0], cache->out_buffer[i][1]);
    }
}

void FRFTEngine::ifft(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    PlanCache* cache = get_or_create_plan(n);

    for (size_t i = 0; i < n; ++i) {
        cache->in_buffer[i][0] = input[i].real();
        cache->in_buffer[i][1] = input[i].imag();
    }

    fftw_execute(cache->backward_plan);

    ensure_size(output, n);
    for (size_t i = 0; i < n; ++i) {
        output[i] = Complex(cache->out_buffer[i][0] / n, cache->out_buffer[i][1] / n);
    }
}

void FRFTEngine::fft_n(const std::vector<Complex>& input, size_t input_size, size_t n, std::vector<Complex>& output) {
    ensure_size(fft_pad_buffer_, n);

    // Zero-pad
    for (size_t i = 0; i < input_size; ++i) {
        fft_pad_buffer_[i] = input[i];
    }
    for (size_t i = input_size; i < n; ++i) {
        fft_pad_buffer_[i] = Complex(0.0, 0.0);
    }

    fft(fft_pad_buffer_, n, output);
}

void FRFTEngine::ifft_n(const std::vector<Complex>& input, size_t input_size, size_t n, std::vector<Complex>& output) {
    ensure_size(fft_pad_buffer_, n);

    // Zero-pad
    for (size_t i = 0; i < input_size; ++i) {
        fft_pad_buffer_[i] = input[i];
    }
    for (size_t i = input_size; i < n; ++i) {
        fft_pad_buffer_[i] = Complex(0.0, 0.0);
    }

    ifft(fft_pad_buffer_, n, output);
}

int FRFTEngine::next_power_of_2(int n) {
    if (n <= 1) return 1;
    return std::pow(2, std::ceil(std::log2(n)));
}

void FRFTEngine::fftshift(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    size_t half = n / 2;
    ensure_size(output, n);

    // Move second half to first half, first half to second half
    // [0 1 2 3 4 5] -> [3 4 5 0 1 2]  (for n=6, half=3)
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[(i + half) % n];
    }
}

void FRFTEngine::ifftshift(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    size_t half = (n + 1) / 2;  // Ceiling division for odd n
    ensure_size(output, n);

    // Inverse of fftshift
    // [3 4 5 0 1 2] -> [0 1 2 3 4 5]  (for n=6)
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[(i + half) % n];
    }
}

// ============================================================================
// FRFTEngineManager Implementation (thread-safe singleton)
// ============================================================================

FRFTEngine* FRFTEngineManager::get_thread_engine() {
    // Use thread_local storage for lock-free access after first initialization
    thread_local FRFTEngine* cached_engine = nullptr;

    if (cached_engine != nullptr) {
        return cached_engine;  // Fast path: no lock needed
    }

    // Slow path: first access from this thread, need to create engine
    std::thread::id tid = std::this_thread::get_id();

    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check if engine was created by another call (rare race)
        auto it = engines_.find(tid);
        if (it != engines_.end()) {
            cached_engine = it->second.get();
            return cached_engine;
        }

        // Create new engine for this thread
        auto engine = std::make_unique<FRFTEngine>();
        bool debug_flag = debug_enabled_.load(std::memory_order_relaxed);
        engine->set_debug(debug_flag);

        FRFTEngine* engine_ptr = engine.get();
        engines_[tid] = std::move(engine);

        if (debug_flag) {
            std::cerr << "✅ Created FRFTEngine for thread " << tid << std::endl;
        }

        cached_engine = engine_ptr;
        return engine_ptr;
    }
}