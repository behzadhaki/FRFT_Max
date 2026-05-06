#include "frft_engine_wasm.h"
#include <algorithm>
#include <cstring>

// ============================================================================
// FRFTEngine Implementation (KissFFT-backed, single-threaded for WASM)
// ============================================================================

FRFTEngine::FRFTEngine() {}

FRFTEngine::~FRFTEngine() {
    cleanup_plans();
}

void FRFTEngine::cleanup_plans() {
    for (auto& cache : plan_cache_) {
        if (cache.forward_cfg)  free(cache.forward_cfg);
        if (cache.backward_cfg) free(cache.backward_cfg);
    }
    plan_cache_.clear();
}

FRFTEngine::PlanCache* FRFTEngine::get_or_create_plan(size_t size) {
    for (auto& cache : plan_cache_) {
        if (cache.size == size) return &cache;
    }

    PlanCache new_cache;
    new_cache.size = size;
    new_cache.forward_cfg  = kiss_fft_alloc(size, 0, nullptr, nullptr);
    new_cache.backward_cfg = kiss_fft_alloc(size, 1, nullptr, nullptr);
    new_cache.in_buf.resize(size);
    new_cache.out_buf.resize(size);

    plan_cache_.push_back(std::move(new_cache));
    return &plan_cache_.back();
}

void FRFTEngine::prepare(int size) {
    get_or_create_plan(size);
    get_or_create_plan(size * 2);
    get_or_create_plan(next_power_of_2(size * 4));

    current_prepared_size_ = size;

    int max_size = size;
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
    if (buffer.size() < size) buffer.resize(size);
}

void FRFTEngine::ensure_size(std::vector<double>& buffer, size_t size) {
    if (buffer.size() < size) buffer.resize(size);
}

// ============================================================================
// FFT helpers — only this section differs from frft_engine.cpp
// ============================================================================

void FRFTEngine::fft(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    PlanCache* cache = get_or_create_plan(n);

    for (size_t i = 0; i < n; ++i) {
        cache->in_buf[i].r = input[i].real();
        cache->in_buf[i].i = input[i].imag();
    }

    kiss_fft(cache->forward_cfg, cache->in_buf.data(), cache->out_buf.data());

    ensure_size(output, n);
    for (size_t i = 0; i < n; ++i) {
        output[i] = Complex(cache->out_buf[i].r, cache->out_buf[i].i);
    }
}

void FRFTEngine::ifft(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    PlanCache* cache = get_or_create_plan(n);

    for (size_t i = 0; i < n; ++i) {
        cache->in_buf[i].r = input[i].real();
        cache->in_buf[i].i = input[i].imag();
    }

    kiss_fft(cache->backward_cfg, cache->in_buf.data(), cache->out_buf.data());

    // KissFFT does not normalise — divide by N to match FFTW behaviour
    ensure_size(output, n);
    double inv_n = 1.0 / static_cast<double>(n);
    for (size_t i = 0; i < n; ++i) {
        output[i] = Complex(cache->out_buf[i].r * inv_n, cache->out_buf[i].i * inv_n);
    }
}

void FRFTEngine::fft_n(const std::vector<Complex>& input, size_t input_size, size_t n, std::vector<Complex>& output) {
    ensure_size(fft_pad_buffer_, n);
    for (size_t i = 0; i < input_size; ++i) fft_pad_buffer_[i] = input[i];
    for (size_t i = input_size; i < n; ++i) fft_pad_buffer_[i] = Complex(0.0, 0.0);
    fft(fft_pad_buffer_, n, output);
}

void FRFTEngine::ifft_n(const std::vector<Complex>& input, size_t input_size, size_t n, std::vector<Complex>& output) {
    ensure_size(fft_pad_buffer_, n);
    for (size_t i = 0; i < input_size; ++i) fft_pad_buffer_[i] = input[i];
    for (size_t i = input_size; i < n; ++i) fft_pad_buffer_[i] = Complex(0.0, 0.0);
    ifft(fft_pad_buffer_, n, output);
}

// ============================================================================
// Algorithm — identical to frft_engine.cpp from here down
// ============================================================================

bool FRFTEngine::compute(const double* real_in, const double* imag_in,
                         double* real_out, double* imag_out,
                         int size, double a_param) {
    if (size % 2 != 0) return false;

    size_t max_size = size;
    ensure_size(work_buffer1_, max_size);
    ensure_size(work_buffer2_, max_size);
    ensure_size(work_buffer3_, max_size);

    for (int i = 0; i < size; ++i) {
        work_buffer1_[i] = Complex(real_in[i], imag_in[i]);
    }

    fftshift(work_buffer1_, size, work_buffer2_);
    std::copy(work_buffer2_.begin(), work_buffer2_.begin() + size, work_buffer1_.begin());

    double a = std::fmod(a_param, 4.0);
    if (a > 2.0)       a -= 4.0;
    else if (a < -2.0) a += 4.0;

    {
        bizinter(work_buffer1_, size, work_buffer2_);
        size_t biz_size = size * 2;

        ensure_size(work_buffer3_, size + biz_size + size);
        std::fill(work_buffer3_.begin(), work_buffer3_.begin() + size, Complex(0.0, 0.0));
        std::copy(work_buffer2_.begin(), work_buffer2_.begin() + biz_size,
                  work_buffer3_.begin() + size);
        std::fill(work_buffer3_.begin() + size + biz_size,
                  work_buffer3_.begin() + size + biz_size + size, Complex(0.0, 0.0));

        size_t fc_expanded_size = size + biz_size + size;

        if ((0 <= a && a < 0.5) || (1.5 < a && a <= 2.0)) {
            corefrmod2(work_buffer3_, fc_expanded_size, 1.0, work_buffer4_);
            std::copy(work_buffer4_.begin(), work_buffer4_.begin() + fc_expanded_size,
                      work_buffer3_.begin());
            a -= 1.0;
        }

        if ((-0.5 < a && a < 0) || (-2.0 <= a && a < -1.5)) {
            corefrmod2(work_buffer3_, fc_expanded_size, -1.0, work_buffer4_);
            std::copy(work_buffer4_.begin(), work_buffer4_.begin() + fc_expanded_size,
                      work_buffer3_.begin());
            a += 1.0;
        }

        corefrmod2(work_buffer3_, fc_expanded_size, a, work_buffer4_);

        ensure_size(work_buffer5_, size * 2);
        std::copy(work_buffer4_.begin() + size, work_buffer4_.begin() + 3 * size,
                  work_buffer5_.begin());

        bizdec(work_buffer5_, size * 2, work_buffer1_);
    }

    // Output in centered order (no ifftshift): DC at N/2, Nyquist artifact at 0.
    // The Hann synthesis window has win[0]=0, so the Nyquist artifact is
    // suppressed for free.  Main signal energy near N/2 gets full window weight.
    for (int i = 0; i < size; ++i) {
        real_out[i] = work_buffer1_[i].real();
        imag_out[i] = work_buffer1_[i].imag();
    }

    // Interpolate the Nyquist boundary artifact at position 0 in centered output.
    real_out[0] = 0.5 * (real_out[size - 1] + real_out[1]);
    imag_out[0] = 0.5 * (imag_out[size - 1] + imag_out[1]);

    return true;
}

void FRFTEngine::dflip(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    if (n == 0) return;
    ensure_size(output, n);
    output[0] = input[0];
    for (size_t i = 1; i < n; ++i) output[i] = input[n - i];
}

void FRFTEngine::bizdec(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    size_t out_size = n / 2 + 1;
    ensure_size(output, out_size);
    size_t j = 0;
    for (size_t i = 0; i < n; i += 2) output[j++] = input[i];
}

void FRFTEngine::bizinter(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    ensure_size(real_work_, n);
    ensure_size(imag_work_, n);
    for (size_t i = 0; i < n; ++i) {
        real_work_[i] = input[i].real();
        imag_work_[i] = input[i].imag();
    }

    size_t result_size = 2 * n;
    ensure_size(work_buffer6_, result_size);
    ensure_size(work_buffer7_, result_size);

    bizinter_real(real_work_, n, work_buffer6_);
    bizinter_real(imag_work_, n, work_buffer7_);

    ensure_size(output, result_size);
    for (size_t i = 0; i < result_size; ++i) {
        output[i] = Complex(work_buffer6_[i].real(), work_buffer7_[i].real());
    }
}

void FRFTEngine::bizinter_real(const std::vector<double>& input, size_t n, std::vector<Complex>& output) {
    size_t N1 = n / 2 + (n % 2);
    size_t N2 = 2 * n - (n / 2);

    ensure_size(work_buffer8_, n * 2);
    for (size_t i = 0; i < n; ++i) work_buffer8_[i] = Complex(input[i], 0.0);

    upsample2(work_buffer8_, n, fft_pad_buffer_);
    fft(fft_pad_buffer_, n * 2, conv_buffer_);

    for (size_t i = N1; i < N2 && i < n * 2; ++i) conv_buffer_[i] = Complex(0.0, 0.0);

    ifft(conv_buffer_, n * 2, output);

    for (size_t i = 0; i < 2 * n; ++i) {
        output[i] = Complex(2.0 * output[i].real(), 0.0);
    }
}

void FRFTEngine::upsample2(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    size_t out_size = n * 2;
    ensure_size(output, out_size);
    for (size_t i = 0; i < n; ++i) {
        output[2 * i]     = input[i];
        output[2 * i + 1] = Complex(0.0, 0.0);
    }
}

void FRFTEngine::corefrmod2(const std::vector<Complex>& signal, size_t N, double a, std::vector<Complex>& output) {
    int Nend   = N / 2;
    int Nstart = -(static_cast<int>(N % 2) + Nend);
    double deltax = std::sqrt(static_cast<double>(N));

    double phi = a * M_PI / 2.0;
    Complex alpha = Complex(0.0, -M_PI * std::tan(phi / 2.0));
    Complex beta  = Complex(0.0, M_PI / std::sin(phi));

    Complex Aphi_num   = std::exp(Complex(0.0, -(M_PI * (std::sin(phi) >= 0 ? 1.0 : -1.0) / 4.0 - phi / 2.0)));
    double  Aphi_denum = std::sqrt(std::abs(std::sin(phi)));
    Complex Aphi = Aphi_num / Aphi_denum;

    ensure_size(chirp_buffer_, N);
    ensure_size(multip_buffer_, N);

    for (int i = 0; i < static_cast<int>(N); ++i) {
        double x = static_cast<double>(Nstart + i) / deltax;
        chirp_buffer_[i]  = std::exp(alpha * x * x);
        multip_buffer_[i] = signal[i] * chirp_buffer_[i];
    }

    size_t t_size = 2 * N - 1;
    ensure_size(hlptc_buffer_, t_size);
    for (int i = 0; i < static_cast<int>(t_size); ++i) {
        double t = static_cast<double>(-static_cast<int>(N) + 1 + i) / deltax;
        hlptc_buffer_[i] = std::exp(beta * t * t);
    }

    int next_pow2 = next_power_of_2(t_size + N - 1);
    ensure_size(work_buffer5_, next_pow2);
    ensure_size(work_buffer6_, next_pow2);
    ensure_size(conv_buffer_, next_pow2);

    fft_n(multip_buffer_, N, next_pow2, work_buffer5_);
    fft_n(hlptc_buffer_, t_size, next_pow2, work_buffer6_);
    vecmul(work_buffer5_, work_buffer6_, next_pow2, conv_buffer_);

    ensure_size(work_buffer7_, next_pow2);
    ifft_n(conv_buffer_, next_pow2, next_pow2, work_buffer7_);

    ensure_size(output, N);
    for (size_t i = 0; i < N; ++i) {
        output[i] = work_buffer7_[N - 1 + i] * Aphi * chirp_buffer_[i] / deltax;
    }

    if (N % 2 == 1) {
        Complex temp = output[0];
        for (size_t i = 0; i < N - 1; ++i) output[i] = output[i + 1];
        output[N - 1] = temp;
    }
}

void FRFTEngine::vecmul(const std::vector<Complex>& tensor, const std::vector<Complex>& vector,
                        size_t n, std::vector<Complex>& output) {
    ensure_size(output, n);
    for (size_t i = 0; i < n; ++i) output[i] = tensor[i] * vector[i];
}

int FRFTEngine::next_power_of_2(int n) {
    if (n <= 1) return 1;
    return std::pow(2, std::ceil(std::log2(n)));
}

void FRFTEngine::fftshift(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    size_t half = n / 2;
    ensure_size(output, n);
    for (size_t i = 0; i < n; ++i) output[i] = input[(i + half) % n];
}

void FRFTEngine::ifftshift(const std::vector<Complex>& input, size_t n, std::vector<Complex>& output) {
    size_t half = (n + 1) / 2;
    ensure_size(output, n);
    for (size_t i = 0; i < n; ++i) output[i] = input[(i + half) % n];
}
