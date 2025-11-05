#include "frft_engine.h"
#include <algorithm>
#include <cstring>

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
    // Check if we already have a plan for this size
    for (auto& cache : plan_cache_) {
        if (cache.size == size) {
            return &cache;
        }
    }

    // Create new plan
    PlanCache new_cache;
    new_cache.size = size;
    new_cache.in_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    new_cache.out_buffer = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);

    // Create plans with FFTW_MEASURE for better performance
    new_cache.forward_plan = fftw_plan_dft_1d(size, new_cache.in_buffer, new_cache.out_buffer,
                                              FFTW_FORWARD, FFTW_MEASURE);
    new_cache.backward_plan = fftw_plan_dft_1d(size, new_cache.in_buffer, new_cache.out_buffer,
                                               FFTW_BACKWARD, FFTW_MEASURE);

    plan_cache_.push_back(new_cache);
    return &plan_cache_.back();
}

void FRFTEngine::prepare(int size) {
    // Pre-create plans for common sizes used in audio
    get_or_create_plan(size);
    get_or_create_plan(size * 2);  // For upsampling
    get_or_create_plan(next_power_of_2(size * 4));  // For convolution
}

bool FRFTEngine::compute(const double* real_in, const double* imag_in,
                         double* real_out, double* imag_out,
                         int size, double a_param) {
    // Validate inputs
    if (size % 2 != 0) {
        return false;  // Signal size must be even
    }

    // Reallocate buffers only if size changed
    if (cached_size_ != static_cast<size_t>(size)) {
        cached_size_ = size;
        fc_.resize(size);
        result_.resize(size);
        zeros_.resize(size);
    }

    // Construct complex signal (reusing fc_ buffer)
    for (int i = 0; i < size; ++i) {
        fc_[i] = Complex(real_in[i], imag_in[i]);
    }

    // Apply fftshift to convert from FFT ordering [0, pos, neg] to centered [-N/2, ..., N/2]
    fftshift(fc_);
    fc_.swap(fftshift_output_);  // Swap instead of copy

    // 4-modulation and shifting to [-2, 2] interval
    double a = std::fmod(a_param, 4.0);
    if (a > 2.0) {
        a -= 4.0;
    } else if (a < -2.0) {
        a += 4.0;
    }

    // Special integer cases
    if (std::abs(a) < 1e-10) {
        result_ = fc_;
    } else if (std::abs(a - 2.0) < 1e-10 || std::abs(a + 2.0) < 1e-10) {
        result_ = dflip(fc_);
    } else {
        // General case
        biz_ = bizinter(fc_);

        // Fill zeros vector (already allocated)
        std::fill(zeros_.begin(), zeros_.end(), Complex(0.0, 0.0));

        // Concatenate: zeros + biz + zeros (reusing fc_expanded_)
        size_t expanded_size = size + biz_.size() + size;
        if (fc_expanded_.size() != expanded_size) {
            fc_expanded_.resize(expanded_size);
        }

        std::copy(zeros_.begin(), zeros_.end(), fc_expanded_.begin());
        std::copy(biz_.begin(), biz_.end(), fc_expanded_.begin() + size);
        std::copy(zeros_.begin(), zeros_.end(), fc_expanded_.begin() + size + biz_.size());

        // Use fc_expanded_ directly - avoid copy to res_
        // Conditional transformations based on a value
        if ((0 < a && a < 0.5) || (1.5 < a && a < 2.0)) {
            res_ = corefrmod2(fc_expanded_, 1.0);
            a -= 1.0;
        } else {
            res_ = fc_expanded_;
        }

        if ((-0.5 < a && a < 0) || (-2.0 < a && a < -1.5)) {
            res_ = corefrmod2(res_, -1.0);
            a += 1.0;
        }

        res_ = corefrmod2(res_, a);

        // Extract elements from index N to 3*N (reusing res_extracted_)
        if (res_extracted_.size() != static_cast<size_t>(2 * size)) {
            res_extracted_.resize(2 * size);
        }
        std::copy(res_.begin() + size, res_.begin() + 3 * size, res_extracted_.begin());

        // Decimate
        res_extracted_ = bizdec(res_extracted_);

        result_ = res_extracted_;
    }

    // Apply ifftshift to convert back from centered to FFT ordering
    ifftshift(result_);
    result_.swap(ifftshift_output_);  // Swap instead of copy

    // Extract real and imaginary parts
    for (size_t i = 0; i < result_.size() && i < static_cast<size_t>(size); ++i) {
        real_out[i] = result_[i].real();
        imag_out[i] = result_[i].imag();
    }

    return true;
}

std::vector<Complex> FRFTEngine::dflip(const std::vector<Complex>& tensor) {
    if (tensor.empty()) return tensor;

    // Reallocate only if size changed
    if (dflip_result_.size() != tensor.size()) {
        dflip_result_.resize(tensor.size());
    }

    dflip_result_[0] = tensor[0];

    for (size_t i = 1; i < tensor.size(); ++i) {
        dflip_result_[i] = tensor[tensor.size() - i];
    }

    return dflip_result_;
}

std::vector<Complex> FRFTEngine::bizdec(const std::vector<Complex>& x) {
    size_t result_size = x.size() / 2 + 1;

    // Reallocate only if size changed
    if (bizdec_result_.size() != result_size) {
        bizdec_result_.resize(result_size);
    }

    size_t idx = 0;
    for (size_t i = 0; i < x.size(); i += 2) {
        bizdec_result_[idx++] = x[i];
    }

    return bizdec_result_;
}

std::vector<Complex> FRFTEngine::bizinter(const std::vector<Complex>& x) {
    // Reallocate only if size changed
    if (real_part_.size() != x.size()) {
        real_part_.resize(x.size());
        imag_part_.resize(x.size());
        bizinter_result_.resize(0); // Will be resized by bizinter_real
    }

    for (size_t i = 0; i < x.size(); ++i) {
        real_part_[i] = x[i].real();
        imag_part_[i] = x[i].imag();
    }

    real_result_ = bizinter_real(real_part_);
    imag_result_ = bizinter_real(imag_part_);

    if (bizinter_result_.size() != real_result_.size()) {
        bizinter_result_.resize(real_result_.size());
    }

    for (size_t i = 0; i < bizinter_result_.size(); ++i) {
        bizinter_result_[i] = Complex(real_result_[i].real(), imag_result_[i].real());
    }

    return bizinter_result_;
}

std::vector<Complex> FRFTEngine::bizinter_real(const std::vector<double>& x) {
    size_t N = x.size();
    size_t N1 = N / 2 + (N % 2);
    size_t N2 = 2 * N - (N / 2);

    // Reallocate only if size changed
    if (x_complex_.size() != N) {
        x_complex_.resize(N);
    }

    for (size_t i = 0; i < N; ++i) {
        x_complex_[i] = Complex(x[i], 0.0);
    }

    upsampled_ = upsample2(x_complex_);
    xf_ = fft(upsampled_);

    for (size_t i = N1; i < N2 && i < xf_.size(); ++i) {
        xf_[i] = Complex(0.0, 0.0);
    }

    bizinter_real_result_ = ifft(xf_);

    // Scale by 2 to compensate for upsampling, preserving full complex precision
    for (size_t i = 0; i < bizinter_real_result_.size(); ++i) {
        bizinter_real_result_[i] *= 2.0;
    }

    return bizinter_real_result_;
}

std::vector<Complex> FRFTEngine::upsample2(const std::vector<Complex>& x) {
    size_t output_size = x.size() * 2;

    // Reallocate only if size changed
    if (upsample2_result_.size() != output_size) {
        upsample2_result_.resize(output_size);
    }

    for (size_t i = 0; i < x.size(); ++i) {
        upsample2_result_[2 * i] = x[i];
        upsample2_result_[2 * i + 1] = Complex(0.0, 0.0);
    }

    return upsample2_result_;
}

std::vector<Complex> FRFTEngine::corefrmod2(const std::vector<Complex>& signal, double a) {
    size_t N = signal.size();
    int Nend = N / 2;
    int Nstart = -(static_cast<int>(N % 2) + Nend);
    double deltax = std::sqrt(static_cast<double>(N));

    double phi = a * M_PI / 2.0;
    Complex alpha = Complex(0.0, -M_PI * std::tan(phi / 2.0));
    Complex beta = Complex(0.0, M_PI / std::sin(phi));

    Complex Aphi_num = std::exp(Complex(0.0, -(M_PI * (std::sin(phi) >= 0 ? 1.0 : -1.0) / 4.0 - phi / 2.0)));
    double Aphi_denum = std::sqrt(std::abs(std::sin(phi)));
    Complex Aphi = Aphi_num / Aphi_denum;

    // Reallocate only if size changed
    if (chirp_.size() != N) {
        chirp_.resize(N);
        multip_.resize(N);
        corefrmod2_result_.resize(N);
    }

    for (int i = 0; i < static_cast<int>(N); ++i) {
        double x = static_cast<double>(Nstart + i) / deltax;
        chirp_[i] = std::exp(alpha * x * x);
        multip_[i] = signal[i] * chirp_[i];
    }

    size_t t_size = 2 * N - 1;
    if (hlptc_.size() != t_size) {
        hlptc_.resize(t_size);
    }

    for (int i = 0; i < static_cast<int>(t_size); ++i) {
        double t = static_cast<double>(-static_cast<int>(N) + 1 + i) / deltax;
        hlptc_[i] = std::exp(beta * t * t);
    }

    int next_pow2 = next_power_of_2(t_size + N - 1);

    multip_fft_ = fft_n(multip_, next_pow2);
    hlptc_fft_ = fft_n(hlptc_, next_pow2);
    conv_fft_ = vecmul(multip_fft_, hlptc_fft_);
    Hc_ = ifft_n(conv_fft_, next_pow2);

    // Reallocate only if size changed
    if (Hc_extracted_.size() != N) {
        Hc_extracted_.resize(N);
    }
    std::copy(Hc_.begin() + N - 1, Hc_.begin() + 2 * N - 1, Hc_extracted_.begin());

    for (size_t i = 0; i < N; ++i) {
        corefrmod2_result_[i] = Hc_extracted_[i] * Aphi * chirp_[i] / deltax;
    }

    if (N % 2 == 1) {
        std::rotate(corefrmod2_result_.begin(), corefrmod2_result_.begin() + 1, corefrmod2_result_.end());
    }

    return corefrmod2_result_;
}

std::vector<Complex> FRFTEngine::vecmul(const std::vector<Complex>& tensor, const std::vector<Complex>& vector) {
    size_t size = std::min(tensor.size(), vector.size());

    // Reallocate only if size changed
    if (vecmul_result_.size() != size) {
        vecmul_result_.resize(size);
    }

    for (size_t i = 0; i < size; ++i) {
        vecmul_result_[i] = tensor[i] * vector[i];
    }

    return vecmul_result_;
}

std::vector<Complex> FRFTEngine::fft(const std::vector<Complex>& input) {
    size_t N = input.size();
    PlanCache* cache = get_or_create_plan(N);

    for (size_t i = 0; i < N; ++i) {
        cache->in_buffer[i][0] = input[i].real();
        cache->in_buffer[i][1] = input[i].imag();
    }

    fftw_execute(cache->forward_plan);

    // Reallocate only if size changed
    if (fft_result_.size() != N) {
        fft_result_.resize(N);
    }

    for (size_t i = 0; i < N; ++i) {
        fft_result_[i] = Complex(cache->out_buffer[i][0], cache->out_buffer[i][1]);
    }

    return fft_result_;
}

std::vector<Complex> FRFTEngine::ifft(const std::vector<Complex>& input) {
    size_t N = input.size();
    PlanCache* cache = get_or_create_plan(N);

    for (size_t i = 0; i < N; ++i) {
        cache->in_buffer[i][0] = input[i].real();
        cache->in_buffer[i][1] = input[i].imag();
    }

    fftw_execute(cache->backward_plan);

    // Reallocate only if size changed
    if (ifft_result_.size() != N) {
        ifft_result_.resize(N);
    }

    for (size_t i = 0; i < N; ++i) {
        ifft_result_[i] = Complex(cache->out_buffer[i][0] / N, cache->out_buffer[i][1] / N);
    }

    return ifft_result_;
}

std::vector<Complex> FRFTEngine::fft_n(const std::vector<Complex>& input, size_t n) {
    // Reallocate only if size changed
    if (fft_n_padded_.size() != n) {
        fft_n_padded_.resize(n);
    }

    // Clear padding area with zeros
    std::fill(fft_n_padded_.begin(), fft_n_padded_.end(), Complex(0.0, 0.0));

    size_t copy_size = std::min(input.size(), n);
    for (size_t i = 0; i < copy_size; ++i) {
        fft_n_padded_[i] = input[i];
    }

    return fft(fft_n_padded_);
}

std::vector<Complex> FRFTEngine::ifft_n(const std::vector<Complex>& input, size_t n) {
    // Reallocate only if size changed
    if (ifft_n_padded_.size() != n) {
        ifft_n_padded_.resize(n);
    }

    // Clear padding area with zeros
    std::fill(ifft_n_padded_.begin(), ifft_n_padded_.end(), Complex(0.0, 0.0));

    size_t copy_size = std::min(input.size(), n);
    for (size_t i = 0; i < copy_size; ++i) {
        ifft_n_padded_[i] = input[i];
    }

    return ifft(ifft_n_padded_);
}

int FRFTEngine::next_power_of_2(int n) {
    if (n <= 1) return 1;
    return std::pow(2, std::ceil(std::log2(n)));
}

std::vector<Complex> FRFTEngine::fftshift(const std::vector<Complex>& input) {
    size_t N = input.size();
    size_t half = N / 2;

    // Reallocate only if size changed
    if (fftshift_output_.size() != N) {
        fftshift_output_.resize(N);
    }

    // Move second half to first half, first half to second half
    // [0 1 2 3 4 5] -> [3 4 5 0 1 2]  (for N=6, half=3)
    for (size_t i = 0; i < N; ++i) {
        fftshift_output_[i] = input[(i + half) % N];
    }

    return fftshift_output_;
}

std::vector<Complex> FRFTEngine::ifftshift(const std::vector<Complex>& input) {
    size_t N = input.size();
    size_t half = (N + 1) / 2;  // Ceiling division for odd N

    // Reallocate only if size changed
    if (ifftshift_output_.size() != N) {
        ifftshift_output_.resize(N);
    }

    // Inverse of fftshift
    // [3 4 5 0 1 2] -> [0 1 2 3 4 5]  (for N=6)
    for (size_t i = 0; i < N; ++i) {
        ifftshift_output_[i] = input[(i + half) % N];
    }

    return ifftshift_output_;
}