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

    // Construct complex signal
    std::vector<Complex> fc(size);
    for (int i = 0; i < size; ++i) {
        fc[i] = Complex(real_in[i], imag_in[i]);
    }

    // Apply fftshift to convert from FFT ordering [0, pos, neg] to centered [-N/2, ..., N/2]
    fc = fftshift(fc);

    // 4-modulation and shifting to [-2, 2] interval
    double a = std::fmod(a_param, 4.0);
    if (a > 2.0) {
        a -= 4.0;
    } else if (a < -2.0) {
        a += 4.0;
    }

    std::vector<Complex> result;

    // Special integer cases
    if (std::abs(a) < 1e-10) {
        result = fc;
    } else if (std::abs(a - 2.0) < 1e-10 || std::abs(a + 2.0) < 1e-10) {
        result = dflip(fc);
    } else {
        // General case
        std::vector<Complex> biz = bizinter(fc);

        // Create zeros vector of size N
        std::vector<Complex> zeros(size, Complex(0.0, 0.0));

        // Concatenate: zeros + biz + zeros
        std::vector<Complex> fc_expanded;
        fc_expanded.reserve(size + biz.size() + size);
        fc_expanded.insert(fc_expanded.end(), zeros.begin(), zeros.end());
        fc_expanded.insert(fc_expanded.end(), biz.begin(), biz.end());
        fc_expanded.insert(fc_expanded.end(), zeros.begin(), zeros.end());

        std::vector<Complex> res = fc_expanded;

        // Conditional transformations based on a value
        if ((0 < a && a < 0.5) || (1.5 < a && a < 2.0)) {
            res = corefrmod2(fc_expanded, 1.0);
            a -= 1.0;
        }

        if ((-0.5 < a && a < 0) || (-2.0 < a && a < -1.5)) {
            res = corefrmod2(fc_expanded, -1.0);
            a += 1.0;
        }

        res = corefrmod2(res, a);

        // Extract elements from index N to 3*N
        std::vector<Complex> res_extracted(res.begin() + size, res.begin() + 3 * size);

        // Decimate
        res_extracted = bizdec(res_extracted);

        result = res_extracted;
    }

    // Apply ifftshift to convert back from centered to FFT ordering
    result = ifftshift(result);

    // Extract real and imaginary parts
    for (size_t i = 0; i < result.size() && i < static_cast<size_t>(size); ++i) {
        real_out[i] = result[i].real();
        imag_out[i] = result[i].imag();
    }

    return true;
}

std::vector<Complex> FRFTEngine::dflip(const std::vector<Complex>& tensor) {
    if (tensor.empty()) return tensor;

    std::vector<Complex> result(tensor.size());
    result[0] = tensor[0];

    for (size_t i = 1; i < tensor.size(); ++i) {
        result[i] = tensor[tensor.size() - i];
    }

    return result;
}

std::vector<Complex> FRFTEngine::bizdec(const std::vector<Complex>& x) {
    std::vector<Complex> result;
    result.reserve(x.size() / 2 + 1);

    for (size_t i = 0; i < x.size(); i += 2) {
        result.push_back(x[i]);
    }

    return result;
}

std::vector<Complex> FRFTEngine::bizinter(const std::vector<Complex>& x) {
    std::vector<double> real_part(x.size());
    std::vector<double> imag_part(x.size());

    for (size_t i = 0; i < x.size(); ++i) {
        real_part[i] = x[i].real();
        imag_part[i] = x[i].imag();
    }

    std::vector<Complex> real_result = bizinter_real(real_part);
    std::vector<Complex> imag_result = bizinter_real(imag_part);

    std::vector<Complex> result(real_result.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = Complex(real_result[i].real(), imag_result[i].real());
    }

    return result;
}

std::vector<Complex> FRFTEngine::bizinter_real(const std::vector<double>& x) {
    size_t N = x.size();
    size_t N1 = N / 2 + (N % 2);
    size_t N2 = 2 * N - (N / 2);

    std::vector<Complex> x_complex(N);
    for (size_t i = 0; i < N; ++i) {
        x_complex[i] = Complex(x[i], 0.0);
    }

    std::vector<Complex> upsampled = upsample2(x_complex);
    std::vector<Complex> xf = fft(upsampled);

    for (size_t i = N1; i < N2 && i < xf.size(); ++i) {
        xf[i] = Complex(0.0, 0.0);
    }

    std::vector<Complex> result = ifft(xf);

    // Scale by 2 to compensate for upsampling, preserving full complex precision
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] *= 2.0;
    }

    return result;
}

std::vector<Complex> FRFTEngine::upsample2(const std::vector<Complex>& x) {
    std::vector<Complex> result(x.size() * 2);

    for (size_t i = 0; i < x.size(); ++i) {
        result[2 * i] = x[i];
        result[2 * i + 1] = Complex(0.0, 0.0);
    }

    return result;
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

    std::vector<Complex> chirp(N);
    std::vector<Complex> multip(N);

    for (int i = 0; i < static_cast<int>(N); ++i) {
        double x = static_cast<double>(Nstart + i) / deltax;
        chirp[i] = std::exp(alpha * x * x);
        multip[i] = signal[i] * chirp[i];
    }

    size_t t_size = 2 * N - 1;
    std::vector<Complex> hlptc(t_size);

    for (int i = 0; i < static_cast<int>(t_size); ++i) {
        double t = static_cast<double>(-static_cast<int>(N) + 1 + i) / deltax;
        hlptc[i] = std::exp(beta * t * t);
    }

    int next_pow2 = next_power_of_2(t_size + N - 1);

    std::vector<Complex> multip_fft = fft_n(multip, next_pow2);
    std::vector<Complex> hlptc_fft = fft_n(hlptc, next_pow2);
    std::vector<Complex> conv_fft = vecmul(multip_fft, hlptc_fft);
    std::vector<Complex> Hc = ifft_n(conv_fft, next_pow2);

    std::vector<Complex> Hc_extracted(Hc.begin() + N - 1, Hc.begin() + 2 * N - 1);

    std::vector<Complex> result(N);
    for (size_t i = 0; i < N; ++i) {
        result[i] = Hc_extracted[i] * Aphi * chirp[i] / deltax;
    }

    if (N % 2 == 1) {
        std::rotate(result.begin(), result.begin() + 1, result.end());
    }

    return result;
}

std::vector<Complex> FRFTEngine::vecmul(const std::vector<Complex>& tensor, const std::vector<Complex>& vector) {
    size_t size = std::min(tensor.size(), vector.size());
    std::vector<Complex> result(size);

    for (size_t i = 0; i < size; ++i) {
        result[i] = tensor[i] * vector[i];
    }

    return result;
}

std::vector<Complex> FRFTEngine::fft(const std::vector<Complex>& input) {
    size_t N = input.size();
    PlanCache* cache = get_or_create_plan(N);

    for (size_t i = 0; i < N; ++i) {
        cache->in_buffer[i][0] = input[i].real();
        cache->in_buffer[i][1] = input[i].imag();
    }

    fftw_execute(cache->forward_plan);

    std::vector<Complex> result(N);
    for (size_t i = 0; i < N; ++i) {
        result[i] = Complex(cache->out_buffer[i][0], cache->out_buffer[i][1]);
    }

    return result;
}

std::vector<Complex> FRFTEngine::ifft(const std::vector<Complex>& input) {
    size_t N = input.size();
    PlanCache* cache = get_or_create_plan(N);

    for (size_t i = 0; i < N; ++i) {
        cache->in_buffer[i][0] = input[i].real();
        cache->in_buffer[i][1] = input[i].imag();
    }

    fftw_execute(cache->backward_plan);

    std::vector<Complex> result(N);
    for (size_t i = 0; i < N; ++i) {
        result[i] = Complex(cache->out_buffer[i][0] / N, cache->out_buffer[i][1] / N);
    }

    return result;
}

std::vector<Complex> FRFTEngine::fft_n(const std::vector<Complex>& input, size_t n) {
    std::vector<Complex> padded(n, Complex(0.0, 0.0));

    size_t copy_size = std::min(input.size(), n);
    for (size_t i = 0; i < copy_size; ++i) {
        padded[i] = input[i];
    }

    return fft(padded);
}

std::vector<Complex> FRFTEngine::ifft_n(const std::vector<Complex>& input, size_t n) {
    std::vector<Complex> padded(n, Complex(0.0, 0.0));

    size_t copy_size = std::min(input.size(), n);
    for (size_t i = 0; i < copy_size; ++i) {
        padded[i] = input[i];
    }

    return ifft(padded);
}

int FRFTEngine::next_power_of_2(int n) {
    if (n <= 1) return 1;
    return std::pow(2, std::ceil(std::log2(n)));
}

std::vector<Complex> FRFTEngine::fftshift(const std::vector<Complex>& input) {
    size_t N = input.size();
    size_t half = N / 2;
    std::vector<Complex> output(N);

    // Move second half to first half, first half to second half
    // [0 1 2 3 4 5] -> [3 4 5 0 1 2]  (for N=6, half=3)
    for (size_t i = 0; i < N; ++i) {
        output[i] = input[(i + half) % N];
    }

    return output;
}

std::vector<Complex> FRFTEngine::ifftshift(const std::vector<Complex>& input) {
    size_t N = input.size();
    size_t half = (N + 1) / 2;  // Ceiling division for odd N
    std::vector<Complex> output(N);

    // Inverse of fftshift
    // [3 4 5 0 1 2] -> [0 1 2 3 4 5]  (for N=6)
    for (size_t i = 0; i < N; ++i) {
        output[i] = input[(i + half) % N];
    }

    return output;
}