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

// Pure C++ FRFT implementation using FFTW
// No PyTorch dependencies - optimized for real-time audio processing

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

private:
    // Core algorithm functions
    std::vector<Complex> dflip(const std::vector<Complex>& tensor);
    std::vector<Complex> bizdec(const std::vector<Complex>& x);
    std::vector<Complex> bizinter(const std::vector<Complex>& x);
    std::vector<Complex> bizinter_real(const std::vector<double>& x);
    std::vector<Complex> upsample2(const std::vector<Complex>& x);
    std::vector<Complex> corefrmod2(const std::vector<Complex>& signal, double a);
    std::vector<Complex> vecmul(const std::vector<Complex>& tensor, const std::vector<Complex>& vector);

    // FFT helpers
    std::vector<Complex> fft(const std::vector<Complex>& input);
    std::vector<Complex> ifft(const std::vector<Complex>& input);
    std::vector<Complex> fft_n(const std::vector<Complex>& input, size_t n);
    std::vector<Complex> ifft_n(const std::vector<Complex>& input, size_t n);

    // Utility
    int next_power_of_2(int n);
    std::vector<Complex> fftshift(const std::vector<Complex>& input);
    std::vector<Complex> ifftshift(const std::vector<Complex>& input);

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

    // Cached buffers to avoid reallocation (reused across compute() calls)
    size_t cached_size_ = 0;
    std::vector<Complex> fc_;
    std::vector<Complex> result_;
    std::vector<Complex> biz_;
    std::vector<Complex> zeros_;
    std::vector<Complex> fc_expanded_;
    std::vector<Complex> res_;
    std::vector<Complex> res_extracted_;

    // Buffers for corefrmod2
    std::vector<Complex> chirp_;
    std::vector<Complex> multip_;
    std::vector<Complex> hlptc_;
    std::vector<Complex> multip_fft_;
    std::vector<Complex> hlptc_fft_;
    std::vector<Complex> conv_fft_;
    std::vector<Complex> Hc_;
    std::vector<Complex> Hc_extracted_;
    std::vector<Complex> corefrmod2_result_;

    // Buffers for bizinter
    std::vector<double> real_part_;
    std::vector<double> imag_part_;
    std::vector<Complex> real_result_;
    std::vector<Complex> imag_result_;
    std::vector<Complex> bizinter_result_;

    // Buffers for bizinter_real
    std::vector<Complex> x_complex_;
    std::vector<Complex> upsampled_;
    std::vector<Complex> xf_;
    std::vector<Complex> bizinter_real_result_;

    // Buffers for upsample2
    std::vector<Complex> upsample2_result_;

    // Buffers for fftshift/ifftshift
    std::vector<Complex> fftshift_output_;
    std::vector<Complex> ifftshift_output_;

    // Buffers for dflip, bizdec, vecmul
    std::vector<Complex> dflip_result_;
    std::vector<Complex> bizdec_result_;
    std::vector<Complex> vecmul_result_;

    // Buffers for fft/ifft operations
    std::vector<Complex> fft_result_;
    std::vector<Complex> ifft_result_;
    std::vector<Complex> fft_n_padded_;
    std::vector<Complex> ifft_n_padded_;
};