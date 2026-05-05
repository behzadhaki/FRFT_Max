// Compiles both engines side-by-side and compares their outputs.
//
// Cross-comparison: FRFTEngineOrig (FFTW) vs FRFTEngine (KissFFT)
// Math properties: round-trip, additivity, a=2 reversal
//
// Build: see Makefile in this directory.

// ---- Include original engine under a renamed class -------------------------
#define FRFTEngine          FRFTEngineOrig
#define FRFTEngineManager   FRFTEngineManagerOrig
#include "../../src/frft_engine.h"
#undef FRFTEngine
#undef FRFTEngineManager

// ---- Include WASM engine normally ------------------------------------------
#include "frft_engine_wasm.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// Helpers
// ============================================================================

static void fill_random(std::vector<double>& r, std::vector<double>& im, int n, unsigned seed = 42) {
    srand(seed);
    r.resize(n); im.resize(n);
    for (int i = 0; i < n; ++i) {
        r[i]  = (double)rand() / RAND_MAX * 2.0 - 1.0;
        im[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
}

static double max_error(const std::vector<double>& a, const std::vector<double>& b,
                        const std::vector<double>& c, const std::vector<double>& d) {
    double err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        err = std::max(err, std::abs(a[i] - b[i]));
        err = std::max(err, std::abs(c[i] - d[i]));
    }
    return err;
}

static bool check(const std::string& label, double err, double tol = 1e-8) {
    bool ok = err < tol;
    std::cout << (ok ? "  PASS" : "  FAIL")
              << "  " << std::left << std::setw(52) << label
              << "  max_err = " << std::scientific << std::setprecision(2) << err
              << "\n";
    return ok;
}

// ============================================================================
// Test 1: Cross-comparison — FFTW engine vs KissFFT engine
// ============================================================================

static bool test_cross_comparison() {
    std::cout << "\n[1] Cross-comparison (FFTW vs KissFFT)\n";
    bool all_ok = true;

    FRFTEngineOrig orig;
    FRFTEngine     wasm;

    const std::vector<int>    sizes  = {64, 128, 256, 512};
    const std::vector<double> alphas = {0.25, 0.5, 0.75, 1.0, 1.5, -0.5};

    for (int n : sizes) {
        std::vector<double> in_r, in_i;
        fill_random(in_r, in_i, n);

        orig.prepare(n);
        wasm.prepare(n);

        std::vector<double> or_r(n), or_i(n), wr_r(n), wr_i(n);

        for (double a : alphas) {
            orig.compute(in_r.data(), in_i.data(), or_r.data(), or_i.data(), n, a);
            wasm.compute(in_r.data(), in_i.data(), wr_r.data(), wr_i.data(), n, a);

            double err = max_error(or_r, wr_r, or_i, wr_i);
            std::string label = "N=" + std::to_string(n) + "  alpha=" + std::to_string(a);
            all_ok &= check(label, err, 1e-8);
        }
    }
    return all_ok;
}

// ============================================================================
// Tests 2-4: Math properties — verified on both engines to confirm they
// behave identically (not that they hit machine epsilon).
// Round-trip / additivity errors at fractional orders are a known property
// of this discrete FRFT algorithm; the cross-comparison in Test 1 is the
// definitive correctness check for the KissFFT port.
// ============================================================================

static double round_trip_error(FRFTEngine& eng, const std::vector<double>& in_r,
                                const std::vector<double>& in_i, int n, double a) {
    std::vector<double> mid_r(n), mid_i(n), out_r(n), out_i(n);
    eng.compute(in_r.data(), in_i.data(), mid_r.data(), mid_i.data(), n,  a);
    eng.compute(mid_r.data(), mid_i.data(), out_r.data(), out_i.data(), n, -a);
    return max_error(in_r, out_r, in_i, out_i);
}

static double round_trip_error(FRFTEngineOrig& eng, const std::vector<double>& in_r,
                                const std::vector<double>& in_i, int n, double a) {
    std::vector<double> mid_r(n), mid_i(n), out_r(n), out_i(n);
    eng.compute(in_r.data(), in_i.data(), mid_r.data(), mid_i.data(), n,  a);
    eng.compute(mid_r.data(), mid_i.data(), out_r.data(), out_i.data(), n, -a);
    return max_error(in_r, out_r, in_i, out_i);
}

static bool test_round_trip() {
    std::cout << "\n[2] Round-trip consistency (FFTW error vs KissFFT error must match)\n";
    bool all_ok = true;

    FRFTEngineOrig orig;
    FRFTEngine     wasm;

    const std::vector<int>    sizes  = {128, 512};
    const std::vector<double> alphas = {0.25, 0.5, 0.75, 1.0, 1.5};

    for (int n : sizes) {
        std::vector<double> in_r, in_i;
        fill_random(in_r, in_i, n);

        for (double a : alphas) {
            double err_orig = round_trip_error(orig, in_r, in_i, n, a);
            double err_wasm = round_trip_error(wasm, in_r, in_i, n, a);
            double diff = std::abs(err_orig - err_wasm);

            std::string label = "N=" + std::to_string(n) + "  alpha=" + std::to_string(a)
                              + "  (FFTW=" + std::to_string(err_orig).substr(0,6)
                              + " KissFFT=" + std::to_string(err_wasm).substr(0,6) + ")";
            all_ok &= check(label, diff, 1e-8);
        }
    }
    return all_ok;
}

static bool test_additivity() {
    std::cout << "\n[3] Additivity consistency (FFTW error vs KissFFT error must match)\n";
    bool all_ok = true;

    FRFTEngineOrig orig;
    FRFTEngine     wasm;
    const int n = 128;

    std::vector<double> in_r, in_i;
    fill_random(in_r, in_i, n);

    const std::vector<std::pair<double,double>> pairs = {{0.3,0.4},{0.5,0.5},{0.2,0.7}};

    auto additivity_err = [&](auto& eng, double a, double b) {
        std::vector<double> mid_r(n), mid_i(n), lhs_r(n), lhs_i(n), rhs_r(n), rhs_i(n);
        eng.compute(in_r.data(), in_i.data(), mid_r.data(), mid_i.data(), n, a);
        eng.compute(mid_r.data(), mid_i.data(), lhs_r.data(), lhs_i.data(), n, b);
        eng.compute(in_r.data(), in_i.data(), rhs_r.data(), rhs_i.data(), n, a + b);
        return max_error(lhs_r, rhs_r, lhs_i, rhs_i);
    };

    for (auto [a, b] : pairs) {
        double err_orig = additivity_err(orig, a, b);
        double err_wasm = additivity_err(wasm, a, b);
        double diff = std::abs(err_orig - err_wasm);

        std::string label = "a=" + std::to_string(a).substr(0,4)
                          + "  b=" + std::to_string(b).substr(0,4)
                          + "  (FFTW=" + std::to_string(err_orig).substr(0,6)
                          + " KissFFT=" + std::to_string(err_wasm).substr(0,6) + ")";
        all_ok &= check(label, diff, 1e-8);
    }
    return all_ok;
}

static bool test_reversal() {
    std::cout << "\n[4] a=2 reversal consistency (FFTW error vs KissFFT error must match)\n";
    bool all_ok = true;

    FRFTEngineOrig orig;
    FRFTEngine     wasm;

    const std::vector<int> sizes = {64, 256};

    auto reversal_err = [](auto& eng, const std::vector<double>& in_r,
                           const std::vector<double>& in_i, int n) {
        std::vector<double> out_r(n), out_i(n), ref_r(n), ref_i(n);
        eng.compute(in_r.data(), in_i.data(), out_r.data(), out_i.data(), n, 2.0);
        ref_r[0] = in_r[0]; ref_i[0] = in_i[0];
        for (int k = 1; k < n; ++k) { ref_r[k] = in_r[n-k]; ref_i[k] = in_i[n-k]; }
        return max_error(out_r, ref_r, out_i, ref_i);
    };

    for (int n : sizes) {
        std::vector<double> in_r, in_i;
        fill_random(in_r, in_i, n);

        double err_orig = reversal_err(orig, in_r, in_i, n);
        double err_wasm = reversal_err(wasm, in_r, in_i, n);
        double diff = std::abs(err_orig - err_wasm);

        std::string label = "N=" + std::to_string(n)
                          + "  (FFTW=" + std::to_string(err_orig).substr(0,6)
                          + " KissFFT=" + std::to_string(err_wasm).substr(0,6) + ")";
        all_ok &= check(label, diff, 1e-8);
    }
    return all_ok;
}

// ============================================================================

int main() {
    std::cout << "FRFT Engine Test: KissFFT (wasm) vs FFTW (original)\n";
    std::cout << std::string(60, '=') << "\n";

    int failed = 0;
    failed += !test_cross_comparison();
    failed += !test_round_trip();
    failed += !test_additivity();
    failed += !test_reversal();

    std::cout << "\n" << std::string(60, '=') << "\n";
    if (failed == 0) {
        std::cout << "All tests passed.\n";
    } else {
        std::cout << failed << " test group(s) failed.\n";
    }
    return failed;
}
