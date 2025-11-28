#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "frft_engine.h"
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// Wrapper function to compute FRFT and return numpy arrays
py::tuple frft_compute(
    py::array_t<double> real_in,
    py::array_t<double> imag_in,
    double a_param) {

    // Get buffer info
    py::buffer_info real_buf = real_in.request();
    py::buffer_info imag_buf = imag_in.request();

    // Validate inputs
    if (real_buf.ndim != 1 || imag_buf.ndim != 1) {
        throw std::runtime_error("Input arrays must be 1-dimensional");
    }

    if (real_buf.shape[0] != imag_buf.shape[0]) {
        throw std::runtime_error("Real and imaginary parts must have same length");
    }

    int size = real_buf.shape[0];

    // Check for even length (required by FRFT algorithm)
    if (size % 2 != 0) {
        throw std::runtime_error("Input size must be even");
    }

    // Get pointers to input data
    double* real_in_ptr = static_cast<double*>(real_buf.ptr);
    double* imag_in_ptr = static_cast<double*>(imag_buf.ptr);

    // Allocate output arrays
    py::array_t<double> real_out(size);
    py::array_t<double> imag_out(size);

    py::buffer_info real_out_buf = real_out.request();
    py::buffer_info imag_out_buf = imag_out.request();

    double* real_out_ptr = static_cast<double*>(real_out_buf.ptr);
    double* imag_out_ptr = static_cast<double*>(imag_out_buf.ptr);

    // Get thread-local engine and compute
    FRFTEngine* engine = FRFTEngineManager::instance().get_thread_engine();

    bool success = engine->compute(
        real_in_ptr, imag_in_ptr,
        real_out_ptr, imag_out_ptr,
        size, a_param
    );

    if (!success) {
        throw std::runtime_error("FRFT computation failed");
    }

    return py::make_tuple(real_out, imag_out);
}

// Convenience function for real-valued signals
py::tuple frft_compute_real(
    py::array_t<double> signal,
    double a_param) {

    py::buffer_info buf = signal.request();

    if (buf.ndim != 1) {
        throw std::runtime_error("Input array must be 1-dimensional");
    }

    int size = buf.shape[0];

    // Create zero imaginary part
    py::array_t<double> imag_in = py::array_t<double>(size);
    py::buffer_info imag_buf = imag_in.request();
    double* imag_ptr = static_cast<double*>(imag_buf.ptr);
    std::fill(imag_ptr, imag_ptr + size, 0.0);

    return frft_compute(signal, imag_in, a_param);
}

// Function to prepare engine for a specific size (optional optimization)
void prepare_engine(int size) {
    FRFTEngine* engine = FRFTEngineManager::instance().get_thread_engine();
    engine->prepare(size);
}

// Debug control functions
void set_debug(bool enable) {
    FRFTEngineManager::instance().set_debug_enabled(enable);
}

bool get_debug() {
    return FRFTEngineManager::instance().get_debug_enabled();
}

PYBIND11_MODULE(frft_cpp, m) {
    m.doc() = "Fast Fractional Fourier Transform (FRFT) C++ implementation";

    m.def("frft_compute", &frft_compute,
          py::arg("real_in"),
          py::arg("imag_in"),
          py::arg("a_param"),
          R"pbdoc(
          Compute the Fractional Fourier Transform of a complex signal.

          Parameters
          ----------
          real_in : np.ndarray
              Real part of input signal (1D array, even length)
          imag_in : np.ndarray
              Imaginary part of input signal (1D array, even length)
          a_param : float
              Fractional order parameter

          Returns
          -------
          tuple of np.ndarray
              (real_out, imag_out) - Real and imaginary parts of FRFT result

          Examples
          --------
          >>> import numpy as np
          >>> import frft_cpp
          >>> signal = np.random.randn(1024)
          >>> imag = np.zeros(1024)
          >>> real_out, imag_out = frft_cpp.frft_compute(signal, imag, 0.5)
          )pbdoc");

    m.def("frft_compute_real", &frft_compute_real,
          py::arg("signal"),
          py::arg("a_param"),
          R"pbdoc(
          Compute the Fractional Fourier Transform of a real-valued signal.

          Parameters
          ----------
          signal : np.ndarray
              Real-valued input signal (1D array, even length)
          a_param : float
              Fractional order parameter

          Returns
          -------
          tuple of np.ndarray
              (real_out, imag_out) - Real and imaginary parts of FRFT result

          Examples
          --------
          >>> import numpy as np
          >>> import frft_cpp
          >>> signal = np.random.randn(1024)
          >>> real_out, imag_out = frft_cpp.frft_compute_real(signal, 0.5)
          )pbdoc");

    m.def("prepare", &prepare_engine,
          py::arg("size"),
          "Pre-allocate buffers for a specific signal size (optional optimization)");

    m.def("set_debug", &set_debug,
          py::arg("enable"),
          "Enable or disable debug output");

    m.def("get_debug", &get_debug,
          "Get current debug state");

    m.attr("__version__") = "0.1.0";
}