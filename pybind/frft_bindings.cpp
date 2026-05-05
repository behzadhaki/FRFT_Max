#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "frft_engine.h"
#include <stdexcept>
#include <iostream>

namespace py = pybind11;

// Wrapper class to expose the FRFTEngine logic to Python
class FrftWrapper {
public:
    FrftWrapper() {
        // The FRFTEngine is managed by a thread-local singleton,
        // which handles setup and thread safety automatically.
    }

    // Main compute function exposed to Python
    void compute(
        py::array_t<double, py::array::c_style> in_real_array,
        py::array_t<double, py::array::c_style> in_imag_array,
        py::array_t<double, py::array::c_style> out_real_array,
        py::array_t<double, py::array::c_style> out_imag_array,
        double alpha
    ) {
        // Check input/output array sizes
        if (in_real_array.size() != out_real_array.size() ||
            in_imag_array.size() != out_imag_array.size() ||
            in_real_array.size() != in_imag_array.size())
        {
            throw std::runtime_error("Input and output arrays must have the same size.");
        }

        size_t size = in_real_array.size();
        if (size == 0 || size % 2 != 0) {
            throw std::runtime_error("Array size must be positive and even.");
        }

        // Get pointers to the data
        const double* in_real = in_real_array.data();
        const double* in_imag = in_imag_array.data();
        double* out_real = out_real_array.mutable_data();
        double* out_imag = out_imag_array.mutable_data();

        // Get the thread-local engine instance
        FRFTEngine* engine = FRFTEngineManager::instance().get_thread_engine();

        // Compute FRFT
        bool success = engine->compute(
            in_real, in_imag,
            out_real, out_imag,
            (int)size, alpha
        );

        if (!success) {
            std::cerr << "FRFT computation failed in C++ engine." << std::endl;
        }
    }
};

PYBIND11_MODULE(frft_cpp, m) {
    m.doc() = "Pybind11 interface for the C++ FRFT implementation"; // module docstring

    py::class_<FrftWrapper>(m, "Frft")
        .def(py::init<>())
        .def("compute", &FrftWrapper::compute,
             "Compute Fractional Fourier Transform (FRFT)",
             py::arg("in_real"), py::arg("in_imag"),
             py::arg("out_real"), py::arg("out_imag"),
             py::arg("alpha"));

    // FIX: Using a lambda function to correctly bind the singleton's method.
    // This avoids taking a pointer to a temporary object.
    m.def("set_debug", [](bool enable) {
        FRFTEngineManager::instance().set_debug_enabled(enable);
    }, py::arg("enable"), "Set debug mode for C++ FRFT engine");
}