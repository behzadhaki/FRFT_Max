#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "frft_engine_wasm.h"

using namespace emscripten;

// ============================================================================
// FRFTProcessor
//
// Thin wrapper around FRFTEngine designed for JS/AudioWorklet use.
//
// Two usage patterns are supported:
//
// (A) Zero-copy — write/read directly to/from WASM heap buffers.
//     Best for AudioWorklet where you already have a Float64Array.
//
//       proc.prepare(512);
//       const inR = new Float64Array(Module.HEAPF64.buffer,
//                                    proc.inputRealPtr(), 512);
//       inR.set(audioData);
//       proc.process(512, alpha);
//       const outR = new Float64Array(Module.HEAPF64.buffer,
//                                     proc.outputRealPtr(), 512);
//
// (B) Convenience — pass JS Float64Arrays in, get a JS object back.
//     Simpler for non-real-time use (e.g. offline file processing).
//
//       const result = proc.processArrays(realIn, imagIn, alpha);
//       result.real   // Float64Array
//       result.imag   // Float64Array
// ============================================================================

class FRFTProcessor {
public:
    FRFTProcessor() = default;

    // Pre-allocate internal buffers for a given block size.
    // Call once before the first process() at that size.
    void prepare(int size) {
        if (size == size_) return;
        in_real_.assign(size, 0.0);
        in_imag_.assign(size, 0.0);
        out_real_.assign(size, 0.0);
        out_imag_.assign(size, 0.0);
        engine_.prepare(size);
        size_ = size;
    }

    // Run the FRFT on the current contents of the input buffers.
    // Returns false if size is 0 or odd.
    bool process(int size, double alpha) {
        if (size != size_) prepare(size);
        return engine_.compute(in_real_.data(), in_imag_.data(),
                               out_real_.data(), out_imag_.data(),
                               size, alpha);
    }

    // ---- Zero-copy buffer access (pattern A) --------------------------------

    // Byte offsets into the WASM heap — use with Module.HEAPF64:
    //   new Float64Array(Module.HEAPF64.buffer, proc.inputRealPtr(), size)
    uintptr_t inputRealPtr()  const { return reinterpret_cast<uintptr_t>(in_real_.data()); }
    uintptr_t inputImagPtr()  const { return reinterpret_cast<uintptr_t>(in_imag_.data()); }
    uintptr_t outputRealPtr() const { return reinterpret_cast<uintptr_t>(out_real_.data()); }
    uintptr_t outputImagPtr() const { return reinterpret_cast<uintptr_t>(out_imag_.data()); }

    int currentSize() const { return size_; }

    // ---- Convenience method (pattern B) -------------------------------------

    // Accepts Float64Arrays from JS, returns { real: Float64Array, imag: Float64Array }.
    val processArrays(val real_in_js, val imag_in_js, double alpha) {
        const int size = real_in_js["length"].as<int>();
        prepare(size);

        // Copy JS typed arrays into internal buffers
        val heap      = val::module_property("HEAPF64");
        val heap_buf  = heap["buffer"];

        // Write input into WASM heap via typed memory view
        for (int i = 0; i < size; ++i) {
            in_real_[i] = real_in_js[i].as<double>();
            in_imag_[i] = imag_in_js[i].as<double>();
        }

        engine_.compute(in_real_.data(), in_imag_.data(),
                        out_real_.data(), out_imag_.data(),
                        size, alpha);

        // Build result Float64Arrays from output buffers
        val Float64Array = val::global("Float64Array");
        val out_real_view = Float64Array.new_(
            typed_memory_view(size, out_real_.data()));
        val out_imag_view = Float64Array.new_(
            typed_memory_view(size, out_imag_.data()));

        val result = val::object();
        result.set("real", out_real_view);
        result.set("imag", out_imag_view);
        return result;
    }

private:
    FRFTEngine engine_;
    std::vector<double> in_real_, in_imag_, out_real_, out_imag_;
    int size_ = 0;
};

// ============================================================================
// Emscripten bindings
// ============================================================================

EMSCRIPTEN_BINDINGS(frft) {
    class_<FRFTProcessor>("FRFTProcessor")
        .constructor<>()
        .function("prepare",       &FRFTProcessor::prepare)
        .function("process",       &FRFTProcessor::process)
        .function("inputRealPtr",  &FRFTProcessor::inputRealPtr)
        .function("inputImagPtr",  &FRFTProcessor::inputImagPtr)
        .function("outputRealPtr", &FRFTProcessor::outputRealPtr)
        .function("outputImagPtr", &FRFTProcessor::outputImagPtr)
        .function("currentSize",   &FRFTProcessor::currentSize)
        .function("processArrays", &FRFTProcessor::processArrays)
        ;
}
