#include "c74_min.h"
#include "frft_engine.h"

using namespace c74::min;

// Define t_pfftpub structure for pfft~ access
// Based on Max SDK's r_pfft.h - field order matters!
// Memory layout determined from actual pfft~ object
extern "C" {
    typedef struct _pfftpub {
        c74::max::t_pxobject x_obj;  // indices 0-7 (t_pxobject is large)
        void* x_dspchain;             // index 8
        void* x_args;                 // index 9
        long x_extras;                // index 10
        long x_fftsize;               // index 11 - FFT size (e.g., 1024)
        long x_hopsize;               // index 12 - Hop size (e.g., 256)
        long x_phase;                 // index 13
        long x_unknown;               // index 14
        long x_mode;                  // index 15 - Mode: 0=half-frame, 1=full-frame
        // We don't need the rest of the fields
    } t_pfftpub;
}

class frft : public object<frft>, public vector_operator<> {
private:
    FRFTEngine engine;
    bool initialized = false;
    bool in_pfft = false;
    int current_buffer_size = 0;
    long fft_size = 0;
    long half_frame_size = 0;
    long hop_size = 0;
    long pfft_mode = 0;
    long overlap_factor = 0;

    // Pre-allocated buffers for real-time processing
    std::vector<double> real_buffer;
    std::vector<double> imag_buffer;

public:
    MIN_DESCRIPTION{"Fractional Fourier Transform using native C++ implementation (pfft~ only)"};
    MIN_TAGS{"spectral, transform, frft, pfft"};
    MIN_AUTHOR{"YourName"};

    inlet<> real_in{this, "(signal) real part input", "signal"};
    inlet<> imag_in{this, "(signal) imaginary part input", "signal"};
    inlet<> alpha_msg_in{this, "(float) alpha parameter"};

    outlet<> real_out{this, "(signal) real part output", "signal"};
    outlet<> imag_out{this, "(signal) imag part output", "signal"};

    attribute<number> alpha{this, "alpha", 0.5,
        description{"Alpha parameter for FRFT (fractional order)"},
        range{-10.0, 10.0}
    };

    frft() {
        // Initialize common symbols (required for Max SDK integration)
        c74::max::common_symbols_init();

        // Check if running inside pfft~ using Max SDK method
        using namespace c74::max;
        t_pfftpub* pfft = (t_pfftpub*)gensym("__pfft~__")->s_thing;

        if (!pfft) {
            cerr << "❌ ERROR: frft~ must be used inside pfft~" << endl;
            cerr << "   This object will not function outside of pfft~" << endl;
            in_pfft = false;
            initialized = false;
            return;
        }

        // Successfully detected pfft~ - extract and store all settings
        in_pfft = true;
        fft_size = pfft->x_fftsize;
        hop_size = pfft->x_hopsize;
        pfft_mode = pfft->x_mode;

        // Calculate half-frame size based on mode
        // mode 0 = half-frame (fft_size/2), mode 1 = full-frame (fft_size)
        half_frame_size = (pfft_mode == 0) ? (fft_size / 2) : fft_size;

        // Calculate overlap factor: overlap = fft_size / hop_size
        overlap_factor = (hop_size > 0) ? (fft_size / hop_size) : 1;

        // Print pfft~ settings
        cout << "✅ frft~ loaded in pfft~" << endl;
        cout << "   FFT Size: " << fft_size << endl;
        cout << "   Spectrum Mode: " << (pfft_mode == 0 ? "Half Spectrum" : "Full Spectrum") << endl;
        cout << "   Frame Size: " << half_frame_size << endl;
        cout << "   Hop Size: " << hop_size << endl;
        cout << "   Overlap Factor: " << overlap_factor << endl;

        initialized = true;
    }

    message<> float_input{this, "float", "Set alpha parameter",
        MIN_FUNCTION {
            if (args.size() > 0) {
                alpha = args[0];
            }
            return {};
        }
    };

    message<> status{this, "status", "Print engine status", MIN_FUNCTION {
        if (!in_pfft) {
            cout << "❌ ERROR: Not running in pfft~" << endl;
            cout << "   frft~ requires pfft~ to operate" << endl;
        } else if (initialized) {
            cout << "✅ frft~ running in pfft~" << endl;
            cout << "\nPFFT~ Settings:" << endl;
            cout << "   FFT Size: " << fft_size << endl;
            cout << "   Spectrum Mode: " << (pfft_mode == 0 ? "Half Spectrum" : "Full Spectrum") << endl;
            cout << "   Frame Size: " << half_frame_size << endl;
            cout << "   Hop Size: " << hop_size << endl;
            cout << "   Overlap Factor: " << overlap_factor << endl;
            cout << "\nFRFT Engine:" << endl;
            cout << "   Current buffer size: " << current_buffer_size << endl;
            cout << "   Current alpha: " << double(alpha) << endl;
        } else {
            cout << "❌ Engine not initialized" << endl;
        }
        return {};
    }};

private:
    void ensure_buffer_size(int vs) {
        if (current_buffer_size != vs) {
            real_buffer.resize(vs);
            imag_buffer.resize(vs);

            // Pre-create FFTW plans for this buffer size
            engine.prepare(vs);

            current_buffer_size = vs;
        }
    }

public:
    void operator()(audio_bundle input, audio_bundle output) {
        auto in_real = input.samples(0);
        auto in_imag = input.samples(1);
        auto out_real = output.samples(0);
        auto out_imag = output.samples(1);

        int vs = input.frame_count();

        // Check if we're in pfft~ before processing
        if (!in_pfft) {
            // Zero output if not in pfft~
            std::fill(out_real, out_real + vs, 0.0);
            std::fill(out_imag, out_imag + vs, 0.0);
            return;
        }

        if (!initialized) {
            // Pass through if not initialized
            std::copy(in_real, in_real + vs, out_real);
            std::copy(in_imag, in_imag + vs, out_imag);
            return;
        }

        // Check if buffer size is even
        if (vs % 2 != 0) {
            static bool error_printed = false;
            if (!error_printed) {
                cerr << "❌ Vector size must be even, got: " << vs << endl;
                cerr << "   Passing signal through unchanged." << endl;
                error_printed = true;
            }
            std::copy(in_real, in_real + vs, out_real);
            std::copy(in_imag, in_imag + vs, out_imag);
            return;
        }

        try {
            // Ensure buffers are allocated
            ensure_buffer_size(vs);

            // Get alpha parameter
            double alpha_param = static_cast<double>(alpha);

            // Determine if we're in half-spectrum mode
            bool is_half_spectrum = (pfft_mode == 0);

            // Compute FRFT
            bool success = engine.compute(
                in_real, in_imag,
                out_real, out_imag,
                vs, alpha_param, is_half_spectrum
            );

            if (!success) {
                static bool compute_error_printed = false;
                if (!compute_error_printed) {
                    cerr << "❌ FRFT computation failed" << endl;
                    compute_error_printed = true;
                }
                // Pass through on error
                std::copy(in_real, in_real + vs, out_real);
                std::copy(in_imag, in_imag + vs, out_imag);
            }
        }
        catch (const std::exception& e) {
            static bool exception_printed = false;
            if (!exception_printed) {
                cerr << "❌ FRFT exception: " << e.what() << endl;
                exception_printed = true;
            }
            // Pass through on exception
            std::copy(in_real, in_real + vs, out_real);
            std::copy(in_imag, in_imag + vs, out_imag);
        }
    }
};

MIN_EXTERNAL(frft);