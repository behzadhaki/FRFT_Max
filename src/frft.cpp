#include "c74_min.h"
#include <torch/script.h>
#include <vector>
#include "shared_external_helpers.h"

using namespace c74::min;

class frft : public object<frft>, public vector_operator<> {
private:
    torch::jit::script::Module model;
    bool model_loaded = false;

    // Pre-allocated tensors for reuse
    torch::Tensor real_tensor;
    torch::Tensor imag_tensor;

    // Pre-allocated IValue vector
    std::vector<torch::jit::IValue> inputs;

    int current_buffer_size = 0;

public:
    MIN_DESCRIPTION{"Fractional Fourier Transform using PyTorch model"};
    MIN_TAGS{"spectral, transform, torch"};
    MIN_AUTHOR{"YourName"};

    inlet<> real_in{this, "(signal) real part input", "signal"};
    inlet<> imag_in{this, "(signal) imaginary part input", "signal"};
    inlet<> alpha_msg_in{this, "(float) alpha parameter"};

    outlet<> real_out{this, "(signal) real part output", "signal"};
    outlet<> imag_out{this, "(signal) imag part output", "signal"};

    BundleResourceLoader resourceLoder;

    attribute<number> alpha{this, "alpha", 0.5,
        description{"Alpha parameter for FRFT"}
    };

    frft() {
        cout << "FRFT external initialized. Use 'modelpath <path>' to load model." << endl;
        try {
            std::string model_path = resourceLoder.get_resource_path("frft_continuous.ts");
            cout << "Loading model from: " << model_path << endl;

            model = torch::jit::load(model_path);
            model.eval();

            cout << "✅ FRFT model loaded successfully" << endl;
            model_loaded = true;

            current_buffer_size = 0;
        }
        catch (const std::exception& e) {
            cerr << "❌ Failed to load FRFT model: " << e.what() << endl;
            model_loaded = false;
        }
    }

    message<> float_input{this, "float", "Set alpha parameter",
        MIN_FUNCTION {
            if (args.size() > 0) {
                alpha = args[0];
            }
            return {};
        }
    };

    // message<> modelpath{this, "modelpath", "Load model from absolute path", MIN_FUNCTION {
    //     if (args.empty()) {
    //         cerr << "modelpath requires a path argument" << endl;
    //         return {};
    //     }
    //
    //     try {
    //         std::string model_path = std::string(args[0]);
    //         cout << "Loading model from: " << model_path << endl;
    //
    //         model = torch::jit::load(model_path);
    //         model.eval();
    //
    //         cout << "✅ FRFT model loaded successfully" << endl;
    //         model_loaded = true;
    //
    //         current_buffer_size = 0;
    //     }
    //     catch (const std::exception& e) {
    //         cerr << "❌ Failed to load FRFT model: " << e.what() << endl;
    //         model_loaded = false;
    //     }
    //
    //     return {};
    // }};

    message<> status{this, "status", "Print model status", MIN_FUNCTION {
        if (model_loaded) {
            cout << "✅ FRFT model is loaded and ready" << endl;
            cout << "   Current buffer size: " << current_buffer_size << endl;
        } else {
            cout << "❌ No model loaded" << endl;
        }
        return {};
    }};

private:

    void ensure_tensor_size(int vs) {
        if (current_buffer_size != vs) {
            // Allocate with explicit options to ensure proper memory layout
            auto options = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(torch::kCPU)
                .requires_grad(false);

            real_tensor = torch::empty({vs}, options);
            imag_tensor = torch::empty({vs}, options);

            inputs.clear();
            inputs.reserve(3);

            current_buffer_size = vs;

            cout << "Allocated tensors for buffer size: " << vs << endl;
        }
    }

public:
    void operator()(audio_bundle input, audio_bundle output) {
        auto in_real = input.samples(0);
        auto in_imag = input.samples(1);
        auto out_real = output.samples(0);
        auto out_imag = output.samples(1);

        int vs = input.frame_count();

        if (!model_loaded) {
            std::copy(in_real, in_real + vs, out_real);
            std::copy(in_imag, in_imag + vs, out_imag);
            return;
        }

        if (vs % 2 != 0) {
            static bool error_printed = false;
            if (!error_printed) {
                cerr << "❌ Vector size must be even, got: " << vs << endl;
                error_printed = true;
            }
            std::copy(in_real, in_real + vs, out_real);
            std::copy(in_imag, in_imag + vs, out_imag);
            return;
        }

        try {
            ensure_tensor_size(vs);

            // Use NoGradGuard - safer than InferenceMode, still provides speedup
            torch::NoGradGuard no_grad;

            // Direct memory access for fast copy
            float* real_ptr = real_tensor.data_ptr<float>();
            float* imag_ptr = imag_tensor.data_ptr<float>();

            // Copy input data
            for (int i = 0; i < vs; i++) {
                real_ptr[i] = static_cast<float>(in_real[i]);
                imag_ptr[i] = static_cast<float>(in_imag[i]);
            }

            float alpha_param = static_cast<float>(alpha);

            // Reuse inputs vector
            inputs.clear();
            inputs.push_back(real_tensor);
            inputs.push_back(imag_tensor);
            inputs.push_back(alpha_param);

            // Run inference
            auto output_tuple = model.forward(inputs).toTuple();

            // Extract outputs
            torch::Tensor result_real = output_tuple->elements()[0].toTensor();
            torch::Tensor result_imag = output_tuple->elements()[1].toTensor();

            // Ensure tensors are contiguous before accessing memory
            if (!result_real.is_contiguous()) {
                result_real = result_real.contiguous();
            }
            if (!result_imag.is_contiguous()) {
                result_imag = result_imag.contiguous();
            }

            // Direct pointer access for output
            const float* result_real_ptr = result_real.data_ptr<float>();
            const float* result_imag_ptr = result_imag.data_ptr<float>();

            // Copy output data
            for (int i = 0; i < vs; i++) {
                out_real[i] = static_cast<double>(result_real_ptr[i]);
                out_imag[i] = static_cast<double>(result_imag_ptr[i]);
            }
        }
        catch (const std::exception& e) {
            cerr << "❌ FRFT inference error: " << e.what() << endl;
            std::copy(in_real, in_real + vs, out_real);
            std::copy(in_imag, in_imag + vs, out_imag);
        }
    }
};

MIN_EXTERNAL(frft);