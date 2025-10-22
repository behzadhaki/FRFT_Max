#include "c74_min.h"
#include <torch/script.h>
#include <vector>

using namespace c74::min;

class frft : public object<frft>, public vector_operator<> {
public:
    MIN_DESCRIPTION{"Fractional Fourier Transform using PyTorch model"};
    MIN_TAGS{"spectral, transform, torch"};
    MIN_AUTHOR{"YourName"};

    inlet<> real_in{this, "(signal) real part input", "signal"};
    inlet<> imag_in{this, "(signal) imaginary part input", "signal"};
    inlet<> alpha_msg_in{this, "(float) alpha parameter"};  // Message inlet for alpha

    outlet<> real_out{this, "(signal) real part output", "signal"};
    outlet<> imag_out{this, "(signal) imaginary part output", "signal"};

    attribute<number> alpha{this, "alpha", 0.5,
        description{"Alpha parameter for FRFT"}
    };

    frft() {
        cout << "FRFT external initialized. Use 'modelpath <path>' to load model." << endl;
    }

    // Handle float messages on the alpha inlet
    message<> float_input{this, "float", "Set alpha parameter",
        MIN_FUNCTION {
            if (args.size() > 0) {
                alpha = args[0];
                cout << "Alpha set to: " << double(alpha) << endl;
            }
            return {};
        }
    };

    message<> modelpath{this, "modelpath", "Load model from absolute path", MIN_FUNCTION {
        if (args.empty()) {
            cerr << "modelpath requires a path argument" << endl;
            return {};
        }

        try {
            std::string model_path = std::string(args[0]);
            cout << "Loading model from: " << model_path << endl;

            model = torch::jit::load(model_path);
            model.eval();

            cout << "✅ FRFT model loaded successfully" << endl;
            model_loaded = true;
        }
        catch (const std::exception& e) {
            cerr << "❌ Failed to load FRFT model: " << e.what() << endl;
            model_loaded = false;
        }

        return {};
    }};

    message<> status{this, "status", "Print model status", MIN_FUNCTION {
        if (model_loaded) {
            cout << "✅ FRFT model is loaded and ready" << endl;
        } else {
            cout << "❌ No model loaded" << endl;
        }
        return {};
    }};

private:
    torch::jit::script::Module model;
    bool model_loaded = false;

public:
    void operator()(audio_bundle input, audio_bundle output) {
        auto in_real = input.samples(0);
        auto in_imag = input.samples(1);
        auto out_real = output.samples(0);
        auto out_imag = output.samples(1);

        int vs = input.frame_count();

        if (!model_loaded) {
            // If model not loaded, pass through input
            for (int i = 0; i < vs; i++) {
                out_real[i] = in_real[i];
                out_imag[i] = in_imag[i];
            }
            return;
        }

        // Check if vector size is even (required by FRFT)
        if (vs % 2 != 0) {
            cerr << "❌ Vector size must be even, got: " << vs << endl;
            for (int i = 0; i < vs; i++) {
                out_real[i] = in_real[i];
                out_imag[i] = in_imag[i];
            }
            return;
        }

        try {
            // Create input tensors from audio buffers
            std::vector<float> real_vec(vs);
            std::vector<float> imag_vec(vs);

            for (int i = 0; i < vs; i++) {
                real_vec[i] = static_cast<float>(in_real[i]);
                imag_vec[i] = static_cast<float>(in_imag[i]);
            }

            // Convert to torch tensors
            torch::Tensor real_tensor = torch::from_blob(
                real_vec.data(),
                {vs},
                torch::kFloat32
            ).clone();

            torch::Tensor imag_tensor = torch::from_blob(
                imag_vec.data(),
                {vs},
                torch::kFloat32
            ).clone();

            // Use the alpha attribute (set via the message inlet)
            float alpha_param = static_cast<float>(alpha);

            // Run inference
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(real_tensor);
            inputs.push_back(imag_tensor);
            inputs.push_back(alpha_param);

            auto output_tuple = model.forward(inputs).toTuple();

            // Extract real and imag outputs
            torch::Tensor result_real = output_tuple->elements()[0].toTensor();
            torch::Tensor result_imag = output_tuple->elements()[1].toTensor();

            // Convert back to audio buffer
            auto real_accessor = result_real.accessor<float, 1>();
            auto imag_accessor = result_imag.accessor<float, 1>();

            for (int i = 0; i < vs; i++) {
                out_real[i] = static_cast<double>(real_accessor[i]);
                out_imag[i] = static_cast<double>(imag_accessor[i]);
            }
        }
        catch (const std::exception& e) {
            cerr << "❌ FRFT inference error: " << e.what() << endl;
            // Pass through on error
            for (int i = 0; i < vs; i++) {
                out_real[i] = in_real[i];
                out_imag[i] = in_imag[i];
            }
        }
    }
};

MIN_EXTERNAL(frft);