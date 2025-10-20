#include "c74_min.h"
#include <torch/script.h>
#include "groove_shared.h"
#include <thread>
#include <atomic>
#include <chrono>

using namespace c74::min;

class FrftIn : public object<FrftIn>, public vector_operator<> {

public:
    MIN_DESCRIPTION{"Groove encoder using TorchScript"};
    MIN_TAGS{"groove, encoder, torch"};
    MIN_AUTHOR{"MusicTechnologyGroup"};

    inlet<> groove_in{this, "(list) input groove vector"};
    outlet<> latent_out{this, "(list) output latent vector"};

    torch::jit::script::Module model;
    torch::Tensor groove_hvo = torch::zeros({1, 32, 3}, torch::kFloat32);
    bool model_loaded = false;

    fifo<torch::Tensor> groove_queue{8192};
    fifo<torch::Tensor> latent_queue{8192};
    fifo <std::string> model_path_queue{4}; // to background thread

    std::thread worker_thread;
    std::atomic<bool> stop_thread{false};

    FrftIn() {
        worker_thread = std::thread([this]() {

            std::string dequeued_model_path = "";
            torch::Tensor dequeued_groove;


            while (!stop_thread.load()) {
                // check if model path is available, load the model if it is
                if (model_path_queue.try_dequeue(dequeued_model_path)) {
                    try {
                        load_model(dequeued_model_path);
                    } catch (const std::exception& e) {
                        std::cerr << "❌ Decoder model load error: " << e.what() << std::endl;
                    }
                }

                // if model is not loaded, wait for it to be loaded
                if (!model_loaded) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                // check if groove input is available, encode it if it is
                if (groove_queue.try_dequeue(dequeued_groove)) {
                    auto encode = model.get_method("encode_all");
                    auto latent = encode({dequeued_groove}).toTuple()->elements()[0].toTensor();
                    latent_queue.enqueue(latent.clone());
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        });
    }

    ~FrftIn() {
        stop_thread = true;
        if (worker_thread.joinable())
            worker_thread.join();
    }

    void load_model(const std::string& path) {
        try {
            model = torch::jit::load(path, torch::kCPU);
            model.eval();
            model_loaded = true;
        } catch (const c10::Error& e) {
            std::cerr << "❌ Encoder: Failed to load model: " << e.what() << std::endl;
        }
    }

    message<> model_path{this, "model_path", "Load TorchScript model",
        MIN_FUNCTION {
            std::cout << "🔄  model request at: " << args[0] << std::endl;
            // push the model path to the queue
            model_path_queue.enqueue(args[0]);
            return {};
        }
    };

    message<> list{
        this, "list", "Receive groove input",
        MIN_FUNCTION {

            if (args.size() % 2 != 0) {
                std::cerr << "❌ Encoder: Input must be time/velocity pairs." << std::endl;
                return {};
            }

            groove_hvo.zero_();
            for (size_t i = 0; i < args.size(); i += 2) {
                float time = static_cast<float>(args[i]);
                float velocity = static_cast<float>(args[i + 1]);

                if (time < 0.0f || time >= 1.0f) continue;

                const float step_dur = 1.0f / 32.0f;
                int step_index = static_cast<int>(std::round(time / step_dur)) % 32;
                float center_time = step_index * step_dur;

                float time_diff = time - center_time;
                if (time_diff > 0.5f) time_diff -= 1.0f;
                if (time_diff < -0.5f) time_diff += 1.0f;

                float microtiming = time_diff / step_dur;

                groove_hvo[0][step_index][0] = 1.0f;
                groove_hvo[0][step_index][1] = velocity / 127.0f;
                groove_hvo[0][step_index][2] = microtiming;
            }

            groove_queue.enqueue(groove_hvo.clone());
            return {};
        }
    };

    void operator()(audio_bundle input, audio_bundle output) {
        // check if the latent queue has data to receive
        // only get the latest one and pop the rest if there are more
        torch::Tensor latent;
        if (latent_queue.try_dequeue(latent)) {
            std::vector<float> latent_vector(latent.data_ptr<float>(), latent.data_ptr<float>() + latent.numel());
            atoms latent_list;
            for (int i = 0; i < latent.size(1); ++i)
                latent_list.push_back(atom(latent[0][i].item<float>()));

            latent_out.send(latent_list);
        }
    }

};

MIN_EXTERNAL(FrftIn);
