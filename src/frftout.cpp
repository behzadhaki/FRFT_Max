#include "c74_min.h"
#include <torch/script.h>
#include <cmath>

using namespace c74::min;

class FrftOut : public object<FrftOut>, public vector_operator<> {
public:
    MIN_DESCRIPTION{"Groove decoder using TorchScript"};
    MIN_TAGS{"groove, torch, generation"};
    MIN_AUTHOR{"MusicTechnologyGroup"};

    inlet<> latent_in{this, "(list) set latent vector"};

    outlet<> kick_out{this, "(tuple) kick pattern"};
    outlet<> snare_out{this, "(tuple) snare pattern"};
    outlet<> chat_out{this, "(tuple) closed hat pattern"};
    outlet<> ohat_out{this, "(tuple) open hat pattern"};
    outlet<> ltom_out{this, "(tuple) low tom pattern"};
    outlet<> mtom_out{this, "(tuple) mid tom pattern"};
    outlet<> htom_out{this, "(tuple) high tom pattern"};
    outlet<> crash_out{this, "(tuple) crash pattern"};
    outlet<> ride_out{this, "(tuple) ride pattern"};

    torch::jit::script::Module model;
    torch::Tensor latent = torch::randn({1, 128});
    torch::Tensor genre = torch::tensor(0, torch::kLong);
    torch::Tensor redir_kick = torch::tensor(0, torch::kLong);
    torch::Tensor redir_snare = torch::tensor(0, torch::kLong);
    torch::Tensor redir_hats = torch::tensor(0, torch::kLong);
    torch::Tensor redir_toms = torch::tensor(0, torch::kLong);
    torch::Tensor redir_cymbals = torch::tensor(0, torch::kLong);
    torch::Tensor per_voice_thresholds = torch::full({9}, 0.5f);
    torch::Tensor per_voice_max_counts = torch::full({9}, 32, torch::kInt64);

    float out_vel_threshold = 0.0f;
    bool model_loaded = false;

    // fifo for model path
    fifo <std::string> model_path_queue{4}; // to background thread
    fifo<std::vector<torch::jit::IValue>> latent_and_controls_queue{8192};             // to background thread
    fifo<std::vector<std::vector<float>>> output_queue{8192};                         // returns generations from background thread

    std::thread worker_thread;
    std::atomic<bool> stop_thread{false};

    FrftOut() {
        worker_thread = std::thread([this]() {

            std::string dequeued_model_path = "";
            std::vector<torch::jit::IValue> dequeued_latent_and_controls;

            while (!stop_thread.load()) {
                // check if model path is available, load the model if it is
                if (model_path_queue.try_dequeue(dequeued_model_path)) {
                    try {
                        load_model(dequeued_model_path);
                    } catch (const std::exception& e) {
                        std::cerr << "❌ Decoder model load error: " << e.what() << std::endl;
                    }
                }

                // make sure we have a model loaded
                if (!model_loaded) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }


                // check if multiple latent vectors are available, pop all but the latest one
                if (latent_and_controls_queue.try_dequeue(dequeued_latent_and_controls)) {
                    try {
                        auto sample = model.get_method("sample")(dequeued_latent_and_controls).toTuple();
                        auto hits = sample->elements()[0].toTensor().squeeze(0);
                        auto velocities = sample->elements()[1].toTensor().squeeze(0);
                        auto microtimings = sample->elements()[2].toTensor().squeeze(0);

                        std::vector<std::vector<float>> streams;
                        streams.resize(9);

                        for (int v = 0; v < 9; ++v) {
                            for (int t = 0; t < hits.size(0); ++t) {
                                float hit = hits[t][v].item<float>();
                                float vel = velocities[t][v].item<float>();
                                float micro = microtimings[t][v].item<float>();

                                if (hit == 0 || vel < out_vel_threshold) continue;

                                float time = (t + micro) * 0.25f;
                                if (time < 0) time += 8;
                                if (time >= 8) time -= 8;

                                time = round_to_2_decimal(time / 8.0f);
                                streams[v].push_back(time);
                                streams[v].push_back(vel);
                            }
                        }
                        output_queue.enqueue(streams);

                    } catch (const std::exception& e) {
                        std::cerr << "❌ Decoder render error: " << e.what() << std::endl;
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        });
    }

    ~FrftOut() {
        stop_thread = true;
        if (worker_thread.joinable())
            worker_thread.join();
    }

    void load_model(const std::string& path) {
        try {
            model = torch::jit::load(path, torch::kCPU);
            model.eval();
            model_loaded = true;
        } catch (const std::exception& e) {
            std::cerr << "❌ Failed to load model: " << e.what() << std::endl;
        }
    }

    float round_to_2_decimal(float value) {
        return std::round(value * 100.0f) / 100.0f;
    }

    void send_latent_and_controls_to_bg_thread() {
        try {
            std::vector<torch::jit::IValue> inputs{
                latent, genre,
                redir_kick, redir_snare, redir_hats, redir_toms, redir_cymbals,
                per_voice_thresholds, per_voice_max_counts
            };

            latent_and_controls_queue.enqueue(inputs);
        } catch (const std::exception& e) {
            std::cerr << "❌ Error sending latent and controls: " << e.what() << std::endl;
        }
    }

    message<> bang{this, "bang", "Render current latent vector",
        MIN_FUNCTION {
            send_latent_and_controls_to_bg_thread();
            return {};
        }
    };

    message<> list{this, "list", "Set latent vector",
        MIN_FUNCTION {
            if (args.size() != 128) {
                std::cerr << "❌ Expected 128 latent dimensions." << std::endl;
                return {};
            }
            std::vector<float> buf(128);
            for (size_t i = 0; i < 128; ++i)
                buf[i] = static_cast<float>(args[i]);
            latent = torch::tensor(buf, torch::kFloat32).view({1, 128}).clone();
            send_latent_and_controls_to_bg_thread();
            return {};
        }
    };

    void operator()(audio_bundle input, audio_bundle output) {
        // check if the output queue has data to send, if multiple outputs are available, pop all but the latest one
        std::vector<std::vector<float>> dequeued_stream;
        while (output_queue.try_dequeue(dequeued_stream)) {
            if (dequeued_stream.size() != 9) {
                std::cerr << "❌ Expected 9 output streams. Received: " << dequeued_stream.size() << std::endl;
                continue;
            }

            outlet<>* outs[9] = {&kick_out, &snare_out, &chat_out, &ohat_out,
                                 &ltom_out, &mtom_out, &htom_out, &crash_out, &ride_out};

            for (int v = 0; v < 9; ++v) {

                // if empty then clear the outlet
                if (dequeued_stream[v].empty()) {
                    outs[v]->send(atom("clear"));
                    continue;
                }

                if (dequeued_stream[v].size() % 2 != 0) {
                    std::cerr << "❌ Output stream must be time/velocity pairs." << std::endl;
                    continue;
                }

                // else send the stream to the corresponding outlet
                atoms outlist;
                for (size_t i = 0; i < dequeued_stream[v].size(); i += 2) {
                    float time = dequeued_stream[v][i];
                    float velocity = dequeued_stream[v][i + 1];
                    outlist.push_back(atom(time));
                    outlist.push_back(atom(velocity));
                }
                outs[v]->send(outlist);
            }
        }
    }







#define CONTROL_PARAM(param_name, tensor_var, cast_type) \
    message<> param_name##_set{this, #param_name, "Set " #param_name, \
    MIN_FUNCTION { \
    try { \
    if (!args.empty()) tensor_var = torch::tensor(static_cast<cast_type>(args[0]), torch::kLong); \
    send_latent_and_controls_to_bg_thread(); \
    } catch (...) { std::cerr << "❌ " #param_name " error." << std::endl; } \
    return {}; \
    } \
    }


    CONTROL_PARAM(genre, genre, int64_t);
    CONTROL_PARAM(redir_kick, redir_kick, int64_t);
    CONTROL_PARAM(redir_snare, redir_snare, int64_t);
    CONTROL_PARAM(redir_hats, redir_hats, int64_t);
    CONTROL_PARAM(redir_toms, redir_toms, int64_t);
    CONTROL_PARAM(redir_cymbals, redir_cymbals, int64_t);

    message<> out_vel_threshold_set{this, "out_vel_threshold", "Set output velocity threshold",
        MIN_FUNCTION {
            try { out_vel_threshold = static_cast<float>(args[0]); send_latent_and_controls_to_bg_thread(); }
            catch (...) { std::cerr << "❌ out_vel_threshold error." << std::endl; }
            return {};
        }
    };

#define THRESH_SETTER(index, name) \
    message<> thresh_##name{this, "thresh_" #name, "Set threshold for " #name, \
        MIN_FUNCTION { \
            try { per_voice_thresholds[index] = static_cast<float>(args[0]); send_latent_and_controls_to_bg_thread(); } \
            catch (...) { std::cerr << "❌ thresh_" #name " error." << std::endl; } \
            return {}; \
        } \
    }

    THRESH_SETTER(0, kick);
    THRESH_SETTER(1, snare);
    THRESH_SETTER(2, chat);
    THRESH_SETTER(3, ohat);
    THRESH_SETTER(4, ltom);
    THRESH_SETTER(5, mtom);
    THRESH_SETTER(6, htom);
    THRESH_SETTER(7, crash);
    THRESH_SETTER(8, ride);

#define MAX_SETTER(index, name) \
    message<> max_##name##_events{this, "max_" #name "_events", "Set max count for " #name, \
        MIN_FUNCTION { \
            try { per_voice_max_counts[index] = static_cast<int64_t>(args[0]); send_latent_and_controls_to_bg_thread(); } \
            catch (...) { std::cerr << "❌ max_" #name " error." << std::endl; } \
            return {}; \
        } \
    }

    MAX_SETTER(0, kick);
    MAX_SETTER(1, snare);
    MAX_SETTER(2, chat);
    MAX_SETTER(3, ohat);
    MAX_SETTER(4, ltom);
    MAX_SETTER(5, mtom);
    MAX_SETTER(6, htom);
    MAX_SETTER(7, crash);
    MAX_SETTER(8, ride);

    message<> model_path{this, "model_path", "Load TorchScript model",
        MIN_FUNCTION {
            std::cout << "🔄  model request at: " << args[0] << std::endl;
            // push the model path to the queue
            model_path_queue.enqueue(args[0]);
            return {};
        }
    };
};

MIN_EXTERNAL(FrftOut);
