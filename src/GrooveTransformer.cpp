#include "c74_min.h"
#include <torch/script.h>
#include <cmath>

using namespace c74::min;

class GrooveTransformer : public object<GrooveTransformer> {
public:
    MIN_DESCRIPTION{"Groove decoder using TorchScript"};
    MIN_TAGS{"groove, torch, generation"};
    MIN_AUTHOR{"MusicTechnologyGroup"};

    inlet<> input{this, "(bang) trigger"};
    inlet<> latent_in{this, "(list) set latent vector"};
    inlet<> groove_in{this, "(list) set groove vector"};


    outlet<> kick_out{this, "(tuple) kick pattern, velocity, microtiming"};
    outlet<> snare_out{this, "(tuple) snare pattern, velocity, microtiming"};
    outlet<> chat_out{this, "(tuple) closed-hat pattern, velocity, microtiming"};
    outlet<> ohat_out{this, "(tuple) open-hat pattern, velocity, microtiming"};
    outlet<> ltom_out{this, "(tuple) low-tom pattern, velocity, microtiming"};
    outlet<> mtom_out{this, "(tuple) mid-tom pattern, velocity, microtiming"};
    outlet<> htom_out{this, "(tuple) high-tom pattern, velocity, microtiming"};
    outlet<> crash_out{this, "(tuple) crash pattern, velocity, microtiming"};
    outlet<> ride_out{this, "(tuple) ride pattern, velocity, microtiming"};
    outlet<> latent_out{this, "(list) current latent vector"};

    torch::jit::script::Module model;
    bool model_loaded = false;
    static constexpr int latent_dim = 128;

    torch::Tensor latent = torch::randn({1, latent_dim});
    torch::Tensor redir_kick = torch::tensor(0, torch::kLong);
    torch::Tensor redir_snare = torch::tensor(0, torch::kLong);
    torch::Tensor redir_hats = torch::tensor(0, torch::kLong);
    torch::Tensor redir_toms = torch::tensor(0, torch::kLong);
    torch::Tensor redir_cymbals = torch::tensor(0, torch::kLong);
    torch::Tensor genre = torch::tensor(0, torch::kLong);
    torch::Tensor per_voice_thresholds = torch::full({9}, 0.5f, torch::kFloat32);
    torch::Tensor per_voice_max_counts = torch::full({9}, 32, torch::kInt64);

    torch::Tensor groove_hvo = torch::zeros({1, 32, 3}, torch::kFloat32);

    float out_vel_threshold = 0.0f;
    bool render_on_any_change = false;

    GrooveTransformer() {
        output_latent_vector();
    }

    void load_model(const std::string& path) {
        try {
            model = torch::jit::load(path, torch::kCPU);
            model.eval();
            model_loaded = true;
        } catch (const c10::Error& e) {
            std::cerr << "❌ Failed to load model: " << e.what() << std::endl;
        }
    }

    void trigger_if_enabled() {
        try {
            if (render_on_any_change) {
                render_pattern();
                output_latent_vector();
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ trigger_if_enabled error: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "❌ trigger_if_enabled unknown error." << std::endl;
        }
    }

    void output_latent_vector() {
        try {
            atoms latent_list;
            auto flat = latent.squeeze();
            for (int i = 0; i < flat.size(0); ++i)
                latent_list.push_back(atom(flat[i].item<float>()));
            latent_out.send(latent_list);
        } catch (const std::exception& e) {
            std::cerr << "❌ output_latent_vector error: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "❌ output_latent_vector unknown error." << std::endl;
        }
    }

    float round_to_2_decimal(float value) {
        return std::round(value * 100.0f) / 100.0f;
    }

    void render_pattern() {
        if (!model_loaded) {
            std::cerr << "❌ Model not loaded." << std::endl;
            return;
        }

        try {
            std::vector<torch::jit::IValue> inputs{
                latent, genre,
                redir_kick, redir_snare, redir_hats, redir_toms, redir_cymbals,
                per_voice_thresholds, per_voice_max_counts
            };

            auto tup = model.get_method("sample")(inputs).toTuple();
            auto hits = tup->elements()[0].toTensor().squeeze(0);
            auto velocities = tup->elements()[1].toTensor().squeeze(0);
            auto microtimings = tup->elements()[2].toTensor().squeeze(0);

            outlet<>* outs[9] = {&kick_out, &snare_out, &chat_out, &ohat_out, &ltom_out,
                                 &mtom_out, &htom_out, &crash_out, &ride_out};

            int T = hits.size(0);
            for (int v = 0; v < 9; ++v) {
                std::vector<float> time_vel_pairs;
                for (int t = 0; t < T; ++t) {
                    float hit = hits[t][v].item<float>();
                    float velocity = velocities[t][v].item<float>();
                    float microtime = microtimings[t][v].item<float>();

                    float actual_time = (t + microtime) * 0.25f;
                    if (actual_time < 0) actual_time += 8;
                    else if (actual_time >= 8) actual_time -= 8;
                    actual_time /= 8;

                    actual_time = round_to_2_decimal(actual_time);
                    if (hit != 0 && velocity != 0 && velocity >= out_vel_threshold) {
                        time_vel_pairs.push_back(actual_time);
                        time_vel_pairs.push_back(velocity);
                    }
                }

                if (!time_vel_pairs.empty()) {
                    atoms time_vel_list;
                    for (const auto& pair : time_vel_pairs)
                        time_vel_list.push_back(atom(pair));
                    outs[v]->send(time_vel_list);
                } else {
                    outs[v]->send(atom("clear"));
                }
            }
        } catch (const c10::Error& e) {
            std::cerr << "❌ TorchScript forward error: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "❌ render_pattern error: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "❌ render_pattern unknown error." << std::endl;
        }
    }

    message<> bang{
        this, "bang", "Render using current latent",
        MIN_FUNCTION {
            try {
                render_pattern();
                output_latent_vector();
            } catch (const std::exception& e) {
                std::cerr << "❌ bang error: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "❌ bang unknown error." << std::endl;
            }
            return {};
        }
    };

    message<> random_pattern{
        this, "random_pattern", "Generate random latent and render",
        MIN_FUNCTION {
            try {
                latent = torch::randn({1, latent_dim});
                render_pattern();
                output_latent_vector();
            } catch (const std::exception& e) {
                std::cerr << "❌ random_pattern error: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "❌ random_pattern unknown error." << std::endl;
            }
            return {};
        }
    };

    message<> output_latent{
        this, "output_latent", "Output latent vector as list",
        MIN_FUNCTION {
            try {
                output_latent_vector();
            } catch (const std::exception& e) {
                std::cerr << "❌ output_latent error: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "❌ output_latent unknown error." << std::endl;
            }
            return {};
        }
    };

    message<> render_on_change_set{
        this, "render_on_any_change", "Enable/disable auto-render",
        MIN_FUNCTION {
            try {
                if (!args.empty())
                    render_on_any_change = static_cast<bool>(args[0]);
            } catch (const std::exception& e) {
                std::cerr << "❌ render_on_any_change error: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "❌ render_on_any_change unknown error." << std::endl;
            }
            return {};
        }
    };

#define SAFE_MSG_HANDLER(name, stmt) \
    MIN_FUNCTION { \
        try { stmt } \
        catch (const std::exception& e) { std::cerr << "❌ " #name " error: " << e.what() << std::endl; } \
        catch (...) { std::cerr << "❌ " #name " unknown error." << std::endl; } \
        return {}; \
    }

    message<> genre_set{
        this, "genre", "Set genre index",
        SAFE_MSG_HANDLER(genre_set,
            if (!args.empty()) genre = torch::tensor(static_cast<int64_t>(args[0]), torch::kLong);
            trigger_if_enabled();
        )
    };

    message<> redir_kick_set{
        this, "redir_kick", "Redirect kick",
        SAFE_MSG_HANDLER(redir_kick_set,
            redir_kick = torch::tensor(static_cast<int64_t>(args[0]), torch::kLong);
            trigger_if_enabled();
        )
    };

    message<> redir_snare_set{
        this, "redir_snare", "Redirect snare",
        SAFE_MSG_HANDLER(redir_snare_set,
            redir_snare = torch::tensor(static_cast<int64_t>(args[0]), torch::kLong);
            trigger_if_enabled();
        )
    };

    message<> redir_hats_set{
        this, "redir_hats", "Redirect hats",
        SAFE_MSG_HANDLER(redir_hats_set,
            redir_hats = torch::tensor(static_cast<int64_t>(args[0]), torch::kLong);
            trigger_if_enabled();
        )
    };

    message<> redir_toms_set{
        this, "redir_toms", "Redirect toms",
        SAFE_MSG_HANDLER(redir_toms_set,
            redir_toms = torch::tensor(static_cast<int64_t>(args[0]), torch::kLong);
            trigger_if_enabled();
        )
    };

    message<> redir_cymbals_set{
        this, "redir_cymbals", "Redirect cymbals",
        SAFE_MSG_HANDLER(redir_cymbals_set,
            redir_cymbals = torch::tensor(static_cast<int64_t>(args[0]), torch::kLong);
            trigger_if_enabled();
        )
    };

    message<> model_path_set{
        this, "model_path", "Set path to model file",
        SAFE_MSG_HANDLER(model_path_set,
            if (!args.empty()) {
                std::string model_path = static_cast<std::string>(args[0]);
                load_model(model_path);
            }
        )
    };

    message<> out_vel_threshold_set{
        this, "out_vel_threshold", "Set velocity threshold",
        SAFE_MSG_HANDLER(out_vel_threshold_set,
            if (!args.empty())
                out_vel_threshold = static_cast<float>(args[0]);
            trigger_if_enabled();
        )
    };

#define THRESH_SETTER(idx, name)                                          \
    message<> thresh_##name{                                              \
        this, "thresh_" #name, "Set threshold for " #name,                \
        SAFE_MSG_HANDLER(thresh_##name,                                   \
            per_voice_thresholds[idx] = static_cast<float>(args[0]);      \
            trigger_if_enabled();                                         \
        )                                                                 \
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

#define MAX_SETTER(idx, name)                                             \
    message<> max_##name##_events{                                        \
        this, "max_" #name "_events", "Set max events for " #name,        \
        SAFE_MSG_HANDLER(max_##name##_events,                             \
            per_voice_max_counts[idx] = static_cast<int64_t>(args[0]);    \
            trigger_if_enabled();                                         \
        )                                                                 \
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

    message<> list{
        this, "list", "Receive latent vector (inlet 1) or groove input (inlet 2)",
        MIN_FUNCTION {
            try {
                // Inlet 1: latent vector
                if (inlet == 1) {
                    if (args.size() != latent_dim) {
                        std::cerr << "❌ Expected " << latent_dim
                                  << " values, got " << args.size() << std::endl;
                        return {};
                    }

                    std::vector<float> buf(latent_dim);
                    for (size_t i = 0; i < latent_dim; ++i)
                        buf[i] = static_cast<float>(args[i]);

                    latent = torch::tensor(buf, torch::kFloat32).view({1, latent_dim}).clone();

                    render_pattern();
                    output_latent_vector();
                }

                // Inlet 2: groove pattern (time, vel) pairs
                else if (inlet == 2) {
                    if (args.size() % 2 != 0) {
                        std::cerr << "❌ Expected pairs of (time, velocity), got odd number of elements." << std::endl;
                        return {};
                    }

                    if (!model_loaded) {
                        std::cerr << "❌ Model not loaded, cannot encode or decode." << std::endl;
                        return {};
                    }

                    // Zero existing pattern
                    groove_hvo.zero_();

                    for (size_t i = 0; i < args.size(); i += 2) {
                        float time = static_cast<float>(args[i]);
                        float velocity = static_cast<float>(args[i + 1]);

                        if (time < 0.0f || time >= 1.0f) continue;

                        const float step_dur = 1.0f / 32.0f;
                        int step_index = static_cast<int>(std::round(time / step_dur)) % 32;
                        float center_time = step_index * step_dur;

                        // Compute circular time difference (wraparound at 1.0)
                        float time_diff = time - center_time;
                        if (time_diff >  0.5f) time_diff -= 1.0f;
                        if (time_diff < -0.5f) time_diff += 1.0f;

                        float microtiming = time_diff / step_dur;  // normalize to [-0.5, 0.5]

                        std::cout << "🟡 Time: " << time
                                  << " → Step: " << step_index
                                  << ", Center: " << center_time
                                  << ", Microtiming: " << microtiming << std::endl;

                        groove_hvo[0][step_index][0] = 1.0f;
                        groove_hvo[0][step_index][1] = velocity;
                        groove_hvo[0][step_index][2] = microtiming;
                    }

                    // Encode
                    auto encode = model.get_method("encode_all");
                    auto enc_inputs = std::vector<torch::jit::IValue>{groove_hvo};
                    auto encoder_output = encode(enc_inputs);
                    latent = encoder_output.toTuple()->elements()[0].toTensor();

                    output_latent_vector();

                    // Decode
                    render_pattern();
                }
            } catch (const std::exception& e) {
                std::cerr << "❌ list error: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "❌ list unknown error." << std::endl;
            }

            return {};
        }
    };


};

MIN_EXTERNAL(GrooveTransformer);

