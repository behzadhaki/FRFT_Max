#pragma once
#include <torch/script.h>
#include <cmath>

constexpr int LATENT_DIM = 128;

inline float round_to_2_decimal(float value) {
    return std::round(value * 100.0f) / 100.0f;
}
