#pragma once
#include <torch/extension.h>

void layernorm(torch::Tensor out, torch::Tensor in, torch::Tensor weights,
               torch::Tensor bias, float eps);

void gelu(torch::Tensor out, torch::Tensor in);

void linear(torch::Tensor out, torch::Tensor in, torch::Tensor weights,
            torch::Tensor bias);
