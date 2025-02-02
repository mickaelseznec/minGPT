#include "gpt.h"
#include <torch/extension.h>

PYBIND11_MODULE(gpt, m) {
  m.def("layernorm", &layernorm, "LayerNorm");
  m.def("gelu", &gelu, "GeLU");
  m.def("linear", &linear, "Linear");
}
