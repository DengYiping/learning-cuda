#include "online_softmax.hpp"
#include "safe_softmax.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("safe_softmax", &safe_softmax, "Safe softmax ops");
  m.def("online_softmax", &online_softmax, "Online softmax ops");
}
