/*
 * Copyright (c) 2024, Fireworks AI.
 */

#include <torch/extension.h>

#include "multi_gpu_barrier.h"

namespace amd_repro {
// These depend on py::bytes and can't be exposed as pytorch operators
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_barrier", create_barrier);
  py::class_<MultiGpuBarrier, std::shared_ptr<MultiGpuBarrier>>(
      m, "MultiGpuBarrier")
      .def(py::init<int, int, int>(), py::call_guard<py::gil_scoped_release>())
      .def("run", &MultiGpuBarrier::run,
           py::call_guard<py::gil_scoped_release>())
      .def("bufferHandle", &MultiGpuBarrier::bufferHandle,
           py::call_guard<py::gil_scoped_release>())
      .def("setPeerBufferHandles", &MultiGpuBarrier::setPeerBufferHandles,
           py::call_guard<py::gil_scoped_release>());
}
}  // namespace amd_repro