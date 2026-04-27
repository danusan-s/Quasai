#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quasai/core/shape.hpp"
#include "quasai/core/tensor.hpp"
#include "quasai/nn/activations.hpp"
#include "quasai/nn/linear.hpp"
#include "quasai/nn/loss.hpp"
#include "quasai/nn/model.hpp"
#include "quasai/nn/module.hpp"
#include "quasai/nn/sequential.hpp"
#include "quasai/optim/sgd.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyquasai, m) {
  // -------------------------
  // Tensor
  // -------------------------
  py::class_<quasai::Tensor, std::shared_ptr<quasai::Tensor>>(m, "Tensor")
      .def(py::init<>())
      .def("shape", [](const quasai::Tensor &t) {
        const auto &s = t.shape();
        std::vector<size_t> dims;
        for (size_t i = 0; i < s.dimensions(); ++i) {
          dims.push_back(s[i]);
        }
        return dims;
      });

  // Factory to create tensor from numpy array
  m.def("tensor", [](py::array_t<float> arr) {
    auto buf = arr.request();
    size_t ndim = static_cast<size_t>(buf.ndim);
    std::vector<size_t> sizes(ndim, 0);
    for (size_t i = 0; i < ndim; ++i) {
      sizes[i] = static_cast<size_t>(buf.shape[i]);
    }
    quasai::Shape shape(sizes.data(), ndim);
    auto t = quasai::Tensor::zeros(shape, quasai::DType::FLOAT32,
                                   quasai::Device::cpu());
    std::memcpy(t.data<float>(), buf.ptr, buf.size * sizeof(float));
    return t;
  });

  // -------------------------
  // Base Module (polymorphic)
  // -------------------------
  py::class_<quasai::Module, std::shared_ptr<quasai::Module>>(m, "Module")
      .def("__call__", &quasai::Module::operator());

  // -------------------------
  // Linear layer
  // -------------------------
  py::class_<quasai::Linear, quasai::Module, std::shared_ptr<quasai::Linear>>(
      m, "Linear")
      .def(py::init<int, int>());

  // -------------------------
  // ReLU activation
  // -------------------------
  py::class_<quasai::ReLU, quasai::Module, std::shared_ptr<quasai::ReLU>>(
      m, "ReLU")
      .def(py::init<>());

  // -------------------------
  // Sequential container
  // -------------------------
  py::class_<quasai::Sequential, quasai::Module,
             std::shared_ptr<quasai::Sequential>>(m, "Sequential")
      .def(py::init<std::vector<std::shared_ptr<quasai::Module>>>());

  py::class_<quasai::Optimizer, std::shared_ptr<quasai::Optimizer>>(
      m, "Optimizer");

  py::class_<quasai::SGD, quasai::Optimizer, std::shared_ptr<quasai::SGD>>(
      m, "SGD")
      .def(py::init<float, float>());

  py::enum_<quasai::Loss>(m, "Loss")
      .value("MSE", quasai::Loss::MSE)
      .value("L1", quasai::Loss::L1)
      .export_values();

  // -------------------------
  // Model
  // -------------------------
  py::class_<quasai::Model, std::shared_ptr<quasai::Model>>(m, "Model")
      .def(py::init<std::shared_ptr<quasai::Module>>())
      .def("compile", &quasai::Model::compile, py::arg("loss_fn"),
           py::arg("optimizer"))
      .def("train", &quasai::Model::train, py::arg("x"), py::arg("y"),
           py::arg("epochs"), py::arg("batch_size") = 32)
      .def("predict", &quasai::Model::predict, py::arg("x"));
}
