#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quasai/core/shape.hpp"
#include "quasai/core/tensor.hpp"
#include "quasai/nn/loss.hpp"
#include "quasai/nn/model.hpp"
#include "quasai/nn/module.hpp"
#include "quasai/nn/modules/activations.hpp"
#include "quasai/nn/modules/linear.hpp"
#include "quasai/nn/modules/sequential.hpp"
#include "quasai/optim/sgd.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyquasai, m) {
  // -------------------------
  // Tensor
  // -------------------------
  py::class_<quasai::core::Tensor, std::shared_ptr<quasai::core::Tensor>>(
      m, "Tensor")
      .def(py::init<>())
      .def("shape", [](const quasai::core::Tensor &t) {
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
    quasai::core::Shape shape(sizes.data(), ndim);
    auto t = quasai::core::Tensor::zeros(shape, quasai::core::DType::FLOAT32,
                                         quasai::core::Device::cpu());
    std::memcpy(t.data<float>(), buf.ptr, buf.size * sizeof(float));
    return t;
  });

  // -------------------------
  // Base Module (polymorphic)
  // -------------------------
  py::class_<quasai::nn::Module, std::shared_ptr<quasai::nn::Module>>(m,
                                                                      "Module")
      .def("__call__", &quasai::nn::Module::operator());

  // -------------------------
  // Linear layer
  // -------------------------
  py::class_<quasai::nn::Linear, quasai::nn::Module,
             std::shared_ptr<quasai::nn::Linear>>(m, "Linear")
      .def(py::init<int, int>());

  // -------------------------
  // ReLU activation
  // -------------------------
  py::class_<quasai::nn::ReLU, quasai::nn::Module,
             std::shared_ptr<quasai::nn::ReLU>>(m, "ReLU")
      .def(py::init<>());

  // -------------------------
  // Sequential container
  // -------------------------
  py::class_<quasai::nn::Sequential, quasai::nn::Module,
             std::shared_ptr<quasai::nn::Sequential>>(m, "Sequential")
      .def(py::init<std::vector<std::shared_ptr<quasai::nn::Module>>>());

  py::class_<quasai::optim::Optimizer,
             std::shared_ptr<quasai::optim::Optimizer>>(m, "Optimizer");

  py::class_<quasai::optim::SGD, quasai::optim::Optimizer,
             std::shared_ptr<quasai::optim::SGD>>(m, "SGD")
      .def(py::init<float, float>());

  py::enum_<quasai::nn::Loss>(m, "Loss")
      .value("MSE", quasai::nn::Loss::MSE)
      .value("L1", quasai::nn::Loss::L1)
      .export_values();

  // -------------------------
  // Model
  // -------------------------
  py::class_<quasai::nn::Model, std::shared_ptr<quasai::nn::Model>>(m, "Model")
      .def(py::init<std::shared_ptr<quasai::nn::Module>>())
      .def("compile", &quasai::nn::Model::compile, py::arg("loss_fn"),
           py::arg("optimizer"))
      .def("train", &quasai::nn::Model::train, py::arg("x"), py::arg("y"),
           py::arg("epochs"), py::arg("batch_size") = 32)
      .def("predict", &quasai::nn::Model::predict, py::arg("x"));
}
