#include "quasai/nn/module.hpp"

namespace quasai::nn {

class Sequential : public Module {
public:
  Sequential(const std::vector<std::shared_ptr<Module>> &modules);

  core::Tensor forward(const core::Tensor &input) override;

  void train() override {
    for (const auto &module : modules_) {
      module->train();
    }
    training_ = true;
  }

  void eval() override {
    for (const auto &module : modules_) {
      module->eval();
    }
    training_ = false;
  }

private:
  std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace quasai::nn
