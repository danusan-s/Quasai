#include "quasai/nn/module.hpp"
#include <memory>

namespace quasai::nn {

class Sequential : public Module {
public:
  Sequential();

  void add_module(std::unique_ptr<Module> module);

  core::Tensor forward(const core::Tensor &input) override;

  void train() override;

  void eval() override;

private:
  std::vector<std::unique_ptr<Module>> modules_;
};

} // namespace quasai::nn
