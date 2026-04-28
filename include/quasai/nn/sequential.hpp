#include "quasai/nn/module.hpp"

namespace quasai::nn {

class Sequential : public Module {
public:
  Sequential(const std::vector<std::shared_ptr<Module>> &modules);

  core::Tensor forward(const core::Tensor &input) override;

private:
  std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace quasai::nn
