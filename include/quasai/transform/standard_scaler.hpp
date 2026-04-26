#include "quasai/core/tensor.hpp"
namespace quasai {

class StandardScaler {
public:
  StandardScaler() = default;

  void fit(const Tensor &data);

  Tensor transform(const Tensor &data) const;

  Tensor inverse_transform(const Tensor &data) const;

private:
  Tensor mean_;
  Tensor std_;
};

} // namespace quasai
