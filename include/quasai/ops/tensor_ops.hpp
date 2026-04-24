#pragma once

#include "quasai/core/tensor.hpp"
#include "quasai/utils/logger.hpp"
#include <functional>
#include <sstream>
namespace quasai {

void add_binary_gradient(const Tensor &a, const Tensor &b, Tensor &result,
                         std::function<Function *()> grad_fn_constructor);
Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mul(const Tensor &a, const Tensor &b);
Tensor div(const Tensor &a, const Tensor &b);

Tensor matmul(const Tensor &a, const Tensor &b);
Tensor transpose(const Tensor &a);

void add_unary_gradient(const Tensor &a, Tensor &result,
                        std::function<Function *()> grad_fn_constructor);
Tensor neg(const Tensor &a);
Tensor relu(const Tensor &a);
Tensor step(const Tensor &a);
Tensor sigmoid(const Tensor &a);
Tensor tanh(const Tensor &a);

Tensor sum(const Tensor &a);
Tensor sum_to_shape(const Tensor &a, const Shape &target);
Tensor broadcast_to_shape(const Tensor &a, const Shape &target);
Tensor mean(const Tensor &a);

template <typename T>
void do_binary_op(const Tensor &a, const Tensor &b, Tensor &result,
                  std::function<T(T, T)> op) {

  const Shape out_shape = result.shape();

  const size_t num_elements = total_size(out_shape);

  const T *data_a = a.data<T>();
  const T *data_b = b.data<T>();
  T *data_result = result.data<T>();

  std::size_t ndim_a = a.shape().dimensions();
  std::size_t ndim_b = b.shape().dimensions();

  for (size_t i = 0; i < num_elements; ++i) {
    Index idx = unravel_index(i, out_shape);
    Index idx_a = get_broadcast_index(idx, a.shape());
    Index idx_b = get_broadcast_index(idx, b.shape());

    data_result[i] = op(data_a[ravel_index(idx_a, a.strides())],
                        data_b[ravel_index(idx_b, b.strides())]);
    std::ostringstream ss;
    ss << "a: " << data_a[ravel_index(idx_a, a.strides())]
       << ", b: " << data_b[ravel_index(idx_b, b.strides())]
       << ", result: " << data_result[i];
    LOG_INFO(ss.str().c_str());
  }
}

template <typename T>
void do_unary_op(const Tensor &a, Tensor &result, std::function<T(T)> op) {
  const Shape out_shape = result.shape();

  const size_t num_elements = total_size(out_shape);
  const T *data_a = a.data<T>();
  T *data_result = result.data<T>();

  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = op(data_a[i]);
  }
}

template <typename T> void do_sum(const Tensor &a, Tensor &result) {
  const size_t num_elements = total_size(a.shape());
  const T *data_a = a.data<T>();
  T *data_result = result.data<T>();

  T sum = 0;
  for (size_t i = 0; i < num_elements; ++i) {
    sum += data_a[i];
  }
  data_result[0] = sum;
}

template <typename T> void do_sum_to_shape(const Tensor &a, Tensor &result) {
  const Shape &a_shape = a.shape();
  const Shape &out_shape = result.shape();
  const Strides &out_strides = result.strides();

  const size_t num_elements = total_size(a_shape);
  const T *data_a = a.data<T>();
  T *data_result = result.data<T>();

  for (size_t i = 0; i < num_elements; ++i) {
    Index idx_a = unravel_index(i, a_shape);
    Index idx_out = get_broadcast_index(idx_a, out_shape);
    data_result[ravel_index(idx_out, out_strides)] += data_a[i];
  }
}

template <typename T>
void do_broadcast_to_shape(const Tensor &a, Tensor &result) {
  const Shape &a_shape = a.shape();
  const Strides &a_strides = a.strides();
  const Shape &out_shape = result.shape();

  const size_t num_elements = total_size(out_shape);
  const T *data_a = a.data<T>();
  T *data_result = result.data<T>();

  for (size_t i = 0; i < num_elements; ++i) {
    Index idx_out = unravel_index(i, out_shape);
    Index idx_a = get_broadcast_index(idx_out, a_shape);
    data_result[i] = data_a[ravel_index(idx_a, a_strides)];
  }
}

template <typename T>
void do_matmul(const Tensor &a, const Tensor &b, Tensor &result) {
  const size_t M = a.shape()[0];
  const size_t K = a.shape()[1];
  const size_t N = b.shape()[1];

  const T *data_a = a.data<T>();
  const T *data_b = b.data<T>();
  T *data_result = result.data<T>();

  // Naive matrix multiplication
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += data_a[i * K + k] * data_b[k * N + j];
      }
      data_result[i * N + j] = sum;
    }
  }

  std::ostringstream ss;

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      ss << "result[" << i << ", " << j << "] = " << data_result[i * N + j];
      LOG_INFO(ss.str().c_str());
    }
  }
}

} // namespace quasai
