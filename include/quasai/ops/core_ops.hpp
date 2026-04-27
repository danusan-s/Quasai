#pragma once

#include "quasai/core/tensor.hpp"
#include "quasai/utils/logger.hpp"
#include <functional>

namespace quasai {

constexpr size_t MIN_NUM_ELEMENTS_FOR_PARALLEL = 10000;
constexpr size_t MIN_NUM_ELEMENTS_FOR_PARALLEL_REDUCTION = 20000;
constexpr size_t MIN_NUM_TILES_FOR_PARALLEL_MATMUL = 64;

inline bool can_use_parallel(size_t num_elements) {
  return num_elements > MIN_NUM_ELEMENTS_FOR_PARALLEL;
}

inline bool can_use_parallel_reduction(size_t num_elements) {
  return num_elements > MIN_NUM_ELEMENTS_FOR_PARALLEL_REDUCTION;
}

inline bool can_use_parallel_matmul(size_t M, size_t N, size_t BS) {
  size_t num_tiles = (M + BS - 1) / BS * (N + BS - 1) / BS;
  return num_tiles > MIN_NUM_TILES_FOR_PARALLEL_MATMUL;
}

// Binary ops
void add_binary_gradient(const Tensor &a, const Tensor &b, Tensor &result,
                         std::function<Function *()> grad_fn_constructor);
Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mul(const Tensor &a, const Tensor &b);
Tensor div(const Tensor &a, const Tensor &b);

Tensor matmul(const Tensor &a, const Tensor &b);

// Unary ops
void add_unary_gradient(const Tensor &a, Tensor &result,
                        std::function<Function *()> grad_fn_constructor);
Tensor neg(const Tensor &a);
Tensor abs(const Tensor &a);
Tensor relu(const Tensor &a);
Tensor heaviside(const Tensor &a);
Tensor signum(const Tensor &a);
Tensor sigmoid(const Tensor &a);
Tensor tanh(const Tensor &a);

// Reduction ops
Tensor sum(const Tensor &a);
Tensor sum_to_shape(const Tensor &a, const Shape &target);
Tensor broadcast_to_shape(const Tensor &a, const Shape &target);
Tensor mean(const Tensor &a);

// View ops
Tensor transpose(const Tensor &a);
Tensor expand(const Tensor &a,
              const Shape &target); // broadcast_to_shape but view and no new
                                    // buffer allocation
Tensor reshape(const Tensor &a, const Shape &target);
Tensor make_contiguous(const Tensor &a);
Tensor slice(const Tensor &a, size_t start, size_t end);

template <typename T>
void do_binary_op(const Tensor &a, const Tensor &b, Tensor &result,
                  std::function<T(T, T)> op) {
  // Take when no broadcasting and both tensors are contiguous as a fast path
  bool fast_path =
      (a.shape() == b.shape()) && a.is_contiguous() && b.is_contiguous();

  if (fast_path) {
    do_binary_fast<T>(a, b, result, op);
  } else {
    do_binary_slow<T>(a, b, result, op);
  }
}
template <typename T>
void do_binary_fast(const Tensor &a, const Tensor &b, Tensor &result,
                    std::function<T(T, T)> op) {
  const size_t num_elements = total_size(result.shape());
  const T *data_a = a.data<T>();
  const T *data_b = b.data<T>();
  T *data_result = result.data<T>();

#pragma omp parallel for if (can_use_parallel(num_elements))
  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = op(data_a[i], data_b[i]);
  }
}

template <typename T>
void do_binary_slow(const Tensor &a, const Tensor &b, Tensor &result,
                    std::function<T(T, T)> op) {
  const Shape out_shape = result.shape();

  const size_t num_elements = total_size(out_shape);
  const T *data_a = a.data<T>();
  const T *data_b = b.data<T>();
  T *data_result = result.data<T>();

#pragma omp parallel for if (can_use_parallel(num_elements))
  for (size_t i = 0; i < num_elements; ++i) {
    Index idx = unravel_index(i, out_shape);
    Index idx_a = get_broadcast_index(idx, a.shape());
    Index idx_b = get_broadcast_index(idx, b.shape());

    data_result[i] = op(data_a[ravel_index(idx_a, a.strides())],
                        data_b[ravel_index(idx_b, b.strides())]);
  }
}

template <typename T>
void do_unary_op(const Tensor &a, Tensor &result, std::function<T(T)> op) {
  const Shape out_shape = result.shape();

  const size_t num_elements = total_size(out_shape);
  const T *data_a = a.data<T>();
  T *data_result = result.data<T>();

#pragma omp parallel for if (can_use_parallel(num_elements))
  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = op(data_a[i]);
  }
}

template <typename T> void do_sum(const Tensor &a, Tensor &result) {
  const size_t num_elements = total_size(a.shape());
  const T *data_a = a.data<T>();
  T *data_result = result.data<T>();

  T sum = 0;
#pragma omp parallel for reduction(                                            \
        + : sum) if (can_use_parallel_reduction(num_elements))
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

  // Dangerous to parallize since multiple threads may write to the same out ind
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

#pragma omp parallel for
  for (size_t i = 0; i < num_elements; ++i) {
    Index idx_out = unravel_index(i, out_shape);
    Index idx_a = get_broadcast_index(idx_out, a_shape);
    data_result[i] = data_a[ravel_index(idx_a, a_strides)];
  }
}

template <typename T>
void do_matmul(const Tensor &a, const Tensor &b, Tensor &result) {
  size_t M = a.shape()[0];
  size_t N = b.shape()[1];

  if (result.shape() != Shape{M, N}) {
    throw std::runtime_error("Result tensor has incorrect shape for matmul");
  }

  T *C = result.data<T>();
  for (size_t i = 0; i < M * N; ++i) {
    C[i] = 0;
  }

  bool fast_path =
      a.is_contiguous() && b.is_contiguous() && result.is_contiguous();

  if (fast_path) {
    do_matmul_fast<T>(a, b, result);
  } else {
    do_matmul_slow<T>(a, b, result);
  }
}

template <typename T>
void do_matmul_fast(const Tensor &a, const Tensor &b, Tensor &result) {
  size_t M = a.shape()[0];
  size_t K = a.shape()[1];
  size_t N = b.shape()[1];

  const T *A = a.data<T>();
  const T *B = b.data<T>();
  T *C = result.data<T>();

  const size_t BS = 32;

#pragma omp parallel for collapse(2) if (can_use_parallel_matmul(M, N, BS))
  for (size_t ii = 0; ii < M; ii += BS) {
    for (size_t jj = 0; jj < N; jj += BS) {
      for (size_t kk = 0; kk < K; kk += BS) {
        for (size_t i = ii; i < std::min(ii + BS, M); ++i) {
          for (size_t k = kk; k < std::min(kk + BS, K); ++k) {
            T a_val = A[i * K + k];
            for (size_t j = jj; j < std::min(jj + BS, N); ++j) {
              C[i * N + j] += a_val * B[k * N + j];
            }
          }
        }
      }
    }
  }
}

template <typename T>
void do_matmul_slow(const Tensor &a, const Tensor &b, Tensor &result) {
  size_t M = a.shape()[0];
  size_t K = a.shape()[1];
  size_t N = b.shape()[1];

  const T *A = a.data<T>();
  const T *B = b.data<T>();
  T *C = result.data<T>();

  auto a_strides = a.strides();
  auto b_strides = b.strides();
  auto c_strides = result.strides();

  const size_t BS = 32;

#pragma omp parallel for collapse(2) if (can_use_parallel_matmul(M, N, BS))
  for (size_t ii = 0; ii < M; ii += BS) {
    for (size_t jj = 0; jj < N; jj += BS) {
      for (size_t kk = 0; kk < K; kk += BS) {
        for (size_t i = ii; i < std::min(ii + BS, M); ++i) {
          for (size_t k = kk; k < std::min(kk + BS, K); ++k) {
            T a_val = A[i * a_strides[0] + k * a_strides[1]];
            for (size_t j = jj; j < std::min(jj + BS, N); ++j) {
              C[i * c_strides[0] + j * c_strides[1]] +=
                  a_val * B[k * b_strides[0] + j * b_strides[1]];
            }
          }
        }
      }
    }
  }
}

template <typename T> void do_contiguous_copy(const Tensor &a, Tensor &result) {
  const Shape &a_shape = a.shape();
  const Strides &a_strides = a.strides();

  const size_t num_elements = total_size(a_shape);
  const T *data_a = a.data<T>();
  T *data_result = result.data<T>();

#pragma omp parallel for if (can_use_parallel(num_elements))
  for (size_t i = 0; i < num_elements; ++i) {
    Index idx_a = unravel_index(i, a_shape);
    data_result[i] = data_a[ravel_index(idx_a, a_strides)];
  }
  return;
}

} // namespace quasai
