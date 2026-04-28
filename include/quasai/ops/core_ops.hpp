#pragma once

#include "quasai/core/tensor.hpp"
#include "quasai/utils/logger.hpp"
#include <functional>

namespace quasai::autograd {
class Function;
} // namespace quasai::autograd

namespace quasai::ops {

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
void add_binary_gradient(
    const core::Tensor &a, const core::Tensor &b, core::Tensor &result,
    std::function<std::unique_ptr<autograd::Function>()> grad_fn_constructor);
core::Tensor add(const core::Tensor &a, const core::Tensor &b);
core::Tensor sub(const core::Tensor &a, const core::Tensor &b);
core::Tensor mul(const core::Tensor &a, const core::Tensor &b);
core::Tensor div(const core::Tensor &a, const core::Tensor &b);

core::Tensor matmul(const core::Tensor &a, const core::Tensor &b);

// Unary ops
void add_unary_gradient(
    const core::Tensor &a, core::Tensor &result,
    std::function<std::unique_ptr<autograd::Function>()> grad_fn_constructor);
core::Tensor neg(const core::Tensor &a);
core::Tensor abs(const core::Tensor &a);
core::Tensor relu(const core::Tensor &a);
core::Tensor heaviside(const core::Tensor &a);
core::Tensor signum(const core::Tensor &a);
core::Tensor sigmoid(const core::Tensor &a);
core::Tensor tanh(const core::Tensor &a);

// Reduction ops
core::Tensor sum(const core::Tensor &a);
core::Tensor sum_to_shape(const core::Tensor &a, const core::Shape &target);
core::Tensor mean(const core::Tensor &a);

// View ops
core::Tensor transpose(const core::Tensor &a);
core::Tensor expand(const core::Tensor &a,
                    const core::Shape &target); // broadcast_to_shape but view
                                                // and no new buffer allocation
core::Tensor reshape(const core::Tensor &a, const core::Shape &target);
core::Tensor make_contiguous(const core::Tensor &a);
core::Tensor slice(const core::Tensor &a, size_t start, size_t end);

template <typename T>
void do_binary_fast(const core::Tensor &a, const core::Tensor &b,
                    core::Tensor &result, std::function<T(T, T)> op) {
  const size_t num_elements = total_size(result.shape());
  const T *data_a = a.data<T>();
  const T *data_b = b.data<T>();
  T *data_result = result.data<T>();

#pragma omp parallel for simd if (can_use_parallel(num_elements))
  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = op(data_a[i], data_b[i]);
  }
}

template <typename T>
void do_binary_slow(const core::Tensor &a, const core::Tensor &b,
                    core::Tensor &result, std::function<T(T, T)> op) {
  const core::Shape out_shape = result.shape();

  const size_t num_elements = total_size(out_shape);
  const T *data_a = a.data<T>();
  const T *data_b = b.data<T>();
  T *data_result = result.data<T>();

#pragma omp parallel for if (can_use_parallel(num_elements))
  for (size_t i = 0; i < num_elements; ++i) {
    core::Index idx = unravel_index(i, out_shape);
    core::Index idx_a = get_broadcast_index(idx, a.shape());
    core::Index idx_b = get_broadcast_index(idx, b.shape());

    data_result[i] = op(data_a[ravel_index(idx_a, a.strides())],
                        data_b[ravel_index(idx_b, b.strides())]);
  }
}

template <typename T>
void do_binary_op(const core::Tensor &a, const core::Tensor &b,
                  core::Tensor &result, std::function<T(T, T)> op) {
  const core::Shape &out_shape = result.shape();

  // Fast path: no broadcasting, both contiguous -> flat access
  if (a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous()) {
    const size_t num_elements = total_size(out_shape);
    const T *data_a = a.data<T>();
    const T *data_b = b.data<T>();
    T *data_result = result.data<T>();

#pragma omp parallel for simd if (can_use_parallel(num_elements))
    for (size_t i = 0; i < num_elements; ++i) {
      data_result[i] = op(data_a[i], data_b[i]);
    }
    return;
  }

  // Slow path: expand both tensors to output shape as views (zero-copy)
  core::Tensor a_exp = ops::expand(a, out_shape);
  core::Tensor b_exp = ops::expand(b, out_shape);

  const size_t num_elements = total_size(out_shape);
  const T *data_a = a_exp.data<T>();
  const T *data_b = b_exp.data<T>();
  T *data_result = result.data<T>();

  const core::Strides &strides_a = a_exp.strides();
  const core::Strides &strides_b = b_exp.strides();
  const size_t ndim = out_shape.dimensions();

#pragma omp parallel for if (can_use_parallel(num_elements))
  for (size_t i = 0; i < num_elements; ++i) {
    core::Index idx = unravel_index(i, out_shape);
    size_t linear_a = 0, linear_b = 0;
    for (size_t d = 0; d < ndim; ++d) {
      linear_a += idx[d] * strides_a[d];
      linear_b += idx[d] * strides_b[d];
    }
    data_result[i] = op(data_a[linear_a], data_b[linear_b]);
  }
}

template <typename T>
void do_unary_op(const core::Tensor &a, core::Tensor &result,
                 std::function<T(T)> op) {
  const core::Shape out_shape = result.shape();

  const size_t num_elements = total_size(out_shape);
  const T *data_a = a.data<T>();
  T *data_result = result.data<T>();

#pragma omp parallel for simd if (can_use_parallel(num_elements))
  for (size_t i = 0; i < num_elements; ++i) {
    data_result[i] = op(data_a[i]);
  }
}

template <typename T> void do_sum(const core::Tensor &a, core::Tensor &result) {
  const size_t num_elements = total_size(a.shape());
  const T *data_a = a.data<T>();
  T *data_result = result.data<T>();

  T sum = 0;
#pragma omp parallel for simd reduction(                                       \
        + : sum) if (can_use_parallel_reduction(num_elements))
  for (size_t i = 0; i < num_elements; ++i) {
    sum += data_a[i];
  }
  data_result[0] = sum;
}

template <typename T>
void do_sum_to_shape(const core::Tensor &a, core::Tensor &result) {
  const core::Shape &a_shape = a.shape();
  const core::Shape &out_shape = result.shape();
  const core::Strides &out_strides = result.strides();

  const size_t num_elements = total_size(a_shape);
  const T *data_a = a.data<T>();
  T *data_result = result.data<T>();

  // Dangerous to parallize since multiple threads may write to the same out ind
  for (size_t i = 0; i < num_elements; ++i) {
    core::Index idx_a = unravel_index(i, a_shape);
    core::Index idx_out = get_broadcast_index(idx_a, out_shape);
    data_result[ravel_index(idx_out, out_strides)] += data_a[i];
  }
}

template <typename T>
void do_matmul_fast(const core::Tensor &a, const core::Tensor &b,
                    core::Tensor &result) {
  size_t M = a.shape()[0];
  size_t K = a.shape()[1];
  size_t N = b.shape()[1];

  const T *A = a.data<T>();
  const T *B = b.data<T>();
  T *C = result.data<T>();

  // Cache-aware blocking: L1=64 bytes, L2~256KB
  // Block sizes tuned for typical cache sizes
  const size_t BM = 64;  // Block rows of A/C
  const size_t BK = 256; // Block cols of A / rows of B (L1 cache)
  const size_t BN = 64;  // Block cols of B/C

#pragma omp parallel for collapse(2) if (can_use_parallel_matmul(M, N, BN))
  for (size_t ii = 0; ii < M; ii += BM) {
    for (size_t jj = 0; jj < N; jj += BN) {
      // Micro-panel: accumulate in registers
      for (size_t kk = 0; kk < K; kk += BK) {
        size_t i_end = std::min(ii + BM, M);
        size_t j_end = std::min(jj + BN, N);
        size_t k_end = std::min(kk + BK, K);

        for (size_t i = ii; i < i_end; ++i) {
          for (size_t k = kk; k < k_end; ++k) {
            T a_val = A[i * K + k];
#pragma omp simd
            for (size_t j = jj; j < j_end; ++j) {
              C[i * N + j] += a_val * B[k * N + j];
            }
          }
        }
      }
    }
  }
}

template <typename T>
void do_matmul_slow(const core::Tensor &a, const core::Tensor &b,
                    core::Tensor &result) {
  size_t M = a.shape()[0];
  size_t K = a.shape()[1];
  size_t N = b.shape()[1];

  const T *A = a.data<T>();
  const T *B = b.data<T>();
  T *C = result.data<T>();

  auto a_strides = a.strides();
  auto b_strides = b.strides();
  auto c_strides = result.strides();

  const size_t BM = 64;
  const size_t BK = 256;
  const size_t BN = 64;

#pragma omp parallel for collapse(2) if (can_use_parallel_matmul(M, N, BN))
  for (size_t ii = 0; ii < M; ii += BM) {
    for (size_t jj = 0; jj < N; jj += BN) {
      for (size_t kk = 0; kk < K; kk += BK) {
        size_t i_end = std::min(ii + BM, M);
        size_t j_end = std::min(jj + BN, N);
        size_t k_end = std::min(kk + BK, K);

        for (size_t i = ii; i < i_end; ++i) {
          size_t a_i_offset = i * a_strides[0];
          size_t c_i_offset = i * c_strides[0];
          for (size_t k = kk; k < k_end; ++k) {
            T a_val = A[a_i_offset + k * a_strides[1]];
            size_t b_k_offset = k * b_strides[0];
#pragma omp simd
            for (size_t j = jj; j < j_end; ++j) {
              C[c_i_offset + j * c_strides[1]] +=
                  a_val * B[b_k_offset + j * b_strides[1]];
            }
          }
        }
      }
    }
  }
}

template <typename T>
void do_matmul(const core::Tensor &a, const core::Tensor &b,
               core::Tensor &result) {
  size_t M = a.shape()[0];
  size_t N = b.shape()[1];

  if (result.shape() != core::Shape{M, N}) {
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
void do_contiguous_copy(const core::Tensor &a, core::Tensor &result) {
  const core::Shape &a_shape = a.shape();
  const core::Strides &a_strides = a.strides();

  const size_t num_elements = total_size(a_shape);
  const T *data_a = a.data<T>();
  T *data_result = result.data<T>();

#pragma omp parallel for if (can_use_parallel(num_elements))
  for (size_t i = 0; i < num_elements; ++i) {
    core::Index idx_a = unravel_index(i, a_shape);
    data_result[i] = data_a[ravel_index(idx_a, a_strides)];
  }
  return;
}

} // namespace quasai::ops
