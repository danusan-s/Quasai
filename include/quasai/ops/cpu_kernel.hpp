#pragma once

#include "quasai/core/tensor.hpp"
#include "quasai/ops/core_ops.hpp"

#ifndef QUASAI_DISABLE_OPENBLAS
extern "C" {
#include <cblas.h>
}
#endif

namespace quasai::ops {

/// @brief Minimum elements for enabling OpenMP parallelization.
constexpr size_t MIN_NUM_ELEMENTS_FOR_PARALLEL = 10000;
/// @brief Minimum elements for parallel reduction.
constexpr size_t MIN_NUM_ELEMENTS_FOR_PARALLEL_REDUCTION = 20000;
/// @brief Minimum tiles for parallel matrix multiplication.
constexpr size_t MIN_NUM_TILES_FOR_PARALLEL_MATMUL = 64;

/// @brief Check if parallelization should be used based on element count.
inline bool can_use_parallel(size_t num_elements) {
  return num_elements > MIN_NUM_ELEMENTS_FOR_PARALLEL;
}

/// @brief Check if parallel reduction should be used.
inline bool can_use_parallel_reduction(size_t num_elements) {
  return num_elements > MIN_NUM_ELEMENTS_FOR_PARALLEL_REDUCTION;
}

/// @brief Check if parallel matmul should be used.
inline bool can_use_parallel_matmul(size_t M, size_t N, size_t BS) {
  size_t num_tiles = (M + BS - 1) / BS * (N + BS - 1) / BS;
  return num_tiles > MIN_NUM_TILES_FOR_PARALLEL_MATMUL;
}

inline bool can_fast_path(const core::Tensor &a, const core::Tensor &b) {
  return (a.shape().dimensions() == 0 || b.shape().dimensions() == 0 ||
          a.shape() == b.shape()) &&
         a.is_contiguous() && b.is_contiguous();
}

/**
 * @brief Execute a binary operation with optional OpenMP parallelization.
 * @tparam T Data type of the tensors.
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param result Output tensor.
 * @param op Binary operation to apply.
 * @note Uses fast path for contiguous same-shape tensors, slow path with
 * broadcasting.
 */
template <typename T>
void do_binary_op(const core::Tensor &a, const core::Tensor &b,
                  core::Tensor &result, std::function<T(T, T)> op) {
  const core::Shape &out_shape = result.shape();

  // Fast path: no broadcasting, both contiguous -> flat access
  if (can_fast_path(a, b)) {
    const size_t num_elements = total_size(out_shape);
    const T *data_a = a.data<T>();
    const T *data_b = b.data<T>();
    T *data_result = result.data<T>();

    bool a_is_scalar = (a.shape().dimensions() == 0);
    bool b_is_scalar = (b.shape().dimensions() == 0);

    if (a_is_scalar && b_is_scalar) {
      data_result[0] = op(data_a[0], data_b[0]);
      return;
    } else if (a_is_scalar) {
      T a_val = data_a[0];
#pragma omp parallel for simd if (can_use_parallel(num_elements))
      for (size_t i = 0; i < num_elements; ++i) {
        data_result[i] = op(a_val, data_b[i]);
      }
      return;
    } else if (b_is_scalar) {
      T b_val = data_b[0];
#pragma omp parallel for simd if (can_use_parallel(num_elements))
      for (size_t i = 0; i < num_elements; ++i) {
        data_result[i] = op(data_a[i], b_val);
      }
      return;
    }
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

/**
 * @brief Execute a unary operation with optional OpenMP parallelization.
 * @tparam T Data type of the tensor.
 * @param a Input tensor.
 * @param result Output tensor.
 * @param op Unary operation to apply.
 */
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

/**
 * @brief Compute sum of all elements with OpenMP parallel reduction.
 * @tparam T Data type of the tensor.
 * @param a Input tensor.
 * @param result Output tensor (scalar).
 */
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

/**
 * @brief Sum elements and broadcast result to a target shape.
 * @tparam T Data type of the tensor.
 * @param a Input tensor.
 * @param result Output tensor (target shape).
 * @note Not parallelized (may have write conflicts).
 */
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

#ifndef QUASAI_DISABLE_OPENBLAS
/**
 * @brief OpenBLAS matrix multiplication for float (contiguous, row-major).
 */
inline void do_matmul_openblas_float(const core::Tensor &a,
                                     const core::Tensor &b,
                                     core::Tensor &result) {
  size_t M = a.shape()[0];
  size_t K = a.shape()[1];
  size_t N = b.shape()[1];

  const float *A = a.data<float>();
  const float *B = b.data<float>();
  float *C = result.data<float>();

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K,
              1.0f, A, (int)K, B, (int)N, 0.0f, C, (int)N);
}

/**
 * @brief OpenBLAS matrix multiplication for double (contiguous, row-major).
 */
inline void do_matmul_openblas_double(const core::Tensor &a,
                                      const core::Tensor &b,
                                      core::Tensor &result) {
  size_t M = a.shape()[0];
  size_t K = a.shape()[1];
  size_t N = b.shape()[1];

  const double *A = a.data<double>();
  const double *B = b.data<double>();
  double *C = result.data<double>();

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K,
              1.0, A, (int)K, B, (int)N, 0.0, C, (int)N);
}
#endif

/**
 * @brief Fast matrix multiplication for contiguous tensors.
 * @tparam T Data type.
 * @param a First tensor (M x K, contiguous).
 * @param b Second tensor (K x N, contiguous).
 * @param result Output tensor (M x N).
 * @note Uses cache-aware blocking for performance.
 */
template <typename T>
void do_matmul_fast(const core::Tensor &a, const core::Tensor &b,
                    core::Tensor &result) {
#ifndef QUASAI_DISABLE_OPENBLAS
  if constexpr (std::is_same_v<T, float>) {
    do_matmul_openblas_float(a, b, result);
    return;
  }
  if constexpr (std::is_same_v<T, double>) {
    do_matmul_openblas_double(a, b, result);
    return;
  }
#endif

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

/**
 * @brief Matrix multiplication for non-contiguous tensors (uses strides).
 * @tparam T Data type.
 * @param a First tensor (M x K).
 * @param b Second tensor (K x N).
 * @param result Output tensor (M x N).
 */
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

/**
 * @brief Matrix multiplication dispatcher (fast or slow path).
 * @tparam T Data type.
 * @param a First tensor (M x K).
 * @param b Second tensor (K x N).
 * @param result Output tensor (M x N).
 * @note Chooses fast path (contiguous) or slow path (strided) automatically.
 */
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

/**
 * @brief Copy a tensor to contiguous memory layout.
 * @tparam T Data type.
 * @param a Input tensor (may be non-contiguous).
 * @param result Output tensor (will be contiguous).
 */
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
