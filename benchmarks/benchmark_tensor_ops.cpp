#include "quasai/core/tensor.hpp"
#include "quasai/ops/tensor_ops.hpp"
#include <benchmark/benchmark.h>
#include <vector>

// Large tensor sizes to trigger OpenMP parallelism
constexpr size_t LARGE_2D = 4096;     // 4096x4096 = 16M elements
constexpr size_t HUGE_2D = 8192;      // 8192x8192 = 64M elements
constexpr size_t LARGE_1D = 10000000; // 10M elements

// Benchmark: Tensor addition (large 2D)
static void BM_TensorAdd_Large2D(benchmark::State &state) {
  const size_t N = state.range(0);
  std::vector<float> data1(N * N, 1.0f);
  std::vector<float> data2(N * N, 2.0f);

  quasai::core::Tensor a = quasai::core::Tensor::from_data(
      data1.data(), quasai::core::Shape{N, N}, quasai::core::DType::FLOAT32);
  quasai::core::Tensor b = quasai::core::Tensor::from_data(
      data2.data(), quasai::core::Shape{N, N}, quasai::core::DType::FLOAT32);

  for (auto _ : state) {
    quasai::core::Tensor c = quasai::ops::add(a, b);
    benchmark::DoNotOptimize(c);
  }

  state.SetItemsProcessed(state.iterations() * N * N);
}
BENCHMARK(BM_TensorAdd_Large2D)->Arg(LARGE_2D)->Arg(HUGE_2D);

// Benchmark: Matrix multiplication (large 2D)
static void BM_TensorMatmul_Large2D(benchmark::State &state) {
  const size_t N = state.range(0);
  std::vector<float> data1(N * N, 1.0f);
  std::vector<float> data2(N * N, 2.0f);

  quasai::core::Tensor a = quasai::core::Tensor::from_data(
      data1.data(), quasai::core::Shape{N, N}, quasai::core::DType::FLOAT32);
  quasai::core::Tensor b = quasai::core::Tensor::from_data(
      data2.data(), quasai::core::Shape{N, N}, quasai::core::DType::FLOAT32);

  for (auto _ : state) {
    quasai::core::Tensor c = quasai::ops::matmul(a, b);
    benchmark::DoNotOptimize(c);
  }

  state.SetItemsProcessed(state.iterations() * N * N * N); // O(N^3)
}
BENCHMARK(BM_TensorMatmul_Large2D)->Arg(LARGE_2D);

// Benchmark: Transpose (large 2D)
static void BM_TensorTranspose_Large2D(benchmark::State &state) {
  const size_t N = state.range(0);
  std::vector<float> data(N * N, 1.0f);

  quasai::core::Tensor a = quasai::core::Tensor::from_data(
      data.data(), quasai::core::Shape{N, N}, quasai::core::DType::FLOAT32);

  for (auto _ : state) {
    quasai::core::Tensor t = quasai::ops::transpose(a);
    benchmark::DoNotOptimize(t);
  }

  state.SetItemsProcessed(state.iterations() * N * N);
}
BENCHMARK(BM_TensorTranspose_Large2D)->Arg(LARGE_2D)->Arg(HUGE_2D);

// Benchmark: Sum reduction (large 2D)
static void BM_TensorSum_Large2D(benchmark::State &state) {
  const size_t N = state.range(0);
  std::vector<float> data(N * N, 1.0f);

  quasai::core::Tensor a = quasai::core::Tensor::from_data(
      data.data(), quasai::core::Shape{N, N}, quasai::core::DType::FLOAT32);

  for (auto _ : state) {
    quasai::core::Tensor s = quasai::ops::sum(a);
    benchmark::DoNotOptimize(s);
  }

  state.SetItemsProcessed(state.iterations() * N * N);
}
BENCHMARK(BM_TensorSum_Large2D)->Arg(LARGE_2D)->Arg(HUGE_2D);

// Benchmark: Scalar broadcast add (large 1D)
static void BM_TensorScalarAdd_Large1D(benchmark::State &state) {
  const size_t N = state.range(0);
  std::vector<float> data(N, 1.0f);
  float scalar_val = 5.0f;

  quasai::core::Tensor a = quasai::core::Tensor::from_data(
      data.data(), quasai::core::Shape{N}, quasai::core::DType::FLOAT32);

  for (auto _ : state) {
    quasai::core::Tensor c = quasai::ops::add(a, scalar_val);
    benchmark::DoNotOptimize(c);
  }

  state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_TensorScalarAdd_Large1D)->Arg(LARGE_1D);

// Benchmark: ReLU activation (large 2D)
static void BM_ReLU_Large2D(benchmark::State &state) {
  const size_t N = state.range(0);
  std::vector<float> data(N * N, -1.0f);

  quasai::core::Tensor a = quasai::core::Tensor::from_data(
      data.data(), quasai::core::Shape{N, N}, quasai::core::DType::FLOAT32);

  for (auto _ : state) {
    quasai::core::Tensor r = quasai::ops::relu(a);
    benchmark::DoNotOptimize(r);
  }

  state.SetItemsProcessed(state.iterations() * N * N);
}
BENCHMARK(BM_ReLU_Large2D)->Arg(LARGE_2D)->Arg(HUGE_2D);

BENCHMARK_MAIN();
