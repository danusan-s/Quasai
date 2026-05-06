#include "quasai/core/dtype.hpp"
#include "quasai/core/tensor.hpp"
#include "quasai/ops/cpu_kernel.hpp"
#include <benchmark/benchmark.h>
#include <vector>

#ifdef WITH_TORCH
#include <torch/torch.h>
#endif

// Large tensor sizes to trigger OpenMP parallelism
constexpr size_t LARGE_2D = 2048;    // 2048x2048 = 4M elements
constexpr size_t HUGE_2D = 4096;     // 4096x4096 = 16M elements
constexpr size_t LARGE_1D = 5000000; // 5M elements

// Benchmark: Tensor addition (large 2D)
static void BM_TensorAdd_Large2D(benchmark::State &state) {
  const size_t N = state.range(0);
  std::vector<float> data1(N * N, 1.0f);
  std::vector<float> data2(N * N, 2.0f);

  quasai::core::Tensor a = quasai::core::Tensor::from_data(
      data1.data(), quasai::core::Shape{N, N}, quasai::core::DType::FLOAT32);
  quasai::core::Tensor b = quasai::core::Tensor::from_data(
      data2.data(), quasai::core::Shape{N, N}, quasai::core::DType::FLOAT32);
  quasai::core::Tensor c = quasai::core::Tensor::empty(
      quasai::core::Shape{N, N}, quasai::core::DType::FLOAT32);

  for (auto _ : state) {
    quasai::core::dispatch_by_dtype(a.dtype(), [&]<typename T>() {
      quasai::ops::do_binary_op<T>(a, b, c,
                                   [](auto x, auto y) { return x + y; });
    });
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
  quasai::core::Tensor c = quasai::core::Tensor::empty(
      quasai::core::Shape{N, N}, quasai::core::DType::FLOAT32);

  for (auto _ : state) {
    quasai::core::dispatch_by_dtype(a.dtype(), [&]<typename T>() {
      quasai::ops::do_matmul_fast<T>(a, b, c);
    });
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
  quasai::core::Tensor b = quasai::core::Tensor::from_scalar(
      scalar_val, quasai::core::DType::FLOAT32);
  quasai::core::Tensor c = quasai::core::Tensor::empty(
      quasai::core::Shape{N}, quasai::core::DType::FLOAT32);

  for (auto _ : state) {
    quasai::core::dispatch_by_dtype(a.dtype(), [&]<typename T>() {
      quasai::ops::do_binary_op<T>(a, b, c,
                                   [](auto x, auto y) { return x + y; });
    });
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
  quasai::core::Tensor b = quasai::core::Tensor::empty(
      quasai::core::Shape{N, N}, quasai::core::DType::FLOAT32);

  for (auto _ : state) {
    quasai::core::dispatch_by_dtype(a.dtype(), [&]<typename T>() {
      quasai::ops::do_unary_op<T>(a, b, [](auto x) { return x > 0 ? x : 0; });
    });
  }

  state.SetItemsProcessed(state.iterations() * N * N);
}
BENCHMARK(BM_ReLU_Large2D)->Arg(LARGE_2D)->Arg(HUGE_2D);

#ifdef WITH_TORCH
// --- PyTorch/LibTorch Comparison Benchmarks ---

// Benchmark: Tensor addition (large 2D) - PyTorch
static void BM_Torch_Add_Large2D(benchmark::State &state) {
  const size_t N = state.range(0);
  torch::Tensor a = torch::ones({(int64_t)N, (int64_t)N}, torch::kFloat32);
  torch::Tensor b =
      torch::full({(int64_t)N, (int64_t)N}, 2.0f, torch::kFloat32);

  for (auto _ : state) {
    torch::Tensor c = a + b;
    benchmark::DoNotOptimize(c.data_ptr());
  }

  state.SetItemsProcessed(state.iterations() * N * N);
}
BENCHMARK(BM_Torch_Add_Large2D)->Arg(LARGE_2D)->Arg(HUGE_2D);

// Benchmark: Matrix multiplication (large 2D) - PyTorch
static void BM_Torch_Matmul_Large2D(benchmark::State &state) {
  const size_t N = state.range(0);
  torch::Tensor a = torch::ones({(int64_t)N, (int64_t)N}, torch::kFloat32);
  torch::Tensor b =
      torch::full({(int64_t)N, (int64_t)N}, 2.0f, torch::kFloat32);

  for (auto _ : state) {
    torch::Tensor c = torch::matmul(a, b);
    benchmark::DoNotOptimize(c.data_ptr());
  }

  state.SetItemsProcessed(state.iterations() * N * N * N);
}
BENCHMARK(BM_Torch_Matmul_Large2D)->Arg(LARGE_2D);

// Benchmark: Transpose (large 2D) - PyTorch
static void BM_Torch_Transpose_Large2D(benchmark::State &state) {
  const size_t N = state.range(0);
  torch::Tensor a = torch::ones({(int64_t)N, (int64_t)N}, torch::kFloat32);

  for (auto _ : state) {
    torch::Tensor t = a.transpose(0, 1);
    benchmark::DoNotOptimize(t.data_ptr());
  }

  state.SetItemsProcessed(state.iterations() * N * N);
}
BENCHMARK(BM_Torch_Transpose_Large2D)->Arg(LARGE_2D)->Arg(HUGE_2D);

// Benchmark: Sum reduction (large 2D) - PyTorch
static void BM_Torch_Sum_Large2D(benchmark::State &state) {
  const size_t N = state.range(0);
  torch::Tensor a = torch::ones({(int64_t)N, (int64_t)N}, torch::kFloat32);

  for (auto _ : state) {
    torch::Tensor s = a.sum();
    benchmark::DoNotOptimize(s.data_ptr());
  }

  state.SetItemsProcessed(state.iterations() * N * N);
}
BENCHMARK(BM_Torch_Sum_Large2D)->Arg(LARGE_2D)->Arg(HUGE_2D);

// Benchmark: Scalar broadcast add (large 1D) - PyTorch
static void BM_Torch_ScalarAdd_Large1D(benchmark::State &state) {
  const size_t N = state.range(0);
  torch::Tensor a = torch::ones({(int64_t)N}, torch::kFloat32);
  float scalar_val = 5.0f;

  for (auto _ : state) {
    torch::Tensor c = a + scalar_val;
    benchmark::DoNotOptimize(c.data_ptr());
  }

  state.SetItemsProcessed(state.iterations() * N);
}
BENCHMARK(BM_Torch_ScalarAdd_Large1D)->Arg(LARGE_1D);

// Benchmark: ReLU activation (large 2D) - PyTorch
static void BM_Torch_ReLU_Large2D(benchmark::State &state) {
  const size_t N = state.range(0);
  torch::Tensor a =
      torch::full({(int64_t)N, (int64_t)N}, -1.0f, torch::kFloat32);

  for (auto _ : state) {
    torch::Tensor r = torch::relu(a);
    benchmark::DoNotOptimize(r.data_ptr());
  }

  state.SetItemsProcessed(state.iterations() * N * N);
}
BENCHMARK(BM_Torch_ReLU_Large2D)->Arg(LARGE_2D)->Arg(HUGE_2D);

#endif // WITH_TORCH

BENCHMARK_MAIN();
