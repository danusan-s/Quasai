include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
  CMAKE_ARGS "-DBENCHMARK_ENABLE_TESTING=OFF"
)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(quasai_benchmarks benchmark_tensor_ops.cpp)
target_link_libraries(quasai_benchmarks PRIVATE quasai benchmark::benchmark)
