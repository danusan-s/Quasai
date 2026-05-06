set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

include(FetchContent)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(quasai_benchmarks benchmark_tensor_ops.cpp)
target_link_libraries(quasai_benchmarks PRIVATE quasai benchmark::benchmark)

option(BUILD_TORCH_BENCHMARKS "Build benchmarks with PyTorch/LibTorch comparison" OFF)

if(BUILD_TORCH_BENCHMARKS)

  include(FetchContent)

  FetchContent_Declare(
    libtorch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
  )

  FetchContent_MakeAvailable(libtorch)

  # LibTorch provides CMake config files inside the download
  list(APPEND CMAKE_PREFIX_PATH ${libtorch_SOURCE_DIR})

  find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)

  target_compile_definitions(quasai_benchmarks PRIVATE WITH_TORCH)

  target_link_libraries(quasai_benchmarks PRIVATE
    ${TORCH_LIBRARIES}
  )

  # This replaces manual RPATH handling
  set_target_properties(quasai_benchmarks PROPERTIES
    BUILD_RPATH "${libtorch_SOURCE_DIR}/lib"
    INSTALL_RPATH "${libtorch_SOURCE_DIR}/lib"
  )

endif()
