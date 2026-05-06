cmake_minimum_required(VERSION 3.16)
project(Quasai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Sanitizers
option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(ENABLE_SANITIZERS)
  set(SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
  add_compile_options(${SANITIZER_FLAGS})
  add_link_options(${SANITIZER_FLAGS})
endif()

set(SRC_FILES
    src/quasai/nn/layer_norm.cpp
    src/quasai/nn/model.cpp
    src/quasai/nn/sequential.cpp
    src/quasai/nn/dropout.cpp
    src/quasai/nn/batch_norm_1d.cpp
    src/quasai/nn/linear.cpp
    src/quasai/nn/loss.cpp
    src/quasai/nn/init.cpp
    src/quasai/ops/unary_ops.cpp
    src/quasai/ops/reduction_ops.cpp
    src/quasai/ops/view_ops.cpp
    src/quasai/ops/linalg_ops.cpp
    src/quasai/ops/binary_ops.cpp
    src/quasai/autograd/unary_func.cpp
    src/quasai/autograd/reduction_func.cpp
    src/quasai/autograd/view_func.cpp
    src/quasai/autograd/linalg_func.cpp
    src/quasai/autograd/binary_func.cpp
    src/quasai/autograd/engine.cpp
    src/quasai/core/tensor.cpp
    src/quasai/transform/standard_scaler.cpp
    src/quasai/storage/buffer.cpp
    src/quasai/storage/allocator.cpp
    src/quasai/optim/sgd.cpp
)

# Library
add_library(quasai STATIC
  ${SRC_FILES}
)
set_target_properties(quasai PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(quasai PRIVATE -march=native)

target_include_directories(quasai PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

include(FetchContent)

find_package(OpenMP QUIET)

if(OpenMP_FOUND AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(quasai PUBLIC OpenMP::OpenMP_CXX)
else()
  message(STATUS "OpenMP not found - building without OpenMP support, Large operations may be slower")
endif()


find_package(BLAS QUIET)

if(BLAS_FOUND AND TARGET BLAS::BLAS)
    target_link_libraries(quasai PUBLIC BLAS::BLAS)
else()
    message(STATUS "BLAS not found - fetching OpenBLAS")

    FetchContent_Declare(
        openblas
        GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
        GIT_TAG v0.3.28
    )

    FetchContent_MakeAvailable(openblas)

    # OpenBLAS target name depends on build, this is typical:
    target_link_libraries(quasai PUBLIC openblas)
endif()

# Benchmarks (optional)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Examples
add_subdirectory(examples)

# Tests
enable_testing()
add_subdirectory(tests)
