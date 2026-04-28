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

find_package(OpenMP)

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Library
add_library(quasai STATIC
  ${SRC_FILES}
)
set_target_properties(quasai PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(quasai PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)


if(OpenMP_CXX_FOUND)
    target_link_libraries(quasai PUBLIC OpenMP::OpenMP_CXX)
endif()

# Benchmarks (optional)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Python bindings
add_subdirectory(bindings)

# Examples
add_subdirectory(examples)

# Tests
enable_testing()
add_subdirectory(tests)
