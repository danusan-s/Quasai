cmake_minimum_required(VERSION 3.16)
project(Quasai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library
add_library(quasai STATIC
    src/quasai/storage/allocator.cpp
    src/quasai/storage/buffer.cpp
    src/quasai/core/tensor.cpp
)

target_include_directories(quasai PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# GoogleTest
enable_testing()
add_subdirectory(tests)
