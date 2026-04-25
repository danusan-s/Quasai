cmake_minimum_required(VERSION 3.16)
project(Quasai LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Library
add_library(quasai STATIC
  ${SRC_FILES}
)

target_include_directories(quasai PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# GoogleTest
enable_testing()
add_subdirectory(tests)

# Examples
add_subdirectory(examples)
