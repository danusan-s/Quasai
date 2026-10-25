cmake_minimum_required(VERSION 3.16)
project(Quasai LANGUAGES CXX VERSION 0.1.0)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -march=native")

# Options
option(QUASAI_BUILD_SHARED      "Build shared library" OFF)
option(BUILD_BENCHMARKS         "Build benchmarks" OFF)
option(BUILD_EXAMPLES           "Build examples" ON)
option(BUILD_TESTS              "Build tests" ON)
option(ENABLE_SANITIZERS        "Enable AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(QUASAI_DEBUG             "Enable debug logging" OFF)
option(QUASAI_OPENMP            "Enable OpenMP parallelization" ON)
option(QUASAI_OPENBLAS          "Use OpenBLAS (fetch if not found)" ON)
option(QUASAI_CUDA              "Enable CUDA support (experimental)" OFF)

if(QUASAI_BUILD_SHARED)
  set(BUILD_SHARED_LIBS ON)
else()
  set(BUILD_SHARED_LIBS OFF)
endif()

# Sanitizers
if(ENABLE_SANITIZERS)
  set(SANITIZE_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer")
  add_compile_options(${SANITIZE_FLAGS})
  add_link_options(${SANITIZE_FLAGS})
  message(STATUS "Sanitizers enabled")
endif()

# Debug logging
if(QUASAI_DEBUG)
  add_compile_definitions(QUASAI_DEBUG=1)
  message(STATUS "Debug logging enabled")
endif()

# Source files (auto-discover)
file(GLOB_RECURSE SRC_FILES
  "src/quasai/*.cpp"
  "src/quasai/**/*.cpp"
)

# Library
add_library(quasai ${SRC_FILES})
set_target_properties(quasai PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

target_include_directories(quasai PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# OpenMP
if(QUASAI_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_FOUND)
    target_link_libraries(quasai PUBLIC OpenMP::OpenMP_CXX)
    add_compile_definitions(QUASAI_HAS_OPENMP=1)
    message(STATUS "OpenMP enabled")
  else()
    message(WARNING "OpenMP requested but not found")
  endif()
endif()

# BLAS / OpenBLAS
if(QUASAI_OPENBLAS)
  find_package(BLAS QUIET)
  if(BLAS_FOUND AND TARGET BLAS::BLAS)
    target_link_libraries(quasai PUBLIC BLAS::BLAS)
    message(STATUS "Using system BLAS")
  else()
    message(STATUS "BLAS not found - fetching OpenBLAS")
    include(FetchContent)
    FetchContent_Declare(
      openblas
      GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
      GIT_TAG v0.3.28
    )
    FetchContent_MakeAvailable(openblas)
    target_link_libraries(quasai PUBLIC openblas)
  endif()
endif()

# CUDA (experimental)
if(QUASAI_CUDA)
  find_package(CUDA REQUIRED)
  if(CUDA_FOUND)
    target_link_libraries(quasai PUBLIC CUDA::cudart)
    add_compile_definitions(QUASAI_CUDA=1)
    message(STATUS "CUDA enabled")
  endif()
endif()

# Install
install(TARGETS quasai
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include
  FILES_MATCHING PATTERN "*.hpp"
)

# Subdirectories
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Packaging
include(CPack)
set(CPACK_PACKAGE_NAME "Quasai")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++ ML Framework")
set(CPACK_GENERATOR "TGZ;ZIP")