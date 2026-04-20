include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)

add_executable(memory_test core/memory.cpp)
target_link_libraries(memory_test PRIVATE quasai GTest::gtest_main)

add_executable(dense_storage_test storage/dense_storage.cpp)
target_link_libraries(dense_storage_test PRIVATE quasai GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(memory_test)
gtest_discover_tests(dense_storage_test)
