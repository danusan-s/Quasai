include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Explicitly list test directories to avoid globbing issues
set(TEST_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/nn
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd
    ${CMAKE_CURRENT_SOURCE_DIR}/storage
    ${CMAKE_CURRENT_SOURCE_DIR}/transform
)

foreach(dir ${TEST_DIRS})
    # Use GLOB_RECURSE to find all .cpp files in subdirectories
    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "${dir}/*.cpp")

    if(TEST_SOURCES)
        get_filename_component(test_name ${dir} NAME)

        add_executable(${test_name}_tests ${TEST_SOURCES})
        target_link_libraries(${test_name}_tests PRIVATE quasai GTest::gtest_main)

        gtest_discover_tests(${test_name}_tests)
    endif()
endforeach()

