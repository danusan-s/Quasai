include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Get all subdirectories in tests/
file(GLOB TEST_DIRS LIST_DIRECTORIES true *)

foreach(dir ${TEST_DIRS})
    if(IS_DIRECTORY ${dir})
        file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${dir}/*.cpp)

        if(TEST_SOURCES)
            get_filename_component(test_name ${dir} NAME)

            add_executable(${test_name}_tests ${TEST_SOURCES})
            target_link_libraries(${test_name}_tests PRIVATE quasai GTest::gtest_main)

            gtest_discover_tests(${test_name}_tests)
        endif()
    endif()
endforeach()

