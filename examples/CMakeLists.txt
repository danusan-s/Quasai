add_executable(linear_regression 
    cpp/linear_regression.cpp
)
target_link_libraries(linear_regression PRIVATE quasai)

add_executable(simple_example 
    cpp/simple_example.cpp
)
target_link_libraries(simple_example PRIVATE quasai)
