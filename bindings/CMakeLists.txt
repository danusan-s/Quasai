find_package(pybind11)

if(pybind11_FOUND)
    pybind11_add_module(pyquasai MODULE
        python/bindings.cpp
    )
    target_link_libraries(pyquasai PRIVATE quasai)
endif()

