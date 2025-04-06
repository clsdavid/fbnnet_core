cmake_minimum_required(VERSION 3.14)
project(fbnet_py)

find_package(pybind11 REQUIRED)

add_library(fbnet_py MODULE fbnet_bindings.cpp)
target_link_libraries(fbnet_py PRIVATE pybind11::module)
target_include_directories(fbnet_py PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_target_properties(fbnet_py PROPERTIES PREFIX "" SUFFIX ".so")
