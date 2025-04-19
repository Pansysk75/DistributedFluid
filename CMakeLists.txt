<<<<<<< HEAD
cmake_minimum_required(VERSION 3.19)
project(spring2025-csc4700-project)

set(CMAKE_CXX_STANDARD 23)

enable_testing()
include(CTest)

find_package(Python3 COMPONENTS Development REQUIRED)

add_executable(project work/project.cpp)
add_test(NAME project COMMAND project)

# Uncomment this if you want to use matplotlib.h
# target_compile_definitions(project PRIVATE WITHOUT_NUMPY)
# target_link_libraries(project Python3::Python Python3::Module)
=======
# Basic cmake file
cmake_minimum_required(VERSION 3.10)
project(stencil_2d CXX)

set(CMAKE_CXX_STANDARD 17)

# Add the executable target
add_executable(stencil_2d main.cpp)
>>>>>>> d7cc90b (Initial commit)
