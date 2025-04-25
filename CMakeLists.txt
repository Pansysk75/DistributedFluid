cmake_minimum_required(VERSION 3.19)
project(spring2025-csc4700-project)

set(CMAKE_CXX_STANDARD 23)

enable_testing()
include(CTest)

find_package(Python3 COMPONENTS Development REQUIRED)
find_package(HPX REQUIRED)

# tile_component
add_library(tile_component work/tile_component.cpp)
target_link_libraries(tile_component HPX::hpx)
target_include_directories(tile_component PRIVATE work/include)

add_executable(project work/project.cpp)
target_link_libraries(project HPX::hpx)
target_include_directories(project PRIVATE work/include)
add_test(NAME project COMMAND project)

add_executable(project_distributed_1 work/project_distributed_1.cpp)
target_link_libraries(project_distributed_1 HPX::hpx tile_component)
target_include_directories(project_distributed_1 PRIVATE work/include)

add_executable(project_distributed_2 work/project_distributed_2.cpp)
target_link_libraries(project_distributed_2 HPX::hpx tile_component)
target_include_directories(project_distributed_2 PRIVATE work/include)

add_executable(project_fluid_1 work/project_fluid_1.cpp)
target_include_directories(project_fluid_1 PRIVATE work/include)

add_executable(project_fluid_2 work/project_fluid_2.cpp)
target_include_directories(project_fluid_2 PRIVATE work/include)

add_executable(project_fluid_3 work/project_fluid_3.cpp)
target_include_directories(project_fluid_3 PRIVATE work/include)

add_executable(project_fluid_4 work/project_fluid_4.cpp)
target_include_directories(project_fluid_4 PRIVATE work/include)

add_executable(project_fluid_5 work/project_fluid_5.cpp)
target_include_directories(project_fluid_5 PRIVATE work/include)

add_executable(project_fluid_5_hpx work/project_fluid_5_hpx.cpp)
target_link_libraries(project_fluid_5_hpx HPX::hpx)
target_include_directories(project_fluid_5_hpx PRIVATE work/include)

add_executable(project_fluid_static work/project_fluid_static.cpp)
target_include_directories(project_fluid_static PRIVATE work/include)

add_executable(project_fluid_5_soa work/project_fluid_5_soa.cpp)
target_include_directories(project_fluid_5_soa PRIVATE work/include)

add_executable(project_distributed_fluid work/project_distributed_fluid.cpp)
target_link_libraries(project_distributed_fluid HPX::hpx tile_component)
target_include_directories(project_distributed_fluid PRIVATE work/include)

# Uncomment this if you want to use matplotlib.h
# target_compile_definitions(project PRIVATE WITHOUT_NUMPY)
# target_link_libraries(project Python3::Python Python3::Module)
