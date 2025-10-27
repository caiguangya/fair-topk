cmake_minimum_required(VERSION 3.12)
project(fair_topk)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Boost REQUIRED)
find_package(GUROBI REQUIRED)
find_package(xtensor REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(LibCDS REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(SCIP REQUIRED)

add_library(common_objs OBJECT src/data_loader.cpp src/utility.cpp src/experiments.cpp)
target_include_directories(common_objs PRIVATE include)
target_link_libraries(common_objs Eigen3::Eigen xtensor Boost::boost CLI11::CLI11)

add_executable(mip_based_method src/mip_based_method.cpp $<TARGET_OBJECTS:common_objs>)
add_executable(klevel_based_method src/klevel_based_method.cpp $<TARGET_OBJECTS:common_objs>)
add_executable(klevel_based_method_2d src/klevel_based_method_2d.cpp $<TARGET_OBJECTS:common_objs>)
add_executable(baseline src/baseline.cpp $<TARGET_OBJECTS:common_objs>)
add_executable(baseline_2d src/baseline_2d.cpp $<TARGET_OBJECTS:common_objs>)

target_include_directories(mip_based_method PRIVATE include)
target_include_directories(klevel_based_method PRIVATE include)
target_include_directories(klevel_based_method_2d PRIVATE include)
target_include_directories(baseline PRIVATE include)
target_include_directories(baseline_2d PRIVATE include)

target_link_libraries(mip_based_method Eigen3::Eigen GUROBI ${SCIP_LIBRARIES})
target_link_libraries(klevel_based_method Boost::boost Eigen3::Eigen LibCDS::cds-s)
target_link_libraries(klevel_based_method_2d Eigen3::Eigen)
target_link_libraries(baseline Boost::boost Eigen3::Eigen)
target_link_libraries(baseline_2d Eigen3::Eigen)