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

add_library(data_loader_obj OBJECT src/data_loader.cpp)
target_include_directories(data_loader_obj PRIVATE include)
target_link_libraries(data_loader_obj Boost::boost Eigen3::Eigen xtensor)

add_library(utility_obj OBJECT src/utility.cpp)
target_include_directories(utility_obj PRIVATE include)
target_link_libraries(utility_obj Eigen3::Eigen CLI11::CLI11 Boost::boost)

add_library(expt_obj OBJECT src/experiments.cpp)
target_include_directories(expt_obj PRIVATE include)
target_link_libraries(expt_obj utility_obj Eigen3::Eigen)

set(COMMON_OBJS $<TARGET_OBJECTS:data_loader_obj> $<TARGET_OBJECTS:utility_obj> $<TARGET_OBJECTS:expt_obj>)

add_executable(mip_based_method src/mip_based_method.cpp ${COMMON_OBJS})
add_executable(klevel_based_method src/klevel_based_method.cpp ${COMMON_OBJS})
add_executable(klevel_based_method_2d src/klevel_based_method_2d.cpp ${COMMON_OBJS})
add_executable(baseline src/baseline.cpp ${COMMON_OBJS})
add_executable(baseline_2d src/baseline_2d.cpp ${COMMON_OBJS})
add_executable(preprocess_data src/preprocess_data.cpp src/data_preprocessor.cpp $<TARGET_OBJECTS:data_loader_obj>)

target_include_directories(mip_based_method PRIVATE include)
target_include_directories(klevel_based_method PRIVATE include)
target_include_directories(klevel_based_method_2d PRIVATE include)
target_include_directories(baseline PRIVATE include)
target_include_directories(baseline_2d PRIVATE include)
target_include_directories(preprocess_data PRIVATE include)

target_link_libraries(mip_based_method Eigen3::Eigen GUROBI ${SCIP_LIBRARIES})
target_link_libraries(klevel_based_method Boost::boost Eigen3::Eigen LibCDS::cds-s)
target_link_libraries(klevel_based_method_2d Eigen3::Eigen)
target_link_libraries(baseline Boost::boost Eigen3::Eigen)
target_link_libraries(baseline_2d Eigen3::Eigen)
target_link_libraries(preprocess_data Eigen3::Eigen GUROBI xtensor)
