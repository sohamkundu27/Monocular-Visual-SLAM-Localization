cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(mvslam
    VERSION 0.1.0
    DESCRIPTION "Classical monocular visual SLAM and localization on KITTI Odometry (C++17 port)"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MVSLAM_BUILD_TESTS "Build the GoogleTest suite" ON)
option(MVSLAM_WERROR "Treat compiler warnings as errors" OFF)
# Off by default: -march=native lets the compiler contract a*b+c into FMA and
# pick different vector widths, which changes floating-point results between
# machines. It must also match how GTSAM was built (Eigen alignment ABI).
option(MVSLAM_NATIVE "Compile with -march=native (non-reproducible numerics)" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

# --------------------------------------------------------------------------- #
# Warnings, applied to this project's targets only (never to dependencies).
# --------------------------------------------------------------------------- #
add_library(mvslam_warnings INTERFACE)
target_compile_options(mvslam_warnings INTERFACE -Wall -Wextra -Wpedantic)
if(MVSLAM_WERROR)
    target_compile_options(mvslam_warnings INTERFACE -Werror)
endif()
if(MVSLAM_NATIVE)
    target_compile_options(mvslam_warnings INTERFACE -march=native)
endif()

# --------------------------------------------------------------------------- #
# Core library: one source file per Python module, same directory layout.
# --------------------------------------------------------------------------- #
set(MVSLAM_SOURCES
    src/config.cpp
    src/pipeline.cpp
    src/reporting.cpp
    src/version.cpp
    src/compat/pyjson.cpp
    src/compat/pyformat.cpp
    src/compat/pyyaml.cpp
    src/compat/numpy.cpp
    src/utils/logging.cpp
    src/utils/seeding.cpp
    src/utils/timing.cpp
    src/utils/cv_eigen.cpp
    src/geometry/transforms.cpp
    src/geometry/pose.cpp
    src/geometry/epipolar.cpp
    src/datasets/calibration.cpp
    src/datasets/kitti.cpp
    src/features/detector.cpp
    src/features/matcher.cpp
    src/odometry/scale.cpp
    src/odometry/visual_odometry.cpp
    src/loop_closure/database.cpp
    src/loop_closure/detector.cpp
    src/optimization/pose_graph.cpp
    src/optimization/gtsam_backend.cpp
    src/evaluation/alignment.cpp
    src/evaluation/metrics.cpp
    src/evaluation/plotting.cpp
    src/cli/common.cpp
    src/cli/run_slam.cpp
    src/cli/run_vo.cpp
    src/cli/evaluate.cpp
    src/cli/report.cpp
)

add_library(mvslam STATIC ${MVSLAM_SOURCES})
add_library(mvslam::mvslam ALIAS mvslam)
target_include_directories(mvslam PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)
target_link_libraries(mvslam
    PUBLIC
        Eigen3::Eigen
        ${MVSLAM_OPENCV_TARGETS}
        gtsam
        nlohmann_json::nlohmann_json
    PRIVATE
        mvslam_warnings
        yaml-cpp::yaml-cpp
        CLI11::CLI11)
target_compile_definitions(mvslam PRIVATE
    MVSLAM_VERSION="${PROJECT_VERSION}"
    MVSLAM_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# --------------------------------------------------------------------------- #
# Executables (replace scripts/*.py and the Python console entry points).
# --------------------------------------------------------------------------- #
foreach(_app IN ITEMS run_slam run_vo evaluate report)
    add_executable(${_app} apps/${_app}.cpp)
    target_link_libraries(${_app} PRIVATE mvslam mvslam_warnings)
endforeach()

if(MVSLAM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
