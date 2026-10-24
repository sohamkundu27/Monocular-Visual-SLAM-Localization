# Dependency discovery.
#
# Heavy, numerics-relevant libraries (OpenCV, GTSAM, Eigen) must be installed
# and are found with find_package; tools/build_deps.sh builds the exact versions
# the Python reference uses. Small header-only / utility libraries are taken
# from the system when present and fetched otherwise.

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# -- Eigen ------------------------------------------------------------------ #
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# -- OpenCV ----------------------------------------------------------------- #
# The Python reference ran against cv2 4.13.0. Any 4.8+ builds, but bit-level
# front-end parity is only expected with the same major.minor (and IPP/LAPACK
# configuration); the parity tests skip themselves otherwise.
# MVSLAM_OPENCV_MIN_VERSION exists so CI can build against the distribution's
# older OpenCV (everything used here exists since 4.5); 4.8 is the supported floor.
set(MVSLAM_OPENCV_MIN_VERSION 4.8 CACHE STRING "Minimum accepted OpenCV version")
find_package(OpenCV ${MVSLAM_OPENCV_MIN_VERSION} REQUIRED
    COMPONENTS core imgproc imgcodecs features2d calib3d)
set(MVSLAM_OPENCV_TARGETS opencv_core opencv_imgproc opencv_imgcodecs opencv_features2d
    opencv_calib3d)
message(STATUS "mvslam: OpenCV ${OpenCV_VERSION} (${OpenCV_DIR})")

# -- GTSAM ------------------------------------------------------------------ #
find_package(GTSAM 4.2 REQUIRED)
message(STATUS "mvslam: GTSAM ${GTSAM_VERSION} (${GTSAM_DIR})")

# -- yaml-cpp --------------------------------------------------------------- #
find_package(yaml-cpp 0.7 QUIET)
if(NOT TARGET yaml-cpp::yaml-cpp)
    if(TARGET yaml-cpp)
        add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
    else()
        FetchContent_Declare(yaml-cpp
            URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
            URL_HASH SHA256=fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16)
        set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
        set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
        set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(yaml-cpp)
        if(NOT TARGET yaml-cpp::yaml-cpp)
            add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
        endif()
    endif()
endif()

# -- nlohmann_json ---------------------------------------------------------- #
find_package(nlohmann_json 3.10 QUIET)
if(NOT TARGET nlohmann_json::nlohmann_json)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# -- CLI11 ------------------------------------------------------------------ #
find_package(CLI11 2.3 QUIET)
if(NOT TARGET CLI11::CLI11)
    FetchContent_Declare(cli11
        URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.tar.gz
        URL_HASH SHA256=f2d893a65c3b1324c50d4e682c0cdc021dd0477ae2c048544f39eed6654b699a)
    set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(cli11)
endif()

# -- GoogleTest ------------------------------------------------------------- #
if(MVSLAM_BUILD_TESTS)
    find_package(GTest 1.12 QUIET)
    if(NOT TARGET GTest::gtest_main)
        FetchContent_Declare(googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
            URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()
