# GoogleTest suite. tests/test_<module>.cpp mirrors tests/test_<module>.py;
# tests/TEST_PORTING.md maps every Python test to its C++ counterpart.

include(GoogleTest)

set(_support_sources "")
foreach(_src IN ITEMS support/golden.cpp support/synthetic.cpp support/kitti_fixture.cpp
                      support/test_utils.cpp)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
        list(APPEND _support_sources "${_src}")
    endif()
endforeach()

if(_support_sources)
    add_library(mvslam_testsupport STATIC ${_support_sources})
    target_include_directories(mvslam_testsupport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(mvslam_testsupport PUBLIC mvslam GTest::gtest PRIVATE mvslam_warnings)
    target_compile_definitions(mvslam_testsupport PUBLIC
        MVSLAM_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
        MVSLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
else()
    add_library(mvslam_testsupport INTERFACE)
    target_include_directories(mvslam_testsupport INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Optional KITTI root for the parity tests (they skip when unset/missing);
# the MVSLAM_KITTI_DATASET environment variable overrides it at run time.
set(MVSLAM_KITTI_DATASET "" CACHE PATH "KITTI odometry root (holding sequences/) for parity tests")

function(mvslam_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE mvslam mvslam_testsupport mvslam_warnings
        GTest::gtest_main)
    target_compile_definitions(${name} PRIVATE
        MVSLAM_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden"
        MVSLAM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        MVSLAM_KITTI_DATASET_DEFAULT="${MVSLAM_KITTI_DATASET}")
    foreach(_app IN ITEMS run_slam run_vo evaluate report)
        if(TARGET ${_app})
            string(TOUPPER ${_app} _APP)
            target_compile_definitions(${name} PRIVATE MVSLAM_EXE_${_APP}="$<TARGET_FILE:${_app}>")
            add_dependencies(${name} ${_app})
        endif()
    endforeach()
    gtest_discover_tests(${name}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        TEST_PREFIX "${name}."
        DISCOVERY_TIMEOUT 60
        PROPERTIES LABELS ${name})
endfunction()

file(GLOB _test_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)
foreach(_test_source IN LISTS _test_sources)
    get_filename_component(_test_name ${_test_source} NAME_WE)
    mvslam_add_test(${_test_name})
endforeach()
