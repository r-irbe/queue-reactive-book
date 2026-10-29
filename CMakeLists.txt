cmake_minimum_required(VERSION 3.16)
project(queue_reactive_book LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(QRLOB_BUILD_EXPERIMENTAL "Build experimental replay, SPSC bridge, Hawkes calibration, and parity surface" ON)

set(QRLOB_ARCH_FLAGS "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    list(APPEND QRLOB_ARCH_FLAGS -mavx2)
endif()

add_library(qrlob_core STATIC
    src/flat_ladder_book.cpp
    src/lob.cpp
)
target_include_directories(qrlob_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<$<BOOL:${QRLOB_BUILD_EXPERIMENTAL}>:${CMAKE_CURRENT_SOURCE_DIR}/experimental>
)
target_compile_options(qrlob_core PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -fno-exceptions ${QRLOB_ARCH_FLAGS}
)
if(QRLOB_BUILD_EXPERIMENTAL)
    target_compile_definitions(qrlob_core PUBLIC QRLOB_BUILD_EXPERIMENTAL=1)
else()
    target_compile_definitions(qrlob_core PUBLIC QRLOB_BUILD_EXPERIMENTAL=0)
endif()

add_executable(sim src/main.cpp)
target_include_directories(sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(sim PRIVATE -fno-exceptions ${QRLOB_ARCH_FLAGS})
target_link_libraries(sim PRIVATE qrlob_core)

add_executable(sim_example examples/basic_sim.cpp)
target_include_directories(sim_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(sim_example PRIVATE -fno-exceptions ${QRLOB_ARCH_FLAGS})
target_link_libraries(sim_example PRIVATE qrlob_core)

add_library(qrlob_ffi SHARED src/qrlob_ffi.cpp)
target_include_directories(qrlob_ffi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(qrlob_ffi PRIVATE -O3 -fno-exceptions ${QRLOB_ARCH_FLAGS})
target_link_libraries(qrlob_ffi PRIVATE qrlob_core)

add_executable(sim_bench bench/main.cpp)
target_include_directories(sim_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/test_shim
)
target_compile_options(sim_bench PRIVATE -fno-exceptions ${QRLOB_ARCH_FLAGS})
target_link_libraries(sim_bench PRIVATE qrlob_core)

enable_testing()

set(QRLOB_TEST_SOURCES
    tests/main.cpp
    tests/test_pillar4_breakthroughs.cpp
    tests/test_lean_gpu_parity.cpp
    tests/test_flat_ladder_prefix.cpp
)
if(QRLOB_BUILD_EXPERIMENTAL)
    list(APPEND QRLOB_TEST_SOURCES tests/test_experimental_q_surface.cpp)
endif()

add_executable(sim_test ${QRLOB_TEST_SOURCES})
target_include_directories(sim_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/test_shim
)
target_compile_options(sim_test PRIVATE ${QRLOB_ARCH_FLAGS})
target_link_libraries(sim_test PRIVATE qrlob_core qrlob_ffi)
add_test(NAME sim_test COMMAND sim_test)

add_custom_target(qrlob_tests DEPENDS sim_test)
