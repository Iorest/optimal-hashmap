cmake_minimum_required(VERSION 3.14)
project(ooh VERSION 1.0.1 LANGUAGES CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Options ───────────────────────────────────────────────────────────────────
option(OOH_BUILD_TESTS    "Build correctness & robustness tests"       ON)
option(OOH_BUILD_BENCH    "Build standalone benchmark (ooh vs ankerl)" OFF)
option(OOH_BUILD_GBENCH   "Build Google Benchmark suite"               OFF)
option(OOH_ENABLE_ASAN    "Enable AddressSanitizer + UBSan"            OFF)
option(OOH_ENABLE_TSAN    "Enable ThreadSanitizer"                     OFF)
option(OOH_ENABLE_COVERAGE "Enable gcov/lcov code coverage"            OFF)

# ── Header-only library target ────────────────────────────────────────────────
add_library(ooh INTERFACE)
add_library(ooh::ooh ALIAS ooh)

target_include_directories(ooh INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(ooh INTERFACE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(ooh INTERFACE
        $<$<CONFIG:Release>:-O3 -march=native>
        $<$<CONFIG:RelWithDebInfo>:-O2 -march=native>
    )
endif()

# ── Sanitizer helper ──────────────────────────────────────────────────────────
function(ooh_add_sanitizers target)
    if(OOH_ENABLE_ASAN AND OOH_ENABLE_TSAN)
        message(FATAL_ERROR "OOH_ENABLE_ASAN and OOH_ENABLE_TSAN are mutually exclusive.")
    endif()
    if(OOH_ENABLE_ASAN)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
        target_link_options(   ${target} PRIVATE -fsanitize=address,undefined)
    endif()
    if(OOH_ENABLE_TSAN)
        target_compile_options(${target} PRIVATE -fsanitize=thread -fno-omit-frame-pointer -g)
        target_link_options(   ${target} PRIVATE -fsanitize=thread)
    endif()
    if(OOH_ENABLE_COVERAGE)
        target_compile_options(${target} PRIVATE --coverage -O0 -g)
        target_link_options(   ${target} PRIVATE --coverage)
    endif()
endfunction()

# ── External dependency: ankerl/unordered_dense ───────────────────────────────
# Declared once and reused by both bench targets.
if(OOH_BUILD_BENCH OR OOH_BUILD_GBENCH)
    include(FetchContent)
    FetchContent_Declare(
        unordered_dense
        GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
        GIT_TAG        v4.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(unordered_dense)
    message(STATUS "ankerl::unordered_dense v4.4.0 fetched")
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(OOH_BUILD_TESTS)
    enable_testing()
    add_executable(test_flat_map tests/test_flat_map.cpp)
    target_link_libraries(test_flat_map PRIVATE ooh::ooh $<$<BOOL:${UNIX}>:pthread>)
    target_compile_options(test_flat_map PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>)
    ooh_add_sanitizers(test_flat_map)
    add_test(NAME flat_map COMMAND test_flat_map)
    set_tests_properties(flat_map PROPERTIES TIMEOUT 180)
endif()

# ── Standalone benchmark (no Google Benchmark dependency) ─────────────────────
if(OOH_BUILD_BENCH)
    add_executable(bench_ooh bench/bench.cpp)
    set_target_properties(bench_ooh PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_link_libraries(bench_ooh PRIVATE ooh::ooh unordered_dense::unordered_dense
        $<$<BOOL:${UNIX}>:pthread>)
    target_compile_features(bench_ooh PRIVATE cxx_std_17)
    target_compile_options(bench_ooh PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>)
endif()

# ── Google Benchmark suite ────────────────────────────────────────────────────
if(OOH_BUILD_GBENCH)
    include(FetchContent)
    set(BENCHMARK_ENABLE_TESTING      OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL      OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.8.4
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(googlebenchmark)

    add_executable(bench_ooh_gbench bench/bench_gbench.cpp)
    set_target_properties(bench_ooh_gbench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_link_libraries(bench_ooh_gbench PRIVATE
        ooh::ooh benchmark::benchmark_main
        unordered_dense::unordered_dense $<$<BOOL:${UNIX}>:pthread>)
    target_compile_features(bench_ooh_gbench PRIVATE cxx_std_17)
    target_compile_options(bench_ooh_gbench PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>)
    message(STATUS "Google Benchmark suite: enabled")
endif()

# ── Install ───────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ooh EXPORT oohTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT oohTargets
    FILE oohTargets.cmake NAMESPACE ooh::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ooh)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/oohConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/oohConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ooh)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/oohConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/oohConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/oohConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ooh)
