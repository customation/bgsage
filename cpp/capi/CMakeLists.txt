# Standalone build of the bgsage C API shared library (CPU-only).
#
# Deliberately self-contained so embedders (engine daemons, FFI hosts)
# can build the shim without CUDA or pybind11. At upstream-PR time this
# folds into the main CMakeLists as an optioned target.
cmake_minimum_required(VERSION 3.18)
project(bgsage_capi LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BGBOT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(MSVC)
    set(BGBOT_OPT_FLAGS /O2 /fp:fast /arch:AVX2)
    add_compile_definitions(NOMINMAX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(BGBOT_OPT_FLAGS -O3 -funsafe-math-optimizations -fno-math-errno -funroll-loops)
    else()
        set(BGBOT_OPT_FLAGS -O3 -ffast-math -funroll-loops -mavx2 -mfma)
    endif()
endif()

set(BGBOT_CORE_SOURCES
    ${BGBOT_ROOT}/src/board.cpp
    ${BGBOT_ROOT}/src/moves.cpp
    ${BGBOT_ROOT}/src/strategy.cpp
    ${BGBOT_ROOT}/src/pubeval.cpp
    ${BGBOT_ROOT}/src/game.cpp
    ${BGBOT_ROOT}/src/benchmark.cpp
    ${BGBOT_ROOT}/src/encoding.cpp
    ${BGBOT_ROOT}/src/neural_net.cpp
    ${BGBOT_ROOT}/src/training.cpp
    ${BGBOT_ROOT}/src/multipy.cpp
    ${BGBOT_ROOT}/src/rollout.cpp
    ${BGBOT_ROOT}/src/cube.cpp
    ${BGBOT_ROOT}/src/cube_eval.cpp
    ${BGBOT_ROOT}/src/match_equity.cpp
    ${BGBOT_ROOT}/src/bearoff.cpp
    ${BGBOT_ROOT}/src/cuda_nn_stub.cpp
)

add_library(bgsage_capi SHARED
    ${BGBOT_CORE_SOURCES}
    ${BGBOT_ROOT}/src/capi.cpp
)
target_include_directories(bgsage_capi PUBLIC ${BGBOT_ROOT}/include)
target_compile_options(bgsage_capi PRIVATE ${BGBOT_OPT_FLAGS})
target_compile_definitions(bgsage_capi PRIVATE BGSAGE_CAPI_BUILD)
find_package(Threads REQUIRED)
target_link_libraries(bgsage_capi PRIVATE Threads::Threads)

# Smoke test binary (not installed).
add_executable(capi_smoke ${CMAKE_CURRENT_SOURCE_DIR}/capi_smoke.c)
target_link_libraries(capi_smoke PRIVATE bgsage_capi)
