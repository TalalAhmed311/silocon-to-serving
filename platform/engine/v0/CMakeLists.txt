# #0 v0 — Tiny Inference Engine (CPU).
#   cmake -S platform/engine/v0 -B build/engine-v0 -DCMAKE_BUILD_TYPE=Release && cmake --build build/engine-v0 -j
#   ./build/engine-v0/s2s-engine --model build/tiny-llama --prompt-ids "1 2 3" --steps 32
# S2S_OPS_DIR lets the P0.5 exercises build this engine against the learner's ops.hpp instead of src/ops.hpp.
cmake_minimum_required(VERSION 3.20)
project(s2s_engine_v0 CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=native)
set(S2S_OPS_DIR "" CACHE PATH "Directory containing an alternative ops.hpp (searched first)")
find_package(Threads REQUIRED)
add_executable(s2s-engine src/main.cpp)
if(S2S_OPS_DIR)
  target_include_directories(s2s-engine BEFORE PRIVATE ${S2S_OPS_DIR})
endif()
target_include_directories(s2s-engine PRIVATE src)
target_link_libraries(s2s-engine Threads::Threads)
