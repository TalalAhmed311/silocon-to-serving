# P1.6 exercises: one CTest test per exercises/<NN-name>/test.cu, plus the examples and the sanitizer target.
# Exercises include "kernel.cuh" from their own folder (your starter) or, with -DS2S_USE_SOLUTIONS=ON, from
# solutions/<NN-name>/. This mirrors course/P5-cuda-deep-dive/CMakeLists.txt, scoped to this module.
#
#   cmake -S course/P1-inference-fundamentals/P1.6-first-cuda-program/exercises -B build/p1.6 -DCMAKE_CUDA_ARCHITECTURES=native
#   cmake --build build/p1.6 -j && ctest --test-dir build/p1.6 --output-on-failure
#   (references)  add -DS2S_USE_SOLUTIONS=ON in a second build directory
# Requires CUDA 12.4+, CMake 3.24+, an NVIDIA GPU with compute capability 7.5 or higher (T2).
cmake_minimum_required(VERSION 3.24)
project(p1_6 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
option(S2S_USE_SOLUTIONS "Build exercise tests against the reference solutions" OFF)
find_package(CUDAToolkit REQUIRED)
enable_testing()
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(INC ${ROOT}/laneB-cuda/harness/include ${ROOT}/course/common/include)
set(MOD ${CMAKE_CURRENT_SOURCE_DIR}/..)

function(s2s_cu target src)
  add_executable(${target} ${src})
  target_include_directories(${target} PRIVATE ${ARGN} ${INC})
  target_link_libraries(${target} PRIVATE CUDA::cudart)
  # -lineinfo keeps source lines in compute-sanitizer reports without the slowdown of -G.
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

# Examples: one executable each.
file(GLOB examples ${MOD}/examples/*.cu)
foreach(e ${examples})
  get_filename_component(n ${e} NAME_WE)
  s2s_cu(p1.6_${n} ${e})
endforeach()

# Exercises with a test: one CTest test each.
file(GLOB tests ${CMAKE_CURRENT_SOURCE_DIR}/*/test.cu)
foreach(t ${tests})
  get_filename_component(dir ${t} DIRECTORY)
  get_filename_component(ex ${dir} NAME)
  if(S2S_USE_SOLUTIONS)
    set(impl ${CMAKE_CURRENT_SOURCE_DIR}/solutions/${ex})
  else()
    set(impl ${dir})
  endif()
  s2s_cu(p1.6_${ex} ${t} ${impl})
  add_test(NAME p1.6_${ex} COMMAND p1.6_${ex})
endforeach()

# Exercise 4 has no test: a provided program with a planted out-of-bounds write, to run under compute-sanitizer.
s2s_cu(p1.6_04-sanitizer-oob ${CMAKE_CURRENT_SOURCE_DIR}/04-sanitizer/oob.cu)
