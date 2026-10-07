# D4 kernels repo: header-only kernels (include/d4/*.cuh), their tests (tests/*.cu, CTest) and one bench runner.
#   cmake -S platform/kernels -B build/d4 -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build/d4 -j
#   ctest --test-dir build/d4 --output-on-failure
#   ./build/d4/d4_bench all          # or: reduce | scan | softmax | rmsnorm | gemm | hgemm | attention
# Baselines: cuBLAS (gemm/hgemm), CUB from CCCL (reduce/scan). Peaks: measured copy bandwidth + cuBLAS.
cmake_minimum_required(VERSION 3.24)
project(d4 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
find_package(CUDAToolkit REQUIRED)
enable_testing()
set(INC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../../laneB-cuda/harness/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../course/common/include)
file(GLOB tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu)
foreach(t ${tests} ${CMAKE_CURRENT_SOURCE_DIR}/bench/d4_bench.cu)
  get_filename_component(n ${t} NAME_WE)
  add_executable(${n} ${t})
  target_include_directories(${n} PRIVATE ${INC})
  target_link_libraries(${n} PRIVATE CUDA::cudart CUDA::cublas)
  target_compile_options(${n} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
  if(NOT n STREQUAL "d4_bench")
    add_test(NAME ${n} COMMAND ${n})
  endif()
endforeach()
