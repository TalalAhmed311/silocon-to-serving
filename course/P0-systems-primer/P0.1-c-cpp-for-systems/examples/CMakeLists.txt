# P0.1 examples. Build: cmake -S . -B ../build/examples -DCMAKE_BUILD_TYPE=Release && cmake --build ../build/examples -j
cmake_minimum_required(VERSION 3.20)
project(p0_1_examples C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(S2S_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
if(S2S_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

set(S2S_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/include)

add_executable(01_layout 01_layout.c)
add_executable(02_strides 02_strides.c)
foreach(t 03_raii_buffer 04_move_vs_copy 05_templates)
  add_executable(${t} ${t}.cpp)
  target_include_directories(${t} PRIVATE ${S2S_COMMON})
endforeach()
