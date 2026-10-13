# R.1 exercises. Each test.cpp includes "impl.hpp"; CMake points the include path at your
# starter (default) or at solutions/ (-DS2S_USE_SOLUTIONS=ON).
#   cmake -S . -B ../build/ex && cmake --build ../build/ex -j && ctest --test-dir ../build/ex --output-on-failure
cmake_minimum_required(VERSION 3.20)
project(r1_exercises CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
option(S2S_USE_SOLUTIONS "Build tests against the reference solutions" OFF)
enable_testing()
set(S2S_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/include)

function(s2s_impl_dir name out)
  if(S2S_USE_SOLUTIONS)
    set(${out} ${CMAKE_CURRENT_SOURCE_DIR}/solutions/${name} PARENT_SCOPE)
  else()
    set(${out} ${CMAKE_CURRENT_SOURCE_DIR}/${name} PARENT_SCOPE)
  endif()
endfunction()

function(s2s_exercise name)
  s2s_impl_dir(${name} impl_dir)
  add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/${name}/test.cpp)
  target_include_directories(${name} PRIVATE ${impl_dir} ${S2S_COMMON})
  target_compile_options(${name} PRIVATE -Wall -Wextra)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

s2s_exercise(01-swap)
s2s_exercise(02-array-sum)
s2s_exercise(03-vec2-class)

# Exercise 4 has two files: stats.hpp (declarations) and stats.cpp (definitions).
# The .cpp is compiled into a library target and the test links it.
s2s_impl_dir(04-split-header split_dir)
add_library(stats_lib STATIC ${split_dir}/stats.cpp)
target_include_directories(stats_lib PUBLIC ${split_dir})
target_compile_options(stats_lib PRIVATE -Wall -Wextra)
add_executable(04-split-header ${CMAKE_CURRENT_SOURCE_DIR}/04-split-header/test.cpp)
target_include_directories(04-split-header PRIVATE ${S2S_COMMON})
target_link_libraries(04-split-header PRIVATE stats_lib)
add_test(NAME 04-split-header COMMAND 04-split-header)
