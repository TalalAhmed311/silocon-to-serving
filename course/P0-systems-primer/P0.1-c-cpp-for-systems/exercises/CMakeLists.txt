# P0.1 exercises. Each test.cpp includes "impl.hpp" (or impl.cpp); CMake points the include path at
# your starter (default) or at solutions/ (-DS2S_USE_SOLUTIONS=ON).
#   cmake -S . -B ../build/ex && cmake --build ../build/ex -j && ctest --test-dir ../build/ex --output-on-failure
cmake_minimum_required(VERSION 3.20)
project(p0_1_exercises CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
option(S2S_USE_SOLUTIONS "Build tests against the reference solutions" OFF)
enable_testing()
set(S2S_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/include)

function(s2s_exercise name)
  if(S2S_USE_SOLUTIONS)
    set(impl_dir ${CMAKE_CURRENT_SOURCE_DIR}/solutions/${name})
  else()
    set(impl_dir ${CMAKE_CURRENT_SOURCE_DIR}/${name})
  endif()
  add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/${name}/test.cpp)
  target_include_directories(${name} PRIVATE ${impl_dir} ${S2S_COMMON})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

s2s_exercise(01-shrink-struct)
s2s_exercise(02-strided-copy)
s2s_exercise(03-tensor-raii)
s2s_exercise(04-blocked-transpose)

# Exercise 5 is only meaningful under the sanitizers: they turn silent corruption into a crash.
s2s_exercise(05-sanitizer-hunt)
target_compile_options(05-sanitizer-hunt PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -g -O1)
target_link_options(05-sanitizer-hunt PRIVATE -fsanitize=address,undefined)
