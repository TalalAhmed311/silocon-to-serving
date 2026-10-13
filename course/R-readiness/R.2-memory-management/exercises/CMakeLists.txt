# R.2 exercises. Every one is built with AddressSanitizer + UndefinedBehaviorSanitizer in a Debug
# build, so silent memory bugs turn into loud failures.
#   cmake -S . -B ../build/ex && cmake --build ../build/ex -j && ctest --test-dir ../build/ex --output-on-failure
# Add -DS2S_USE_SOLUTIONS=ON to build the reference solutions instead of your starters.
cmake_minimum_required(VERSION 3.20)
project(r2_exercises CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
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
  target_compile_options(${name} PRIVATE -Wall -Wextra
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all -g -O0)
  target_link_options(${name} PRIVATE -fsanitize=address,undefined)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

s2s_exercise(01-fix-the-leak)
s2s_exercise(02-dangling-return)
s2s_exercise(03-owning-buffer)
s2s_exercise(04-sanitizer-read)
