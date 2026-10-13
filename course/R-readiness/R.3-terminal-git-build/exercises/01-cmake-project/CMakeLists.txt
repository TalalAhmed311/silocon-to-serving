# Exercise 1 starter. Three TODOs. Build with:
#   cmake -S . -B /tmp/r3-build && cmake --build /tmp/r3-build -j && ctest --test-dir /tmp/r3-build --output-on-failure
cmake_minimum_required(VERSION 3.20)
project(r3_cmake_project CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# TODO 1: add a library target named `mathlib` built from mathlib.cpp, and make its folder
#         (this directory) a PUBLIC include directory so anything that links it can #include "mathlib.hpp".
#         Hint: add_library(...) and target_include_directories(... PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(app main.cpp)
# TODO 2: link `app` to `mathlib`. Until you do, the build stops at the linker with "undefined reference".
#         Hint: target_link_libraries(app PRIVATE ...)

# TODO 3: switch on testing, build `test_mathlib` from test_mathlib.cpp, link it to `mathlib`,
#         and register it with CTest under the name `mathlib`.
#         Hint: enable_testing(), add_executable(...), target_link_libraries(...), add_test(NAME ... COMMAND ...)
