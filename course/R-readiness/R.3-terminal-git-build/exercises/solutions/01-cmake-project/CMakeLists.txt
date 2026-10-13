# Exercise 1 reference solution. The sources live next to the starter, so point at them.
cmake_minimum_required(VERSION 3.20)
project(r3_cmake_project CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(SRC ${CMAKE_CURRENT_LIST_DIR}/../../01-cmake-project)

# A library target: compiled once, shared by every target that links it.
add_library(mathlib ${SRC}/mathlib.cpp)
target_include_directories(mathlib PUBLIC ${SRC})   # PUBLIC: users of mathlib get this include path too

add_executable(app ${SRC}/main.cpp)
target_link_libraries(app PRIVATE mathlib)

enable_testing()
add_executable(test_mathlib ${SRC}/test_mathlib.cpp)
target_link_libraries(test_mathlib PRIVATE mathlib)
add_test(NAME mathlib COMMAND test_mathlib)
