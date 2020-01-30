cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)

# LANGUAGE CUDA ==> REQUIRES CMALE 3.10+
project( modern_cuda_cmake LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Set Include path
set(CUWRAP_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(CUWRAP_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
include_directories(${CUWRAP_INCLUDE_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

### >>> [Thread]
find_package(Threads REQUIRED)

### >>> [CUDA BUILD] Generating shared lib for cuda codes
# Getting all cuda files
file(GLOB_RECURSE CUWRAP_CUDA_FILES ${CUWRAP_SOURCE_DIR}/*.c*)
message(STATUS ">>> [CUDA INCLUDE] files: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message(STATUS ">>> [SRC] files: ${CUWRAP_CUDA_FILES}")

# Cmake needs to know which dirs need to be included
# Compile all cuda codes to a shared library as `cuwrap`
add_library(cuwrap SHARED ${CUWRAP_CUDA_FILES})
set_target_properties(cuwrap PROPERTIES # Just know that CUDA 10 only supports CXX14. 
    CMAKE_CXX_STANDARD 14               # If you want CXX17 features in you device code you can use clang cuda frontend.
    CUDA_STANDARD 14
)
set_target_properties(cuwrap PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

### >>> [TEST] Generating test codes
# Establish gtest environment
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.8.0
)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

enable_testing()
# Getting all test files
file(GLOB_RECURSE CUWRAP_TESTS ${CMAKE_SOURCE_DIR}/tests/*spec.c*)

# Deal with each test files
foreach(TEST_FULL_PATH ${CUWRAP_TESTS})
  message(STATUS ">>> [TEST] TO BUILD ${TEST_FULL_PATH}")
  get_filename_component(TEST_NAME ${TEST_FULL_PATH} NAME_WE) 
  # ~ NAME_WE means filename without directory | longest extension
  # ~ See more details at https://cmake.org/cmake/help/v3.0/command/get_filename_component.html

  set(TEST_TAR ${TEST_NAME}.spec)
  add_executable(${TEST_TAR} ${TEST_FULL_PATH})
  set_property(
    TARGET ${TEST_TAR}
    PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(${TEST_TAR} PRIVATE cuwrap gtest_main Threads::Threads)
  add_test(NAME ${TEST_TAR} COMMAND ${TEST_TAR})
  
endforeach()