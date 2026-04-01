cmake_minimum_required(VERSION 3.5)

project(tensorrt)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -MMD -Wall -Wextra -Winit-self")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

find_package(CUDA REQUIRED)
find_package (Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
# find_package(CudaToolkit REQUIRED)

# set includes and sources
include_directories(
    /usr/local/cuda/include
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_LIST_DIR}/include 
    ${CMAKE_CURRENT_LIST_DIR}/../include
    ${TENSORRT_INCLUDE_DIR}
)

link_directories(/usr/local/cuda/lib64 ~/builds/TensorRT-10.14.1.48/lib)

set(PROJECT_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/detector.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/tensorrt_base.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/logger.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/oc_sort.cpp
    ${CMAKE_CURRENT_LIST_DIR}/src/lapjv.cpp
)

# set TensorRT version
if(JETSON)
  set(CMAKE_CXX_STANDARD 14)
  set(TRT 8)
  message(STATUS "TensorRT API 8")
else()
  set(CMAKE_CXX_STANDARD 17)
  set(TRT 10)
  message(STATUS "TensorRT API 10")
endif()

# set conditional sources
if (TRT EQUAL 8) 
    list(APPEND PROJECT_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/tensorrt_v8.cpp)
else()
    list(APPEND PROJECT_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/tensorrt_v10.cpp)
endif()

# build library
add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCES})
target_link_directories(${PROJECT_NAME} PRIVATE ${TENSORRT_LIB_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} cudart nvinfer yaml-cpp::yaml-cpp)

if(PERF_TRACKER)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PERF_TRACKER=ON)
endif()

# set definitions
if(JETSON)
  target_compile_definitions(${PROJECT_NAME} PRIVATE TRT=8)
else()
  target_compile_definitions(${PROJECT_NAME} PRIVATE TRT=10)
endif()

if (BUILD_TEST_PROGRAM)
  add_executable(test_detector ${CMAKE_CURRENT_LIST_DIR}/src/test_detector.cpp)
  add_executable(test_tracker ${CMAKE_CURRENT_LIST_DIR}/src/test_tracker.cpp)

  target_link_libraries(test_detector PRIVATE ${PROJECT_NAME})
  target_link_libraries(test_tracker PRIVATE ${PROJECT_NAME})
  if(REFLECT)
      target_compile_definitions(test_detector PRIVATE REFLECT=ON)
  endif()
  if(KITTI_RESULTS)
      target_compile_definitions(test_tracker PRIVATE KITTI_RESULTS=ON)
  endif()
endif()

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml DESTINATION ${CMAKE_BINARY_DIR}/tensorrt)