cmake_minimum_required(VERSION 3.5)

project(measure_onnx)


set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 13)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_LIST_DIR}/include 
    ${CMAKE_CURRENT_LIST_DIR}/../include
)

set(PROJECT_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/src/main.cpp

    ${CMAKE_CURRENT_LIST_DIR}/src/inference.cpp
    ${CMAKE_CURRENT_LIST_DIR}/include/inference.h
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

if(REFLECT)
    target_compile_definitions(${PROJECT_NAME} PRIVATE REFLECT=ON)
endif()
