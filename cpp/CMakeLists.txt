cmake_minimum_required(VERSION 3.10)
project(opencv_car_count)

set(cmake_minimum_required 11)

find_package(OpenCV 4.8.0 REQUIRED)
message("OpenCV version: ${OpenCV_VERSION}")

set(NCNN_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240410-ubuntu-2004/lib/cmake/ncnn)
include_directories(${NCNN_DIR})
find_package(ncnn REQUIRED HINTS ${NCNN_DIR})

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

