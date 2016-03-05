cmake_minimum_required(VERSION 3.3)
project(Retinex)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
set(SOURCE_FILES main.cpp retinex.h)
add_executable(Retinex ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})