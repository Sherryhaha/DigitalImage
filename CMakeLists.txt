cmake_minimum_required(VERSION 3.3)
project(Retinex)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
find_package(OpenCV REQUIRED)
add_executable(Retinex ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})