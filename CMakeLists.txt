cmake_minimum_required(VERSION 3.22)

project(
  VisualPoseTracking
  VERSION 1.0
  LANGUAGES CXX)
 

# `pkg-config --cflags --libs opencv4`

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)

add_executable("${PROJECT_NAME}" tracking.cpp) 

target_link_libraries("${PROJECT_NAME}" tracking)

