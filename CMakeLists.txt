cmake_minimum_required(VERSION 2.8.8)

find_package(OpenCV)
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS}) # ex. -DCPU_ONLY

add_subdirectory(src/xor)
add_subdirectory(src/shape)
