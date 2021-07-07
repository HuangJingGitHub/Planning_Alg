cmake_minimum_required(VERSION 2.8)
project(Planning_Module)
find_package(OpenCV REQUIRED)
# set(CMAKE_CXX_STANDARD 17)
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")


add_executable(RRT_DOM_demo RRT_DOM_demo.cpp)
target_link_libraries(RRT_DOM_demo ${OpenCV_LIBS})

add_executable(RRTStar_DOM_demo RRTStar_DOM_demo.cpp)
target_link_libraries(RRTStar_DOM_demo ${OpenCV_LIBS})

add_executable(RRT_DOM_demo_test RRT_DOM_demo_test.cpp)
target_link_libraries(RRT_DOM_demo_test ${OpenCV_LIBS})

add_executable(RRT_DOM_Vec RRT_DOM_Vec.cpp)
target_link_libraries(RRT_DOM_Vec ${OpenCV_LIBS})

add_executable(path_smoothing_test path_smoothing_test.cpp)
target_link_libraries(path_smoothing_test ${OpenCV_LIBS})