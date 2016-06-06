cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(ESD)

find_package(PCL 1.7 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CMAKE_CXX_FLAGS "-std=c++0x")

add_executable (ESD ESD.cpp)
target_link_libraries (ESD ${PCL_LIBRARIES})
add_executable (extraPlanarRefinements extraPlanarRefinements.cpp)
target_link_libraries (extraPlanarRefinements ${PCL_LIBRARIES})
add_executable (viewerESD viewerESD.cpp)
target_link_libraries (viewerESD ${PCL_LIBRARIES})
