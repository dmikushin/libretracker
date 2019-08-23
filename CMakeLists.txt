cmake_minimum_required (VERSION 3.1)

include(CheckIncludeFiles)

project(libretracker)

# Default to debug build
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

option (OPENCL_ENABLED "Use OpenCL" OFF) 

if (OPENCL_ENABLED)
add_definitions(-DOPENCL_ENABLED)
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/eigen)
find_package(OpenCV REQUIRED core imgproc imgcodecs highgui videoio calib3d)
set(FLTK_SKIP_FLUID true)
find_package(FLTK REQUIRED)

if (OPENCL_ENABLED)
find_package(Boost REQUIRED)
find_package(OpenCL REQUIRED)
include_directories (${Boost_INCLUDE_DIRS})
include_directories (${OpenCL_INCLUDE_DIRS})
endif ()

file(GLOB SRC "src/*.cpp")
add_executable(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PRIVATE "${FLTK_INCLUDE_DIR}")
target_include_directories(${PROJECT_NAME} PRIVATE "${OpenCV_INCLUDE_DIRS}")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/eigen")
target_link_libraries(${PROJECT_NAME} ${FLTK_LIBRARIES} GL pthread ${OpenCV_LIBRARIES})

if (OPENCL_ENABLED)
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARY})
endif ()

