cmake_minimum_required (VERSION 3.1)

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

if (OPENCL_ENABLED)
find_package(Boost REQUIRED)
find_package(OpenCL REQUIRED)
endif ()

file(GLOB SRC "src/*.cpp")
add_executable(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PRIVATE "${OpenCV_INCLUDE_DIRS}")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/eigen")
target_link_libraries(${PROJECT_NAME} GL pthread ${OpenCV_LIBRARIES} stdc++fs)

if (OPENCL_ENABLED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARY})
endif ()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/data.tar.gz
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data.tar.gz
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Unpacking sample image set")

