cmake_minimum_required (VERSION 3.1)

project(libretracker)

# Enable C++ 11 support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to debug build
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")

option (OPENCL_ENABLED "Use OpenCL" OFF) 

if (OPENCL_ENABLED)
add_definitions(-DOPENCL_ENABLED)
endif ()

find_package(OpenCV REQUIRED core imgproc imgcodecs highgui videoio calib3d)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/agner)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/eigen EXCLUDE_FROM_ALL)

set(TBB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/tbb EXCLUDE_FROM_ALL)
set_target_properties(tbb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties(tbbmalloc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties(tbbmalloc_proxy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if (OPENCL_ENABLED)
find_package(Boost REQUIRED)
find_package(OpenCL REQUIRED)
endif ()

file(GLOB SRC "src/*.cpp")
add_executable(${PROJECT_NAME} ${SRC})
target_include_directories(${PROJECT_NAME} PRIVATE "${OpenCV_INCLUDE_DIRS}")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/eigen")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/agner")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/tbb/include")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} agner tbb stdc++fs)

if (OPENCL_ENABLED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARY})
endif ()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-msse4.1 SSE41_IS_SUPPORTED)
if (${SSE41_IS_SUPPORTED})
target_compile_definitions(${PROJECT_NAME} PRIVATE SSE41_ENABLED)
set_source_files_properties("src/sse41_kernel.cpp" PROPERTIES COMPILE_FLAGS -msse4.1)
else()
message(WARNING "${CMAKE_CXX_COMPILER} does not support SSE4.1, not building AVX kernel")
endif()

check_cxx_compiler_flag(-mavx AVX_IS_SUPPORTED)
if (${AVX_IS_SUPPORTED})
target_compile_definitions(${PROJECT_NAME} PRIVATE AVX_ENABLED)
set_source_files_properties("src/avx_kernel.cpp" PROPERTIES COMPILE_FLAGS -mavx)
else()
message(WARNING "${CMAKE_CXX_COMPILER} does not support AVX, not building AVX kernel")
endif()

check_cxx_compiler_flag(-mavx2 AVX2_IS_SUPPORTED)
if (${AVX2_IS_SUPPORTED})
target_compile_definitions(${PROJECT_NAME} PRIVATE AVX2_ENABLED)
set_source_files_properties("src/avx2_kernel.cpp" PROPERTIES COMPILE_FLAGS -mavx2)
else()
message(WARNING "${CMAKE_CXX_COMPILER} does not support AVX2, not building AVX2 kernel")
endif()

check_cxx_compiler_flag(-march=skylake-avx512 AVX512_IS_SUPPORTED)
check_cxx_compiler_flag(-mprefer-vector-width=512 AVX512_VECTOR_LENGTH_IS_SUPPORTED)
if (${AVX512_IS_SUPPORTED} AND ${AVX512_VECTOR_LENGTH_IS_SUPPORTED})
target_compile_definitions(${PROJECT_NAME} PRIVATE AVX512_ENABLED)
set_source_files_properties("src/avx512_kernel.cpp" PROPERTIES COMPILE_FLAGS "-march=skylake-avx512 -mprefer-vector-width=512")
else()
message(WARNING "${CMAKE_CXX_COMPILER} does not support AVX512, not building AVX512 kernel")
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/data.tar.gz
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data.tar.gz
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Unpacking sample image set")

