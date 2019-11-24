cmake_minimum_required (VERSION 3.1)

project(libretracker)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")

# Enable C++ 11 support
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

# Search path for CMake include files.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
	${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/gpurt/cmake)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")

find_package(CUDA)

find_package(OpenCV REQUIRED core imgproc imgcodecs)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/agner)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/eigen EXCLUDE_FROM_ALL)

set(TBB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/tbb EXCLUDE_FROM_ALL)
set_target_properties(tbb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties(tbbmalloc PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties(tbbmalloc_proxy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if (CUDA_FOUND)
file(GLOB SRC "src/*.cpp" "src/*.cu")
find_package(GPURT REQUIRED)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/gpurt)
set_property(TARGET gpurt PROPERTY POSITION_INDEPENDENT_CODE ON)
add_opencl_kernel("${CMAKE_CURRENT_SOURCE_DIR}/src/opencl_kernel.cl" SRC)
cuda_add_library(${PROJECT_NAME} SHARED ${SRC})
else ()
file(GLOB SRC "src/*.cpp")
add_library(${PROJECT_NAME} SHARED ${SRC})
endif ()
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")

target_include_directories(${PROJECT_NAME} PRIVATE "${OpenCV_INCLUDE_DIRS}")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/eigen")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/agner")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/tbb/include")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} agner tbb)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
target_link_libraries(${PROJECT_NAME} stdc++fs)
endif()

if (CUDA_FOUND)
target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_GPU)
target_include_directories(${PROJECT_NAME} PRIVATE "${GPURT_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME} ${GPURT_LIBRARIES})
endif ()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-msse4.1 SSE41_IS_SUPPORTED)
if ("X${SSE41_IS_SUPPORTED}" STREQUAL "X")
set(SSE41_IS_SUPPORTED 0)
endif()
if (${SSE41_IS_SUPPORTED})
target_compile_definitions(${PROJECT_NAME} PRIVATE SSE41_ENABLED)
set_source_files_properties("src/sse41_kernel.cpp" PROPERTIES COMPILE_FLAGS -msse4.1)
else()
message(WARNING "${CMAKE_CXX_COMPILER} does not support SSE4.1, not building AVX kernel")
endif()

check_cxx_compiler_flag(-mavx AVX_IS_SUPPORTED)
if ("X${AVX_IS_SUPPORTED}" STREQUAL "X")
set(AVX_IS_SUPPORTED 0)
endif()
if (${AVX_IS_SUPPORTED})
target_compile_definitions(${PROJECT_NAME} PRIVATE AVX_ENABLED)
set_source_files_properties("src/avx_kernel.cpp" PROPERTIES COMPILE_FLAGS -mavx)
else()
message(WARNING "${CMAKE_CXX_COMPILER} does not support AVX, not building AVX kernel")
endif()

check_cxx_compiler_flag(-mavx2 AVX2_IS_SUPPORTED)
if ("X${AVX2_IS_SUPPORTED}" STREQUAL "X")
set(AVX2_IS_SUPPORTED 0)
endif()
if (${AVX2_IS_SUPPORTED})
target_compile_definitions(${PROJECT_NAME} PRIVATE AVX2_ENABLED)
set_source_files_properties("src/avx2_kernel.cpp" PROPERTIES COMPILE_FLAGS -mavx2)
else()
message(WARNING "${CMAKE_CXX_COMPILER} does not support AVX2, not building AVX2 kernel")
endif()

check_cxx_compiler_flag(-march=skylake-avx512 AVX512_IS_SUPPORTED)
check_cxx_compiler_flag(-mprefer-vector-width=512 AVX512_VECTOR_LENGTH_IS_SUPPORTED)
if ("X${AVX512_IS_SUPPORTED}" STREQUAL "X")
set(AVX512_IS_SUPPORTED 0)
endif()
if ("X${AVX512_VECTOR_LENGTH_IS_SUPPORTED}" STREQUAL "X")
set(AVX512_VECTOR_LENGTH_IS_SUPPORTED 0)
endif()
if (${AVX512_IS_SUPPORTED} AND ${AVX512_VECTOR_LENGTH_IS_SUPPORTED})
target_compile_definitions(${PROJECT_NAME} PRIVATE AVX512_ENABLED)
set_source_files_properties("src/avx512_kernel.cpp" PROPERTIES COMPILE_FLAGS "-march=skylake-avx512 -mprefer-vector-width=512")
else()
message(WARNING "${CMAKE_CXX_COMPILER} does not support AVX512, not building AVX512 kernel")
endif()

add_executable(${PROJECT_NAME}_mezzanine "src/mezzanine/main.cpp")
target_include_directories(${PROJECT_NAME}_mezzanine PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(${PROJECT_NAME}_mezzanine ${PROJECT_NAME})

if (CUDA_FOUND)
target_compile_definitions(${PROJECT_NAME}_mezzanine PRIVATE ENABLE_GPU)
endif()

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_SOURCE_DIR}/data.tar.gz
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/data.tar.gz
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Unpacking sample image set")

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(TARGETS ${PROJECT_NAME}_mezzanine DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/pupil_tracker.h DESTINATION include)

string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
string(TOLOWER ${CMAKE_SYSTEM_NAME} OS)
set(ARCHIVE "${PROJECT_NAME}-${ARCH}-${OS}-${BUILD_TYPE}")
set(CPACK_PACKAGE_FILE_NAME ${ARCHIVE})
set(CPACK_GENERATOR "ZIP")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
set(CPACK_OUTPUT_FILE_PREFIX ${CMAKE_INSTALL_PREFIX})

include(CPack)
