cmake_minimum_required (VERSION 3.1)

project(agner)

add_library(${PROJECT_NAME} STATIC instrset.cpp)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

