cmake_minimum_required(VERSION 3.25)

project(matrix_multiprication LANGUAGES CXX CUDA)

add_executable(mat_mul
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_mat_mul.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/naive_mat_mul.hpp
)

target_compile_options(mat_mul PRIVATE -Wall -Wextra -O3 -arch=sm_80)

target_compile_features(mat_mul PRIVATE cxx_std_14)
