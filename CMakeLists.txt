cmake_minimum_required(VERSION 3.25)

project(matrix_multiprication LANGUAGES CXX CUDA)

add_executable(mat_mul
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_mat_mul.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/naive_mat_mul.hpp
)

target_compile_options(mat_mul PRIVATE -Wall -Wextra)

target_compile_features(mat_mul PUBLIC cxx_std_14)
