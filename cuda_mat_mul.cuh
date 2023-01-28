#ifndef _CUDA_MAT_MUL_
#define _CUDA_MAT_MUL_

#include <iostream>
#include <cstdlib>
#include <vector>
#include <chrono>

// #define ERROR_CHECK(call)                                                   /
// {                                                                           /
//     const cudaError_t error = call;                                         /
//     if (error != cudaSuccess)                                               /
//     {                                                                       /
//         std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << std::endl; /
//         std::cerr << "Code: " << error << ", reason: "                      /
//             << cudaGetErrorString(error) << std::endl;                      /
//         std::exit(EXIT_FAILURE);                                            /
//     }                                                                       /
// }

template<typename T>
__global__ void matrix_multiprication_kernel(const T* A, const T* B, T* C, size_t m, size_t n, size_t p)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m || j >= p) return;

    // printf("(%ld, %ld) : (%d, %d, %d), (%d, %d, %d)\n", i, j, blockIdx.y, blockDim.y, threadIdx.y, blockIdx.x, blockDim.x, threadIdx.x);

    T res = 0.;
    for (size_t k = 0; k < n; k++)
    {
        res += A[i * n + k] * B[k * p + j];
    }
    C[i * p + j] = res;
}

template<typename T>
auto matrix_multiprication_cuda(const T* A, const T* B, T* C, const size_t m, const size_t n, const size_t p)
{
    auto begin = std::chrono::high_resolution_clock::now();

    T *device_A, *device_B, *device_C;
    cudaMalloc(&device_A, sizeof(T) * m * n);
    cudaMalloc(&device_B, sizeof(T) * n * p);
    cudaMalloc(&device_C, sizeof(T) * m * p);

    cudaMemcpy(device_A, A, sizeof(T) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, sizeof(T) * n * p, cudaMemcpyHostToDevice);

    dim3 threads_per_block = {32, 32};
    dim3 blocks_per_grid = {1, 1};
    blocks_per_grid.x = (int)std::ceil(static_cast<double>(p) / threads_per_block.x);
    blocks_per_grid.y = (int)std::ceil(static_cast<double>(m) / threads_per_block.y);

    // std::cout << blocks_per_grid.x << ", " << blocks_per_grid.y << std::endl;
    // std::cout << threads_per_block.x << ", " << threads_per_block.y << std::endl;

    matrix_multiprication_kernel<<<blocks_per_grid, threads_per_block>>>(device_A, device_B, device_C, m, n, p);

    cudaDeviceSynchronize();

    cudaMemcpy(C, device_C, sizeof(T) * m * p, cudaMemcpyDeviceToHost);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());

    return elapsed_time;
}

#endif
