#ifndef _CUDA_MAT_MUL_
#define _CUDA_MAT_MUL_

#include <iostream>
#include <cstdlib>
#include <vector>
#include <chrono>

template<typename T>
__global__ void matrix_multiplication_kernel(const T* A, const T* B, T* C, const size_t m, const size_t n, const size_t p)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m || j >= p) return;

    // printf("(%ld, %ld) : (%d, %d, %d), (%d, %d, %d)\n", i, j, blockIdx.y, blockDim.y, threadIdx.y, blockIdx.x, blockDim.x, threadIdx.x);

    T result = 0;

    for (size_t k = 0; k < n; k++)
    {
        result += A[i * n + k] * B[k * p + j];
    }

    C[i * p + j] = result;
}

template<typename T>
__global__ void matrix_multiplication_shared_kernel(const T* A, const T* B, T* C, const size_t m, const size_t n, const size_t p)
{
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    T result = 0;

    for (size_t i = 0; i < n / blockDim.x; i++)
    {
        __shared__ T sub_matrix_A[32][32];
        __shared__ T sub_matrix_B[32][32];

        sub_matrix_A[threadIdx.y][threadIdx.x] = A[y * n + blockDim.x * i + threadIdx.x];
        sub_matrix_B[threadIdx.y][threadIdx.x] = B[(blockDim.y * i + threadIdx.y) * p + x];

        __syncthreads();

        for (size_t j = 0; j < blockDim.x; j++)
        {
            result += sub_matrix_A[threadIdx.y][j] * sub_matrix_B[j][threadIdx.x];
        }

        __syncthreads();
    }

    for (size_t i = n / blockDim.x; i < (n + blockDim.x - 1) / blockDim.x; i++)
    {
        __shared__ T sub_matrix_A[32][32];
        __shared__ T sub_matrix_B[32][32];

        if (y * n + blockDim.x * i + threadIdx.x < n * m)
            sub_matrix_A[threadIdx.y][threadIdx.x] = A[y * n + blockDim.x * i + threadIdx.x];

        if ((blockDim.y * i + threadIdx.y) * p + x < m * p)
            sub_matrix_B[threadIdx.y][threadIdx.x] = B[(blockDim.y * i + threadIdx.y) * p + x];

        __syncthreads();

        for (size_t j = 0; j < blockDim.x; j++)
        {
            result += sub_matrix_A[threadIdx.y][j] * sub_matrix_B[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (y < m && x < p)
    {
        C[y * p + x] = result;
    }
}

template<typename T, bool shared_mode>
auto matrix_multiplication_cuda(const T* A, const T* B, T* C, const size_t m, const size_t n, const size_t p)
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
    blocks_per_grid.x = static_cast<int>(std::ceil(static_cast<double>(p) / threads_per_block.x));
    blocks_per_grid.y = static_cast<int>(std::ceil(static_cast<double>(m) / threads_per_block.y));

    if constexpr (shared_mode)
    {
        matrix_multiplication_shared_kernel<<<blocks_per_grid, threads_per_block>>>(device_A, device_B, device_C, m, n, p);
    }
    else
    {
        matrix_multiplication_kernel<<<blocks_per_grid, threads_per_block>>>(device_A, device_B, device_C, m, n, p);
    }

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
