#ifndef _NAIVE_MAT_MUL_
#define _NAIVE_MAT_MUL_

#include <chrono>

template<typename T>
auto matrix_multiprication_naive(const T* A, const T* B, T* C, const size_t m, const size_t n, const size_t p)
{
    auto begin = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            for (size_t k = 0; k < n; k++)
            {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());

    return elapsed_time;
}

#endif
