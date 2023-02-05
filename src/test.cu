#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <cassert>

#include "naive_mat_mul.hpp"
#include "cuda_mat_mul.cuh"

void print_device_information()
{
    int dev = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);
}

template<typename T>
void print_matrix(const std::vector<T> &vec, const size_t height, const size_t width)
{
    size_t idx = 0;

    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            std::cerr << vec.at(idx++) << " ";
        }
        std::cerr << std::endl;
    }
    std::cerr << std::endl;
}

template<typename T>
void set_random_number(std::vector<T> &A, std::vector<T> &B)
{
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<> dist1(-10., 10);

    for (size_t i = 0; i < A.size(); i++)
    {
#ifndef DEBUG_MODE
        A.at(i) = static_cast<T>(dist1(engine));
#else
        A.at(i) = 1;
#endif
    }

    for (size_t i = 0; i < B.size(); i++)
    {
#ifndef DEBUG_MODE
        B.at(i) = static_cast<T>(dist1(engine));
#else
        B.at(i) = 1;
#endif
    }
}

template<typename T>
bool check_result(std::vector<T> &naive_C, std::vector<T> &cuda_C)
{
    bool ok = true;
    assert(naive_C.size() == cuda_C.size());

    auto eps = 0.01;

    for (size_t i = 0; i < naive_C.size(); i++)
    {
        if (std::abs(naive_C.at(i) - cuda_C.at(i)) > eps)
        {
            std::cerr << "index:" << i << std::endl;
            std::cerr << naive_C.at(i) << ", " << cuda_C.at(i) << std::endl;

            ok = false;
        }
    }

    return ok;
}

template<typename T>
void test()
{
    constexpr size_t n = 2001;

    // std::cout << "size,cuda,cuda-shared" << std::endl;

    for (size_t i = 2000; i < n; i += 100)
    {
        std::vector<double> elapsed_time_list[4];

        std::vector<T> A(i * i);
        std::vector<T> B(i * i);

        for (size_t j = 0; j < 5; j++)
        {
            set_random_number(A, B);

            // std::vector<T> naive_C(i * i);
            // auto naive_elapsed_time = matrix_multiplication_naive(A.data(), B.data(), naive_C.data(), i, i, i);

            // std::vector<T> cuda_naive_C(i * i);
            // auto cuda_naive_elapsed_time = matrix_multiplication_cuda<T, 0>(A.data(), B.data(), cuda_naive_C.data(), i, i, i);

            // std::vector<T> cuda_C(i * i);
            // auto cuda_elapsed_time = matrix_multiplication_cuda<T, 1>(A.data(), B.data(), cuda_C.data(), i, i, i);

            std::vector<T> cuda_shared_C(i * i);
            auto cuda_shared_elapsed_time = matrix_multiplication_cuda<T, 2>(A.data(), B.data(), cuda_shared_C.data(), i, i, i);

#ifdef DEBUG_MODE
            if (!check_result(cuda_C, cuda_shared_C))
            {
                std::cout << "Result failed." << std::endl;
                std::cerr << "naive : \n";
                print_matrix(cuda_C, i, i);
                std::cerr << "cuda :  \n";
                print_matrix(cuda_shared_C, i, i);
                return;
            }
#endif

            // elapsed_time_list[0].push_back(naive_elapsed_time);
            // elapsed_time_list[1].push_back(cuda_naive_elapsed_time);
            // elapsed_time_list[2].push_back(cuda_elapsed_time);
            elapsed_time_list[3].push_back(cuda_shared_elapsed_time);
        }

        for (int j = 0; j < 4; j++)
        {
            std::sort(elapsed_time_list[j].begin(), elapsed_time_list[j].end());
        }

        std::cout << i << ",";
        // std::cout << elapsed_time_list[0].at(2) << ",";
        // std::cout << elapsed_time_list[1].at(2) <<",";
        // std::cout << elapsed_time_list[2].at(2) <<",";
        std::cout << elapsed_time_list[3].at(2) << std::endl;
        std::cerr << i << " ok" << std::endl;
    }
}

int main()
{
    // print_device_information();

    test<float>();
}
