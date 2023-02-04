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
        A.at(i) = static_cast<T>(dist1(engine));
        // A.at(i) = 1;
    }

    for (size_t i = 0; i < B.size(); i++)
    {
        B.at(i) = static_cast<T>(dist1(engine));
        // B.at(i) = 1;
    }
}

template<typename T>
bool check_result(std::vector<T> &naive_C, std::vector<T> &cuda_C)
{
    assert(naive_C.size() == cuda_C.size());

    auto eps = std::numeric_limits<T>::epsilon();

    for (size_t i = 0; i < naive_C.size(); i++)
    {
        if (std::abs(naive_C.at(i) - cuda_C.at(i)) > eps)
        {
            std::cerr << "index:" << i << std::endl;
            std::cerr << naive_C.at(i) << ", " << cuda_C.at(i) << std::endl;

            return false;
        }
    }

    return true;
}

template<typename T>
void test()
{
    constexpr size_t n = 101;

    std::cout << "size,naive,cuda,cuda-shared" << std::endl;

    for (size_t i = 1; i < n; i += 10)
    {
        std::vector<double> naive_elapsed_time_list;
        std::vector<double> cuda_elapsed_time_list;
        std::vector<double> cuda_shared_elapsed_time_list;

        std::vector<T> A(i * i);
        std::vector<T> B(i * i);

        set_random_number(A, B);

        for (size_t j = 0; j < 5; j++)
        {
            std::vector<T> naive_C(i * i);
            auto naive_elapsed_time = matrix_multiplication_naive(A.data(), B.data(), naive_C.data(), i, i, i);

            std::vector<T> cuda_C(i * i);
            auto cuda_elapsed_time = matrix_multiplication_cuda<T, false>(A.data(), B.data(), cuda_C.data(), i, i, i);

            std::vector<T> cuda_shared_C(i * i);
            auto cuda_shared_elapsed_time = matrix_multiplication_cuda<T, true>(A.data(), B.data(), cuda_shared_C.data(), i, i, i);

            if (!check_result(cuda_C, cuda_shared_C))
            {
                std::cout << "Result failed." << std::endl;
                // std::cerr << "naive : \n";
                print_matrix(naive_C, i, i);
                // std::cerr << "cuda :  \n";
                print_matrix(cuda_shared_C, i, i);
                return;
            }

            naive_elapsed_time_list.push_back(naive_elapsed_time);
            cuda_elapsed_time_list.push_back(cuda_elapsed_time);
            cuda_shared_elapsed_time_list.push_back(cuda_shared_elapsed_time);
        }

        std::sort(naive_elapsed_time_list.begin(), naive_elapsed_time_list.end());
        std::sort(cuda_elapsed_time_list.begin(), cuda_elapsed_time_list.end());
        std::sort(cuda_shared_elapsed_time_list.begin(), cuda_shared_elapsed_time_list.end());

        std::cout << i << ",";
        std::cout << naive_elapsed_time_list.at(2) << ",";
        std::cout << cuda_elapsed_time_list.at(2) <<",";
        std::cout << cuda_shared_elapsed_time_list.at(2) << std::endl;
        std::cerr << i << " ok" << std::endl;
    }
}

int main()
{
    // print_device_information();

    test<float>();
}
