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
void print_matrix(const std::vector<T> &vec)
{
    for (auto i : vec)
    {
        std::cerr << std::setprecision(10) << i << " ";
    }
    std::cerr << std::endl;
}

template<typename T>
void set_random_number(std::vector<T> &A, std::vector<T> &B)
{
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<> dist1(-10., 10.);

    for (int i = 0; i < A.size(); i++)
    {
        A.at(i) = static_cast<T>(dist1(engine));
    }

    for (int i = 0; i < B.size(); i++)
    {
        B.at(i) = static_cast<T>(dist1(engine));
    }
}

template<typename T>
bool check_result(std::vector<T> &naive_C, std::vector<T> &cuda_C, size_t m, size_t n)
{
    assert(naive_C.size() == cuda_C.size());

    auto eps = std::numeric_limits<T>::epsilon();

    for (int i = 0; i < naive_C.size(); i++)
    {
        if (std::abs(naive_C.at(i) - cuda_C.at(i)) > eps)
        {
            std::cerr << i << std::endl;
            return false;
        }
    }

    return true;
}

template<typename T>
void test()
{
    constexpr size_t n = 300;

    std::cout << "size,naive,cuda" << std::endl;

    for (size_t i = 1; i < 2 * n; i++)
    {
        std::vector<double> naive_elapsed_time_list;
        std::vector<double> cuda_elapsed_time_list;

        std::vector<T> A(i * i);
        std::vector<T> B(i * i);

        set_random_number(A, B);

        for (size_t j = 0; j < 5; j++)
        {
            std::vector<T> naive_C(i * i);
            auto naive_elapsed_time = matrix_multiprication_naive(A.data(), B.data(), naive_C.data(), i, i, i);

            std::vector<T> cuda_C(i * i);
            auto cuda_elapsed_time = matrix_multiprication_cuda(A.data(), B.data(), cuda_C.data(), i, i, i);

            if (!check_result(naive_C, cuda_C, i, i))
            {
                std::cout << "Result failed." << std::endl;
                std::cerr << "naive : ";
                print_matrix(naive_C);
                std::cerr << "cuda :  ";
                print_matrix(cuda_C);
                return;
            }

            naive_elapsed_time_list.push_back(naive_elapsed_time);
            cuda_elapsed_time_list.push_back(cuda_elapsed_time);
        }

        std::sort(naive_elapsed_time_list.begin(), naive_elapsed_time_list.end());
        std::sort(cuda_elapsed_time_list.begin(), cuda_elapsed_time_list.end());

        std::cout << i << "," << naive_elapsed_time_list[2] << "," << cuda_elapsed_time_list[2] << std::endl;
    }
}

int main()
{
    print_device_information();

    test<uint8_t>();
}
