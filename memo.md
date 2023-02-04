# CUDA で行列積

## はじめに

SIMD に続いて GPGPU にもあまり馴染みがないので、触ってみたかった。
お題は「行列積」

## naive

```c
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
```

### 性能

環境：手持ちの AMD Ryzen 7 7700X 8Core

size = 2000
time = 4455630[micro sec]

3590962445.2658772[FLOPS] = 3.6[MFLOPS]

## CUDA 使うと

```c
template<typename T>
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

    matrix_multiplication_kernel<<<blocks_per_grid, threads_per_block>>>(device_A, device_B, device_C, m, n, p);

    cudaDeviceSynchronize();

    cudaMemcpy(C, device_C, sizeof(T) * m * p, cudaMemcpyDeviceToHost);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());

    return elapsed_time;
}
```
host から device に情報を送る。
各スレッドで行列積の1要素分の計算をする。
カーネルはこんな感じ。

```c
template<typename T>
__global__ void matrix_multiplication_kernel(const T* A, const T* B, T* C, const size_t m, const size_t n, const size_t p)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m || j >= p) return;

    T result = 0;

    for (size_t k = 0; k < n; k++)
    {
        result += A[i * n + k] * B[k * p + j];
    }

    C[i * p + j] = result;
}
```

### 性能

環境：NVIDIA GeForce RTX 3060 Ti

size = 2000^3 * 2
time = 192594[micro sec]

83076315980.76784[flops] = 83[MFLOPS]

すごい増えた。すごい。

## shared memory

shared memory を意識して書くともっと速くなると聞いた。

どういう風に計算するかというと、、、
TODO: 図


```c
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
```
### 性能

環境：NVIDIA GeForce RTX 3060 Ti

size = 2000^3 * 2
time = 23478[micro sec]

681489053582.0769[flops] = 681[MFLOPS]

うわ、すごい増えた。すごい。

## 感想

- ちゃんと性能が高くなったという意味では成功
- SIMD

## 参考

- http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/?CUDA%A4%C7%B9%D4%CE%F3%B1%E9%BB%BB%A1%A7%BE%E8%BB%BB%28%A5%B7%A5%A7%A5%A2%A1%BC%A5%C9%A5%E1%A5%E2%A5%EA%BB%C8%CD%D1%C8%C7%29
- https://www.slideshare.net/ssuserf87701/2015gpgpu9-59179722
- http://www.butsuri.it-chiba.ac.jp/~yasutake/matter/tominaga.pdf
- https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
