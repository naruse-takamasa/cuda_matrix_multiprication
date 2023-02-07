# CUDA で行列積

## はじめに

GPGPU にもあまり馴染みがないので、触ってみたかった。
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

size = 2000 * 2000
time = 4338260[micro sec]

3688114589.7203026[FLOPS] = 3.69[GFLOPS]

## CUDA を使う

host から device に情報を送って、各スレッドに色々計算させる。
ブロックのサイズやらグリッドのサイズやらはひとまず放置。

```c
template<typename T>
void matrix_multiplication(const T* A, const T* B, T* C, const size_t m, const size_t n, const size_t p)
{
    T *device_A, *device_B, *device_C;
    cudaMalloc(&device_A, sizeof(T) * m * n);
    cudaMalloc(&device_B, sizeof(T) * n * p);
    cudaMalloc(&device_C, sizeof(T) * m * p);

    cudaMemcpy(device_A, A, sizeof(T) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, B, sizeof(T) * n * p, cudaMemcpyHostToDevice);

    dim3 threads_per_block = {?, ?};
    dim3 blocks_per_grid = {?, ?};

    kernel<<<blocks_per_grid, threads_per_block>>>(device_A, device_B, device_C, m, n, p);

    cudaDeviceSynchronize();

    cudaMemcpy(C, device_C, sizeof(T) * m * p, cudaMemcpyDeviceToHost);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
}
```

## 1スレッドに全部解かせる

```c
kernel<<<{1, 1}, {1, 1}>>>(device_A, device_B, device_C, m, n, p);
```

として、1 つのスレッドに行列積を解いてもらう。

```c
template<typename T>
__global__ void kernel(const T* A, const T* B, T* C, const size_t m, const size_t n, const size_t p)
{
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
}
```

### 性能

環境：NVIDIA GeForce RTX 3060 Ti

size = 500 * 500
time = 6324170[micro sec]

39530879.150939964[FLOPS] = 39.53[MFLOPS]

CPU のときより性能が低い。

## 1 スレッドが 1 要素分だけ解く

各スレッドで行列積の1要素分の計算をする。
カーネルは自分の担当する要素の結果だけを計算。

何も考えずにブロックのサイズを`{32, 32}` にして計算してみる。

```c
template<typename T>
__global__ void kernel(const T* A, const T* B, T* C, const size_t m, const size_t n, const size_t p)
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

size = 2000 * 2000
time = 29576[micro sec]

540979172301.86633[FLOPS] = 540.98[GFLOPS]

大分性能が上がった。

### ブロックのサイズについて

適当にブロックのサイズをいじってみた。
- `{32, 16}` だと 24812 [micro sec]
- `{1024, 1}` だと 37734
- `{512, 2}` だと 29710

## shared memory を意識する

shared memory を意識して書くともっと速くなるという情報を得たのでやってみたくなった。

どういう風に計算するかというと、、、
TODO: 図


```c
__global__ void kernel(const T* A, const T* B, T* C, const size_t m, const size_t n, const size_t p)
{
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;

    T result = 0;

    __shared__ T sub_matrix_A[32][32];
    __shared__ T sub_matrix_B[32][32];

    for (size_t i = 0; i < n / blockDim.x; i++)
    {
        sub_matrix_A[threadIdx.y][threadIdx.x] = A[y * n + blockDim.x * i + threadIdx.x];
        sub_matrix_B[threadIdx.y][threadIdx.x] = B[(blockDim.y * i + threadIdx.y) * p + x];

        __syncthreads();

        for (size_t j = 0; j < blockDim.x; j++)
        {
            result += sub_matrix_A[threadIdx.y][j] * sub_matrix_B[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (n % blockDim.x != 0)
    {
        size_t i = n / blockDim.x;

        sub_matrix_A[threadIdx.y][threadIdx.x] = 0;
        sub_matrix_B[threadIdx.y][threadIdx.x] = 0;

        if (y * n + blockDim.x * i + threadIdx.x < m * n)
        {
            sub_matrix_A[threadIdx.y][threadIdx.x] = A[y * n + blockDim.x * i + threadIdx.x];
        }

        if ((blockDim.y * i + threadIdx.y) * p + x < n * p)
        {
            sub_matrix_B[threadIdx.y][threadIdx.x] = B[(blockDim.y * i + threadIdx.y) * p + x];
        }

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

size = 2000 * 2000
time = 23828[micro sec]

671478932348.4976[FLOPS] = 671.48[GFLOPS]

確かに増えた。

## 感想

- ちゃんと性能が高くなったという意味では成功
  - 1スレッド分の性能は CPU 以下っぽいということも分かった
- shared memory 使う方は端数処理で少しハマった
- block のサイズをいろいろ変えると速度に大きな影響を与えることは実験中に知った
  - けど、具体的にどんな値にすると速いとかは分かってないし、まだ試せていないのでまた今度やる


```
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 3060 Ti"
  CUDA Driver Version / Runtime Version          11.8 / 11.5
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 8192 MBytes (8589410304 bytes)
MapSMtoCores for SM 8.6 is undefined.  Default to use 128 Cores/SM
MapSMtoCores for SM 8.6 is undefined.  Default to use 128 Cores/SM
  (38) Multiprocessors, (128) CUDA Cores/MP:     4864 CUDA Cores
  GPU Max Clock rate:                            1665 MHz (1.66 GHz)
  Memory Clock rate:                             7001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 3145728 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.8, CUDA Runtime Version = 11.5, NumDevs = 1, Device0 = NVIDIA GeForce RTX 3060 Ti
Result = PASS
```

```
lscpu
アーキテクチャ:                        x86_64
  CPU 操作モード:                      32-bit, 64-bit
  Address sizes:                       48 bits physical, 48 bits virtual
  バイト順序:                          Little Endian
CPU:                                   16
  オンラインになっている CPU のリスト: 0-15
ベンダー ID:                           AuthenticAMD
  モデル名:                            AMD Ryzen 7 7700X 8-Core Processor
    CPU ファミリー:                    25
    モデル:                            97
    コアあたりのスレッド数:            2
    ソケットあたりのコア数:            8
    ソケット数:                        1
    ステッピング:                      2
    BogoMIPS:                          8999.81
    フラグ:                            fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush m
                                       mx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep
                                       _good nopl tsc_reliable nonstop_tsc cpuid extd_apicid pni pclmulqdq ssse3 fma cx1
                                       6 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_leg
                                       acy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext ssbd ibrs ibp
                                       b stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed
                                        adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt x
                                       savec xgetbv1 xsaves avx512_bf16 clzero xsaveerptr arat npt nrip_save tsc_scale v
                                       mcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload avx51
                                       2vbmi umip avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpo
                                       pcntdq rdpid fsrm
Virtualization features:
  仮想化:                              AMD-V
  ハイパーバイザのベンダー:            Microsoft
  仮想化タイプ:                        完全仮想化
Caches (sum of all):
  L1d:                                 256 KiB (8 instances)
  L1i:                                 256 KiB (8 instances)
  L2:                                  8 MiB (8 instances)
  L3:                                  32 MiB (1 instance)
Vulnerabilities:
  Itlb multihit:                       Not affected
  L1tf:                                Not affected
  Mds:                                 Not affected
  Meltdown:                            Not affected
  Spec store bypass:                   Mitigation; Speculative Store Bypass disabled via prctl and seccomp
  Spectre v1:                          Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:                          Mitigation; Full AMD retpoline, IBPB conditional, IBRS_FW, STIBP conditional, RSB
                                        filling
  Srbds:                               Not affected
  Tsx async abort:                     Not affected
```

## 参考

- http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/?CUDA%A4%C7%B9%D4%CE%F3%B1%E9%BB%BB%A1%A7%BE%E8%BB%BB%28%A5%B7%A5%A7%A5%A2%A1%BC%A5%C9%A5%E1%A5%E2%A5%EA%BB%C8%CD%D1%C8%C7%29
- https://www.slideshare.net/ssuserf87701/2015gpgpu9-59179722
- https://www.slideshare.net/ssuserf87701/2015gpgpu10-59179759
- http://www.butsuri.it-chiba.ac.jp/~yasutake/matter/tominaga.pdf
- https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
