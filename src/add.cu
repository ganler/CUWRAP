#include <cuwrap/kernels/add.hpp>
#include <initializer_list>
#include <tuple>

namespace cuwrap {

template <typename T>
__global__ void kadd(T* lhs, T* rhs, T* out, std::size_t maxn)
{
    int index = (blockDim.x * blockIdx.x) + threadIdx.x; // Thread id in one grid.
    int stride = gridDim.x * blockDim.x; // Thread num for each grid.
    for (std::size_t i = index; i < maxn; i += stride)
        out[i] = lhs[i] + rhs[i];
}

// (std::size_t n, const T* lhs, const T* rhs, T* out, const kparam_t& param = kparam_t{})
template <typename T>
void add_impl_t<T>::operator()(std::size_t n, T* lhs, T* rhs, T* out, kparam_t param) // But well, there will be a lot of time wasted during each kernel section.
{
    if (param.is_default_initialized())
        param.adapt_amount(n);

    T *cl, *cr;
    CUWRAP_IF_CUDA_ERR(cudaMalloc(&cl, n * sizeof(T)));
    if (lhs == rhs)
        cr = cl;
    else
        CUWRAP_IF_CUDA_ERR(cudaMalloc(&cr, n * sizeof(T)));

    CUWRAP_IF_CUDA_ERR(cudaMemcpy(cl, lhs, n * sizeof(T), cudaMemcpyHostToDevice));
    if (lhs != rhs)
        CUWRAP_IF_CUDA_ERR(cudaMemcpy(cr, rhs, n * sizeof(T), cudaMemcpyHostToDevice));

    kadd<<<param.blocks, param.threads_per_block, param.shared_size, (CUstream_st*)param.cuda_stream>>>(cl, cr, cr, n);
    CUWRAP_IF_CUDA_ERR(cudaMemcpy(out, cr, n * sizeof(T), cudaMemcpyDeviceToHost));
    cudaFree(cl);
    if (lhs != rhs)
        cudaFree(cr);

    // int mygpu = cudaGetDevice(&mygpu); // TODO: Specify the custom setting for GPU choice.

    // CUWRAP_IF_CUDA_ERR(cudaMallocManaged(&lhs, sizeof(T) * n));
    // if (lhs != rhs)
    //     CUWRAP_IF_CUDA_ERR(cudaMallocManaged(&rhs, sizeof(T) * n));
    // if (lhs != out && rhs != out)
    //     CUWRAP_IF_CUDA_ERR(cudaMallocManaged(&out, sizeof(T) * n));

    // CUWRAP_IF_CUDA_ERR(cudaMemPrefetchAsync(lhs, sizeof(T) * n, mygpu)); // => GPU
    // CUWRAP_IF_CUDA_ERR(cudaMemPrefetchAsync(rhs, sizeof(T) * n, mygpu)); // => GPU
    // CUWRAP_IF_CUDA_ERR(cudaMemPrefetchAsync(out, sizeof(T) * n, mygpu)); // => GPU

    // kadd<<<param.blocks, param.threads_per_block, param.shared_size, (CUstream_st*)param.cuda_stream>>>(lhs, rhs, out, n);
    // CUWRAP_IF_CUDA_ERR(cudaDeviceSynchronize());
    // CUWRAP_IF_CUDA_ERR(cudaMemPrefetchAsync(out, sizeof(T) * n, cudaCpuDeviceId)); // => CPU

    // cudaFree(lhs);
    // if (lhs != rhs)
    //     cudaFree(rhs);
    // if (lhs != out && rhs != out)
    //     cudaFree(out);
}

template <typename... Ts>
static void force_initialization__()
{
    // (add_impl<Ts>(std::size_t{}, nullptr, nullptr, nullptr), ...); // CUDA: We do not support CXX17 currently.
    std::initializer_list<std::nullptr_t>{ ((add_impl_t<Ts>{})(std::size_t{}, nullptr, nullptr, nullptr, kparam_t{}), nullptr)... };
}

void force_initialization()
{
    force_initialization__<CUWRAP_ARITHMETIC_TS>();
}

} // namespace cuwrap

// ----------------------------------------------------------------------------------------------------------------
// Here're some notes that you may just ignore:

// CPU适合粗粒度的并行，而GPU适合细粒度的并行
// 因为CPU在进行上下文切换的时候需要把寄存器的数据丢到RAM，而GPU只需要通过寄存器组调度者选择对应的内容即可

// CPU的失速问题：对于大量小任务，CPU的执行时间大都会浪费在上下文切换上，而且基于时间片的调度策略（将时间平
// 均分片给每个线程）导致当线程多起来的时候上下文切换增加，从而导致效率降低。
// GPU的实现往往更高效，有效的使用`工作池`来保证其一直有事情做，当当前指令一直带进行等待的时候，SM会切换到另
// 外一个指令流，之后再执行被堵塞的指令

// cuda里面自动和host同步的：
// - cudaMalloc
// - cudaDeviceSynchronize
// - cudaMemcpy
// - Free

// 跑kernel是不需要同步的
// 所以在跑完kernel后一定要在·使用前·调用有同步功能的代码。

// Unified Memory is a single memory address space accessible from any processor in a system.
// 简单来说就是多个处理器的共享内存。

// The important point here is that the Pascal GPU architecture is the first with hardware
// support for virtual memory page faulting and migration, via its Page Migration Engine.
// Older GPUs based on the Kepler and Maxwell architectures also support a more limited form
// of Unified Memory. (1070Ti ~ Pascal)

// 对于Pascal GPU，只有当access(by GPU/CPU等) cudaMallocManaged()产生的内存的时候，他才会真正的分配内存。

// 减少migration overhead的tips:
// - Init data on GPU.
// - Unified Memory.

// CUBLAS VERSION ==================================================================

// #pragma once

// #include "../utils/util.hpp"

// #include <algorithm>
// #include <cublas_v2.h>
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <vector>

// void test_cublas_vecadd()
// {
//     using type = float;
//     std::size_t size = 1 << 20;
//     std::vector<type> cpu_data(size);
//     std::generate(cpu_data.begin(), cpu_data.end(), std::rand);

//     type *cuda_a, *cuda_b;
//     cudaMalloc(&cuda_a, sizeof(type) * size);
//     cudaMalloc(&cuda_b, sizeof(type) * size);

//     // Cublas initialization
//     cublasHandle_t cublas_hander;
//     cublasCreate_v2(&cublas_hander);

//     // mem : cpu => gpu
//     cublasSetVector(size, sizeof(float), cpu_data.data(), 1, cuda_a, 1);
//     cublasSetVector(size, sizeof(float), cpu_data.data(), 1, cuda_b, 1);

//     // @@
//     ganler::timer t; // CuBlas is 5x faster than my code.
//     // Now we got: a, b => we want: c = a*scale +b;
//     constexpr type scale = 2.0;
//     cublasSaxpy_v2(cublas_hander, size, &scale, cuda_a, 1, cuda_b, 1);
//     t.print_milli();

//     cublasDestroy_v2(cublas_hander);
//     cudaFree(cuda_a);
//     cudaFree(cuda_b);
// }