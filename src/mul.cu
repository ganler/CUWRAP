#include <cuwrap/kernels/mul.hpp>
#include <initializer_list>
#include <tuple>

namespace cuwrap {

template <typename T>
__global__ void kmul(T* lhs, T* rhs, T* out, std::size_t maxn)
{
    int index = (blockDim.x * blockIdx.x) + threadIdx.x; // Thread id in one grid.
    int stride = gridDim.x * blockDim.x; // Thread num for each grid.
    for (std::size_t i = index; i < maxn; i += stride)
        out[i] = lhs[i] * rhs[i];
}

// (std::size_t n, const T* lhs, const T* rhs, T* out, const kparam_t& param = kparam_t{})
template <typename T>
void mul_impl_t<T>::operator()(std::size_t n, T* lhs, T* rhs, T* out, kparam_t param) // But well, there will be a lot of time wasted during each kernel section.
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

    kmul<<<param.blocks, param.threads_per_block, param.shared_size, (CUstream_st*)param.cuda_stream>>>(cl, cr, cr, n);
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
    std::initializer_list<std::nullptr_t>{ ((mul_impl_t<Ts>{})(std::size_t{}, nullptr, nullptr, nullptr, kparam_t{}), nullptr)... };
}

void force_initialization_()
{
    force_initialization__<CUWRAP_ARITHMETIC_TS>();
}

} // namespace cuwrap

// MAT MUL ==> TO be reimplemented =================================

// #pragma once

// #include "../utils/util.hpp"
// #include "init.cuh"

// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

// #include <type_traits>

// template <
// 	typename T,
// 	typename = std::enable_if_t<std::is_arithmetic_v<T>>> __global__
// void tiled_matmul(const T* lhs, const T* rhs, T* out, const int n, const int tile_sz)
// {   // This is what ONE thread is doing !!!
// 	// For one thread => get one value in the output matrix;
// 	// For one block  => get one tiel in the output matrix;
// 	constexpr std::size_t mat_sz = 1024; // As my GPU supports 1024 threads per block :)
// 	__shared__ T sA[mat_sz], sB[mat_sz]; // Shared in the block.

// 	// Position for the thread: C[row][col];
// 	int row = blockIdx.x * tile_sz + threadIdx.x;
// 	int col = blockIdx.y * tile_sz + threadIdx.y;
// 	// Tx, Ty \in [0, 31].

// 	T temp = 0;
// 	for (int i = 0; i < n / tile_sz; ++i) // cros all tiles. ONE block 2 tile in one loop.
// 	{   // In one `block x block` section.
// 		// To calculate C[row][col] we need to initialize: tiles => A, B // Note that y ~ row, x ~ col.
// 		sA[tile_sz * threadIdx.y + threadIdx.x] = lhs[row * n + i * tile_sz + threadIdx.x];
// 		sB[tile_sz * threadIdx.y + threadIdx.x] = rhs[col + i * tile_sz * n + threadIdx.y * n]; // A better way maybe trans it.
// 		__syncthreads(); // BLOCK_lhs & BLOCK_rhs >>> shared memory prepared.

// 		for (int j = 0; j < tile_sz; ++j) // Micro kernel. Consider sA & sB only.
// 			temp += sA[tile_sz * threadIdx.y + j] * sB[j * tile_sz + threadIdx.x];
// 		__syncthreads();
// 	}
// 	out[row * n + col] = temp;
// }

// template <
// 	typename T,
// 	typename = std::enable_if_t<std::is_arithmetic_v<T>>> __global__
// void naive_matmul(const T* lhs, const T* rhs, T* out, const int n)
// {
// 	int row = blockDim.y * blockIdx.y + threadIdx.y;
// 	int col = blockDim.x * blockIdx.x + threadIdx.x;

// 	T tem = 0;
// 	if (row < n && col < n)
// 	{
// 		for (int k = 0; k < n; ++k)
// 			tem += lhs[row * n + k] * rhs[k * n + col];
// 		out[row * n + col] = tem;
// 	}
// }

// void tiled_matmul_test()
// {
// 	using type = float;

// 	int mygpu = cudaGetDevice(&mygpu);
// 	type* lhs, * rhs, * dst;

// 	// It's 2-D. But we just consider 1-D first.
// 	constexpr int matsz = (1 << 12);
// 	cudaMallocManaged(&lhs, sizeof(type) * matsz * matsz);
// 	cudaMallocManaged(&rhs, sizeof(type) * matsz * matsz);
// 	cudaMallocManaged(&dst, sizeof(type) * matsz * matsz);

// 	cudaMemPrefetchAsync(lhs, sizeof(type) * matsz * matsz, mygpu);
// 	cudaMemPrefetchAsync(rhs, sizeof(type) * matsz * matsz, mygpu);
// 	cudaMemPrefetchAsync(dst, sizeof(type) * matsz * matsz, mygpu);

// 	constexpr int threads_per_blk = 16; // 32 x 32 => 1024
// 	constexpr int blks_per_grid = (matsz + threads_per_blk - 1) / threads_per_blk;

// 	dim3 blks(blks_per_grid, blks_per_grid), thres(threads_per_blk, threads_per_blk);

// 	constexpr int init_blks = (matsz * matsz + 1023) / 1024;
// 	init << <init_blks, 1024 >> > (lhs, 1.f, matsz * matsz);
// 	init << <init_blks, 1024 >> > (rhs, 1.f, matsz * matsz);
// 	cudaDeviceSynchronize();

// 	std::cout << "NAIVE_MATMAL: \n";
// 	ganler::timer t;
// 	naive_matmul << <blks, thres >> > (lhs, rhs, dst, matsz);
// 	cudaDeviceSynchronize();
// 	t.print_milli();

// 	std::cout << "TILED_MATMAL: \n";
// 	t.reset();
// 	tiled_matmul << <blks, thres >> > (lhs, rhs, dst, matsz, threads_per_blk); // Slower ? Why ?
// 	cudaDeviceSynchronize();
// 	t.print_milli();

// 	cudaMemPrefetchAsync(dst, sizeof(type) * matsz * matsz, cudaCpuDeviceId);

// 	std::cout << dst[matsz * matsz - 1] << '\n';

// 	cudaFree(lhs);
// 	cudaFree(rhs);
// 	cudaFree(dst);
// }