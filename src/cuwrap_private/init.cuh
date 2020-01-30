#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuwrap {
namespace priv {
    template <typename T>
    __global__ void init(T* src, T val, int sz)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        int stride = gridDim.x * blockDim.x;
        for (int i = index; i < sz; i += stride)
            src[i] = val;
    }
} // namespace priv

} // namespace cuwrap
