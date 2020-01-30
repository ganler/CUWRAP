#include <cuwrap/cuwrap.hpp>
#include <cuwrap/utility/device.hpp>
#include <mutex>
#include <vector>

namespace cuwrap {

namespace util {

    static std::vector<cudaDeviceProp> cached_dev_properties;
    static std::vector<uint32_t> dev_max_thread_per_block_impl; // For the access of C++ code... It sucks.
    static void call_once_to_get_properties()
    {
        int n_devices = 0;
        CUWRAP_IF_CUDA_ERR(cudaGetDeviceCount(&n_devices))
        cached_dev_properties = decltype(cached_dev_properties)(n_devices);
        dev_max_thread_per_block_impl.reserve(n_devices);
        const auto peak_mem_band = [](auto bus_width, auto clock_rate) -> float {
            return 2.0 * clock_rate * (bus_width / 8) / 1.0e+6;
        };
        for (auto&& property : cached_dev_properties) {
            cudaGetDeviceProperties(&property, dev_max_thread_per_block_impl.size());
            dev_max_thread_per_block_impl.emplace_back(property.maxThreadsPerBlock);
        }
    }

    static std::once_flag dev_max_thread_per_block_impl_flag{};
    const std::vector<uint32_t>& dev_max_thread_per_block()
    {
        // So, to avoit the overhead produced by CXX11's atomic access of static var in functions, I just took the `once_flag` out of the scope.
        std::call_once(dev_max_thread_per_block_impl_flag, call_once_to_get_properties); // I do not know why this sh*t is such long...
        return dev_max_thread_per_block_impl;
    }

    int devinfo(bool show)
    {
        const static auto peak_mem_band = [](auto bus_width, auto clock_rate) -> float {
            return 2.0 * clock_rate * (bus_width / 8) / 1.0e+6;
        };
        dev_max_thread_per_block();
        if (show) {
            for (const auto& property : cached_dev_properties) {
                std::cout << ">>> Device name:\t" << property.name << '\n';
                std::cout << "\t@Memory clock rate (KHz):\t" << property.memoryClockRate << '\n';
                std::cout << "\t@Memory Bus Width (bits):\t" << property.memoryBusWidth << '\n';
                std::cout << "\t@Peak Memory Bandwidth (GB/s):\t" << peak_mem_band(property.memoryBusWidth, property.memoryClockRate) << '\n';
                std::cout << "\t@Max Grid Size:\t" << property.maxGridSize << '\n';
                std::cout << "\t@Max Threads Per Block:\t" << property.maxThreadsPerBlock << '\n';
                std::cout << "\t@Max Threads Per Dim:\t" << property.maxThreadsDim << '\n';
                std::cout << "\t@Warp Size:\t" << property.warpSize << '\n';
            }
        }
        return cached_dev_properties.size();
    }

} // namespace util

} // namespace cuwrap