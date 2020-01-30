#pragma once

#include <cstddef>

namespace cuwrap {

namespace defaults {

    constexpr auto n_blocks = 0; // A hint that it is not initialized.
    constexpr auto n_threads_per_block = 32; // For most cuda devices, 32 <= max_thread_num_per_block.
    constexpr auto n_shared_size = 48 * (1 << 10);
    constexpr auto n_cuda_stream = 0;

} // namespace defaults

} // namespace cuwrap
