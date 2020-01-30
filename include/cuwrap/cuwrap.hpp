#pragma once

// I'll just follow the CXX14 standard. Lol.

#include <cuwrap/defaults.hpp>
#include <cuwrap/utility/macro.hpp>

namespace cuwrap {

struct kparam_t {
    std::size_t threads_per_block{ defaults::n_threads_per_block };
    std::size_t blocks{ defaults::n_blocks /* 0 -> A hint that it is not initialized */ };
    std::size_t shared_size{ defaults::n_shared_size };
    std::size_t cuda_stream{ defaults::n_cuda_stream };
    kparam_t() noexcept = default;
    kparam_t(int, int = 0);
    void adapt_amount(int, int = 0);
    bool is_default_initialized() const noexcept;
};

} // namespace cuwrap
