#include <cuwrap/cuwrap.hpp>
#include <cuwrap/utility/device.hpp>

namespace cuwrap {

kparam_t::kparam_t(int amount, int dev_hint)
    : threads_per_block(util::dev_max_thread_per_block()[dev_hint])
    , blocks((amount + threads_per_block - 1) / threads_per_block)
{
}

void kparam_t::adapt_amount(int n, int devid)
{
    threads_per_block = util::dev_max_thread_per_block()[devid];
    blocks = (n + threads_per_block - 1) / threads_per_block;
}

bool kparam_t::is_default_initialized() const noexcept
{
    return blocks == defaults::n_blocks;
}

} // namespace cuwrap
