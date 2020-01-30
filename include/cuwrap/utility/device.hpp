#pragma once

#include <cuwrap/utility/macro.hpp>
#include <iostream>
#include <vector>

namespace cuwrap {

namespace util {

    const std::vector<uint32_t>& dev_max_thread_per_block();
    int devinfo(bool show = true);

} // namespace info

} // namespace cuwrap
