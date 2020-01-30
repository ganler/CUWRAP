#include "test.hpp"
#include <cuwrap/utility/device.hpp>

TEST(Test, PrintDevices)
{
    EXPECT_TRUE(::cuwrap::util::devinfo(true) > 0);
}
