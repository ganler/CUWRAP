#include "test.hpp"
#include <cuwrap/cuwrap.hpp>

TEST(Test, ParamClass)
{
    EXPECT_TRUE(cuwrap::kparam_t{}.is_default_initialized());
}