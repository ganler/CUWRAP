#include "test.hpp"
#include <cuwrap/kernels/mul.hpp>

TEST(Test, MulTwoVectors)
{
    constexpr std::size_t N = 1 << 20;
    int *a = new int[N](),
        *b = new int[N](),
        *r = new int[N]();
    for (int i = 0; i < N; i++) {
        a[i] = 2.0f;
        b[i] = 2.0f;
    }

    cuwrap::mul(N, a, b, r);

    for (int i = 0; i < N; i++)
        ASSERT_EQ(r[i], 4);

    delete[] a;
    delete[] b;
    delete[] r;
}
