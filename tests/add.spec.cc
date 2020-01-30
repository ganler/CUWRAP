#include "test.hpp"

#include <cuwrap/kernels/add.hpp>

TEST(Test, AddTwoVectors)
{
    std::size_t N = 1 << 20;
    float *a = new float[N],
          *b = new float[N],
          *r = new float[N];

    for (int i = 0; i < N; i++)
        a[i] = b[i] = 2.0f;

    cuwrap::add(N, a, b, r);

    for (int i = 0; i < N; i++)
        ASSERT_EQ(r[i], 4.0f);

    delete[] a;
    delete[] b;
    delete[] r;
}
