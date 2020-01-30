#pragma once

#include <cuwrap/cuwrap.hpp>
#include <cuwrap/type_traits.hpp>

namespace cuwrap {

template <typename T>
struct mul_impl_t {
    void operator()(std::size_t, T*, T*, T*, kparam_t); // def but not impl.
};

template <typename T>
void mul(std::size_t n, T* lhs, T* rhs, T* out, kparam_t param = kparam_t{})
{
    static_assert(belong_to_arithm<T>::value, "[[T] is not implemented in the cuwrap library.] ==> Check `CUWRAP_ARITHMETIC_TS` in cuwrap/utility/macro.hpp");
    mul_impl_t<T>{}(n, lhs, rhs, out, param);
}

} // namespace cuwrap
