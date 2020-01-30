#pragma once

#include <cstddef>
#include <iostream>

#define CUWRAP_IF_CUDA_ERR(err_code)                                                                                         \
    {                                                                                                                        \
        auto&& err_code_ = err_code;                                                                                         \
        if (err_code_ != cudaSuccess) {                                                                                      \
            std::cerr << "==> [ERROR] ==> [CUDA] ==> [LINE]" << __LINE__ << " ==> [CALLING]" << __PRETTY_FUNCTION__ << '\n'; \
            std::cerr << "\t ==> [DETAILS]" << cudaGetErrorString(cudaError_t(err_code_)) << '\n';                           \
            std::exit(err_code_);                                                                                            \
        }                                                                                                                    \
    }

#define CUWRAP_ARITHMETIC_TS std::uint8_t, std::uint16_t,  \
                             std::uint32_t, std::uint64_t, \
                             std::int8_t, std::int16_t,    \
                             std::int32_t, std::int64_t,   \
                             float, double, bool

#define CUWRAP_DEF_FUNC_IMPL(func_name) \
    template <typename T>               \
    static void func_name##_impl(std::size_t n, const T* lhs, const T* rhs, T* out, const cuwrap::kparam_t& param = cuwrap::kparam_t{})

#define CUWRAP_DEF_FUNC(func_name)                                                                                                                          \
    CUWRAP_DEF_FUNC_IMPL(func_name);                                                                                                                        \
    template <typename T>                                                                                                                                   \
    void func_name(std::size_t n, const T* lhs, const T* rhs, T* out, const cuwrap::kparam_t& param = cuwrap::kparam_t{})                                   \
    {                                                                                                                                                       \
        static_assert("[[T] is not implemented in the cuwrap library.] ==> Check `CUWRAP_ARITHMETIC_TS` in cuwrap/utility/macro.hpp", is_arithm_impl_v<T>); \
        func_name##_impl(n, lhs, rhs, out, param);                                                                                                          \
    }
