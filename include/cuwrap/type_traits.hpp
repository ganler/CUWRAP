#pragma once

#include <cstddef>
#include <cuwrap/utility/macro.hpp>
#include <type_traits>

namespace cuwrap {

// ---------------------------------------------------------------------------------------------
// I hope one day cuda could support cxx17 by default.

// template <typename T, typename... Ts>
// inline constexpr bool belong_to_v = std::disjunction_v<std::is_same<T, Ts>...>;

// template <typename T>
// inline constexpr bool belong_to_arithm_v = belong_to_v<T, CUWRAP_ARITHMETIC_TS>;
// ----------------------------------------------------------------------------------------------

template <typename U, typename V, typename... Ts>
struct belong_to {
    static constexpr bool value = std::is_same<U, V>::value || belong_to<U, Ts...>::value;
};

template <typename U, typename V>
struct belong_to<U, V> {
    static constexpr bool value = std::is_same<U, V>::value;
};

template <typename T>
struct belong_to_arithm {
    static constexpr bool value = belong_to<T, CUWRAP_ARITHMETIC_TS>::value;
};

} // namespace cuwrap
