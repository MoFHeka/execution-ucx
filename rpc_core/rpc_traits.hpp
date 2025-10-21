/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.
 *
 *Licensed under the Apache License Version 2.0 with LLVM Exceptions
 *(the "License"); you may not use this file except in compliance with
 *the License. You may obtain a copy of the License at
 *
 *    https://llvm.org/LICENSE.txt
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 *==============================================================================*/

#pragma once

#ifndef RPC_CORE_RPC_TRAITS_HPP_
#define RPC_CORE_RPC_TRAITS_HPP_

#include <cista.h>

#include <tuple>
#include <type_traits>

#include "rpc_core/rpc_types.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace rpc {

// =============================================================================
// SFINAE Type Traits for C++17 Compatibility
// =============================================================================

// Checks if a type is a std::pair
template <typename T>
struct is_pair : std::false_type {};
template <typename T1, typename T2>
struct is_pair<std::pair<T1, T2>> : std::true_type {};

// Checks if a type is RpcResponseHeader
template <typename T>
struct is_rpc_response_header : std::false_type {};
template <>
struct is_rpc_response_header<RpcResponseHeader> : std::true_type {};

// Checks if a type is cista::strong
template <typename T>
struct is_cista_strong : std::false_type {};
template <typename T, typename Tag>
struct is_cista_strong<cista::strong<T, Tag>> : std::true_type {};

// Helper to get the underlying type of a cista::strong
template <typename T>
struct get_cista_strong_underlying_type;

template <typename T, typename Tag>
struct get_cista_strong_underlying_type<cista::strong<T, Tag>> {
  using type = T;
};

// Helper to safely get the first type of a pair
template <typename T>
struct get_pair_first_type {
  using type = void;
};
template <typename T1, typename T2>
struct get_pair_first_type<std::pair<T1, T2>> {
  using type = T1;
};

// C++20 function traits for deducing properties of callables.
template <typename T>
struct function_traits;

template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
  using return_type = R;
  using args_tuple = std::tuple<Args...>;
  static constexpr size_t arity = sizeof...(Args);
};

template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...)> {
  using return_type = R;
  using class_type = C;
  using args_tuple = std::tuple<Args...>;
  static constexpr size_t arity = sizeof...(Args);
};

template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...) const> {
  using return_type = R;
  using class_type = C;
  using args_tuple = std::tuple<Args...>;
  static constexpr size_t arity = sizeof...(Args);
};

template <typename Lambda>
struct function_traits : function_traits<decltype(&Lambda::operator())> {};

// Helper to check if a type is present in a std::variant.
template <typename T, typename Variant>
struct is_in_variant;

template <typename T, typename... Types>
struct is_in_variant<T, std::variant<Types...>>
  : std::disjunction<std::is_same<T, Types>...> {};

// Helper to check for specific payload types by deriving from PayloadVariant.
template <typename T>
struct is_payload_type {
  static constexpr bool value =
    !std::is_same_v<std::decay_t<T>, std::monostate>
    && is_in_variant<std::decay_t<T>, PayloadVariant>::value;
};

template <typename T>
constexpr bool is_payload_type_v = is_payload_type<T>::value;

template <typename T>
constexpr bool is_payload_v = is_payload_type_v<T>;

// Helper to deduce the input context type from the function signature.
template <typename T>
struct is_data_vector : std::false_type {};
template <typename T>
struct is_data_vector<data::vector<T>> : std::true_type {};

template <typename T>
constexpr ParamType get_param_type() {
  using DecayedT = std::decay_t<T>;

  if constexpr (
    is_payload_v<DecayedT> || std::is_same_v<DecayedT, RpcRequestHeader>) {
    return ParamType::UNKNOWN;  // These are filtered out
  } else if constexpr (
    std::is_arithmetic_v<DecayedT> || std::is_enum_v<DecayedT>) {
    return get_primitive_param_info<DecayedT>().type;
  } else if constexpr (is_cista_strong<DecayedT>::value) {
    using UnderlyingType =
      typename get_cista_strong_underlying_type<DecayedT>::type;
    return get_primitive_param_info<UnderlyingType>().type;
  } else if constexpr (std::is_same_v<DecayedT, data::string>) {
    return ParamType::STRING;
  } else if constexpr (std::is_same_v<DecayedT, TensorMeta>) {
    return ParamType::TENSOR_META;
  } else if constexpr (is_data_vector<DecayedT>::value) {
    using ElementType = typename DecayedT::value_type;
    return get_vector_param_info<ElementType>().type;
  } else {
    return ParamType::UNKNOWN;
  }
}

template <typename T>
constexpr bool is_serializable_v = get_param_type<T>() != ParamType::UNKNOWN;

template <typename Tuple, size_t Index = 0>
struct payload_finder;

template <typename Tuple, size_t Index>
struct payload_finder {
  using current_element_raw = std::tuple_element_t<Index, Tuple>;
  using current_element = std::decay_t<current_element_raw>;

  using type = std::conditional_t<
    is_payload_type_v<current_element>, current_element,
    typename payload_finder<Tuple, Index + 1>::type>;
};

template <typename Tuple>
struct payload_finder<Tuple, std::tuple_size_v<Tuple>> {
  using type = void;
};

}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_RPC_TRAITS_HPP_
