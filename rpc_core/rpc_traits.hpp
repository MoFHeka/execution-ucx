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
#include <vector>

#include "rpc_core/rpc_types.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace rpc {

class RpcResponseBuilder;

// =============================================================================
// SFINAE Type Traits for C++17 Compatibility
// =============================================================================

// Checks if a type is a std::pair
template <typename T>
struct is_pair : std::false_type {};
template <typename T1, typename T2>
struct is_pair<std::pair<T1, T2>> : std::true_type {};

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

template <typename R, typename... Args>
struct function_traits<R(Args...)> {
  using return_type = R;
  using args_tuple = std::tuple<Args...>;
  static constexpr size_t arity = sizeof...(Args);
};

namespace detail {
template <typename R, typename T>
struct function_signature_builder;

template <typename R, typename... Args>
struct function_signature_builder<R, std::tuple<Args...>> {
  using type = R(Args...);
};
}  // namespace detail

// Helper to deduce function signature R(Args...) from any callable type T
template <typename T>
struct signature_from_traits {
  using traits = function_traits<std::decay_t<T>>;
  using type = typename detail::function_signature_builder<
    typename traits::return_type,
    typename traits::args_tuple>::type;
};

template <typename T>
struct is_std_function : std::false_type {};
template <typename R, typename... Args>
struct is_std_function<std::function<R(Args...)>> : std::true_type {};

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

template <typename T>
constexpr PayloadType get_payload_type() {
  using DecayedT = std::decay_t<T>;
  if constexpr (std::is_same_v<DecayedT, ucxx::UcxBuffer>) {
    return PayloadType::UCX_BUFFER;
  } else if constexpr (std::is_same_v<DecayedT, ucxx::UcxBufferVec>) {
    return PayloadType::UCX_BUFFER_VEC;
  } else {
    return PayloadType::MONOSTATE;
  }
}

// Helper to deduce the input context type from the function signature.
template <typename T>
struct is_data_vector : std::false_type {};
template <typename T>
struct is_data_vector<data::vector<T>> : std::true_type {};

template <typename T>
struct is_std_vector : std::false_type {};
template <typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

template <typename T>
constexpr ParamType get_param_type() {
  using DecayedT = std::decay_t<T>;

  // Payloads are handled separately and not serialized as parameters.
  if constexpr (is_payload_v<DecayedT>) {
    return ParamType::UNKNOWN;
  } else if constexpr (
    std::is_arithmetic_v<DecayedT> || std::is_enum_v<DecayedT>) {
    return get_primitive_param_info<DecayedT>().type;
  } else if constexpr (is_cista_strong<DecayedT>::value) {
    using UnderlyingType =
      typename get_cista_strong_underlying_type<DecayedT>::type;
    return get_primitive_param_info<UnderlyingType>().type;
  } else if constexpr (
    std::is_same_v<DecayedT, data::string>
    || std::is_same_v<DecayedT, std::string>
    || std::is_same_v<DecayedT, std::string_view>
    || std::is_same_v<DecayedT, const char*>) {
    return ParamType::STRING;
  } else if constexpr (std::is_same_v<DecayedT, TensorMeta>) {
    return ParamType::TENSOR_META;
  } else if constexpr (
    is_data_vector<DecayedT>::value || is_std_vector<DecayedT>::value) {
    using ElementType = typename DecayedT::value_type;
    return get_vector_param_info<ElementType>().type;
  } else {
    // Types like RpcRequestHeader, RpcResponseBuilder, and any other
    // unsupported types will fall through to here.
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
    is_payload_type_v<current_element>,
    current_element,
    typename payload_finder<Tuple, Index + 1>::type>;
};

template <typename Tuple>
struct payload_finder<Tuple, std::tuple_size_v<Tuple>> {
  using type = void;
};

// Finds the index and count of context types in a type pack.
namespace detail {
template <size_t I, typename... Ts>
struct payload_finder_impl;

template <size_t I, typename T, typename... Ts>
struct payload_finder_impl<I, T, Ts...> {
  static constexpr bool is_current_payload = is_payload_v<T>;
  using next_finder = payload_finder_impl<I + 1, Ts...>;
  static constexpr size_t payload_count =
    (is_current_payload ? 1 : 0) + next_finder::payload_count;
  static constexpr size_t index = is_current_payload ? I : next_finder::index;
};

template <size_t I>
struct payload_finder_impl<I> {
  static constexpr size_t payload_count = 0;
  static constexpr size_t index = -1;
};
}  // namespace detail

template <typename... Args>
struct PayloadExtractor {
 private:
  using finder = detail::payload_finder_impl<0, std::decay_t<Args>...>;

 public:
  static constexpr size_t payload_count = finder::payload_count;
  static constexpr size_t payload_index = finder::index;
};

// Merges two tuples into a new tuple of a specified type.
// The new tuple's types must be a superset of the types in the source tuples.
template <typename ResultTuple, typename Tuple1, typename Tuple2>
ResultTuple tuple_merge_by_type(Tuple1&& t1, Tuple2&& t2) {
  return std::apply(
    [&](auto&&... args1) {
      return std::apply(
        [&](auto&&... args2) {
          return ResultTuple{std::get<std::decay_t<decltype(args1)>>(
            std::forward_as_tuple(args1..., args2...))...};
        },
        std::forward<Tuple2>(t2));
    },
    std::forward<Tuple1>(t1));
}

// Filters a tuple's types based on a type trait.
template <typename Tuple, template <typename> class Trait, typename ResultTuple>
struct filter_tuple_by_trait;

template <
  typename Head,
  typename... Tail,
  template <typename>
  class Trait,
  typename... ResultArgs>
struct filter_tuple_by_trait<
  std::tuple<Head, Tail...>,
  Trait,
  std::tuple<ResultArgs...>> {
  using type = std::conditional_t<
    Trait<Head>::value,
    typename filter_tuple_by_trait<
      std::tuple<Tail...>,
      Trait,
      std::tuple<ResultArgs..., Head>>::type,
    typename filter_tuple_by_trait<
      std::tuple<Tail...>,
      Trait,
      std::tuple<ResultArgs...>>::type>;
};

template <template <typename> class Trait, typename... ResultArgs>
struct filter_tuple_by_trait<std::tuple<>, Trait, std::tuple<ResultArgs...>> {
  using type = std::tuple<ResultArgs...>;
};

}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_RPC_TRAITS_HPP_
