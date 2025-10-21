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

#ifndef RPC_CORE_RPC_REQUEST_BUILDER_HPP_
#define RPC_CORE_RPC_REQUEST_BUILDER_HPP_

#include <atomic>
#include <optional>
#include <tuple>
#include <utility>

#include "rpc_core/rpc_traits.hpp"
#include "rpc_core/rpc_types.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace rpc {

namespace detail {

template <typename T>
void pack_arg(RpcRequestHeader& header, T&& value) {
  using DecayedT = std::decay_t<T>;
  static_assert(
    is_serializable_v<DecayedT> || is_payload_v<DecayedT>
      || std::is_same_v<DecayedT, RpcRequestHeader>,
    "Unsupported argument type passed to prepare_request. Must be a "
    "serializable primitive, string, vector, TensorMeta, or a valid payload "
    "type (UcxBuffer, UcxBufferVec).");

  if constexpr (
    is_payload_v<DecayedT> || std::is_same_v<DecayedT, RpcRequestHeader>) {
    return;
  }

  ParamMeta meta;
  meta.type = get_param_type<DecayedT>();

  constexpr bool is_arithmetic_or_enum =
    std::is_arithmetic_v<DecayedT> || std::is_enum_v<DecayedT>;

  if constexpr (is_arithmetic_or_enum) {
    meta.value.template emplace<PrimitiveValue>(std::forward<T>(value));
  } else if constexpr (is_cista_strong<DecayedT>::value) {
    meta.value.template emplace<PrimitiveValue>(value.v_);
  } else if constexpr (std::is_same_v<DecayedT, data::string>) {
    meta.value.template emplace<data::string>(std::forward<T>(value));
  } else if constexpr (std::is_same_v<DecayedT, TensorMeta>) {
    meta.value.template emplace<TensorMeta>(std::forward<T>(value));
  } else if constexpr (is_data_vector<DecayedT>::value) {
    meta.value.template emplace<VectorValue>(std::forward<T>(value));
  }

  header.add_param(std::move(meta));
}

template <typename... Args>
void pack_args(RpcRequestHeader& header, Args&&... args) {
  (pack_arg(header, std::forward<Args>(args)), ...);
}

// Finds the index and count of context types in a type pack.
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

template <typename... Args>
struct PayloadExtractor {
 private:
  using finder = payload_finder_impl<0, std::decay_t<Args>...>;

 public:
  static constexpr size_t payload_count = finder::payload_count;
  static constexpr size_t payload_index = finder::index;
};

}  // namespace detail

class RpcRequestBuilder {
 public:
  RpcRequestBuilder() = default;

  template <typename... Args>
  std::pair<RpcRequestHeader, std::optional<PayloadVariant>> prepare_request(
    function_id_t function_id, session_id_t session_id, Args&&... args) {
    using Extractor = detail::PayloadExtractor<Args...>;
    static_assert(
      Extractor::payload_count <= 1,
      "RPC calls can have at most one payload argument.");

    RpcRequestHeader header;
    header.function_id = function_id;
    header.session_id = session_id;

    auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);

    std::apply(
      [&header](auto&&... a) {
        detail::pack_args(header, std::forward<decltype(a)>(a)...);
      },
      args_tuple);

    if constexpr (Extractor::payload_count == 1) {
      auto payload = std::get<Extractor::payload_index>(std::move(args_tuple));
      return {
        std::move(header),
        std::make_optional(PayloadVariant(std::move(payload)))};
    } else {
      return {std::move(header), std::nullopt};
    }
  }

  template <typename... Args>
  std::pair<RpcRequestHeader, std::optional<PayloadVariant>> prepare_request(
    function_id_t function_id, RpcFunctionSignature signature,
    session_id_t session_id, Args&&... args) {
    using Extractor = detail::PayloadExtractor<Args...>;
    static_assert(
      Extractor::payload_count <= 1,
      "RPC calls can have at most one payload argument.");

    RpcRequestHeader header;
    header.function_id = function_id;
    header.session_id = session_id;

    auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);

    std::apply(
      [&header](auto&&... a) {
        detail::pack_args(header, std::forward<decltype(a)>(a)...);
      },
      args_tuple);

    if constexpr (Extractor::payload_count == 1) {
      auto payload = std::get<Extractor::payload_index>(std::move(args_tuple));
      return {
        std::move(header),
        std::make_optional(PayloadVariant(std::move(payload)))};
    } else {
      return {std::move(header), std::nullopt};
    }
  }

 private:
};

}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_RPC_REQUEST_BUILDER_HPP_
