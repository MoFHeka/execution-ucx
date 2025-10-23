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

#ifndef RPC_CORE_RPC_RESPONSE_BUILDER_HPP_
#define RPC_CORE_RPC_RESPONSE_BUILDER_HPP_

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
void pack_result(RpcResponseHeader& header, T&& value) {
  using DecayedT = std::decay_t<T>;
  static_assert(
    is_serializable_v<DecayedT> || is_payload_v<DecayedT>
      || std::is_same_v<DecayedT, RpcResponseHeader>,
    "Unsupported argument type passed to prepare_response. Must be a "
    "serializable primitive, string, vector, TensorMeta, or a valid payload "
    "type (UcxBuffer, UcxBufferVec).");

  if constexpr (
    is_payload_v<DecayedT> || std::is_same_v<DecayedT, RpcResponseHeader>) {
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
  } else if constexpr (
    std::is_same_v<DecayedT, data::string>
    || std::is_same_v<DecayedT, std::string>
    || std::is_same_v<DecayedT, std::string_view>
    || std::is_same_v<DecayedT, const char*>) {
    meta.value.template emplace<data::string>(data::string{value});
  } else if constexpr (std::is_same_v<DecayedT, TensorMeta>) {
    meta.value.template emplace<TensorMeta>(std::forward<T>(value));
  } else if constexpr (is_data_vector<DecayedT>::value) {
    meta.value.template emplace<VectorValue>(std::forward<T>(value));
  } else if constexpr (is_std_vector<DecayedT>::value) {
    using ElementType = typename DecayedT::value_type;
    data::vector<ElementType> cista_vec;
    cista_vec.set(value.begin(), value.end());
    meta.value.template emplace<VectorValue>(std::move(cista_vec));
  }

  header.add_result(std::move(meta));
}

template <typename... Args>
void pack_results(RpcResponseHeader& header, Args&&... args) {
  (pack_result(header, std::forward<Args>(args)), ...);
}

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

class RpcResponseBuilder {
 public:
  RpcResponseBuilder() = default;

  template <typename... Args>
  auto prepare_response(
    session_id_t session_id, request_id_t request_id, Args&&... args) const {
    using Extractor = detail::PayloadExtractor<Args...>;
    static_assert(
      Extractor::payload_count <= 1,
      "RPC responses can have at most one payload argument.");

    RpcResponseHeader header;
    header.session_id = session_id;
    header.request_id = request_id;

    auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);

    std::apply(
      [&header](auto&&... a) {
        detail::pack_results(header, std::forward<decltype(a)>(a)...);
      },
      args_tuple);

    if constexpr (Extractor::payload_count == 1) {
      auto payload = std::get<Extractor::payload_index>(std::move(args_tuple));
      return std::make_pair(
        std::move(header), PayloadVariant(std::move(payload)));
    } else {
      return header;
    }
  }
};

}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_RPC_RESPONSE_BUILDER_HPP_
