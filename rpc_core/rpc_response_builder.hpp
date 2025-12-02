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

#include <string>
#include <tuple>
#include <utility>

#include "rpc_core/rpc_traits.hpp"
#include "rpc_core/rpc_types.hpp"

namespace eux {
namespace rpc {

namespace detail {

template <typename T>
void PackResult(RpcResponseHeader& header, T&& value) {
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
  constexpr bool is_string_type = std::is_same_v<DecayedT, data::string>
                                  || std::is_same_v<DecayedT, std::string>
                                  || std::is_same_v<DecayedT, std::string_view>
                                  || std::is_same_v<DecayedT, const char*>;

  if constexpr (is_arithmetic_or_enum) {
    meta.value.template emplace<PrimitiveValue>(std::forward<T>(value));
  } else if constexpr (is_cista_strong<DecayedT>::value) {
    meta.value.template emplace<PrimitiveValue>(value.v_);
  } else if constexpr (is_string_type) {
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

  header.AddResult(std::move(meta));
}

template <typename... Args>
void PackResults(RpcResponseHeader& header, Args&&... args) {
  (PackResult(header, std::forward<Args>(args)), ...);
}

}  // namespace detail

class RpcResponseBuilder {
 public:
  RpcResponseBuilder() = default;

  template <typename... Args>
  auto PrepareResponse(
    session_id_t session_id, request_id_t request_id, Args&&... args) const {
    using Extractor = PayloadExtractor<Args...>;
    static_assert(
      Extractor::payload_count <= 1,
      "RPC responses can have at most one payload argument.");

    RpcResponseHeader header;
    header.session_id = session_id;
    header.request_id = request_id;
    header.status = RpcStatus(std::error_code{});

    auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);

    std::apply(
      [&header](auto&&... a) {
        detail::PackResults(header, std::forward<decltype(a)>(a)...);
      },
      args_tuple);

    if constexpr (Extractor::payload_count == 1) {
      using RawTuple = decltype(args_tuple);
      // Handle both value and (const) reference cases for args_tuple
      if constexpr (
        std::is_lvalue_reference_v<RawTuple>
        || std::is_const_v<std::remove_reference_t<RawTuple>>) {
        auto&& payload = std::get<Extractor::payload_index>(args_tuple);
        using PayloadType = std::decay_t<decltype(payload)>;
        // For move-only types, we need to remove const before moving
        if constexpr (std::is_const_v<
                        std::remove_reference_t<decltype(payload)>>) {
          // Move-only types can't be copied, so we cast away const
          // and move (safe since we're transferring ownership)
          auto& non_const_payload =
            const_cast<std::remove_const_t<PayloadType>&>(payload);
          return std::make_pair(
            std::move(header),
            PayloadVariant(
              std::in_place_type<PayloadType>, std::move(non_const_payload)));
        } else {
          return std::make_pair(
            std::move(header),
            PayloadVariant(
              std::in_place_type<PayloadType>,
              std::forward<decltype(payload)>(payload)));
        }
      } else {
        auto&& payload =
          std::get<Extractor::payload_index>(std::move(args_tuple));
        using PayloadType = std::decay_t<decltype(payload)>;
        // For move-only types, we need to remove const before moving
        if constexpr (std::is_const_v<
                        std::remove_reference_t<decltype(payload)>>) {
          // Move-only types can't be copied, so we cast away const
          // and move (safe since we're transferring ownership)
          auto& non_const_payload =
            const_cast<std::remove_const_t<PayloadType>&>(payload);
          return std::make_pair(
            std::move(header),
            PayloadVariant(
              std::in_place_type<PayloadType>, std::move(non_const_payload)));
        } else {
          return std::make_pair(
            std::move(header),
            PayloadVariant(
              std::in_place_type<PayloadType>,
              std::forward<decltype(payload)>(payload)));
        }
      }
    } else {
      return header;
    }
  }
};

}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_RPC_RESPONSE_BUILDER_HPP_
