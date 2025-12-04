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

#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "rpc_core/rpc_traits.hpp"
#include "rpc_core/rpc_types.hpp"

namespace eux {
namespace rpc {

namespace detail {

template <typename T>
void PackArg(RpcRequestHeader& header, T&& value) {
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

  constexpr bool is_string_convertable =
    std::is_same_v<DecayedT, data::string>
    || std::is_same_v<DecayedT, std::string>
    || std::is_same_v<DecayedT, std::string_view>
    || std::is_same_v<DecayedT, const char*>;

  if constexpr (is_arithmetic_or_enum) {
    meta.value.template emplace<PrimitiveValue>(std::forward<T>(value));
  } else if constexpr (is_cista_strong<DecayedT>::value) {
    meta.value.template emplace<PrimitiveValue>(value.v_);
  } else if constexpr (is_string_convertable) {
    meta.value.template emplace<data::string>(data::string{value});
  } else if constexpr (std::is_same_v<DecayedT, TensorMeta>) {
    meta.value.template emplace<TensorMeta>(std::forward<T>(value));
  } else if constexpr (is_data_vector<DecayedT>::value) {
    using ElementType = typename DecayedT::value_type;
    // Special case: data::vector<TensorMeta> is TensorMetaVec
    if constexpr (std::is_same_v<ElementType, TensorMeta>) {
      meta.value.template emplace<TensorMetaVecValue>(std::forward<T>(value));
    } else {
      meta.value.template emplace<VectorValue>(std::forward<T>(value));
    }
  } else if constexpr (is_std_vector<DecayedT>::value) {
    using ElementType = typename DecayedT::value_type;
    // Special case: std::vector<TensorMeta> is TensorMetaVec
    if constexpr (std::is_same_v<ElementType, TensorMeta>) {
      TensorMetaVecValue cista_vec;
      cista_vec.set(value.begin(), value.end());
      meta.value.template emplace<TensorMetaVecValue>(std::move(cista_vec));
    } else {
      data::vector<ElementType> cista_vec;
      cista_vec.set(value.begin(), value.end());
      meta.value.template emplace<VectorValue>(std::move(cista_vec));
    }
  }

  header.AddParam(std::move(meta));
}

template <typename... Args>
void PackArgs(RpcRequestHeader& header, Args&&... args) {
  (PackArg(header, std::forward<Args>(args)), ...);
}

template <typename... Args>
struct first_arg_is_param_meta_vector : std::false_type {};

template <typename First, typename... Rest>
struct first_arg_is_param_meta_vector<First, Rest...> {
  static constexpr bool value =
    std::is_same_v<std::decay_t<First>, data::vector<ParamMeta>>;
};

template <typename... Args>
struct first_arg_is_signature : std::false_type {};

template <typename First, typename... Rest>
struct first_arg_is_signature<First, Rest...> {
  static constexpr bool value =
    std::is_same_v<std::decay_t<First>, RpcFunctionSignature>;
};

}  // namespace detail

struct RpcRequestBuilderOptions {
  session_id_t session_id;
  request_id_t request_id;
  function_id_t function_id;
  utils::HybridLogicalClock hlc{};
  utils::workflow_id_t workflow_id{};
};

class RpcRequestBuilder {
 public:
  RpcRequestBuilder() = default;

  template <typename... Args>
    requires(
      !detail::first_arg_is_param_meta_vector<Args...>::value
      && !detail::first_arg_is_signature<Args...>::value)
  auto PrepareRequest(
    const RpcRequestBuilderOptions& options, Args&&... args) const {
    using Extractor = PayloadExtractor<Args...>;
    static_assert(
      Extractor::payload_count <= 1,
      "RPC calls can have at most one payload argument.");

    RpcRequestHeader header;
    header.session_id = options.session_id;
    header.request_id = options.request_id;
    header.function_id = options.function_id;
    header.hlc = options.hlc;
    header.workflow_id = options.workflow_id;

    auto args_tuple = std::forward_as_tuple(std::forward<Args>(args)...);

    std::apply(
      [&header](auto&&... a) {
        detail::PackArgs(header, std::forward<decltype(a)>(a)...);
      },
      args_tuple);

    if constexpr (Extractor::payload_count == 1) {
      auto payload = std::get<Extractor::payload_index>(std::move(args_tuple));
      return std::make_pair(std::move(header), std::move(payload));
    } else if constexpr (Extractor::payload_count == 0) {
      return header;
    } else {
      static_assert(
        Extractor::payload_count == 0,
        "RPC calls can have at most one payload argument.");
    }
  }

  template <typename... Args>
    requires(!detail::first_arg_is_param_meta_vector<Args...>::value)
  auto PrepareRequest(
    const RpcRequestBuilderOptions& options,
    const RpcFunctionSignature& signature, Args&&... args) const {
    auto result = PrepareRequest(options, std::forward<Args>(args)...);

    if constexpr (std::is_same_v<
                    std::decay_t<decltype(result)>, RpcRequestHeader>) {
      Validate(result.params, signature);
    } else {
      Validate(result.first.params, signature);
    }
    return result;
  }

  // Dynamic construction
  template <typename PayloadT = std::monostate>
  auto PrepareRequest(
    const RpcRequestBuilderOptions& options, data::vector<ParamMeta>&& params,
    PayloadT&& payload = PayloadT{}) const {
    RpcRequestHeader header;
    header.session_id = options.session_id;
    header.request_id = options.request_id;
    header.function_id = options.function_id;
    header.hlc = options.hlc;
    header.workflow_id = options.workflow_id;
    header.params = std::move(params);

    if constexpr (rpc::is_payload_v<std::decay_t<PayloadT>>) {
      return std::make_pair(std::move(header), std::forward<PayloadT>(payload));
    } else {
      return header;
    }
  }

  template <typename PayloadT = std::monostate>
  auto PrepareRequest(
    const RpcRequestBuilderOptions& options,
    const RpcFunctionSignature& signature, data::vector<ParamMeta>&& params,
    PayloadT&& payload = PayloadT{}) const {
    Validate(params, signature);
    return PrepareRequest(
      options, std::move(params), std::forward<PayloadT>(payload));
  }

 private:
  void Validate(
    const data::vector<ParamMeta>& params,
    const RpcFunctionSignature& signature) const {
    if (params.size() != signature.param_types.size()) {
      throw std::invalid_argument(
        "Parameter count mismatch: expected "
        + std::to_string(signature.param_types.size()) + ", got "
        + std::to_string(params.size()));
    }
    for (size_t i = 0; i < params.size(); ++i) {
      if (params[i].type != signature.param_types[i]) {
        throw std::invalid_argument(
          "Parameter type mismatch at index " + std::to_string(i));
      }
    }
  }
};

}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_RPC_REQUEST_BUILDER_HPP_
