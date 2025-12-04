/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#ifndef AXON_CORE_UTILS_AXON_MESSAGE_HPP_
#define AXON_CORE_UTILS_AXON_MESSAGE_HPP_

#include <cstddef>
#include <functional>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <vector>

#include "axon/utils/tensor.hpp"
#include "rpc_core/rpc_payload_types.hpp"
#include "rpc_core/rpc_types.hpp"

namespace eux {
namespace axon {
namespace utils {

using eux::rpc::PayloadVariant;
using eux::rpc::RpcRequestHeader;
using eux::rpc::RpcResponseHeader;

namespace detail {
template <typename T>
constexpr bool IsUcxBuffer =
  std::is_same_v<T, std::variant_alternative_t<1, PayloadVariant>>;
template <typename T>
constexpr bool IsUcxBufferVec =
  std::is_same_v<T, std::variant_alternative_t<2, PayloadVariant>>;
}  // namespace detail

using AxonMessageID = uint64_t;

// Helper function to find tensor parameter index
// Returns the index of the tensor parameter (either TENSOR_META or
// TENSOR_META_VEC) Throws if there are zero or more than one tensor parameters
inline std::optional<size_t> ExtractTensorParamIndex(const auto& params) {
  std::optional<size_t> tensor_index;
  for (auto [i, param] : std::views::enumerate(params)) {
    if (
      param.type == rpc::ParamType::TENSOR_META
      || param.type == rpc::ParamType::TENSOR_META_VEC) {
      if (tensor_index.has_value()) {
        throw std::runtime_error(
          "Multiple tensor parameters found. Only one TENSOR_META or "
          "TENSOR_META_VEC parameter is allowed.");
      }
      tensor_index = i;
    }
  }
  return tensor_index;
}

inline std::vector<std::reference_wrapper<const rpc::TensorMeta>>
ExtractTensorMetas(const auto& params) {
  std::vector<std::reference_wrapper<const rpc::TensorMeta>> tensor_metas;
  // Find the first TENSOR_META or TENSOR_META_VEC parameter
  for (const auto& param : params) {
    if (param.type == rpc::ParamType::TENSOR_META) {
      // Single TensorMeta: return vector with one element
      tensor_metas.push_back(
        std::ref(cista::get<rpc::TensorMeta>(param.value)));
      return tensor_metas;
    } else if (param.type == rpc::ParamType::TENSOR_META_VEC) {
      // TensorMetaVecValue: return all metas from the vector
      const auto& tensor_meta_vec =
        cista::get<rpc::TensorMetaVecValue>(param.value);
      tensor_metas.reserve(tensor_meta_vec.size());
      for (const auto& meta : tensor_meta_vec) {
        tensor_metas.push_back(std::ref(meta));
      }
      return tensor_metas;
    }
  }
  // No tensor meta parameter found
  return tensor_metas;
}

template <typename HeaderType>
struct AxonMessage {
  HeaderType header;
  PayloadVariant payload;

  AxonMessage(HeaderType&& msg_header, PayloadVariant&& payload_data);
  AxonMessage(
    HeaderType&& msg_header, PayloadVariant&& payload_data,
    std::optional<size_t> tensor_param_index);

  AxonMessage(const AxonMessage&) = delete;
  AxonMessage& operator=(const AxonMessage&) = delete;
  AxonMessage(AxonMessage&&) = default;
  AxonMessage& operator=(AxonMessage&&) = default;

  size_t GetTensorCount() const;
  std::optional<size_t> GetTensorParamIndex() const;
  // It's only a dlpack-like tensor view, not a copy.
  utils::Tensor<> GetTensor(size_t tensor_index);
  // It's only a dlpack-like tensor view, not a copy.
  std::vector<utils::Tensor<>> GetTensorVec();
  const PayloadVariant& GetPayload() const;
  PayloadVariant& GetPayload();

 private:
  std::optional<size_t> tensor_param_index_;
};

using ArrivalRequest = AxonMessage<RpcRequestHeader>;
using ArrivalResponse = AxonMessage<RpcResponseHeader>;

// Axon naming aliases
using AxonRequest = ArrivalRequest;
using AxonResponse = ArrivalResponse;

}  // namespace utils
}  // namespace axon
}  // namespace eux

#endif  // AXON_CORE_UTILS_AXON_MESSAGE_HPP_
