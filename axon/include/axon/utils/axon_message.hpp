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
#include <ranges>
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

// Helper function to find tensor parameter indices
inline std::vector<size_t> ExtractTensorParamIndices(const auto& params) {
  std::vector<size_t> tensor_indices;
  for (auto [i, param] : std::views::enumerate(params)) {
    if (param.type == rpc::ParamType::TENSOR_META) {
      tensor_indices.push_back(i);
    }
  }
  return tensor_indices;
}

inline std::vector<std::reference_wrapper<const rpc::TensorMeta>>
ExtractTensorMetas(const auto& params) {
  std::vector<std::reference_wrapper<const rpc::TensorMeta>> tensor_metas;
  for (auto& param : params) {
    if (param.type == rpc::ParamType::TENSOR_META) {
      tensor_metas.push_back(
        std::ref(cista::get<rpc::TensorMeta>(param.value)));
    }
  }
  return tensor_metas;
}

template <typename HeaderType>
struct AxonMessage {
  HeaderType header;
  PayloadVariant payload;

  AxonMessage(HeaderType&& msg_header, PayloadVariant&& payload_data);
  AxonMessage(
    HeaderType&& msg_header, PayloadVariant&& payload_data,
    std::vector<size_t>&& tensor_param_indices);

  AxonMessage(const AxonMessage&) = delete;
  AxonMessage& operator=(const AxonMessage&) = delete;
  AxonMessage(AxonMessage&&) = default;
  AxonMessage& operator=(AxonMessage&&) = default;

  size_t GetTensorCount() const;
  const std::vector<size_t>& GetTensorParamIndices() const;
  // It's only a dlpack-like tensor view, not a copy.
  utils::Tensor<> GetTensor(size_t tensor_index);
  // It's only a dlpack-like tensor view, not a copy.
  std::vector<utils::Tensor<>> GetTensorVec();
  const PayloadVariant& GetPayload() const;
  PayloadVariant& GetPayload();

 private:
  std::vector<size_t> tensor_param_indices_;
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
