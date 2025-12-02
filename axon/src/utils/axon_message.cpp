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

#include "axon/utils/axon_message.hpp"

#include <cassert>
#include <utility>
#include <variant>
#include <vector>

#include "rpc_core/rpc_payload_types.hpp"

namespace eux {
namespace axon {
namespace utils {

template <typename HeaderType>
AxonMessage<HeaderType>::AxonMessage(
  HeaderType&& msg_header, PayloadVariant&& payload_data)
  : header(std::move(msg_header)), payload(std::move(payload_data)) {
  const auto& params = header.GetParamsContainer();
  tensor_param_indices_ = ExtractTensorParamIndices(params);
  size_t payload_size = std::visit(
    [](const auto& p) {
      using T = std::decay_t<decltype(p)>;
      if constexpr (detail::IsUcxBuffer<T>) {
        return static_cast<size_t>(p.get()->data == nullptr ? 0 : 1);
      } else if constexpr (detail::IsUcxBufferVec<T>) {
        return p.size();
      }
      return static_cast<size_t>(0);
    },
    payload);
  assert(tensor_param_indices_.size() == payload_size);
}

template <typename HeaderType>
AxonMessage<HeaderType>::AxonMessage(
  HeaderType&& msg_header,
  PayloadVariant&& payload_data,
  std::vector<size_t>&& tensor_param_indices)
  : header(std::move(msg_header)),
    payload(std::move(payload_data)),
    tensor_param_indices_(std::move(tensor_param_indices)) {
  size_t payload_size = std::visit(
    [](const auto& p) {
      using T = std::decay_t<decltype(p)>;
      if constexpr (detail::IsUcxBuffer<T>) {
        return static_cast<size_t>(p.get()->data == nullptr ? 0 : 1);
      } else if constexpr (detail::IsUcxBufferVec<T>) {
        return p.size();
      }
      return static_cast<size_t>(0);
    },
    payload);
  assert(tensor_param_indices_.size() == payload_size);
}

template <typename HeaderType>
size_t AxonMessage<HeaderType>::GetTensorCount() const {
  return tensor_param_indices_.size();
}

template <typename HeaderType>
const std::vector<size_t>& AxonMessage<HeaderType>::GetTensorParamIndices()
  const {
  return tensor_param_indices_;
}

template <typename ParamT, typename PayloadT>
static utils::Tensor<> GetTensorImpl(
  size_t tensor_index,
  const std::vector<size_t>& tensor_param_indices,
  ParamT& header_params,
  PayloadT& payload) {
  if (tensor_index >= tensor_param_indices.size())
    throw std::out_of_range("Tensor index out of range.");
  const size_t param_index = tensor_param_indices[tensor_index];
  auto& param = header_params[param_index];
  auto& meta = cista::get<utils::TensorMeta>(param.value);

  DLTensor dltensor;
  dltensor.device = meta.device;
  dltensor.ndim = meta.ndim;
  dltensor.dtype = meta.dtype;
  dltensor.shape = const_cast<int64_t*>(meta.shape.data());
  dltensor.strides =
    meta.strides.empty() ? nullptr : const_cast<int64_t*>(meta.strides.data());

  std::visit(
    [&](auto& p) {
      using T = std::decay_t<decltype(p)>;
      ucx_buffer_t* am_data;
      void* data_ptr = nullptr;
      if constexpr (detail::IsUcxBuffer<T>) {
        if (tensor_index > 0) {
          throw std::out_of_range("Tensor index out of range for UcxBuffer.");
        }
        am_data = p.get();
        data_ptr = static_cast<char*>(am_data->data) + meta.byte_offset;
      } else if constexpr (detail::IsUcxBufferVec<T>) {
        am_data = &(p.at(tensor_index));
        data_ptr = static_cast<char*>(am_data->data) + meta.byte_offset;
      }

      dltensor.data = data_ptr;
      dltensor.byte_offset = 0;
    },
    payload);

  utils::Tensor<> tensor;
  tensor.assign_view(dltensor);
  return tensor;
}

template <typename HeaderType>
utils::Tensor<> AxonMessage<HeaderType>::GetTensor(size_t tensor_index) {
  return GetTensorImpl(
    tensor_index, tensor_param_indices_, header.GetParamsContainer(), payload);
}

template <typename ParamT, typename PayloadT>
static std::vector<utils::Tensor<>> GetTensorVecImpl(
  const std::vector<size_t>& tensor_param_indices,
  ParamT& header_params,
  PayloadT& payload) {
  std::vector<utils::Tensor<>> tensors;
  tensors.reserve(tensor_param_indices.size());
  for (size_t i = 0; i < tensor_param_indices.size(); ++i) {
    tensors.emplace_back(
      GetTensorImpl(i, tensor_param_indices, header_params, payload));
  }
  return tensors;
}

template <typename HeaderType>
std::vector<utils::Tensor<>> AxonMessage<HeaderType>::GetTensorVec() {
  return GetTensorVecImpl(
    tensor_param_indices_, header.GetParamsContainer(), payload);
}

template <typename HeaderType>
const PayloadVariant& AxonMessage<HeaderType>::GetPayload() const {
  return payload;
}

template <typename HeaderType>
PayloadVariant& AxonMessage<HeaderType>::GetPayload() {
  return payload;
}

template struct AxonMessage<RpcRequestHeader>;
template struct AxonMessage<RpcResponseHeader>;

}  // namespace utils
}  // namespace axon
}  // namespace eux
