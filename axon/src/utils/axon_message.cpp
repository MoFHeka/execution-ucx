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

namespace eux {
namespace axon {
namespace utils {

template <typename HeaderType>
AxonMessage<HeaderType>::AxonMessage(
  HeaderType&& msg_header, PayloadVariant&& payload_data)
  : header(std::move(msg_header)), payload(std::move(payload_data)) {
  const auto& params = header.GetParamsContainer();
  tensor_param_index_ = ExtractTensorParamIndex(params);
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
  // Check that payload size matches the actual tensor count
  size_t expected_tensor_count = 0;
  if (tensor_param_index_.has_value()) {
    const size_t param_index = tensor_param_index_.value();
    if (param_index < params.size()) {
      const auto& param = params[param_index];
      if (param.type == rpc::ParamType::TENSOR_META) {
        expected_tensor_count = 1;
      } else if (param.type == rpc::ParamType::TENSOR_META_VEC) {
        const auto& tensor_meta_vec =
          cista::get<rpc::TensorMetaVecValue>(param.value);
        expected_tensor_count = tensor_meta_vec.size();
      }
    }
  }
  // If there's no tensor parameter, payload should be empty (monostate or
  // empty) If there's a tensor parameter, payload size should match tensor
  // count
  if (tensor_param_index_.has_value()) {
    assert(payload_size == expected_tensor_count);
  } else {
    // No tensor parameter: payload should be empty (monostate or empty
    // UcxBufferVec)
    assert(
      payload_size == 0 || std::holds_alternative<std::monostate>(payload));
  }
}

template <typename HeaderType>
AxonMessage<HeaderType>::AxonMessage(
  HeaderType&& msg_header,
  PayloadVariant&& payload_data,
  std::optional<size_t>
    tensor_param_index)
  : header(std::move(msg_header)),
    payload(std::move(payload_data)),
    tensor_param_index_(tensor_param_index) {
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
  // Verify tensor_param_index matches the actual parameters in header
  const auto& params = header.GetParamsContainer();
  auto actual_tensor_index = ExtractTensorParamIndex(params);
  // If tensor_param_index is provided, it should match the actual tensor
  // parameter
  if (tensor_param_index_.has_value()) {
    assert(
      actual_tensor_index.has_value()
      && actual_tensor_index.value() == tensor_param_index_.value());
  } else {
    // If tensor_param_index is not provided, there should be no tensor
    // parameter
    assert(!actual_tensor_index.has_value());
  }
  // Check that payload size matches the actual tensor count
  size_t expected_tensor_count = 0;
  if (tensor_param_index_.has_value()) {
    const size_t param_index = tensor_param_index_.value();
    if (param_index < params.size()) {
      const auto& param = params[param_index];
      if (param.type == rpc::ParamType::TENSOR_META) {
        expected_tensor_count = 1;
      } else if (param.type == rpc::ParamType::TENSOR_META_VEC) {
        const auto& tensor_meta_vec =
          cista::get<rpc::TensorMetaVecValue>(param.value);
        expected_tensor_count = tensor_meta_vec.size();
      }
    }
  }
  // If there's no tensor parameter, payload should be empty (monostate or
  // empty) If there's a tensor parameter, payload size should match tensor
  // count
  if (tensor_param_index_.has_value()) {
    assert(payload_size == expected_tensor_count);
  } else {
    // No tensor parameter: payload should be empty (monostate or empty
    // UcxBufferVec)
    assert(
      payload_size == 0 || std::holds_alternative<std::monostate>(payload));
  }
}

template <typename HeaderType>
size_t AxonMessage<HeaderType>::GetTensorCount() const {
  if (!tensor_param_index_.has_value()) {
    return 0;
  }
  const size_t param_index = tensor_param_index_.value();
  const auto& params = header.GetParamsContainer();
  if (param_index >= params.size()) {
    return 0;
  }
  const auto& param = params[param_index];
  if (param.type == rpc::ParamType::TENSOR_META) {
    return 1;
  } else if (param.type == rpc::ParamType::TENSOR_META_VEC) {
    const auto& tensor_meta_vec =
      cista::get<rpc::TensorMetaVecValue>(param.value);
    return tensor_meta_vec.size();
  }
  return 0;
}

template <typename HeaderType>
std::optional<size_t> AxonMessage<HeaderType>::GetTensorParamIndex() const {
  return tensor_param_index_;
}

template <typename ParamT, typename PayloadT>
static utils::Tensor<> GetTensorImpl(
  size_t tensor_index,
  const std::optional<size_t>& tensor_param_index,
  ParamT& header_params,
  PayloadT& payload) {
  if (!tensor_param_index.has_value())
    throw std::out_of_range("No tensor parameter found.");
  const size_t param_index = tensor_param_index.value();
  auto& param = header_params[param_index];

  const utils::TensorMeta* meta_ptr = nullptr;
  if (param.type == rpc::ParamType::TENSOR_META) {
    if (tensor_index > 0)
      throw std::out_of_range(
        "Tensor index out of range for single TensorMeta.");
    meta_ptr = &cista::get<utils::TensorMeta>(param.value);
  } else if (param.type == rpc::ParamType::TENSOR_META_VEC) {
    const auto& tensor_meta_vec =
      cista::get<rpc::TensorMetaVecValue>(param.value);
    if (tensor_index >= tensor_meta_vec.size())
      throw std::out_of_range("Tensor index out of range for TensorMetaVec.");
    meta_ptr = &tensor_meta_vec[tensor_index];
  } else {
    throw std::runtime_error("Parameter is not a tensor type.");
  }
  const auto& meta = *meta_ptr;

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
    tensor_index, tensor_param_index_, header.GetParamsContainer(), payload);
}

template <typename ParamT, typename PayloadT>
static std::vector<utils::Tensor<>> GetTensorVecImpl(
  const std::optional<size_t>& tensor_param_index,
  ParamT& header_params,
  PayloadT& payload) {
  std::vector<utils::Tensor<>> tensors;
  if (!tensor_param_index.has_value()) {
    return tensors;
  }
  const size_t param_index = tensor_param_index.value();
  if (param_index >= header_params.size()) {
    return tensors;
  }
  const auto& param = header_params[param_index];
  size_t tensor_count = 0;
  if (param.type == rpc::ParamType::TENSOR_META) {
    tensor_count = 1;
  } else if (param.type == rpc::ParamType::TENSOR_META_VEC) {
    const auto& tensor_meta_vec =
      cista::get<rpc::TensorMetaVecValue>(param.value);
    tensor_count = tensor_meta_vec.size();
  }
  tensors.reserve(tensor_count);
  for (size_t i = 0; i < tensor_count; ++i) {
    tensors.emplace_back(
      GetTensorImpl(i, tensor_param_index, header_params, payload));
  }
  return tensors;
}

template <typename HeaderType>
std::vector<utils::Tensor<>> AxonMessage<HeaderType>::GetTensorVec() {
  return GetTensorVecImpl(
    tensor_param_index_, header.GetParamsContainer(), payload);
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
