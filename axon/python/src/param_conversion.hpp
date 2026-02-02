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

#ifndef AXON_PYTHON_PARAM_CONVERSION_HPP_
#define AXON_PYTHON_PARAM_CONVERSION_HPP_

#include <cista.h>
#include <nanobind/nanobind.h>

#include <type_traits>

#include "rpc_core/rpc_types.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;
namespace rpc = eux::rpc;
namespace ucxx = eux::ucxx;

// Converts a single result ParamMeta to a Python object (for non-tensor types)
nb::object ResultMetaToPython(
  const rpc::ParamMeta& param, const void* buffer_data = nullptr,
  size_t buffer_size = 0);

// Convert Python object to ParamMeta
rpc::ParamMeta PythonToParamMeta(
  nb::object py_obj, rpc::ParamType expected_type);

// Infer ParamMeta from Python object
rpc::ParamMeta InferParamMeta(nb::object py_obj);

// Convert RpcRequestHeader to Python dict
nb::dict HeaderToDict(const rpc::RpcRequestHeader& header);

// Convert RpcResponseHeader to Python dict
nb::dict HeaderToDict(const rpc::RpcResponseHeader& header);

// Helper functions for tensor conversion (implemented in cpp)
nb::object ConvertTensorResult(
  const rpc::ParamMeta& param, ucxx::UcxBuffer&& payload,
  const nb::object& from_dlpack_fn);

nb::object ConvertTensorResult(
  const rpc::ParamMeta& param, ucxx::UcxBufferVec&& payload,
  const nb::object& from_dlpack_fn);

// Converts a vector of ParamMeta (results) to a Python list
// Optional from_dlpack_fn: callable to convert DLPack capsule wrappers to
// tensors
template <typename PayloadT = std::monostate>
nb::tuple ResultsToPythonTuple(
  const cista::offset::vector<rpc::ParamMeta>& params,
  PayloadT&& payload = std::monostate{},
  const nb::object& from_dlpack_fn = nb::none()) {
  using PayloadType = std::decay_t<PayloadT>;

  // Handle PayloadVariant by verifying the active alternative
  if constexpr (std::is_same_v<PayloadType, rpc::PayloadVariant>) {
    return std::visit(
      [&](auto&& inner) {
        return ResultsToPythonTuple(
          params, std::forward<decltype(inner)>(inner), from_dlpack_fn);
      },
      std::forward<PayloadT>(payload));
  }

  nb::list result;

  for (const auto& param : params) {
    if (param.type == rpc::ParamType::TENSOR_META) {
      if constexpr (std::is_same_v<std::decay_t<PayloadT>, ucxx::UcxBuffer>) {
        // Single tensor payload
        nb::object tensor =
          ConvertTensorResult(param, std::move(payload), from_dlpack_fn);
        result.append(std::move(tensor));
      } else {
        // No valid payload for tensor
        throw std::runtime_error(
          "Invalid payload type for tensor when result type is TENSOR_META");
      }
    } else if (param.type == rpc::ParamType::TENSOR_META_VEC) {
      if constexpr (std::is_same_v<
                      std::decay_t<PayloadT>, ucxx::UcxBufferVec>) {
        // Multiple tensor payload
        nb::object tensors =
          ConvertTensorResult(param, std::move(payload), from_dlpack_fn);
        result.extend(std::move(tensors));
      } else {
        // No valid payload for tensor
        throw std::runtime_error(
          "Invalid payload type for tensor when result type is "
          "TENSOR_META_VEC");
      }
    } else {
      // Handle non-tensor types
      result.append(ResultMetaToPython(param));
    }
  }

  return nb::tuple(result);
}

template <typename PayloadT = std::monostate>
nb::object ResultsToPython(
  const cista::offset::vector<rpc::ParamMeta>& params,
  PayloadT&& payload = std::monostate{},
  const nb::object& from_dlpack_fn = nb::none()) {
  using PayloadType = std::decay_t<PayloadT>;
  // Handle PayloadVariant by verifying the active alternative
  if constexpr (std::is_same_v<PayloadType, rpc::PayloadVariant>) {
    return std::visit(
      [&](auto&& inner) {
        return ResultsToPython(
          params, std::forward<decltype(inner)>(inner), from_dlpack_fn);
      },
      std::forward<PayloadT>(payload));
  }

  if (params.size() == 1) {
    if (params[0].type == rpc::ParamType::TENSOR_META) {
      if constexpr (!std::is_same_v<PayloadType, ucxx::UcxBuffer>) {
        throw std::runtime_error(
          "Invalid payload type for tensor when result type is TENSOR_META");
      } else {
        return ConvertTensorResult(
          params[0], std::move(payload), from_dlpack_fn);
      }
    } else if (params[0].type == rpc::ParamType::TENSOR_META_VEC) {
      if constexpr (!std::is_same_v<PayloadType, ucxx::UcxBufferVec>) {
        throw std::runtime_error(
          "Invalid payload type for tensor when result type is "
          "TENSOR_META_VEC");
      } else {
        return ConvertTensorResult(
          params[0], std::move(payload), from_dlpack_fn);
      }
    } else {
      return ResultMetaToPython(params[0]);
    }
  } else if (params.size() > 1) {
    return ResultsToPythonTuple(
      params, std::forward<PayloadT>(payload), from_dlpack_fn);
  }
  return nb::none();
}

}  // namespace python
}  // namespace axon
}  // namespace eux

#endif  // AXON_PYTHON_PARAM_CONVERSION_HPP_
