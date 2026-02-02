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

#include "axon/python/src/param_conversion.hpp"

#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <csignal>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "axon/python/src/dlpack_helpers.hpp"
#include "axon/python/src/memory_policy_helpers.hpp"
#include "axon/python/src/python_module.hpp"
#include "rpc_core/rpc_types.hpp"

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;
namespace rpc = eux::rpc;

nb::object ResultMetaToPython(
  const rpc::ParamMeta& param, const void* buffer_data, size_t buffer_size) {
  namespace data = cista::offset;

  switch (param.type) {
    case rpc::ParamType::PRIMITIVE_BOOL:
      return nb::cast(
        cista::get<bool>(cista::get<rpc::PrimitiveValue>(param.value)));
    case rpc::ParamType::PRIMITIVE_INT8:
      return nb::cast(
        cista::get<int8_t>(cista::get<rpc::PrimitiveValue>(param.value)));
    case rpc::ParamType::PRIMITIVE_INT16:
      return nb::cast(
        cista::get<int16_t>(cista::get<rpc::PrimitiveValue>(param.value)));
    case rpc::ParamType::PRIMITIVE_INT32:
      return nb::cast(
        cista::get<int32_t>(cista::get<rpc::PrimitiveValue>(param.value)));
    case rpc::ParamType::PRIMITIVE_INT64:
      return nb::cast(
        cista::get<int64_t>(cista::get<rpc::PrimitiveValue>(param.value)));
    case rpc::ParamType::PRIMITIVE_UINT8:
      return nb::cast(
        cista::get<uint8_t>(cista::get<rpc::PrimitiveValue>(param.value)));
    case rpc::ParamType::PRIMITIVE_UINT16:
      return nb::cast(
        cista::get<uint16_t>(cista::get<rpc::PrimitiveValue>(param.value)));
    case rpc::ParamType::PRIMITIVE_UINT32:
      return nb::cast(
        cista::get<uint32_t>(cista::get<rpc::PrimitiveValue>(param.value)));
    case rpc::ParamType::PRIMITIVE_UINT64:
      return nb::cast(
        cista::get<uint64_t>(cista::get<rpc::PrimitiveValue>(param.value)));
    case rpc::ParamType::PRIMITIVE_FLOAT32:
      return nb::cast(
        cista::get<float>(cista::get<rpc::PrimitiveValue>(param.value)));
    case rpc::ParamType::PRIMITIVE_FLOAT64:
      return nb::cast(
        cista::get<double>(cista::get<rpc::PrimitiveValue>(param.value)));
    case rpc::ParamType::STRING: {
      const data::string& str_ref = cista::get<data::string>(param.value);
      // Create Python string directly using nb::str constructor
      // This avoids nb::cast which throws std::bad_cast
      return nb::str(str_ref.data(), str_ref.size());
    }
    case rpc::ParamType::VECTOR_BOOL: {
      const auto& vec = cista::get<data::vector<bool>>(
        cista::get<rpc::VectorValue>(param.value));
      std::vector<bool> result;
      result.reserve(vec.size());
      result.assign(vec.begin(), vec.end());
      return nb::cast(std::move(result));
    }
    case rpc::ParamType::VECTOR_INT32: {
      const auto& vec = cista::get<data::vector<int32_t>>(
        cista::get<rpc::VectorValue>(param.value));
      std::vector<int32_t> result;
      result.reserve(vec.size());
      result.assign(vec.begin(), vec.end());
      return nb::cast(std::move(result));
    }
    case rpc::ParamType::VECTOR_INT64: {
      const auto& vec = cista::get<data::vector<int64_t>>(
        cista::get<rpc::VectorValue>(param.value));
      std::vector<int64_t> result;
      result.reserve(vec.size());
      result.assign(vec.begin(), vec.end());
      return nb::cast(std::move(result));
    }
    case rpc::ParamType::VECTOR_FLOAT32: {
      const auto& vec = cista::get<data::vector<float>>(
        cista::get<rpc::VectorValue>(param.value));
      std::vector<float> result;
      result.reserve(vec.size());
      result.assign(vec.begin(), vec.end());
      return nb::cast(std::move(result));
    }
    case rpc::ParamType::VECTOR_FLOAT64: {
      const auto& vec = cista::get<data::vector<double>>(
        cista::get<rpc::VectorValue>(param.value));
      std::vector<double> result;
      result.reserve(vec.size());
      result.assign(vec.begin(), vec.end());
      return nb::cast(std::move(result));
    }
    case rpc::ParamType::TENSOR_META:
      throw std::runtime_error(
        "TENSOR_META should be handled by ResultsToPythonList, not "
        "ResultMetaToPython");
    case rpc::ParamType::TENSOR_META_VEC:
      throw std::runtime_error(
        "TENSOR_META_VEC should be handled by ResultsToPythonList, not "
        "ResultMetaToPython");
    case rpc::ParamType::VOID:
      return nb::none();
    default:
      throw std::runtime_error("Unsupported ParamType in ResultMetaToPython");
  }
}

rpc::ParamMeta PythonToParamMeta(
  nb::object py_obj, rpc::ParamType expected_type) {
  namespace data = cista::offset;

  rpc::ParamMeta meta;
  meta.type = expected_type;

  try {
    switch (expected_type) {
      case rpc::ParamType::PRIMITIVE_BOOL:
        meta.value.template emplace<rpc::PrimitiveValue>(
          nb::cast<bool>(py_obj));
        break;
      case rpc::ParamType::PRIMITIVE_INT32:
        meta.value.template emplace<rpc::PrimitiveValue>(
          nb::cast<int32_t>(py_obj));
        break;
      case rpc::ParamType::PRIMITIVE_INT64:
        meta.value.template emplace<rpc::PrimitiveValue>(
          nb::cast<int64_t>(py_obj));
        break;
      case rpc::ParamType::PRIMITIVE_FLOAT32:
        meta.value.template emplace<rpc::PrimitiveValue>(
          nb::cast<float>(py_obj));
        break;
      case rpc::ParamType::PRIMITIVE_FLOAT64:
        meta.value.template emplace<rpc::PrimitiveValue>(
          nb::cast<double>(py_obj));
        break;
      case rpc::ParamType::STRING: {
        std::string str = nb::cast<std::string>(py_obj);
        meta.value.template emplace<data::string>(data::string{str});
        break;
      }
      case rpc::ParamType::VECTOR_INT32: {
        std::vector<int32_t> vec = nb::cast<std::vector<int32_t>>(py_obj);
        data::vector<int32_t> cista_vec;
        cista_vec.set(vec.begin(), vec.end());
        meta.value.template emplace<rpc::VectorValue>(std::move(cista_vec));
        break;
      }
      case rpc::ParamType::VECTOR_INT64: {
        std::vector<int64_t> vec = nb::cast<std::vector<int64_t>>(py_obj);
        data::vector<int64_t> cista_vec;
        cista_vec.set(vec.begin(), vec.end());
        meta.value.template emplace<rpc::VectorValue>(std::move(cista_vec));
        break;
      }
      case rpc::ParamType::VECTOR_FLOAT32: {
        std::vector<float> vec = nb::cast<std::vector<float>>(py_obj);
        data::vector<float> cista_vec;
        cista_vec.set(vec.begin(), vec.end());
        meta.value.template emplace<rpc::VectorValue>(std::move(cista_vec));
        break;
      }
      case rpc::ParamType::VECTOR_FLOAT64: {
        std::vector<double> vec = nb::cast<std::vector<double>>(py_obj);
        data::vector<double> cista_vec;
        cista_vec.set(vec.begin(), vec.end());
        meta.value.template emplace<rpc::VectorValue>(std::move(cista_vec));
        break;
      }
      default:
        throw std::runtime_error("Unsupported ParamType in PythonToParamMeta");
    }
  } catch (const nb::python_error& e) {
    throw std::runtime_error(
      "Failed to convert Python object to ParamMeta: " + std::string(e.what()));
  }

  return meta;
}

rpc::ParamMeta InferParamMeta(nb::object py_obj) {
  namespace data = cista::offset;
  rpc::ParamMeta meta;

  // Get cached builtins module to check types
  nb::module_ builtins = GetBuiltinsModule();

  if (IsDlpackTensor(py_obj)) {
    meta.type = rpc::ParamType::TENSOR_META;
    meta.value.template emplace<rpc::utils::TensorMeta>(
      ExtractTensorMetaFromDlpack(py_obj));
  } else if (nb::isinstance<bool>(py_obj)) {
    meta.type = rpc::ParamType::PRIMITIVE_BOOL;
    meta.value.template emplace<rpc::PrimitiveValue>(nb::cast<bool>(py_obj));
  } else if (nb::isinstance<int>(py_obj)) {
    // Default to INT64 for Python int
    meta.type = rpc::ParamType::PRIMITIVE_INT64;
    meta.value.template emplace<rpc::PrimitiveValue>(nb::cast<int64_t>(py_obj));
  } else if (nb::isinstance(py_obj, builtins.attr("float"))) {
    // Default to FLOAT64 for Python float
    meta.type = rpc::ParamType::PRIMITIVE_FLOAT64;
    meta.value.template emplace<rpc::PrimitiveValue>(nb::cast<double>(py_obj));
  } else if (nb::isinstance<nb::str>(py_obj)) {
    meta.type = rpc::ParamType::STRING;
    std::string str = nb::cast<std::string>(py_obj);
    meta.value.template emplace<data::string>(data::string{str});
  } else if (
    nb::isinstance<nb::list>(py_obj) || nb::isinstance<nb::tuple>(py_obj)) {
    // Infer vector type from first element
    nb::sequence seq = nb::cast<nb::sequence>(py_obj);
    size_t seq_len = nb::len(seq);

    if (seq_len == 0) {
      // Empty vector - assume INT64
      meta.type = rpc::ParamType::VECTOR_INT64;
      data::vector<int64_t> cista_vec;
      meta.value.template emplace<rpc::VectorValue>(std::move(cista_vec));
    } else {
      nb::object first = seq[0];
      if (nb::isinstance<int>(first)) {
        meta.type = rpc::ParamType::VECTOR_INT64;
        std::vector<int64_t> vec = nb::cast<std::vector<int64_t>>(py_obj);
        data::vector<int64_t> cista_vec;
        cista_vec.set(vec.begin(), vec.end());
        meta.value.template emplace<rpc::VectorValue>(std::move(cista_vec));
      } else if (nb::isinstance(first, builtins.attr("float"))) {
        meta.type = rpc::ParamType::VECTOR_FLOAT64;
        std::vector<double> vec = nb::cast<std::vector<double>>(py_obj);
        data::vector<double> cista_vec;
        cista_vec.set(vec.begin(), vec.end());
        meta.value.template emplace<rpc::VectorValue>(std::move(cista_vec));
      } else {
        throw std::runtime_error("Unsupported list element type for inference");
      }
    }
  } else {
    throw std::runtime_error(
      "Unsupported type for inference: "
      + nb::cast<std::string>(nb::str(py_obj.type())));
  }

  return meta;
}

nb::dict HeaderToDict(const rpc::RpcRequestHeader& header) {
  nb::dict result;
  result["session_id"] = nb::cast(header.session_id.v_);
  result["request_id"] = nb::cast(header.request_id.v_);
  result["function_id"] = nb::cast(header.function_id.v_);
  result["workflow_id"] = nb::cast(header.workflow_id.v_);
  result["hlc"] = nb::cast(header.hlc.Raw());
  // params will be handled separately with ParamsToPythonList
  result["params"] = nb::cast(header.params);
  return result;
}

nb::dict HeaderToDict(const rpc::RpcResponseHeader& header) {
  nb::dict result;
  result["session_id"] = nb::cast(header.session_id.v_);
  result["request_id"] = nb::cast(header.request_id.v_);
  result["function_id"] =
    nb::cast(0);  // Response header doesn't have function_id
  result["workflow_id"] = nb::cast(header.workflow_id.v_);
  result["hlc"] = nb::cast(header.hlc.Raw());
  result["status"] = nb::cast(
    static_cast<int>(static_cast<std::error_code>(header.status).value()));
  // results will be handled separately with ResultsToPythonList
  result["results"] = nb::cast(header.results);
  return result;
}

// ConvertTensorResult implementations

nb::object ConvertTensorResult(
  const rpc::ParamMeta& param, ucxx::UcxBuffer&& buffer,
  const nb::object& from_dlpack_fn) {
  auto meta = cista::get<rpc::utils::TensorMeta>(param.value);

  // Try to get custom buffer (RNDV path with memory policy)
  nb::object custom_obj = TryGetCustomBuffer(buffer.data());
  if (!custom_obj.is_none()) {
    // If it's a custom buffer, we return the object directly.
    return custom_obj;
  }

  auto result = TensorMetaToDlpack(std::move(meta), std::move(buffer));
  if (!from_dlpack_fn.is_none()) {
    return from_dlpack_fn(result);
  }
  return result;
}

nb::object ConvertTensorResult(
  const rpc::ParamMeta& param, ucxx::UcxBufferVec&& buffer_vec,
  const nb::object& from_dlpack_fn) {
  auto meta_vec = cista::get<rpc::TensorMetaVec>(param.value);
  size_t num_tensors = meta_vec.size();
  const auto& num_buffers = buffer_vec.size();
  if (num_tensors > num_buffers) {
    throw std::runtime_error(
      "Not enough buffers for tensor meta vec when "
      "converting tensor result");
  }

  // Check custom buffer on first element
  // Assuming all-or-nothing for custom memory policy
  bool is_custom = false;
  if (buffer_vec.size() > 0) {
    if (!TryGetCustomBuffer(buffer_vec.buffers()[0].data).is_none()) {
      is_custom = true;
    }
  }

  if (is_custom) {
    nb::list result;
    const auto& buffers = buffer_vec.buffers();
    // Iterate up to num_tensors
    for (size_t i = 0; i < num_tensors; ++i) {
      nb::object obj = TryGetCustomBuffer(buffers[i].data);
      if (obj.is_none()) {
        throw std::runtime_error(
          "Mixed custom/non-custom buffers in UcxBufferVec result not "
          "supported");
      }
      result.append(obj);
    }
    return result;
  }

  auto tensors =
    TensorMetaVecToDlpack(std::move(meta_vec), std::move(buffer_vec));
  if (!from_dlpack_fn.is_none()) {
    nb::list result;
    for (const auto& tensor : tensors) {
      result.append(from_dlpack_fn(tensor));
    }
    return result;
  }
  return tensors;
}

}  // namespace python
}  // namespace axon
}  // namespace eux
