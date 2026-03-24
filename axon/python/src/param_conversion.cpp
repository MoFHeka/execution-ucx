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
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "axon/python/src/dlpack_helpers.hpp"
#include "axon/python/src/memory_policy_helpers.hpp"
#include "axon/python/src/python_module.hpp"
#include "rpc_core/rpc_types.hpp"

// ---------------------------------------------------------------------------
// Nested list binary encoding helpers
// Format: [uint32 group_count][uint32 sizes[group_count]][T flat_data[...]]
// All values are native-endian (C++ struct layout on host).
// Sub-type tag stored in ParamMeta::name: "i64", "f64", "b"
// ---------------------------------------------------------------------------
namespace {

namespace data = cista::offset;
namespace nb = nanobind;

template <typename T>
data::string EncodeNestedList(const std::vector<std::vector<T>>& groups) {
  uint32_t gc = static_cast<uint32_t>(groups.size());
  uint32_t total_elems = 0;
  for (const auto& g : groups) total_elems += static_cast<uint32_t>(g.size());
  std::string blob;
  blob.resize(sizeof(uint32_t) * (1 + gc) + sizeof(T) * total_elems);
  char* p = blob.data();
  std::memcpy(p, &gc, sizeof(uint32_t));
  p += sizeof(uint32_t);
  for (const auto& g : groups) {
    uint32_t sz = static_cast<uint32_t>(g.size());
    std::memcpy(p, &sz, sizeof(uint32_t));
    p += sizeof(uint32_t);
  }
  for (const auto& g : groups) {
    std::memcpy(p, g.data(), sizeof(T) * g.size());
    p += sizeof(T) * g.size();
  }
  return data::string(blob.data(), static_cast<uint32_t>(blob.size()));
}

template <typename T>
nb::list DecodeNestedListT(const char* d, size_t sz) {
  if (sz < sizeof(uint32_t))
    throw std::runtime_error("NestedList blob too small for group_count");
  uint32_t gc = 0;
  std::memcpy(&gc, d, sizeof(uint32_t));
  size_t off = sizeof(uint32_t);
  if (sz < off + sizeof(uint32_t) * gc)
    throw std::runtime_error("NestedList blob too small for sizes");
  std::vector<uint32_t> sizes(gc);
  for (uint32_t i = 0; i < gc; ++i) {
    std::memcpy(&sizes[i], d + off, sizeof(uint32_t));
    off += sizeof(uint32_t);
  }
  nb::list outer;
  for (uint32_t i = 0; i < gc; ++i) {
    nb::list inner;
    for (uint32_t j = 0; j < sizes[i]; ++j) {
      if (off + sizeof(T) > sz)
        throw std::runtime_error("NestedList blob truncated");
      T v;
      std::memcpy(&v, d + off, sizeof(T));
      off += sizeof(T);
      if constexpr (std::is_same_v<T, uint8_t>) {
        inner.append(nb::bool_(v != 0));
      } else {
        inner.append(nb::cast(v));
      }
    }
    outer.append(std::move(inner));
  }
  return outer;
}

nb::list DecodeNestedListWithTag(
  const data::string& blob, const data::string& tag) {
  const char* d = blob.data();
  size_t sz = blob.size();
  if (tag == data::string("nli64")) return DecodeNestedListT<int64_t>(d, sz);
  if (tag == data::string("nlf64")) return DecodeNestedListT<double>(d, sz);
  if (tag == data::string("nlb")) return DecodeNestedListT<uint8_t>(d, sz);
  throw std::runtime_error(
    "DecodeNestedListWithTag: unknown tag '"
    + std::string(tag.data(), tag.size()) + "'");
}

// nb::cast<std::string> triggers RTTI dynamic_cast under linkstatic nanobind.
// This helper uses nb::str::c_str() which bypasses that code path.
data::string NbStrToCistaString(nb::handle py_obj) {
  nb::str s(py_obj);
  const char* cs = s.c_str();
  return data::string{cs, static_cast<uint32_t>(strlen(cs))};
}

nb::list DecodeVectorString(const char* d, size_t sz) {
  if (sz < sizeof(uint32_t))
    throw std::runtime_error("VectorString blob too small");
  uint32_t count = 0;
  std::memcpy(&count, d, sizeof(uint32_t));
  size_t off = sizeof(uint32_t);
  nb::list result;
  for (uint32_t i = 0; i < count; ++i) {
    if (off + sizeof(uint32_t) > sz)
      throw std::runtime_error("VectorString blob truncated at length");
    uint32_t len = 0;
    std::memcpy(&len, d + off, sizeof(uint32_t));
    off += sizeof(uint32_t);
    if (off + len > sz)
      throw std::runtime_error("VectorString blob truncated at data");
    result.append(nb::str(d + off, len));
    off += len;
  }
  return result;
}

}  // namespace

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
      if (
        param.name == data::string("nli64")
        || param.name == data::string("nlf64")
        || param.name == data::string("nlb")) {
        return DecodeNestedListBlob(param);
      } else if (param.name == data::string("vs")) {
        return DecodeVectorStringBlob(param);
      }
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
        if (
          nb::isinstance<nb::list>(py_obj)
          || nb::isinstance<nb::tuple>(py_obj)) {
          rpc::ParamMeta inferred = InferParamMeta(py_obj);
          if (inferred.type == rpc::ParamType::STRING) {
            meta = std::move(inferred);
            break;
          } else {
            throw std::runtime_error(
              "PythonToParamMeta: Expected list to infer as encoded STRING "
              "type.");
          }
        }
        meta.value.template emplace<data::string>(NbStrToCistaString(py_obj));
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
    meta.value.template emplace<data::string>(NbStrToCistaString(py_obj));
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
      if (nb::isinstance<nb::list>(first) || nb::isinstance<nb::tuple>(first)) {
        return EncodeNestedListToParamMeta(py_obj);
      } else if (nb::isinstance<nb::str>(first)) {
        return EncodeVectorStringToParamMeta(py_obj);
      } else if (nb::isinstance<bool>(first)) {
        meta.type = rpc::ParamType::VECTOR_BOOL;
        nb::sequence bseq = nb::cast<nb::sequence>(py_obj);
        data::vector<bool> cista_vec;
        for (size_t bi = 0; bi < nb::len(bseq); ++bi) {
          cista_vec.push_back(nb::cast<bool>(bseq[bi]));
        }
        meta.value.template emplace<rpc::VectorValue>(std::move(cista_vec));
      } else if (nb::isinstance<int>(first)) {
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

rpc::ParamMeta EncodeNestedListToParamMeta(nb::object py_obj) {
  namespace data = cista::offset;
  rpc::ParamMeta meta;
  meta.type = rpc::ParamType::STRING;

  nb::sequence outer = nb::cast<nb::sequence>(py_obj);
  size_t outer_len = nb::len(outer);

  // Detect element type from first non-empty inner list
  nb::module_ builtins = GetBuiltinsModule();
  std::string tag;
  bool tag_found = false;
  for (size_t gi = 0; gi < outer_len; ++gi) {
    nb::sequence inner = nb::cast<nb::sequence>(outer[gi]);
    if (nb::len(inner) > 0) {
      nb::object first = inner[0];
      if (nb::isinstance<bool>(first)) {
        tag = "nlb";
      } else if (nb::isinstance<int>(first)) {
        tag = "nli64";
      } else if (nb::isinstance(first, builtins.attr("float"))) {
        tag = "nlf64";
      } else {
        std::string type_name = nb::cast<std::string>(nb::str(first.type()));
        throw std::runtime_error(
          "EncodeNestedListToParamMeta: unsupported element type '" + type_name
          + "'. List[List[...]] only supports int, float, bool elements.");
      }
      tag_found = true;
      break;
    }
  }
  if (!tag_found) {
    throw std::runtime_error(
      "EncodeNestedListToParamMeta: cannot infer element type from"
      " List[List[...]] when all inner lists are empty."
      " Use register_function with type annotations (e.g."
      " List[List[int]]) instead of runtime inference.");
  }

  std::vector<nb::sequence> inners(outer_len);
  std::vector<uint32_t> sizes(outer_len);
  uint32_t total_elems = 0;
  for (size_t gi = 0; gi < outer_len; ++gi) {
    inners[gi] = nb::cast<nb::sequence>(outer[gi]);
    sizes[gi] = static_cast<uint32_t>(nb::len(inners[gi]));
    total_elems += sizes[gi];
  }

  size_t elem_size = (tag == "nli64")   ? sizeof(int64_t)
                     : (tag == "nlf64") ? sizeof(double)
                                        : sizeof(uint8_t);
  size_t header_bytes = sizeof(uint32_t) * (1 + outer_len);
  size_t blob_size = header_bytes + elem_size * total_elems;
  std::string blob;
  blob.resize(blob_size);
  char* p = blob.data();

  uint32_t gc = static_cast<uint32_t>(outer_len);
  std::memcpy(p, &gc, sizeof(uint32_t));
  p += sizeof(uint32_t);
  std::memcpy(p, sizes.data(), sizeof(uint32_t) * outer_len);
  p += sizeof(uint32_t) * outer_len;

  for (size_t gi = 0; gi < outer_len; ++gi) {
    const auto& inner = inners[gi];
    for (size_t j = 0; j < sizes[gi]; ++j) {
      if (tag == "nli64") {
        int64_t v = nb::cast<int64_t>(inner[j]);
        std::memcpy(p, &v, sizeof(int64_t));
        p += sizeof(int64_t);
      } else if (tag == "nlf64") {
        double v = nb::cast<double>(inner[j]);
        std::memcpy(p, &v, sizeof(double));
        p += sizeof(double);
      } else {
        uint8_t v = nb::cast<bool>(inner[j]) ? uint8_t(1) : uint8_t(0);
        std::memcpy(p, &v, sizeof(uint8_t));
        p += sizeof(uint8_t);
      }
    }
  }

  meta.value.template emplace<data::string>(
    data::string(blob.data(), static_cast<uint32_t>(blob.size())));
  meta.name = data::string(tag);
  return meta;
}

rpc::ParamMeta EncodeVectorStringToParamMeta(nb::object py_obj) {
  namespace data = cista::offset;
  nb::sequence seq = nb::cast<nb::sequence>(py_obj);
  size_t seq_len = nb::len(seq);

  struct Entry {
    nb::str s;
    size_t len;
  };
  std::vector<Entry> entries;
  entries.reserve(seq_len);
  size_t total = sizeof(uint32_t);
  for (size_t i = 0; i < seq_len; ++i) {
    nb::str s(seq[i]);
    size_t len = strlen(s.c_str());
    total += sizeof(uint32_t) + len;
    entries.push_back({std::move(s), len});
  }

  std::string blob;
  blob.resize(total);
  char* p = blob.data();
  uint32_t count = static_cast<uint32_t>(seq_len);
  std::memcpy(p, &count, sizeof(uint32_t));
  p += sizeof(uint32_t);
  for (const auto& e : entries) {
    uint32_t len = static_cast<uint32_t>(e.len);
    std::memcpy(p, &len, sizeof(uint32_t));
    p += sizeof(uint32_t);
    std::memcpy(p, e.s.c_str(), e.len);
    p += e.len;
  }

  rpc::ParamMeta meta;
  meta.type = rpc::ParamType::STRING;
  meta.name = data::string("vs");
  meta.value.template emplace<data::string>(
    data::string(blob.data(), static_cast<uint32_t>(blob.size())));
  return meta;
}

nb::list DecodeNestedListBlob(const rpc::ParamMeta& param) {
  namespace data = cista::offset;
  if (!cista::holds_alternative<data::string>(param.value)) {
    throw std::runtime_error(
      "Expected STRING-encoded NestedList blob, but received different "
      "ParamMeta type. Ensure client uses type-hints or sends correct data.");
  }
  return DecodeNestedListWithTag(
    cista::get<data::string>(param.value), param.name);
}

nb::list DecodeVectorStringBlob(const rpc::ParamMeta& param) {
  namespace data = cista::offset;
  if (!cista::holds_alternative<data::string>(param.value)) {
    throw std::runtime_error(
      "Expected STRING-encoded VectorString blob, but received different "
      "ParamMeta type.");
  }
  const data::string& blob = cista::get<data::string>(param.value);
  return DecodeVectorString(blob.data(), blob.size());
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
