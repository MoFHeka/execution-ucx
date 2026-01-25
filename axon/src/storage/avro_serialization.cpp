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

#include "axon/storage/avro_serialization.hpp"

#include <cista.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <avro/Compiler.hh>
#include <avro/Decoder.hh>
#include <avro/Encoder.hh>
#include <avro/Generic.hh>

#include "axon/storage/avro_schema.hpp"
#include "axon/utils/axon_message.hpp"
#include "rpc_core/rpc_payload_types.hpp"
#include "rpc_core/rpc_types.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace axon {
namespace storage {

namespace data = cista::offset;

using eux::rpc::function_id_t;
using eux::rpc::ParamMeta;
using eux::rpc::ParamType;
using eux::rpc::PayloadVariant;
using eux::rpc::PrimitiveValue;
using eux::rpc::request_id_t;
using eux::rpc::RpcRequestHeader;
using eux::rpc::session_id_t;
using eux::rpc::VectorValue;
using eux::rpc::utils::HybridLogicalClock;
using eux::rpc::utils::TensorMeta;
using eux::rpc::utils::workflow_id_t;
using eux::ucxx::UcxBuffer;
using eux::ucxx::UcxBufferVec;
using eux::ucxx::UcxMemoryResourceManager;

// Helper function prototypes
static void ConvertRpcRequestHeaderToAvro(
  const RpcRequestHeader& header, avro::GenericRecord& header_rec);
static void ConvertParamsToAvro(
  const data::vector<ParamMeta>& params, avro::GenericArray& params_array);
static void ConvertParamMetaToAvro(
  const ParamMeta& param, avro::GenericRecord& param_rec);
static void ConvertPayloadToAvro(
  const PayloadVariant& payload, avro::GenericDatum& payload_union,
  UcxMemoryResourceManager& mr);

// Helper functions for vector serialization/deserialization
static auto& SelectVectorUnionBranch(
  avro::GenericDatum& value_union, int branch) {
  value_union.selectBranch(branch);
  auto& record = value_union.value<avro::GenericRecord>();
  return record.field("items").value<avro::GenericArray>().value();
}

template <typename SchemaType>
static auto& SelectPrimitiveUnionBranch(
  avro::GenericDatum& value_union, int branch) {
  value_union.selectBranch(branch);
  return value_union.value<SchemaType>();
}

template <typename T>
  requires std::is_integral_v<T> && (sizeof(T) <= 4)
           && (!std::is_same_v<T, bool>)
static void SerializePrimitiveToAvro(
  const ParamMeta& param, avro::GenericDatum& value_union, int branch) {
  SelectPrimitiveUnionBranch<int32_t>(value_union, branch) =
    static_cast<int32_t>(
      cista::get<T>(cista::get<PrimitiveValue>(param.value)));
}

template <typename T>
  requires std::is_integral_v<T> && (sizeof(T) > 4)
static void SerializePrimitiveToAvro(
  const ParamMeta& param, avro::GenericDatum& value_union, int branch) {
  SelectPrimitiveUnionBranch<int64_t>(value_union, branch) =
    static_cast<int64_t>(
      cista::get<T>(cista::get<PrimitiveValue>(param.value)));
}

template <typename T>
  requires std::is_same_v<T, bool> || std::is_floating_point_v<T>
static void SerializePrimitiveToAvro(
  const ParamMeta& param, avro::GenericDatum& value_union, int branch) {
  SelectPrimitiveUnionBranch<T>(value_union, branch) =
    static_cast<T>(cista::get<T>(cista::get<PrimitiveValue>(param.value)));
}

template <typename T>
  requires std::is_integral_v<T> && (sizeof(T) <= 4)
           && (!std::is_same_v<T, bool>)
static void SerializeVectorToAvro(
  const ParamMeta& param, avro::GenericDatum& value_union, int branch) {
  auto& arr = SelectVectorUnionBranch(value_union, branch);
  const auto& vec =
    cista::get<data::vector<T>>(cista::get<VectorValue>(param.value));
  arr.reserve(vec.size());
  for (const auto& v : vec) {
    arr.push_back(avro::GenericDatum(static_cast<int32_t>(v)));
  }
}

template <typename T>
  requires std::is_integral_v<T> && (sizeof(T) > 4)
static void SerializeVectorToAvro(
  const ParamMeta& param, avro::GenericDatum& value_union, int branch) {
  auto& arr = SelectVectorUnionBranch(value_union, branch);
  const auto& vec =
    cista::get<data::vector<T>>(cista::get<VectorValue>(param.value));
  arr.reserve(vec.size());
  for (const auto& v : vec) {
    arr.push_back(avro::GenericDatum(static_cast<int64_t>(v)));
  }
}

template <typename T>
  requires std::is_same_v<T, float> || std::is_same_v<T, double>
           || std::is_same_v<T, bool>
static void SerializeVectorToAvro(
  const ParamMeta& param, avro::GenericDatum& value_union, int branch) {
  auto& arr = SelectVectorUnionBranch(value_union, branch);
  const auto& vec =
    cista::get<data::vector<T>>(cista::get<VectorValue>(param.value));
  arr.reserve(vec.size());
  for (const auto& v : vec) {
    arr.push_back(avro::GenericDatum(static_cast<T>(v)));
  }
}

template <typename T>
  requires std::is_integral_v<T> && (sizeof(T) <= 4)
           && (!std::is_same_v<T, bool>)
static T DeserializePrimitiveFromAvro(const avro::GenericDatum& value_union) {
  return static_cast<T>(value_union.value<int32_t>());
}

template <typename T>
  requires std::is_integral_v<T> && (sizeof(T) > 4)
static T DeserializePrimitiveFromAvro(const avro::GenericDatum& value_union) {
  return static_cast<T>(value_union.value<int64_t>());
}

template <typename T>
static T DeserializePrimitiveFromAvro(const avro::GenericDatum& value_union) {
  return static_cast<T>(value_union.value<T>());
}

template <typename T>
  requires std::is_integral_v<T> && (sizeof(T) <= 4)
           && (!std::is_same_v<T, bool>)
static data::vector<T> DeserializeVectorFromAvro(
  const avro::GenericDatum& value_union) {
  const auto& record = value_union.value<const avro::GenericRecord>();
  const auto& arr =
    record.field("items").value<const avro::GenericArray>().value();
  data::vector<T> vec;
  vec.reserve(arr.size());
  for (const auto& item : arr) {
    vec.emplace_back(static_cast<T>(item.value<int32_t>()));
  }
  return vec;
}

template <typename T>
  requires std::is_integral_v<T> && (sizeof(T) > 4)
static data::vector<T> DeserializeVectorFromAvro(
  const avro::GenericDatum& value_union) {
  const auto& record = value_union.value<const avro::GenericRecord>();
  const auto& arr =
    record.field("items").value<const avro::GenericArray>().value();
  data::vector<T> vec;
  vec.reserve(arr.size());
  for (const auto& item : arr) {
    vec.emplace_back(static_cast<T>(item.value<int64_t>()));
  }
  return vec;
}

template <typename T>
  requires std::is_same_v<T, float> || std::is_same_v<T, double>
           || std::is_same_v<T, bool>
static data::vector<T> DeserializeVectorFromAvro(
  const avro::GenericDatum& value_union) {
  const auto& record = value_union.value<const avro::GenericRecord>();
  const auto& arr =
    record.field("items").value<const avro::GenericArray>().value();
  data::vector<T> vec;
  vec.reserve(arr.size());
  for (const auto& item : arr) {
    vec.emplace_back(static_cast<T>(item.value<T>()));
  }
  return vec;
}

static RpcRequestHeader ConvertAvroToRpcRequestHeader(
  const avro::GenericRecord& header_rec);
static data::vector<ParamMeta> ConvertAvroToParams(
  const avro::GenericArray& params_array);
static ParamMeta ConvertAvroToParamMeta(const avro::GenericRecord& param_rec);
static PayloadVariant ConvertAvroToPayloadVariant(
  const avro::GenericDatum& payload_datum, UcxMemoryResourceManager& mr);

std::string SerializeToJson(const AxonRequest& request) {
  throw std::runtime_error("SerializeToJson not implemented");
}

avro::GenericDatum AvroSerialization::Serialize(
  const AxonRequest& request, UcxMemoryResourceManager& mr) {
  avro::ValidSchema schema;
  try {
    schema = avro::compileJsonSchemaFromString(
      AvroSchema::GetArrivalRequestSchema().data());
  } catch (const std::exception& e) {
    throw std::runtime_error(
      std::string("Failed to compile Avro schema: ") + e.what());
  }

  avro::GenericDatum datum(schema);
  avro::GenericRecord& root = datum.value<avro::GenericRecord>();

  // Header
  avro::GenericRecord& header_rec =
    root.field("header").value<avro::GenericRecord>();
  ConvertRpcRequestHeaderToAvro(request.header, header_rec);

  // Payload
  avro::GenericDatum& payload_union = root.field("payload");
  ConvertPayloadToAvro(request.payload, payload_union, mr);

  return datum;
}

AxonRequest AvroSerialization::DeserializeFromJson(
  const std::string& json_data) {
  throw std::runtime_error("DeserializeFromJson not implemented");
}

std::shared_ptr<AxonRequest> AvroSerialization::Deserialize(
  const avro::GenericDatum& data, UcxMemoryResourceManager& mr) {
  const avro::GenericRecord& root = data.value<const avro::GenericRecord>();

  // Header
  const avro::GenericRecord& header_rec =
    root.field("header").value<const avro::GenericRecord>();
  RpcRequestHeader header = ConvertAvroToRpcRequestHeader(header_rec);

  // Payload
  const avro::GenericDatum& payload_union = root.field("payload");
  PayloadVariant payload = ConvertAvroToPayloadVariant(payload_union, mr);

  // Extract tensor param index from header parameters
  std::optional<size_t> tensor_param_index =
    utils::ExtractTensorParamIndex(header.params);

  return std::make_shared<AxonRequest>(
    std::move(header), std::move(payload), tensor_param_index);
}

// Helper function implementations
void ConvertRpcRequestHeaderToAvro(
  const RpcRequestHeader& header, avro::GenericRecord& header_rec) {
  header_rec.field("function_id").value<int>() =
    static_cast<int>(cista::to_idx(header.function_id));
  header_rec.field("session_id").value<int>() =
    static_cast<int>(cista::to_idx(header.session_id));
  header_rec.field("request_id").value<int>() =
    static_cast<int>(cista::to_idx(header.request_id));

  avro::GenericArray& params_array =
    header_rec.field("params").value<avro::GenericArray>();
  ConvertParamsToAvro(header.params, params_array);

  avro::GenericRecord& hlc_rec =
    header_rec.field("hlc").value<avro::GenericRecord>();
  hlc_rec.field("raw").value<int64_t>() =
    static_cast<int64_t>(header.hlc.Raw());
  header_rec.field("workflow_id").value<int>() =
    static_cast<int>(cista::to_idx(header.workflow_id));
}

void ConvertParamsToAvro(
  const data::vector<ParamMeta>& params, avro::GenericArray& params_array) {
  auto& params_array_raw = params_array.value();
  params_array_raw.reserve(params.size());
  for (const auto& param : params) {
    avro::GenericDatum param_datum(params_array.schema()->leafAt(0));
    ConvertParamMetaToAvro(param, param_datum.value<avro::GenericRecord>());
    params_array_raw.push_back(param_datum);
  }
}

void ConvertParamMetaToAvro(
  const ParamMeta& param, avro::GenericRecord& param_rec) {
  param_rec.field("type").value<int>() = static_cast<int>(param.type);
  param_rec.field("name").value<std::string>() = param.name;
  avro::GenericDatum& value_union = param_rec.field("value");

  switch (param.type) {
    case ParamType::VOID:
      value_union.selectBranch(0);
      break;
    case ParamType::PRIMITIVE_BOOL:
      SerializePrimitiveToAvro<bool>(param, value_union, 1);
      break;
    case ParamType::PRIMITIVE_INT8:
      SerializePrimitiveToAvro<int8_t>(param, value_union, 2);
      break;
    case ParamType::PRIMITIVE_INT16:
      SerializePrimitiveToAvro<int16_t>(param, value_union, 2);
      break;
    case ParamType::PRIMITIVE_INT32:
      SerializePrimitiveToAvro<int32_t>(param, value_union, 2);
      break;
    case ParamType::PRIMITIVE_UINT8:
      SerializePrimitiveToAvro<uint8_t>(param, value_union, 2);
      break;
    case ParamType::PRIMITIVE_UINT16:
      SerializePrimitiveToAvro<uint16_t>(param, value_union, 2);
      break;
    case ParamType::PRIMITIVE_UINT32:
      SerializePrimitiveToAvro<uint32_t>(param, value_union, 2);
      break;
    case ParamType::PRIMITIVE_INT64:
      SerializePrimitiveToAvro<int64_t>(param, value_union, 3);
      break;
    case ParamType::PRIMITIVE_UINT64:
      SerializePrimitiveToAvro<uint64_t>(param, value_union, 3);
      break;
    case ParamType::PRIMITIVE_FLOAT32:
      SerializePrimitiveToAvro<float>(param, value_union, 4);
      break;
    case ParamType::PRIMITIVE_FLOAT64:
      SerializePrimitiveToAvro<double>(param, value_union, 5);
      break;
    case ParamType::STRING:
      value_union.selectBranch(6);
      value_union.value<std::string>() = cista::get<data::string>(param.value);
      break;
    case ParamType::VECTOR_BOOL: {
      SerializeVectorToAvro<bool>(param, value_union, 7);
    } break;
    case ParamType::VECTOR_INT8: {
      SerializeVectorToAvro<int8_t>(param, value_union, 8);
    } break;
    case ParamType::VECTOR_INT16: {
      SerializeVectorToAvro<int16_t>(param, value_union, 8);
    } break;
    case ParamType::VECTOR_INT32: {
      SerializeVectorToAvro<int32_t>(param, value_union, 8);
    } break;
    case ParamType::VECTOR_UINT8: {
      SerializeVectorToAvro<uint8_t>(param, value_union, 8);
    } break;
    case ParamType::VECTOR_UINT16: {
      SerializeVectorToAvro<uint16_t>(param, value_union, 8);
    } break;
    case ParamType::VECTOR_UINT32: {
      SerializeVectorToAvro<uint32_t>(param, value_union, 8);
    } break;
    case ParamType::VECTOR_INT64: {
      SerializeVectorToAvro<int64_t>(param, value_union, 9);
    } break;
    case ParamType::VECTOR_UINT64: {
      SerializeVectorToAvro<uint64_t>(param, value_union, 9);
    } break;
    case ParamType::VECTOR_FLOAT32: {
      SerializeVectorToAvro<float>(param, value_union, 10);
    } break;
    case ParamType::VECTOR_FLOAT64: {
      SerializeVectorToAvro<double>(param, value_union, 11);
    } break;
    case ParamType::TENSOR_META: {
      value_union.selectBranch(12);
      auto& tensor_rec = value_union.value<avro::GenericRecord>();
      const auto& arg = cista::get<TensorMeta>(param.value);

      auto& device_rec =
        tensor_rec.field("device").value<avro::GenericRecord>();
      device_rec.field("device_type").value<int>() = arg.device.device_type;
      device_rec.field("device_id").value<int>() = arg.device.device_id;

      tensor_rec.field("ndim").value<int>() = arg.ndim;

      auto& dtype_rec = tensor_rec.field("dtype").value<avro::GenericRecord>();
      dtype_rec.field("code").value<int>() = arg.dtype.code;
      dtype_rec.field("bits").value<int>() = arg.dtype.bits;
      dtype_rec.field("lanes").value<int>() = arg.dtype.lanes;

      tensor_rec.field("byte_offset").value<int64_t>() = arg.byte_offset;

      auto& shape_array =
        tensor_rec.field("shape").value<avro::GenericArray>().value();
      shape_array.reserve(arg.shape.size());
      for (auto v : arg.shape) shape_array.push_back(avro::GenericDatum(v));

      auto& strides_array =
        tensor_rec.field("strides").value<avro::GenericArray>().value();
      strides_array.reserve(arg.strides.size());
      for (auto v : arg.strides) strides_array.push_back(avro::GenericDatum(v));
    } break;
    case ParamType::TENSOR_META_VEC: {
      value_union.selectBranch(13);
      auto& tensor_vec_rec = value_union.value<avro::GenericRecord>();
      const auto& arg_vec = cista::get<rpc::TensorMetaVec>(param.value);
      auto& tensor_array =
        tensor_vec_rec.field("items").value<avro::GenericArray>().value();
      tensor_array.reserve(arg_vec.size());
      for (const auto& tensor_meta : arg_vec) {
        avro::GenericDatum tensor_datum(tensor_vec_rec.field("items")
                                          .value<avro::GenericArray>()
                                          .schema()
                                          ->leafAt(0));
        auto& tensor_rec = tensor_datum.value<avro::GenericRecord>();

        auto& device_rec =
          tensor_rec.field("device").value<avro::GenericRecord>();
        device_rec.field("device_type").value<int>() =
          tensor_meta.device.device_type;
        device_rec.field("device_id").value<int>() =
          tensor_meta.device.device_id;

        tensor_rec.field("ndim").value<int>() = tensor_meta.ndim;

        auto& dtype_rec =
          tensor_rec.field("dtype").value<avro::GenericRecord>();
        dtype_rec.field("code").value<int>() = tensor_meta.dtype.code;
        dtype_rec.field("bits").value<int>() = tensor_meta.dtype.bits;
        dtype_rec.field("lanes").value<int>() = tensor_meta.dtype.lanes;

        tensor_rec.field("byte_offset").value<int64_t>() =
          tensor_meta.byte_offset;

        auto& shape_array =
          tensor_rec.field("shape").value<avro::GenericArray>().value();
        shape_array.reserve(tensor_meta.shape.size());
        for (auto v : tensor_meta.shape)
          shape_array.push_back(avro::GenericDatum(v));

        auto& strides_array =
          tensor_rec.field("strides").value<avro::GenericArray>().value();
        strides_array.reserve(tensor_meta.strides.size());
        for (auto v : tensor_meta.strides)
          strides_array.push_back(avro::GenericDatum(v));

        tensor_array.push_back(tensor_datum);
      }
    } break;
    default:
      throw std::runtime_error(
        "Unsupported ParamType for Avro serialization: "
        + std::to_string(static_cast<int>(param.type)));
  }
}

void ConvertPayloadToAvro(
  const PayloadVariant& payload, avro::GenericDatum& payload_union,
  UcxMemoryResourceManager& mr) {
  std::visit(
    [&](const auto& p) {
      using T = std::decay_t<decltype(p)>;
      if constexpr (std::is_same_v<
                      T, std::variant_alternative_t<0, PayloadVariant>>) {
        payload_union.selectBranch(0);  // null
      } else if constexpr (std::is_same_v<
                             T,
                             std::variant_alternative_t<1, PayloadVariant>>) {
        payload_union.selectBranch(1);  // UcxBuffer
        avro::GenericRecord& buf_rec =
          payload_union.value<avro::GenericRecord>();
        auto* buf = p.get();
        const auto* data_ptr = static_cast<const uint8_t*>(buf->data);
        auto size = buf->size;
        auto& dst_vec = buf_rec.field("data").value<std::vector<uint8_t>>();
        dst_vec.resize(size);
        auto dst_ptr = dst_vec.data();
        mr.memcpy(ucx_memory_type::HOST, dst_ptr, p.type(), data_ptr, size);
        buf_rec.field("size").value<int64_t>() = size;
      } else if constexpr (std::is_same_v<
                             T,
                             std::variant_alternative_t<2, PayloadVariant>>) {
        payload_union.selectBranch(2);  // array<UcxBuffer>
        avro::GenericArray& payload_array =
          payload_union.value<avro::GenericArray>();
        payload_array.value().reserve(p.size());
        for (const auto& buf : p) {
          avro::GenericDatum buf_datum(payload_array.schema()->leafAt(0));
          avro::GenericRecord& buf_rec = buf_datum.value<avro::GenericRecord>();
          const auto* data_ptr = static_cast<const uint8_t*>(buf.data);
          auto size = buf.size;
          auto& dst_vec = buf_rec.field("data").value<std::vector<uint8_t>>();
          dst_vec.resize(size);
          auto dst_ptr = dst_vec.data();
          mr.memcpy(ucx_memory_type::HOST, dst_ptr, p.type(), data_ptr, size);
          buf_rec.field("size").value<int64_t>() = size;
          payload_array.value().push_back(buf_datum);
        }
      }
    },
    payload);
}

RpcRequestHeader ConvertAvroToRpcRequestHeader(
  const avro::GenericRecord& header_rec) {
  RpcRequestHeader header;
  header.function_id = function_id_t{
    static_cast<uint32_t>(header_rec.field("function_id").value<int>())};
  header.session_id = session_id_t{
    static_cast<uint32_t>(header_rec.field("session_id").value<int>())};
  header.request_id = request_id_t{
    static_cast<uint32_t>(header_rec.field("request_id").value<int>())};
  header.params = ConvertAvroToParams(
    header_rec.field("params").value<const avro::GenericArray>());

  const auto& hlc_rec =
    header_rec.field("hlc").value<const avro::GenericRecord>();
  header.hlc = HybridLogicalClock(
    static_cast<uint64_t>(hlc_rec.field("raw").value<int64_t>()));
  header.workflow_id = workflow_id_t{
    static_cast<uint32_t>(header_rec.field("workflow_id").value<int>())};
  return header;
}

data::vector<ParamMeta> ConvertAvroToParams(
  const avro::GenericArray& params_array) {
  data::vector<ParamMeta> params;
  auto& params_array_raw = params_array.value();
  params.reserve(params_array_raw.size());
  for (const auto& param_datum : params_array_raw) {
    params.emplace_back(
      ConvertAvroToParamMeta(param_datum.value<const avro::GenericRecord>()));
  }
  return params;
}

ParamMeta ConvertAvroToParamMeta(const avro::GenericRecord& param_rec) {
  ParamMeta param;
  param.type = static_cast<ParamType>(param_rec.field("type").value<int>());
  param.name = param_rec.field("name").value<std::string>();

  // const auto& value_union =
  //   param_rec.field("value").value<const avro::GenericUnion>();
  const auto& value_union = param_rec.field("value");

  switch (param.type) {
    case ParamType::VOID:
      param.value = nullptr;
      break;
    case ParamType::PRIMITIVE_BOOL:
      param.value =
        PrimitiveValue{DeserializePrimitiveFromAvro<bool>(value_union)};
      break;
    case ParamType::PRIMITIVE_INT8:
      param.value =
        PrimitiveValue{DeserializePrimitiveFromAvro<int8_t>(value_union)};
      break;
    case ParamType::PRIMITIVE_INT16:
      param.value =
        PrimitiveValue{DeserializePrimitiveFromAvro<int16_t>(value_union)};
      break;
    case ParamType::PRIMITIVE_INT32:
      param.value =
        PrimitiveValue{DeserializePrimitiveFromAvro<int32_t>(value_union)};
      break;
    case ParamType::PRIMITIVE_UINT8:
      param.value =
        PrimitiveValue{DeserializePrimitiveFromAvro<uint8_t>(value_union)};
      break;
    case ParamType::PRIMITIVE_UINT16:
      param.value =
        PrimitiveValue{DeserializePrimitiveFromAvro<uint16_t>(value_union)};
      break;
    case ParamType::PRIMITIVE_UINT32:
      param.value =
        PrimitiveValue{DeserializePrimitiveFromAvro<uint32_t>(value_union)};
      break;
    case ParamType::PRIMITIVE_INT64:
      param.value =
        PrimitiveValue{DeserializePrimitiveFromAvro<int64_t>(value_union)};
      break;
    case ParamType::PRIMITIVE_UINT64:
      param.value =
        PrimitiveValue{DeserializePrimitiveFromAvro<uint64_t>(value_union)};
      break;
    case ParamType::PRIMITIVE_FLOAT32:
      param.value =
        PrimitiveValue{DeserializePrimitiveFromAvro<float>(value_union)};
      break;
    case ParamType::PRIMITIVE_FLOAT64:
      param.value =
        PrimitiveValue{DeserializePrimitiveFromAvro<double>(value_union)};
      break;
    case ParamType::STRING:
      param.value =
        data::string{DeserializePrimitiveFromAvro<std::string>(value_union)};
      break;
    case ParamType::VECTOR_BOOL: {
      param.value = VectorValue{DeserializeVectorFromAvro<bool>(value_union)};
      break;
    }
    case ParamType::VECTOR_INT8: {
      param.value = VectorValue{DeserializeVectorFromAvro<int8_t>(value_union)};
      break;
    }
    case ParamType::VECTOR_INT16: {
      param.value =
        VectorValue{DeserializeVectorFromAvro<int16_t>(value_union)};
      break;
    }
    case ParamType::VECTOR_INT32: {
      param.value =
        VectorValue{DeserializeVectorFromAvro<int32_t>(value_union)};
      break;
    }
    case ParamType::VECTOR_UINT8: {
      param.value =
        VectorValue{DeserializeVectorFromAvro<uint8_t>(value_union)};
      break;
    }
    case ParamType::VECTOR_UINT16: {
      param.value =
        VectorValue{DeserializeVectorFromAvro<uint16_t>(value_union)};
      break;
    }
    case ParamType::VECTOR_UINT32: {
      param.value =
        VectorValue{DeserializeVectorFromAvro<uint32_t>(value_union)};
      break;
    }
    case ParamType::VECTOR_INT64: {
      param.value =
        VectorValue{DeserializeVectorFromAvro<int64_t>(value_union)};
      break;
    }
    case ParamType::VECTOR_UINT64: {
      param.value =
        VectorValue{DeserializeVectorFromAvro<uint64_t>(value_union)};
      break;
    }
    case ParamType::VECTOR_FLOAT32: {
      param.value = VectorValue{DeserializeVectorFromAvro<float>(value_union)};
      break;
    }
    case ParamType::VECTOR_FLOAT64: {
      param.value = VectorValue{DeserializeVectorFromAvro<double>(value_union)};
      break;
    }
    case ParamType::TENSOR_META: {
      const auto& tensor_rec = value_union.value<const avro::GenericRecord>();
      TensorMeta tm;

      const auto& device_rec =
        tensor_rec.field("device").value<const avro::GenericRecord>();
      tm.device.device_type =
        static_cast<DLDeviceType>(device_rec.field("device_type").value<int>());
      tm.device.device_id = device_rec.field("device_id").value<int>();

      tm.ndim = tensor_rec.field("ndim").value<int>();

      const auto& dtype_rec =
        tensor_rec.field("dtype").value<const avro::GenericRecord>();
      tm.dtype.code = dtype_rec.field("code").value<int>();
      tm.dtype.bits = dtype_rec.field("bits").value<int>();
      tm.dtype.lanes = dtype_rec.field("lanes").value<int>();

      tm.byte_offset = tensor_rec.field("byte_offset").value<int64_t>();

      const auto& shape_arr =
        tensor_rec.field("shape").value<const avro::GenericArray>().value();
      tm.shape.reserve(shape_arr.size());
      for (const auto& item : shape_arr) {
        tm.shape.push_back(item.value<int64_t>());
      }
      const auto& strides_arr =
        tensor_rec.field("strides").value<const avro::GenericArray>().value();
      tm.strides.reserve(strides_arr.size());
      for (const auto& item : strides_arr) {
        tm.strides.push_back(item.value<int64_t>());
      }
      param.value = tm;
      break;
    }
    case ParamType::TENSOR_META_VEC: {
      const auto& tensor_vec_rec =
        value_union.value<const avro::GenericRecord>();
      const auto& tensor_array =
        tensor_vec_rec.field("items").value<const avro::GenericArray>().value();
      rpc::TensorMetaVec tensor_vec;
      tensor_vec.reserve(tensor_array.size());
      for (const auto& tensor_datum : tensor_array) {
        const auto& tensor_rec =
          tensor_datum.value<const avro::GenericRecord>();
        TensorMeta tm;

        const auto& device_rec =
          tensor_rec.field("device").value<const avro::GenericRecord>();
        tm.device.device_type = static_cast<DLDeviceType>(
          device_rec.field("device_type").value<int>());
        tm.device.device_id = device_rec.field("device_id").value<int>();

        tm.ndim = tensor_rec.field("ndim").value<int>();

        const auto& dtype_rec =
          tensor_rec.field("dtype").value<const avro::GenericRecord>();
        tm.dtype.code =
          static_cast<DLDataTypeCode>(dtype_rec.field("code").value<int>());
        tm.dtype.bits = dtype_rec.field("bits").value<int>();
        tm.dtype.lanes = dtype_rec.field("lanes").value<int>();

        tm.byte_offset = tensor_rec.field("byte_offset").value<int64_t>();

        const auto& shape_arr =
          tensor_rec.field("shape").value<const avro::GenericArray>().value();
        tm.shape.reserve(shape_arr.size());
        for (const auto& item : shape_arr) {
          tm.shape.push_back(item.value<int64_t>());
        }

        const auto& strides_arr =
          tensor_rec.field("strides").value<const avro::GenericArray>().value();
        tm.strides.reserve(strides_arr.size());
        for (const auto& item : strides_arr) {
          tm.strides.push_back(item.value<int64_t>());
        }

        tensor_vec.push_back(tm);
      }
      param.value = tensor_vec;
      break;
    }
    default:
      throw std::runtime_error(
        "Unsupported ParamType for Avro deserialization: "
        + std::to_string(static_cast<int>(param.type)));
  }

  return param;
}

PayloadVariant ConvertAvroToPayloadVariant(
  const avro::GenericDatum& payload_datum, UcxMemoryResourceManager& mr) {
  switch (payload_datum.type()) {
    case avro::AVRO_NULL: {
      return std::monostate{};
    }
    case avro::AVRO_RECORD: {
      const auto& buf_rec = payload_datum.value<const avro::GenericRecord>();
      const auto& data_vec =
        buf_rec.field("data").value<std::vector<uint8_t>>();
      size_t size = buf_rec.field("size").value<int64_t>();
      if (size != data_vec.size()) {
        size = size >= data_vec.size() ? size : data_vec.size();
      }
      constexpr auto mem_type = ucx_memory_type::HOST;
      void* buffer_data = mr.allocate(mem_type, size);
      mr.memcpy(mem_type, buffer_data, mem_type, data_vec.data(), size);
      return ucxx::UcxBuffer(mr, mem_type, {buffer_data, size}, nullptr, true);
    }
    case avro::AVRO_ARRAY: {
      const auto& payload_array =
        payload_datum.value<const avro::GenericArray>();
      std::vector<ucx_buffer_t> buffers;
      buffers.reserve(payload_array.value().size());

      constexpr auto mem_type = ucx_memory_type::HOST;
      for (const auto& buf_datum : payload_array.value()) {
        const auto& buf_rec = buf_datum.value<const avro::GenericRecord>();
        const auto& data_vec =
          buf_rec.field("data").value<std::vector<uint8_t>>();
        size_t size = buf_rec.field("size").value<int64_t>();
        if (size != data_vec.size()) {
          size = size >= data_vec.size() ? size : data_vec.size();
        }
        void* buffer_data = mr.allocate(mem_type, size);
        mr.memcpy(mem_type, buffer_data, mem_type, data_vec.data(), size);
        buffers.push_back({buffer_data, size});
      }
      return ucxx::UcxBufferVec(
        mr, mem_type, std::move(buffers), nullptr, true);
    }
    default:
      throw std::runtime_error(
        "Unsupported payload type (expected null/record/array)");
  }
}

}  // namespace storage
}  // namespace axon
}  // namespace eux
