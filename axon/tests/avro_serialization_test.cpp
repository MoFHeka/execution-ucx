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
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "axon/utils/axon_message.hpp"
#include "rpc_core/rpc_types.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

namespace eux {
namespace axon {
namespace storage {

namespace data = cista::offset;

using eux::rpc::ParamMeta;
using eux::rpc::ParamType;
using eux::rpc::PrimitiveValue;
using eux::rpc::request_id_t;
using eux::rpc::RpcRequestHeader;
using eux::rpc::TensorMetaVecValue;
using eux::rpc::VectorValue;
using eux::rpc::utils::HybridLogicalClock;
using eux::rpc::utils::TensorMeta;
using eux::rpc::utils::workflow_id_t;
using eux::ucxx::DefaultUcxMemoryResourceManager;
using eux::ucxx::UcxBuffer;
using eux::ucxx::UcxBufferVec;
using eux::ucxx::UcxMemoryResourceManager;

class AvroSerializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mr_ = std::make_unique<DefaultUcxMemoryResourceManager>();
  }

  void TearDown() override { mr_.reset(); }

  template <typename T>
  void ComparePrimitive(const ParamMeta& p1, const ParamMeta& p2) {
    EXPECT_EQ(
      cista::get<T>(cista::get<PrimitiveValue>(p1.value)),
      cista::get<T>(cista::get<PrimitiveValue>(p2.value)));
  }

  template <typename T>
  void CompareVector(const ParamMeta& p1, const ParamMeta& p2) {
    const auto& v1 =
      cista::get<data::vector<T>>(cista::get<VectorValue>(p1.value));
    const auto& v2 =
      cista::get<data::vector<T>>(cista::get<VectorValue>(p2.value));
    ASSERT_EQ(v1.size(), v2.size());
    for (size_t i = 0; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]);
    }
  }

  std::unique_ptr<UcxMemoryResourceManager> mr_;
};

TEST_F(AvroSerializationTest, SerializeDeserializeAllTypes) {
  // 1. Prepare test data
  data::vector<ParamMeta> params;

  // VOID
  params.push_back({ParamType::VOID, nullptr, "void_param"});
  // PRIMITIVE types
  params.push_back(
    {ParamType::PRIMITIVE_BOOL, PrimitiveValue{true}, "bool_param"});
  params.push_back(
    {ParamType::PRIMITIVE_INT8, PrimitiveValue{int8_t{-8}}, "int8_param"});
  params.push_back(
    {ParamType::PRIMITIVE_INT16, PrimitiveValue{int16_t{-16}}, "int16_param"});
  params.push_back(
    {ParamType::PRIMITIVE_INT32, PrimitiveValue{int32_t{-32}}, "int32_param"});
  params.push_back(
    {ParamType::PRIMITIVE_INT64, PrimitiveValue{int64_t{-64}}, "int64_param"});
  params.push_back(
    {ParamType::PRIMITIVE_UINT8, PrimitiveValue{uint8_t{8}}, "uint8_param"});
  params.push_back(
    {ParamType::PRIMITIVE_UINT16, PrimitiveValue{uint16_t{16}},
     "uint16_param"});
  params.push_back(
    {ParamType::PRIMITIVE_UINT32, PrimitiveValue{uint32_t{32}},
     "uint32_param"});
  params.push_back(
    {ParamType::PRIMITIVE_UINT64, PrimitiveValue{uint64_t{64}},
     "uint64_param"});
  params.push_back(
    {ParamType::PRIMITIVE_FLOAT32, PrimitiveValue{32.5f}, "float32_param"});
  params.push_back(
    {ParamType::PRIMITIVE_FLOAT64, PrimitiveValue{64.5}, "float64_param"});
  // STRING
  params.push_back(
    {ParamType::STRING, data::string{"hello world"}, "string_param"});
  // VECTOR types
  params.push_back(
    {ParamType::VECTOR_BOOL, VectorValue{data::vector<bool>{true, false}},
     "vec_bool_param"});
  params.push_back(
    {ParamType::VECTOR_INT8, VectorValue{data::vector<int8_t>{-1, 2}},
     "vec_int8_param"});
  params.push_back(
    {ParamType::VECTOR_INT16, VectorValue{data::vector<int16_t>{-10, 20}},
     "vec_int16_param"});
  params.push_back(
    {ParamType::VECTOR_INT32, VectorValue{data::vector<int32_t>{-100, 200}},
     "vec_int32_param"});
  params.push_back(
    {ParamType::VECTOR_INT64, VectorValue{data::vector<int64_t>{-1000, 2000}},
     "vec_int64_param"});
  params.push_back(
    {ParamType::VECTOR_UINT8, VectorValue{data::vector<uint8_t>{1, 2}},
     "vec_uint8_param"});
  params.push_back(
    {ParamType::VECTOR_UINT16, VectorValue{data::vector<uint16_t>{10, 20}},
     "vec_uint16_param"});
  params.push_back(
    {ParamType::VECTOR_UINT32, VectorValue{data::vector<uint32_t>{100, 200}},
     "vec_uint32_param"});
  params.push_back(
    {ParamType::VECTOR_UINT64, VectorValue{data::vector<uint64_t>{1000, 2000}},
     "vec_uint64_param"});
  params.push_back(
    {ParamType::VECTOR_FLOAT32, VectorValue{data::vector<float>{1.5f, 2.5f}},
     "vec_float32_param"});
  params.push_back(
    {ParamType::VECTOR_FLOAT64, VectorValue{data::vector<double>{10.5, 20.5}},
     "vec_float64_param"});
  // TENSOR_META_VEC (only one tensor parameter allowed now)
  TensorMeta tm1{};
  tm1.device = {kDLCPU, 1};
  tm1.ndim = 2;
  tm1.dtype = {kDLInt, 32, 1};
  tm1.shape = data::vector<int64_t>{10, 20};
  tm1.strides = data::vector<int64_t>{20, 1};
  tm1.byte_offset = 0;

  TensorMeta tm2{};
  tm2.device = {kDLCPU, 0};
  tm2.ndim = 1;
  tm2.dtype = {kDLFloat, 64, 1};
  tm2.shape = data::vector<int64_t>{20};
  tm2.strides = data::vector<int64_t>{1};
  tm2.byte_offset = 10;

  data::vector<TensorMeta> tensor_meta_vec;
  tensor_meta_vec.push_back(tm1);
  tensor_meta_vec.push_back(tm2);
  params.push_back(
    {ParamType::TENSOR_META_VEC, tensor_meta_vec, "tensor_param_vec"});

  RpcRequestHeader original_header;
  original_header.function_id = rpc::function_id_t{12345};
  original_header.session_id = rpc::session_id_t{67890};
  original_header.request_id = rpc::request_id_t{1};
  original_header.params = std::move(params);
  original_header.hlc.TickLocal();
  original_header.workflow_id = rpc::workflow_id_t{456};

  std::vector<ucx_buffer_t> payload_buffers;
  payload_buffers.resize(2);
  payload_buffers[0].data = mr_->allocate(ucx_memory_type::HOST, 10);
  payload_buffers[0].size = 10;
  memset(payload_buffers[0].data, 'A', 10);
  payload_buffers[1].data = mr_->allocate(ucx_memory_type::HOST, 20);
  payload_buffers[1].size = 20;
  memset(payload_buffers[1].data, 'B', 20);

  // Not owning the payload buffers for easy testing.
  UcxBufferVec original_payload(
    *mr_, ucx_memory_type::HOST, payload_buffers, nullptr, false);

  auto test_header = RpcRequestHeader(original_header);
  auto test_payload =
    UcxBufferVec(*mr_, ucx_memory_type::HOST, payload_buffers, nullptr, false);
  // Extract tensor param index (should be the last parameter)
  auto tensor_param_index = utils::ExtractTensorParamIndex(test_header.params);
  AxonRequest request(
    std::move(test_header), std::move(test_payload), tensor_param_index);

  // 2. Serialize
  avro::GenericDatum datum = AvroSerialization::Serialize(request, *mr_);

  // 3. Deserialize
  auto deserialized_request = AvroSerialization::Deserialize(datum, *mr_);

  // 4. Verify Header
  EXPECT_EQ(
    cista::to_idx(original_header.function_id),
    cista::to_idx(deserialized_request->header.function_id));
  EXPECT_EQ(
    cista::to_idx(original_header.session_id),
    cista::to_idx(deserialized_request->header.session_id));
  EXPECT_EQ(
    cista::to_idx(original_header.request_id),
    cista::to_idx(deserialized_request->header.request_id));
  EXPECT_EQ(original_header.hlc.Raw(), deserialized_request->header.hlc.Raw());
  EXPECT_EQ(
    cista::to_idx(original_header.workflow_id),
    cista::to_idx(deserialized_request->header.workflow_id));

  ASSERT_EQ(
    original_header.params.size(), deserialized_request->header.params.size());

  for (size_t i = 0; i < original_header.params.size(); ++i) {
    const auto& p1 = original_header.params[i];
    const auto& p2 = deserialized_request->header.params[i];
    EXPECT_EQ(p1.type, p2.type) << "Mismatch at index " << i;
    EXPECT_EQ(p1.name, p2.name) << "Mismatch at index " << i;

    switch (p1.type) {
      case ParamType::VOID:
        EXPECT_TRUE(cista::holds_alternative<std::nullptr_t>(p2.value));
        break;
      case ParamType::PRIMITIVE_BOOL:
        ComparePrimitive<bool>(p1, p2);
        break;
      case ParamType::PRIMITIVE_INT8:
        ComparePrimitive<int8_t>(p1, p2);
        break;
      case ParamType::PRIMITIVE_INT16:
        ComparePrimitive<int16_t>(p1, p2);
        break;
      case ParamType::PRIMITIVE_INT32:
        ComparePrimitive<int32_t>(p1, p2);
        break;
      case ParamType::PRIMITIVE_INT64:
        ComparePrimitive<int64_t>(p1, p2);
        break;
      case ParamType::PRIMITIVE_UINT8:
        ComparePrimitive<uint8_t>(p1, p2);
        break;
      case ParamType::PRIMITIVE_UINT16:
        ComparePrimitive<uint16_t>(p1, p2);
        break;
      case ParamType::PRIMITIVE_UINT32:
        ComparePrimitive<uint32_t>(p1, p2);
        break;
      case ParamType::PRIMITIVE_UINT64:
        ComparePrimitive<uint64_t>(p1, p2);
        break;
      case ParamType::PRIMITIVE_FLOAT32:
        ComparePrimitive<float>(p1, p2);
        break;
      case ParamType::PRIMITIVE_FLOAT64:
        ComparePrimitive<double>(p1, p2);
        break;
      case ParamType::STRING:
        EXPECT_EQ(
          cista::get<data::string>(p1.value),
          cista::get<data::string>(p2.value));
        break;
      case ParamType::VECTOR_BOOL:
        CompareVector<bool>(p1, p2);
        break;
      case ParamType::VECTOR_INT8:
        CompareVector<int8_t>(p1, p2);
        break;
      case ParamType::VECTOR_INT16:
        CompareVector<int16_t>(p1, p2);
        break;
      case ParamType::VECTOR_INT32:
        CompareVector<int32_t>(p1, p2);
        break;
      case ParamType::VECTOR_INT64:
        CompareVector<int64_t>(p1, p2);
        break;
      case ParamType::VECTOR_UINT8:
        CompareVector<uint8_t>(p1, p2);
        break;
      case ParamType::VECTOR_UINT16:
        CompareVector<uint16_t>(p1, p2);
        break;
      case ParamType::VECTOR_UINT32:
        CompareVector<uint32_t>(p1, p2);
        break;
      case ParamType::VECTOR_UINT64:
        CompareVector<uint64_t>(p1, p2);
        break;
      case ParamType::VECTOR_FLOAT32:
        CompareVector<float>(p1, p2);
        break;
      case ParamType::VECTOR_FLOAT64:
        CompareVector<double>(p1, p2);
        break;
      case ParamType::TENSOR_META: {
        const auto& tm1 = cista::get<TensorMeta>(p1.value);
        const auto& tm2 = cista::get<TensorMeta>(p2.value);
        EXPECT_EQ(tm1.device.device_type, tm2.device.device_type);
        EXPECT_EQ(tm1.device.device_id, tm2.device.device_id);
        EXPECT_EQ(tm1.ndim, tm2.ndim);
        EXPECT_EQ(tm1.dtype.code, tm2.dtype.code);
        EXPECT_EQ(tm1.dtype.bits, tm2.dtype.bits);
        EXPECT_EQ(tm1.dtype.lanes, tm2.dtype.lanes);
        EXPECT_EQ(tm1.byte_offset, tm2.byte_offset);
        ASSERT_EQ(tm1.shape.size(), tm2.shape.size());
        for (size_t j = 0; j < tm1.shape.size(); ++j) {
          EXPECT_EQ(tm1.shape[j], tm2.shape[j]);
        }
        ASSERT_EQ(tm1.strides.size(), tm2.strides.size());
        for (size_t j = 0; j < tm1.strides.size(); ++j) {
          EXPECT_EQ(tm1.strides[j], tm2.strides[j]);
        }
        break;
      }
      case ParamType::TENSOR_META_VEC: {
        const auto& vec1 = cista::get<rpc::TensorMetaVecValue>(p1.value);
        const auto& vec2 = cista::get<rpc::TensorMetaVecValue>(p2.value);
        ASSERT_EQ(vec1.size(), vec2.size());
        for (size_t i = 0; i < vec1.size(); ++i) {
          const auto& tm1 = vec1[i];
          const auto& tm2 = vec2[i];
          EXPECT_EQ(tm1.device.device_type, tm2.device.device_type);
          EXPECT_EQ(tm1.device.device_id, tm2.device.device_id);
          EXPECT_EQ(tm1.ndim, tm2.ndim);
          EXPECT_EQ(tm1.dtype.code, tm2.dtype.code);
          EXPECT_EQ(tm1.dtype.bits, tm2.dtype.bits);
          EXPECT_EQ(tm1.dtype.lanes, tm2.dtype.lanes);
          EXPECT_EQ(tm1.byte_offset, tm2.byte_offset);
          ASSERT_EQ(tm1.shape.size(), tm2.shape.size());
          for (size_t j = 0; j < tm1.shape.size(); ++j) {
            EXPECT_EQ(tm1.shape[j], tm2.shape[j]);
          }
          ASSERT_EQ(tm1.strides.size(), tm2.strides.size());
          for (size_t j = 0; j < tm1.strides.size(); ++j) {
            EXPECT_EQ(tm1.strides[j], tm2.strides[j]);
          }
        }
        break;
      }
      default:
        FAIL() << "Unhandled ParamType in test verification: "
               << static_cast<int>(p1.type);
    }
  }

  // 5. Verify Payload
  ASSERT_TRUE(
    std::holds_alternative<UcxBufferVec>(deserialized_request->payload));
  const auto& deserialized_payload =
    std::get<UcxBufferVec>(deserialized_request->payload);
  ASSERT_EQ(original_payload.size(), deserialized_payload.size());
  for (size_t i = 0; i < original_payload.size(); ++i) {
    EXPECT_EQ(original_payload[i].size, deserialized_payload[i].size);
    EXPECT_EQ(
      0,
      memcmp(
        original_payload[i].data, deserialized_payload[i].data,
        original_payload[i].size));
  }
}

TEST_F(AvroSerializationTest, SerializeDeserialize_SingleBuffer) {
  // 1. Prepare test data
  data::vector<ParamMeta> params;
  TensorMeta tm0{};
  tm0.device = {kDLCPU, 1};
  tm0.ndim = 2;
  tm0.dtype = {kDLInt, 32, 1};
  tm0.shape = data::vector<int64_t>{10, 20};
  tm0.strides = data::vector<int64_t>{20, 1};
  tm0.byte_offset = 0;
  params.push_back({ParamType::TENSOR_META, tm0, "tensor_param_0"});

  RpcRequestHeader original_header;
  original_header.function_id = rpc::function_id_t{12345};
  original_header.session_id = rpc::session_id_t{67890};
  original_header.request_id = rpc::request_id_t{1};
  original_header.params = std::move(params);
  original_header.hlc.TickLocal();
  original_header.workflow_id = rpc::workflow_id_t{456};

  ucx_buffer_t payload_buffer;
  payload_buffer.data = mr_->allocate(ucx_memory_type::HOST, 10);
  payload_buffer.size = 10;
  memset(payload_buffer.data, 'C', 10);

  UcxBuffer original_payload(
    *mr_, ucx_memory_type::HOST, payload_buffer, nullptr, false);

  auto test_header = RpcRequestHeader(original_header);
  ucx_buffer_t test_payload_buffer;
  test_payload_buffer.data = payload_buffer.data;
  test_payload_buffer.size = payload_buffer.size;
  UcxBuffer test_payload(
    *mr_, ucx_memory_type::HOST, test_payload_buffer, nullptr, false);
  AxonRequest request(
    std::move(test_header), std::move(test_payload), std::optional<size_t>(0));

  // 2. Serialize
  avro::GenericDatum datum = AvroSerialization::Serialize(request, *mr_);

  // 3. Deserialize
  auto deserialized_request = AvroSerialization::Deserialize(datum, *mr_);

  // 4. Verify Header
  EXPECT_EQ(
    cista::to_idx(original_header.function_id),
    cista::to_idx(deserialized_request->header.function_id));
  ASSERT_EQ(
    original_header.params.size(), deserialized_request->header.params.size());
  const auto& p1 = original_header.params[0];
  const auto& p2 = deserialized_request->header.params[0];
  EXPECT_EQ(p1.type, p2.type);
  EXPECT_EQ(p1.name, p2.name);
  const auto& tm1 = cista::get<TensorMeta>(p1.value);
  const auto& tm2 = cista::get<TensorMeta>(p2.value);
  EXPECT_EQ(tm1.device.device_type, tm2.device.device_type);
  EXPECT_EQ(tm1.device.device_id, tm2.device.device_id);
  EXPECT_EQ(tm1.ndim, tm2.ndim);
  EXPECT_EQ(tm1.dtype.code, tm2.dtype.code);

  // 5. Verify Payload
  ASSERT_TRUE(std::holds_alternative<UcxBuffer>(deserialized_request->payload));
  const auto& deserialized_payload =
    std::get<UcxBuffer>(deserialized_request->payload);
  EXPECT_EQ(original_payload.size(), deserialized_payload.size());
  EXPECT_EQ(
    0,
    memcmp(
      original_payload.data(), deserialized_payload.data(),
      original_payload.size()));

  mr_->deallocate(
    ucx_memory_type::HOST, payload_buffer.data, payload_buffer.size);
}

TEST_F(AvroSerializationTest, SerializeDeserialize_NoPayload) {
  // 1. Prepare test data
  RpcRequestHeader original_header;
  original_header.function_id = rpc::function_id_t{1};
  original_header.session_id = rpc::session_id_t{2};
  original_header.request_id = rpc::request_id_t{3};
  original_header.hlc.TickLocal();
  original_header.workflow_id = rpc::workflow_id_t{4};

  AxonRequest request(
    std::move(original_header), std::monostate{}, std::nullopt);

  // 2. Serialize
  avro::GenericDatum datum = AvroSerialization::Serialize(request, *mr_);

  // 3. Deserialize
  auto deserialized_request = AvroSerialization::Deserialize(datum, *mr_);

  // 4. Verify
  ASSERT_TRUE(
    std::holds_alternative<std::monostate>(deserialized_request->payload));
  EXPECT_EQ(
    cista::to_idx(request.header.function_id),
    cista::to_idx(deserialized_request->header.function_id));
}

}  // namespace storage
}  // namespace axon
}  // namespace eux
