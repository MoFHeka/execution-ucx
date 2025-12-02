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

#include "rpc_core/rpc_request_builder.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "rpc_core/rpc_types.hpp"
#include "ucx_context/ucx_context_data.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

namespace eux {
namespace rpc {

class RpcRequestBuilderTest : public ::testing::Test {};

TEST_F(RpcRequestBuilderTest, PrepareRequestSimple) {
  RpcRequestBuilder client;

  RpcRequestBuilderOptions options{
    .session_id = session_id_t{123},
    .request_id = request_id_t{1001},
    .function_id = function_id_t{1},
  };
  auto request = client.PrepareRequest(options, 5, 10);

  static_assert(std::is_same_v<decltype(request), RpcRequestHeader>);
  EXPECT_EQ(request.function_id.v_, 1);
  EXPECT_EQ(request.session_id.v_, 123);
  ASSERT_EQ(request.request_id.v_, 1001);
  ASSERT_EQ(request.params.size(), 2);

  EXPECT_EQ(request.GetPrimitive<int>(0), 5);
  EXPECT_EQ(request.GetPrimitive<int>(1), 10);

  RpcRequestBuilderOptions options2{
    .session_id = session_id_t{123},
    .request_id = request_id_t{1002},
    .function_id = function_id_t{2},
  };
  auto request2 = client.PrepareRequest(options2);
  EXPECT_EQ(request2.request_id.v_, 1002);
  static_assert(std::is_same_v<decltype(request2), RpcRequestHeader>);
}

TEST_F(RpcRequestBuilderTest, PrepareRequestWithPayload) {
  RpcRequestBuilder client;
  data::string test_string = "hello";
  ucxx::DefaultUcxMemoryResourceManager mr;
  ucxx::UcxBufferVec payload(mr, ucx_memory_type_t::HOST, {10, 20});

  RpcRequestBuilderOptions options3{
    .session_id = session_id_t{456},
    .request_id = request_id_t{1003},
    .function_id = function_id_t{3},
  };
  auto [request, returned_payload] =
    client.PrepareRequest(options3, 42, test_string, std::move(payload));

  EXPECT_EQ(request.function_id.v_, 3);
  EXPECT_EQ(request.session_id.v_, 456);
  EXPECT_EQ(request.request_id.v_, 1003);

  // Payload argument is not serialized
  ASSERT_EQ(request.params.size(), 2);

  EXPECT_EQ(request.GetPrimitive<int>(0), 42);
  EXPECT_EQ(request.GetString(1), test_string);

  ASSERT_TRUE((std::is_same_v<ucxx::UcxBufferVec, decltype(returned_payload)>));
  EXPECT_EQ(returned_payload.size(), 2);
  EXPECT_EQ(returned_payload[0].size, 10);
}

TEST_F(RpcRequestBuilderTest, PrepareRequestWithMovedPayload) {
  RpcRequestBuilder client;
  ucxx::DefaultUcxMemoryResourceManager mr;
  auto payload_ptr =
    std::make_unique<ucxx::UcxBuffer>(mr, ucx_memory_type_t::HOST, 128);

  // This is not a valid payload type and will not be extracted.
  // We'll test moving a valid one instead.
  ucxx::UcxBuffer payload(mr, ucx_memory_type_t::HOST, 123);

  RpcRequestBuilderOptions options4{
    .session_id = session_id_t{101},
    .request_id = request_id_t{1004},
    .function_id = function_id_t{5},
  };
  auto [request, returned_payload] =
    client.PrepareRequest(options4, 1, std::move(payload), 2);

  ASSERT_TRUE((std::is_same_v<ucxx::UcxBuffer, decltype(returned_payload)>));
  auto moved_payload = std::move(returned_payload);
  EXPECT_EQ(moved_payload.size(), 123);
  // The original payload should be empty after being moved from.
  EXPECT_EQ(returned_payload.size(), 0);
}

TEST_F(RpcRequestBuilderTest, PrepareRequestWithComprehensiveTypes) {
  RpcRequestBuilder builder;
  ucxx::DefaultUcxMemoryResourceManager mr;

  // Types to test
  bool val_bool = true;
  int8_t val_int8 = -10;
  uint64_t val_uint64 = 123456789012345ULL;
  double val_double = 3.14159;
  function_id_t val_strong = function_id_t{99};
  data::vector<int16_t> val_vec = {-100, 200, -300};
  TensorMeta val_tensor{};
  val_tensor.ndim = 2;
  ucxx::UcxBuffer val_payload(mr, ucx_memory_type_t::HOST, 256);

  RpcRequestBuilderOptions options5{
    .session_id = session_id_t{777},
    .request_id = request_id_t{1005},
    .function_id = function_id_t{10},
  };
  auto [req, payload] = builder.PrepareRequest(
    options5, val_bool, val_int8, val_uint64, val_double, val_strong, val_vec,
    val_tensor, std::move(val_payload));

  // Verify header
  EXPECT_EQ(req.function_id.v_, 10);
  EXPECT_EQ(req.session_id.v_, 777);
  EXPECT_EQ(req.request_id.v_, 1005);
  ASSERT_EQ(req.params.size(), 7);

  // Verify params
  EXPECT_EQ(req.GetPrimitive<bool>(0), val_bool);
  EXPECT_EQ(req.GetPrimitive<int8_t>(1), val_int8);
  EXPECT_EQ(req.GetPrimitive<uint64_t>(2), val_uint64);
  EXPECT_EQ(req.GetPrimitive<double>(3), val_double);
  EXPECT_EQ(req.GetPrimitive<uint32_t>(4), val_strong.v_);

  const auto& vec = req.GetVector<int16_t>(5);
  ASSERT_EQ(vec.size(), 3);
  EXPECT_EQ(vec[0], -100);
  EXPECT_EQ(vec[2], -300);

  const auto& tensor = req.GetTensor(6);
  EXPECT_EQ(tensor.ndim, 2);

  // Verify payload
  ASSERT_TRUE((std::is_same_v<ucxx::UcxBuffer, decltype(payload)>));
  EXPECT_EQ(payload.size(), 256);
}

TEST_F(RpcRequestBuilderTest, PrepareRequestWithSignatureValidation) {
  RpcRequestBuilder builder;

  RpcRequestBuilderOptions options6{
    .session_id = session_id_t{888},
    .request_id = request_id_t{1006},
    .function_id = function_id_t{20},
  };
  // Create a mock signature.
  RpcFunctionSignature sig;
  sig.id = function_id_t{20};
  sig.param_types.push_back(ParamType::PRIMITIVE_INT32);
  sig.param_types.push_back(ParamType::STRING);

  ucxx::DefaultUcxMemoryResourceManager mr;
  ucxx::UcxBuffer payload(mr, ucx_memory_type_t::HOST, 128);

  // This call should succeed as the types match the signature.
  auto [req, p] =
    builder.PrepareRequest(options6, sig, 123, "test", std::move(payload));

  EXPECT_EQ(req.function_id.v_, 20);
  EXPECT_EQ(req.request_id.v_, 1006);
  ASSERT_EQ(req.params.size(), 2);
  EXPECT_EQ(req.GetPrimitive<int32_t>(0), 123);
  EXPECT_EQ(req.GetString(1), "test");
  ASSERT_FALSE((std::is_same_v<std::monostate, decltype(p)>));

  // Failure cases
  RpcRequestBuilderOptions options_fail{
    .session_id = session_id_t{889},
    .request_id = request_id_t{1007},
    .function_id = function_id_t{21},
  };

  // 1. Mismatch count
  EXPECT_THROW(
    builder.PrepareRequest(options_fail, sig, 123), std::invalid_argument);

  // 2. Mismatch type
  EXPECT_THROW(
    builder.PrepareRequest(options_fail, sig, 123, 456), std::invalid_argument);
}

TEST_F(RpcRequestBuilderTest, PrepareRequestDynamic) {
  RpcRequestBuilder builder;
  RpcRequestBuilderOptions options{
    .session_id = session_id_t{999},
    .request_id = request_id_t{2000},
    .function_id = function_id_t{30},
  };

  // Case 1: No payload
  {
    data::vector<ParamMeta> params;
    ParamMeta p1;
    p1.type = ParamType::PRIMITIVE_INT32;
    p1.value.emplace<PrimitiveValue>(100);
    params.push_back(std::move(p1));

    auto req = builder.PrepareRequest(options, std::move(params));
    static_assert(std::is_same_v<decltype(req), RpcRequestHeader>);
    EXPECT_EQ(req.params.size(), 1);
    EXPECT_EQ(req.GetPrimitive<int32_t>(0), 100);
  }

  // Case 2: With payload
  {
    ucxx::DefaultUcxMemoryResourceManager mr;
    ucxx::UcxBuffer payload(mr, ucx_memory_type_t::HOST, 64);

    data::vector<ParamMeta> params;
    ParamMeta p1;
    p1.type = ParamType::PRIMITIVE_BOOL;
    p1.value.emplace<PrimitiveValue>(true);
    params.push_back(std::move(p1));

    auto [req, returned_payload] =
      builder.PrepareRequest(options, std::move(params), std::move(payload));
    EXPECT_EQ(req.params.size(), 1);
    EXPECT_EQ(req.GetPrimitive<bool>(0), true);
    EXPECT_TRUE((std::is_same_v<ucxx::UcxBuffer, decltype(returned_payload)>));
  }
}

TEST_F(RpcRequestBuilderTest, PrepareRequestDynamicWithSignature) {
  RpcRequestBuilder builder;
  RpcRequestBuilderOptions options{
    .session_id = session_id_t{1000},
    .request_id = request_id_t{3000},
    .function_id = function_id_t{40},
  };

  RpcFunctionSignature sig;
  sig.id = function_id_t{40};
  sig.param_types.push_back(ParamType::PRIMITIVE_INT32);

  // Success
  {
    data::vector<ParamMeta> params;
    ParamMeta p1;
    p1.type = ParamType::PRIMITIVE_INT32;
    p1.value.emplace<PrimitiveValue>(123);
    params.push_back(std::move(p1));

    auto req = builder.PrepareRequest(options, sig, std::move(params));
    static_assert(std::is_same_v<decltype(req), RpcRequestHeader>);
    EXPECT_EQ(req.GetPrimitive<int32_t>(0), 123);
  }

  // Failure
  {
    data::vector<ParamMeta> params;
    ParamMeta p1;
    p1.type = ParamType::PRIMITIVE_FLOAT64;  // Mismatch
    p1.value.emplace<PrimitiveValue>(3.14);
    params.push_back(std::move(p1));

    EXPECT_THROW(
      builder.PrepareRequest(options, sig, std::move(params)),
      std::invalid_argument);
  }
}

}  // namespace rpc
}  // namespace eux
