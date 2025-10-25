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

  auto request = client.PrepareRequest(
    function_id_t{1}, session_id_t{123}, request_id_t{1001}, 5, 10);

  EXPECT_EQ(request.function_id.v_, 1);
  EXPECT_EQ(request.session_id.v_, 123);
  ASSERT_EQ(request.request_id.v_, 1001);
  ASSERT_EQ(request.params.size(), 2);

  EXPECT_EQ(request.GetPrimitive<int>(0), 5);
  EXPECT_EQ(request.GetPrimitive<int>(1), 10);

  auto request2 = client.PrepareRequest(
    function_id_t{2}, session_id_t{123}, request_id_t{1002});
  EXPECT_EQ(request2.request_id.v_, 1002);
}

TEST_F(RpcRequestBuilderTest, PrepareRequestWithPayload) {
  RpcRequestBuilder client;
  data::string test_string = "hello";
  ucxx::DefaultUcxMemoryResourceManager mr;
  ucxx::UcxBufferVec payload(mr, ucx_memory_type_t::HOST, {10, 20});

  auto [request, returned_payload] = client.PrepareRequest(
    function_id_t{3}, session_id_t{456}, request_id_t{1003}, 42, test_string,
    std::move(payload));

  EXPECT_EQ(request.function_id.v_, 3);
  EXPECT_EQ(request.session_id.v_, 456);
  EXPECT_EQ(request.request_id.v_, 1003);

  // Payload argument is not serialized
  ASSERT_EQ(request.params.size(), 2);

  EXPECT_EQ(request.GetPrimitive<int>(0), 42);
  EXPECT_EQ(request.GetString(1), test_string);

  ASSERT_FALSE(std::holds_alternative<std::monostate>(returned_payload));
  ASSERT_TRUE(std::holds_alternative<ucxx::UcxBufferVec>(returned_payload));
  auto& casted_payload = std::get<ucxx::UcxBufferVec>(returned_payload);
  EXPECT_EQ(casted_payload.size(), 2);
  EXPECT_EQ(casted_payload[0].size, 10);
}

TEST_F(RpcRequestBuilderTest, PrepareRequestWithMovedPayload) {
  RpcRequestBuilder client;
  ucxx::DefaultUcxMemoryResourceManager mr;
  auto payload_ptr =
    std::make_unique<ucxx::UcxBuffer>(mr, ucx_memory_type_t::HOST, 128);

  // This is not a valid payload type and will not be extracted.
  // We'll test moving a valid one instead.
  ucxx::UcxBuffer payload(mr, ucx_memory_type_t::HOST, 123);

  auto [request, returned_payload] = client.PrepareRequest(
    function_id_t{5}, session_id_t{101}, request_id_t{1004}, 1,
    std::move(payload), 2);

  ASSERT_FALSE(std::holds_alternative<std::monostate>(returned_payload));
  ASSERT_TRUE(std::holds_alternative<ucxx::UcxBuffer>(returned_payload));
  auto moved_payload = std::get<ucxx::UcxBuffer>(std::move(returned_payload));
  EXPECT_EQ(moved_payload.size(), 123);
  // The original payload should be empty after being moved from.
  EXPECT_EQ(payload.size(), 0);
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

  auto [req, payload] = builder.PrepareRequest(
    function_id_t{10}, session_id_t{777}, request_id_t{1005}, val_bool,
    val_int8, val_uint64, val_double, val_strong, val_vec, val_tensor,
    std::move(val_payload));

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
  ASSERT_FALSE(std::holds_alternative<std::monostate>(payload));
  ASSERT_TRUE(std::holds_alternative<ucxx::UcxBuffer>(payload));
  auto& buf = std::get<ucxx::UcxBuffer>(payload);
  EXPECT_EQ(buf.size(), 256);
}

TEST_F(RpcRequestBuilderTest, PrepareRequestWithSignatureValidation) {
  RpcRequestBuilder builder;

  // Create a mock signature.
  RpcFunctionSignature sig;
  sig.id = function_id_t{20};
  sig.param_types.push_back(ParamType::PRIMITIVE_INT32);
  sig.param_types.push_back(ParamType::STRING);
  sig.takes_context = true;

  ucxx::DefaultUcxMemoryResourceManager mr;
  ucxx::UcxBuffer payload(mr, ucx_memory_type_t::HOST, 128);

  // This call should succeed as the types match the signature.
  auto [req, p] = builder.PrepareRequest(
    function_id_t{20}, sig, session_id_t{888}, request_id_t{1006}, 123, "test",
    std::move(payload));

  EXPECT_EQ(req.function_id.v_, 20);
  EXPECT_EQ(req.request_id.v_, 1006);
  ASSERT_EQ(req.params.size(), 2);
  EXPECT_EQ(req.GetPrimitive<int32_t>(0), 123);
  EXPECT_EQ(req.GetString(1), "test");
  ASSERT_FALSE(std::holds_alternative<std::monostate>(p));

  // This test demonstrates the assert check. In debug builds, it would fail.
  // We can't easily test for assert failure in gtest without DEATH tests,
  // so we document it here.
  //
  // RpcFunctionSignature bad_sig;
  // bad_sig.param_types.push_back(ParamType::PRIMITIVE_FLOAT64);
  //
  // builder.PrepareRequest(function_id_t{21}, bad_sig, 123); // Mismatched
  // type
}

/*
namespace {
struct InvalidType {};
}  // namespace

TEST_F(RpcRequestBuilderTest, FailsToCompileWithInvalidType) {
  RpcRequestBuilder builder(session_id_t{888});
  InvalidType invalid_arg;
  // This line should fail to compile due to the static_assert in pack_arg.
  builder.PrepareRequest(function_id_t{11}, invalid_arg);
}
*/

}  // namespace rpc
}  // namespace eux
