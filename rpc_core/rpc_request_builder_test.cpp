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
#include <string>
#include <utility>
#include <vector>

#include "rpc_core/rpc_types.hpp"
#include "ucx_context/ucx_context_data.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

namespace eux {
namespace rpc {

class RpcRequestBuilderTest : public ::testing::Test {};

TEST_F(RpcRequestBuilderTest, PrepareRequestSimple) {
  RpcRequestBuilder client(session_id_t{123});

  auto [request, payload] = client.prepare_request(function_id_t{1}, 5, 10);

  EXPECT_EQ(request.function_id.v_, 1);
  EXPECT_EQ(request.session_id.v_, 123);
  EXPECT_EQ(request.request_id.v_, 0);  // First request
  ASSERT_EQ(request.params.size(), 2);

  EXPECT_EQ(request.get_primitive<int>(0), 5);
  EXPECT_EQ(request.get_primitive<int>(1), 10);
  EXPECT_FALSE(payload.has_value());

  auto [request2, payload2] = client.prepare_request(function_id_t{2});
  EXPECT_EQ(request2.request_id.v_, 1);  // Second request
  EXPECT_FALSE(payload2.has_value());
}

TEST_F(RpcRequestBuilderTest, PrepareRequestWithPayload) {
  RpcRequestBuilder client(session_id_t{456});
  data::string test_string = "hello";
  ucxx::DefaultUcxMemoryResourceManager mr;
  ucxx::UcxBufferVec payload(mr, ucx_memory_type_t::HOST, {10, 20});

  auto [request, returned_payload] = client.prepare_request(
    function_id_t{3}, 42, test_string, std::move(payload));

  EXPECT_EQ(request.function_id.v_, 3);
  EXPECT_EQ(request.session_id.v_, 456);
  EXPECT_EQ(request.request_id.v_, 0);

  // Payload argument is not serialized
  ASSERT_EQ(request.params.size(), 2);

  EXPECT_EQ(request.get_primitive<int>(0), 42);
  EXPECT_EQ(request.get_string(1), test_string);

  ASSERT_TRUE(returned_payload.has_value());
  ASSERT_TRUE(
    std::holds_alternative<ucxx::UcxBufferVec>(returned_payload.value()));
  auto& casted_payload = std::get<ucxx::UcxBufferVec>(returned_payload.value());
  EXPECT_EQ(casted_payload.size(), 2);
  EXPECT_EQ(casted_payload[0].size, 10);
}

TEST_F(RpcRequestBuilderTest, PrepareRequestWithMovedPayload) {
  RpcRequestBuilder client(session_id_t{101});
  ucxx::DefaultUcxMemoryResourceManager mr;
  auto payload_ptr =
    std::make_unique<ucxx::UcxBuffer>(mr, ucx_memory_type_t::HOST, 128);

  // This is not a valid payload type and will not be extracted.
  // We'll test moving a valid one instead.
  ucxx::UcxBuffer payload(mr, ucx_memory_type_t::HOST, 123);

  auto [request, returned_payload] =
    client.prepare_request(function_id_t{5}, 1, std::move(payload), 2);

  ASSERT_TRUE(returned_payload.has_value());
  ASSERT_TRUE(
    std::holds_alternative<ucxx::UcxBuffer>(returned_payload.value()));
  auto moved_payload =
    std::get<ucxx::UcxBuffer>(std::move(returned_payload.value()));
  EXPECT_EQ(moved_payload.size(), 123);
  // The original payload should be empty after being moved from.
  EXPECT_EQ(payload.size(), 0);
}

TEST_F(RpcRequestBuilderTest, PrepareRequestWithComprehensiveTypes) {
  RpcRequestBuilder builder(session_id_t{777});
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

  auto [req, payload] = builder.prepare_request(
    function_id_t{10}, val_bool, val_int8, val_uint64, val_double, val_strong,
    val_vec, val_tensor, std::move(val_payload));

  // Verify header
  EXPECT_EQ(req.function_id.v_, 10);
  EXPECT_EQ(req.session_id.v_, 777);
  EXPECT_EQ(req.request_id.v_, 0);
  ASSERT_EQ(req.params.size(), 7);

  // Verify params
  EXPECT_EQ(req.get_primitive<bool>(0), val_bool);
  EXPECT_EQ(req.get_primitive<int8_t>(1), val_int8);
  EXPECT_EQ(req.get_primitive<uint64_t>(2), val_uint64);
  EXPECT_EQ(req.get_primitive<double>(3), val_double);
  EXPECT_EQ(req.get_primitive<uint32_t>(4), val_strong.v_);

  const auto& vec = req.get_vector<int16_t>(5);
  ASSERT_EQ(vec.size(), 3);
  EXPECT_EQ(vec[0], -100);
  EXPECT_EQ(vec[2], -300);

  const auto& tensor = req.get_tensor(6);
  EXPECT_EQ(tensor.ndim, 2);

  // Verify payload
  ASSERT_TRUE(payload.has_value());
  ASSERT_TRUE(std::holds_alternative<ucxx::UcxBuffer>(payload.value()));
  auto& buf = std::get<ucxx::UcxBuffer>(payload.value());
  EXPECT_EQ(buf.size(), 256);
}

/*
namespace {
struct InvalidType {};
}  // namespace

TEST_F(RpcRequestBuilderTest, FailsToCompileWithInvalidType) {
  RpcRequestBuilder builder(session_id_t{888});
  InvalidType invalid_arg;
  // This line should fail to compile due to the static_assert in pack_arg.
  builder.prepare_request(function_id_t{11}, invalid_arg);
}
*/

}  // namespace rpc
}  // namespace eux
