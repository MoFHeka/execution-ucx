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

#include "rpc_core/rpc_response_builder.hpp"

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

class RpcResponseBuilderTest : public ::testing::Test {};

TEST_F(RpcResponseBuilderTest, PrepareResponseSimple) {
  RpcResponseBuilder builder;

  auto response =
    builder.PrepareResponse(session_id_t{123}, request_id_t{456}, 5, "hello");

  EXPECT_EQ(response.session_id.v_, 123);
  EXPECT_EQ(response.request_id.v_, 456);
  ASSERT_EQ(response.results.size(), 2);

  EXPECT_EQ(response.GetPrimitive<int>(0), 5);
  EXPECT_EQ(response.GetString(1), "hello");
}

TEST_F(RpcResponseBuilderTest, PrepareResponseWithPayload) {
  RpcResponseBuilder builder;
  data::string test_string = "world";
  ucxx::DefaultUcxMemoryResourceManager mr;
  ucxx::UcxBufferVec payload(mr, ucx_memory_type_t::HOST, {10, 20});

  auto [response, returned_payload] = builder.PrepareResponse(
    session_id_t{789}, request_id_t{101}, 42, test_string, std::move(payload));

  EXPECT_EQ(response.session_id.v_, 789);
  EXPECT_EQ(response.request_id.v_, 101);

  // Payload argument is not serialized into results
  ASSERT_EQ(response.results.size(), 2);
  EXPECT_EQ(response.GetPrimitive<int>(0), 42);
  EXPECT_EQ(response.GetString(1), test_string);

  ASSERT_FALSE(std::holds_alternative<std::monostate>(returned_payload));
  ASSERT_TRUE(std::holds_alternative<ucxx::UcxBufferVec>(returned_payload));
  auto& casted_payload = std::get<ucxx::UcxBufferVec>(returned_payload);
  EXPECT_EQ(casted_payload.size(), 2);
  EXPECT_EQ(casted_payload[0].size, 10);
}

TEST_F(RpcResponseBuilderTest, PrepareResponseNoResults) {
  RpcResponseBuilder builder;

  auto response = builder.PrepareResponse(session_id_t{1}, request_id_t{2});

  EXPECT_EQ(response.session_id.v_, 1);
  EXPECT_EQ(response.request_id.v_, 2);
  EXPECT_EQ(response.results.size(), 0);
}

TEST_F(RpcResponseBuilderTest, PrepareResponseWithComprehensiveTypes) {
  RpcResponseBuilder builder;
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

  auto [res, payload] = builder.PrepareResponse(
    session_id_t{100}, request_id_t{200}, val_bool, val_int8, val_uint64,
    val_double, val_strong, val_vec, val_tensor, std::move(val_payload));

  // Verify header
  EXPECT_EQ(res.session_id.v_, 100);
  EXPECT_EQ(res.request_id.v_, 200);
  ASSERT_EQ(res.results.size(), 7);

  // Verify params
  EXPECT_EQ(res.GetPrimitive<bool>(0), val_bool);
  EXPECT_EQ(res.GetPrimitive<int8_t>(1), val_int8);
  EXPECT_EQ(res.GetPrimitive<uint64_t>(2), val_uint64);
  EXPECT_EQ(res.GetPrimitive<double>(3), val_double);
  EXPECT_EQ(res.GetPrimitive<uint32_t>(4), val_strong.v_);

  const auto& vec = res.GetVector<int16_t>(5);
  ASSERT_EQ(vec.size(), 3);
  EXPECT_EQ(vec[0], -100);
  EXPECT_EQ(vec[2], -300);

  const auto& tensor = res.GetTensor(6);
  EXPECT_EQ(tensor.ndim, 2);

  // Verify payload
  ASSERT_FALSE(std::holds_alternative<std::monostate>(payload));
  ASSERT_TRUE(std::holds_alternative<ucxx::UcxBuffer>(payload));
  auto& buf = std::get<ucxx::UcxBuffer>(payload);
  EXPECT_EQ(buf.size(), 256);
}

}  // namespace rpc
}  // namespace eux
