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

#include "rpc_core/async_rpc_dispatcher.hpp"

#include <gtest/gtest.h>

#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <unifex/any_sender_of.hpp>
#include <unifex/just.hpp>
#include <unifex/just_done.hpp>
#include <unifex/just_error.hpp>
#include <unifex/just_from.hpp>
#include <unifex/let_error.hpp>
#include <unifex/let_value.hpp>
#include <unifex/let_value_with.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/then.hpp>

#include "rpc_core/rpc_status.hpp"
#include "ucx_context/ucx_context_data.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

namespace eux {
namespace rpc {

// =============================================================================
// Test Functions and Types
// =============================================================================

// Define the concrete async functions for testing
unifex::any_sender_of<int32_t> add_async(int32_t a, int32_t b) {
  return unifex::just(a + b);
}

unifex::any_sender_of<> void_return_async() { return unifex::just(); }

unifex::any_sender_of<int32_t> error_return_async() {
  return unifex::just_error(
    std::make_exception_ptr(std::runtime_error("Test error")));
}

unifex::any_sender_of<data::string> concat_async(
  const data::string& a, const data::string& b) {
  return unifex::just(data::string(
    std::string(a.data(), a.size()) + std::string(b.data(), b.size())));
}

unifex::any_sender_of<int32_t> sum_vec_async(const data::vector<int32_t>& vec) {
  int32_t sum = 0;
  for (int32_t v : vec) {
    sum += v;
  }
  return unifex::just(sum);
}

unifex::any_sender_of<ucxx::UcxBuffer> return_ucx_buffer_async(
  ucxx::UcxMemoryResourceManager& mr) {
  return unifex::just(
    std::move(ucxx::UcxBuffer(mr, ucx_memory_type::HOST, 1024)));
}

unifex::any_sender_of<ucxx::UcxBufferVec> return_ucx_buffer_vec_async(
  ucxx::UcxMemoryResourceManager& mr) {
  return unifex::just(
    ucxx::UcxBufferVec(mr, ucx_memory_type::HOST, {128, 256}));
}

unifex::any_sender_of<uint64_t> process_buffer_vec_async(
  ucxx::UcxBufferVec&& in_vec) {
  uint64_t total_size = 0;
  for (const auto& buf : in_vec) {
    total_size += buf.size;
  }
  return unifex::just(total_size);
}

unifex::any_sender_of<ucxx::UcxBufferVec> echo_buffer_vec_async(
  ucxx::UcxMemoryResourceManager& mr, const ucxx::UcxBufferVec& in_vec) {
  // Create a new UcxBufferVec with the same sizes as input
  std::vector<size_t> sizes;
  for (const auto& buf : in_vec) {
    sizes.push_back(buf.size);
  }
  return unifex::just(ucxx::UcxBufferVec(mr, ucx_memory_type::HOST, sizes));
}

unifex::any_sender_of<std::pair<RpcResponseHeader, ucxx::UcxBufferVec>>
process_mixed_request_async(
  ucxx::UcxMemoryResourceManager& mr, const RpcRequestHeader& req,
  int multiplier, const data::string& tag,
  const ucxx::UcxBufferVec& input_vec) {
  RpcResponseHeader resp;
  resp.session_id = req.session_id;
  resp.request_id = req.request_id;

  // Calculate a value based on serializable and context inputs
  uint64_t total_input_size = 0;
  for (const auto& buf : input_vec) {
    total_input_size += buf.size;
  }
  uint64_t final_value = total_input_size * multiplier;

  // Pack serializable results into the header
  ParamMeta p1;
  p1.type = ParamType::PRIMITIVE_UINT64;
  p1.value.emplace<PrimitiveValue>(final_value);
  resp.results.emplace_back(std::move(p1));

  ParamMeta p2;
  p2.type = ParamType::STRING;
  p2.value.emplace<data::string>(tag);
  resp.results.emplace_back(std::move(p2));

  // Create a new UcxBufferVec to return as context
  ucxx::UcxBufferVec output_vec(
    mr, ucx_memory_type::HOST, std::vector<size_t>{final_value});

  return unifex::just(std::make_pair(std::move(resp), std::move(output_vec)));
}

unifex::any_sender_of<TensorMeta> mutate_tensor_meta_async(TensorMeta meta) {
  for (auto& dim : meta.shape) {
    dim += 1;
  }
  meta.byte_offset += 8;
  return unifex::just(std::move(meta));
}

unifex::any_sender_of<int64_t> tensor_meta_numel_async(const TensorMeta& meta) {
  int64_t total = 1;
  for (auto dim : meta.shape) {
    total *= dim;
  }
  return unifex::just(total);
}

class MyService {
 public:
  unifex::any_sender_of<int> multiply_async(int a, int b) {
    return unifex::just(a * b);
  }
};

unifex::any_sender_of<RpcResponseHeader> custom_response_header_async(
  session_id_t session_id) {
  RpcResponseHeader response{};
  response.session_id = session_id;
  ParamMeta result_meta{};
  result_meta.type = ParamType::STRING;
  result_meta.value.emplace<data::string>("custom_header");
  response.results.emplace_back(std::move(result_meta));
  return unifex::just(std::move(response));
}

unifex::any_sender_of<std::tuple<int, std::string, float>>
return_multiple_values_async() {
  return unifex::just(std::make_tuple(123, std::string("hello"), 3.14f));
}

unifex::any_sender_of<std::tuple<int, ucxx::UcxBufferVec>>
return_multiple_with_payload_async(ucxx::UcxMemoryResourceManager& mr) {
  return unifex::just(std::make_tuple(
    456, ucxx::UcxBufferVec(mr, ucx_memory_type::HOST, {16, 32})));
}

TensorMeta make_test_tensor_meta() {
  TensorMeta meta{};
  meta.device = DLDevice{kDLCPU, 0};
  meta.ndim = 2;
  meta.dtype = DLDataType{kDLFloat, 32, 1};
  meta.byte_offset = 16;
  meta.shape.emplace_back(2);
  meta.shape.emplace_back(3);
  meta.strides.emplace_back(3);
  meta.strides.emplace_back(1);
  return meta;
}

// --- Test Fixture ---
class AsyncRpcDispatcherTest : public ::testing::Test {
 protected:
  AsyncRpcDispatcherTest() : dispatcher_("test_instance"), mr_() {}

  AsyncRpcDispatcher dispatcher_;
  RpcResponseBuilder builder_;
  ucxx::DefaultUcxMemoryResourceManager mr_;

  void SetUp() override {
    // Register the concrete functions
    dispatcher_.RegisterFunction(function_id_t{1}, &add_async, "add_async");
    dispatcher_.RegisterFunction(
      function_id_t{2},
      [this](const ucxx::UcxBufferVec& in_vec) {
        return echo_buffer_vec_async(mr_, in_vec);
      },
      "echo_buffer_vec_async");
    dispatcher_.RegisterFunction(
      function_id_t{4}, &void_return_async, "void_return_async");
    dispatcher_.RegisterFunction(
      function_id_t{5}, &error_return_async, "error_return_async");
    dispatcher_.RegisterFunction(
      function_id_t{6}, &concat_async, "concat_async");
    dispatcher_.RegisterFunction(
      function_id_t{7}, &sum_vec_async, "sum_vec_async");
    dispatcher_.RegisterFunction(
      function_id_t{8}, [this]() { return return_ucx_buffer_async(mr_); },
      "return_ucx_buffer_async");
    dispatcher_.RegisterFunction(
      function_id_t{9}, [this]() { return return_ucx_buffer_vec_async(mr_); },
      "return_ucx_buffer_vec_async");
    dispatcher_.RegisterFunction(
      function_id_t{10}, &process_buffer_vec_async, "process_buffer_vec_async");
    dispatcher_.RegisterFunction(
      function_id_t{11},
      [this](
        const RpcRequestHeader& req, int multiplier, const data::string& tag,
        const ucxx::UcxBufferVec& input_vec) {
        return process_mixed_request_async(
          mr_, req, multiplier, tag, input_vec);
      },
      "process_mixed_request_async");
    dispatcher_.RegisterFunction(
      function_id_t{12}, &tensor_meta_numel_async, "tensor_meta_numel_async");
    dispatcher_.RegisterFunction(
      function_id_t{13}, &custom_response_header_async,
      "custom_response_header_async");
    dispatcher_.RegisterFunction(
      function_id_t{14}, &return_multiple_values_async,
      "return_multiple_values_async");
    dispatcher_.RegisterFunction(
      function_id_t{15},
      [this]() { return return_multiple_with_payload_async(mr_); },
      "return_multiple_with_payload_async");
  }
};

// --- Test Cases ---

TEST_F(AsyncRpcDispatcherTest, RegisterAndInvokeSimpleFunction) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{1};
  request.session_id = session_id_t{100};
  request.request_id = request_id_t{1000};
  request.AddParam(ParamMeta{ParamType::PRIMITIVE_INT32, PrimitiveValue(10)});
  request.AddParam(ParamMeta{ParamType::PRIMITIVE_INT32, PrimitiveValue(5)});

  auto sender = dispatcher_.InvokeAsyncFunction(
    std::move(request), RpcContextPtr(nullptr, [](void*) {}), builder_);

  auto result = unifex::sync_wait(std::move(sender));
  ASSERT_TRUE(result.has_value());
  auto [header, payload] = std::move(result.value());

  ASSERT_FALSE(static_cast<bool>(header.status.value));
  ASSERT_EQ(header.results.size(), 1);
  EXPECT_EQ(header.results[0].type, ParamType::PRIMITIVE_INT32);
  EXPECT_EQ(header.GetPrimitive<int32_t>(0), 15);
}

TEST_F(AsyncRpcDispatcherTest, RegisterAndInvokeWithContext) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{2};
  request.session_id = session_id_t{100};
  request.request_id = request_id_t{1001};

  std::vector<size_t> sizes = {128, 256};
  ucxx::UcxBufferVec buffer_vec(mr_, ucx_memory_type::HOST, sizes);
  RpcContextPtr ctx_ptr(&buffer_vec, [](void*) { /* non-owning */ });

  auto sender = dispatcher_.InvokeAsyncFunction(
    std::move(request), std::move(ctx_ptr), builder_);
  auto result = unifex::sync_wait(std::move(sender));
  ASSERT_TRUE(result.has_value());
  auto [header, payload] = std::move(result.value());

  ASSERT_FALSE(static_cast<bool>(header.status.value));
  ASSERT_TRUE(std::holds_alternative<ucxx::UcxBufferVec>(payload));
  auto& returned_vec = std::get<ucxx::UcxBufferVec>(payload);
  EXPECT_EQ(returned_vec.size(), 2);
  EXPECT_EQ(returned_vec[0].size, 128);
  EXPECT_EQ(returned_vec[1].size, 256);
}

TEST_F(AsyncRpcDispatcherTest, InvokeVoidReturnFunction) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{4};
  request.session_id = session_id_t{100};
  request.request_id = request_id_t{1002};

  auto sender = dispatcher_.InvokeAsyncFunction(
    std::move(request), RpcContextPtr(nullptr, [](void*) {}), builder_);
  auto result = unifex::sync_wait(std::move(sender));
  ASSERT_TRUE(result.has_value());
  auto [header, payload] = std::move(result.value());

  ASSERT_FALSE(static_cast<bool>(header.status.value));
  EXPECT_EQ(header.results.size(), 0);
}

TEST_F(AsyncRpcDispatcherTest, FunctionNotFound) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{99};
  request.session_id = session_id_t{100};
  request.request_id = request_id_t{1003};

  auto sender = dispatcher_.InvokeAsyncFunction(
    std::move(request), RpcContextPtr(nullptr, [](void*) {}), builder_);

  auto result = unifex::sync_wait(
    std::move(sender) | unifex::let_error([&](std::exception_ptr e) {
      RpcResponseHeader err_header;
      err_header.session_id = request.session_id;
      err_header.request_id = request.request_id;
      err_header.status = RpcStatus(make_error_code(rpc::RpcErrc::NOT_FOUND));
      return unifex::just(
        std::make_pair(std::move(err_header), ReturnedPayload{}));
    }));

  ASSERT_TRUE(result.has_value());
  auto [header, payload] = std::move(result.value());
  EXPECT_EQ(
    header.status.value, make_error_code(rpc::RpcErrc::NOT_FOUND).value());
}

TEST_F(AsyncRpcDispatcherTest, FunctionReturnsError) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{5};
  request.session_id = session_id_t{100};
  request.request_id = request_id_t{1004};

  auto sender = dispatcher_.InvokeAsyncFunction(
    std::move(request), RpcContextPtr(nullptr, [](void*) {}), builder_);

  data::string error_message{"Unknown error"};

  auto result = std::move(sender)
                | unifex::let_error([&](std::exception_ptr e) {
                    try {
                      std::rethrow_exception(e);
                    } catch (const std::exception& e) {
                      error_message = data::string(e.what());
                    }
                    return unifex::just_done();
                  })
                | unifex::sync_wait();
  EXPECT_EQ(error_message, "Test error");
}

TEST_F(AsyncRpcDispatcherTest, DispatchSimpleFunction) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{1};
  request.session_id = session_id_t{100};
  request.request_id = request_id_t{1000};
  request.AddParam(ParamMeta{ParamType::PRIMITIVE_INT32, PrimitiveValue(10)});
  request.AddParam(ParamMeta{ParamType::PRIMITIVE_INT32, PrimitiveValue(5)});

  auto sender = dispatcher_.Dispatch<std::monostate>(std::move(request));
  auto result = unifex::sync_wait(std::move(sender)).value();

  EXPECT_EQ(result.header.GetPrimitive<int32_t>(0), 15);
  ASSERT_FALSE(static_cast<bool>(result.header.status.value));
}

TEST_F(AsyncRpcDispatcherTest, DispatchWithContext) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{2};
  request.session_id = session_id_t{100};
  request.request_id = request_id_t{1001};

  std::vector<size_t> sizes = {128, 256};
  ucxx::UcxBufferVec buffer_vec(mr_, ucx_memory_type::HOST, sizes);

  auto sender =
    dispatcher_.Dispatch<ucxx::UcxBufferVec>(std::move(request), buffer_vec);
  auto result = unifex::sync_wait(std::move(sender)).value();

  ASSERT_TRUE(typeid(result.payload) == typeid(ucxx::UcxBufferVec));
  auto& returned_vec = result.payload;
  EXPECT_EQ(returned_vec.size(), 2);
  EXPECT_EQ(returned_vec[0].size, 128);
  EXPECT_EQ(returned_vec[1].size, 256);
}

TEST_F(AsyncRpcDispatcherTest, DispatchStringFunction) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{6};
  request.session_id = session_id_t{102};
  request.request_id = request_id_t{1006};

  ParamMeta p1{};
  p1.type = ParamType::STRING;
  p1.value.emplace<data::string>("hello ");
  request.AddParam(std::move(p1));

  ParamMeta p2{};
  p2.type = ParamType::STRING;
  p2.value.emplace<data::string>("world");
  request.AddParam(std::move(p2));

  auto sender = dispatcher_.Dispatch<std::monostate>(std::move(request));
  auto result = unifex::sync_wait(std::move(sender)).value();

  ASSERT_FALSE(static_cast<bool>(result.header.status.value));
  EXPECT_EQ(result.header.GetString(0), data::string("hello world"));
}

TEST_F(AsyncRpcDispatcherTest, DispatchVectorFunction) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{7};
  request.session_id = session_id_t{103};
  request.request_id = request_id_t{1007};

  data::vector<int32_t> v;
  v.push_back(10);
  v.push_back(20);
  v.push_back(30);

  ParamMeta p{};
  p.type = ParamType::VECTOR_INT32;
  p.value.emplace<VectorValue>(std::move(v));
  request.AddParam(std::move(p));

  auto sender = dispatcher_.Dispatch<std::monostate>(std::move(request));
  auto result = unifex::sync_wait(std::move(sender)).value();

  ASSERT_FALSE(static_cast<bool>(result.header.status.value));
  EXPECT_EQ(result.header.GetPrimitive<int32_t>(0), 60);
}

TEST_F(AsyncRpcDispatcherTest, DispatchWithReturnContext) {
  // Test returning UcxBuffer
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{8};
    request.session_id = session_id_t{200};
    request.request_id = request_id_t{2000};

    auto sender = dispatcher_.Dispatch<ucxx::UcxBuffer>(std::move(request));
    auto result = unifex::sync_wait(std::move(sender)).value();

    ASSERT_FALSE(static_cast<bool>(result.header.status.value));
    ASSERT_TRUE(typeid(result.payload) == typeid(ucxx::UcxBuffer));
    auto& buffer = result.payload;
    EXPECT_EQ(buffer.size(), 1024);
  }

  // Test returning UcxBufferVec
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{9};
    request.session_id = session_id_t{201};
    request.request_id = request_id_t{2001};

    auto sender = dispatcher_.Dispatch<ucxx::UcxBufferVec>(std::move(request));
    auto result = unifex::sync_wait(std::move(sender)).value();

    ASSERT_FALSE(static_cast<bool>(result.header.status.value));
    ASSERT_TRUE(typeid(result.payload) == typeid(ucxx::UcxBufferVec));
    auto& buffer_vec = result.payload;
    EXPECT_EQ(buffer_vec.size(), 2);
    EXPECT_EQ(buffer_vec[0].size, 128);
    EXPECT_EQ(buffer_vec[1].size, 256);
  }
}

TEST_F(AsyncRpcDispatcherTest, DispatchWithUcxBufferVecContext) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{10};
  request.session_id = session_id_t{108};
  request.request_id = request_id_t{1010};

  std::vector<size_t> sizes = {1024, 2048, 4096};
  ucxx::UcxBufferVec input_vec(mr_, ucx_memory_type::HOST, sizes);

  auto sender = dispatcher_.DispatchMove<std::monostate, ucxx::UcxBufferVec>(
    std::move(request), std::move(input_vec));
  auto result = unifex::sync_wait(std::move(sender)).value();
  EXPECT_EQ(result.header.GetPrimitive<uint64_t>(0), 7168);  // 1024+2048+4096
}

TEST_F(AsyncRpcDispatcherTest, DispatchWithMixedInputsAndOutputs) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{11};
  request.session_id = session_id_t{400};

  ParamMeta p1;
  p1.type = ParamType::PRIMITIVE_INT32;
  p1.value.emplace<PrimitiveValue>(10);
  request.AddParam(std::move(p1));

  ParamMeta p2;
  p2.type = ParamType::STRING;
  p2.value.emplace<data::string>("processed");
  request.AddParam(std::move(p2));

  // Non-serializable context input
  ucxx::UcxBufferVec input_vec(
    mr_, ucx_memory_type::HOST, {10, 20});  // Total size 30

  auto sender = dispatcher_.Dispatch<ucxx::UcxBufferVec>(
    std::move(request), std::move(input_vec));
  auto result = unifex::sync_wait(std::move(sender)).value();

  // Verify serializable outputs from the header
  EXPECT_EQ(result.header.session_id.v_, 400);
  ASSERT_EQ(result.header.results.size(), 2);
  EXPECT_EQ(result.header.GetPrimitive<uint64_t>(0), 300);  // 30 * 10
  EXPECT_EQ(result.header.GetString(1), data::string("processed"));

  // Verify non-serializable context output
  ASSERT_TRUE(std::holds_alternative<ucxx::UcxBufferVec>(result.payload));
  auto& output_vec = std::get<ucxx::UcxBufferVec>(result.payload);
  EXPECT_EQ(output_vec.size(), 1);
  EXPECT_EQ(output_vec[0].size, 300);
}

TEST_F(AsyncRpcDispatcherTest, DispatchTensorMetadataParameter) {
  TensorMeta input_meta = make_test_tensor_meta();

  RpcRequestHeader request{};
  request.function_id = function_id_t{12};
  request.session_id = session_id_t{200};
  request.request_id = request_id_t{1012};

  ParamMeta tensor_param{};
  tensor_param.type = ParamType::TENSOR_META;
  tensor_param.value.emplace<TensorMeta>(input_meta);
  request.AddParam(std::move(tensor_param));

  auto sender = dispatcher_.Dispatch<std::monostate>(std::move(request));
  auto result = unifex::sync_wait(std::move(sender)).value();

  ASSERT_FALSE(static_cast<bool>(result.header.status.value));
  int64_t expected_numel = 1;
  for (auto dim : input_meta.shape) {
    expected_numel *= dim;
  }
  EXPECT_EQ(result.header.GetPrimitive<int64_t>(0), expected_numel);
}

TEST_F(AsyncRpcDispatcherTest, DispatchReturnRpcResponseHeader) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{13};
  request.session_id = session_id_t{107};
  // The function takes session_id as an argument for this test
  ParamMeta p{};
  p.type = ParamType::PRIMITIVE_UINT32;
  p.value.emplace<PrimitiveValue>(request.session_id.v_);
  request.AddParam(std::move(p));

  auto sender = dispatcher_.Dispatch<std::monostate>(std::move(request));
  auto result = unifex::sync_wait(std::move(sender)).value();

  EXPECT_EQ(result.header.session_id.v_, 107);
  ASSERT_EQ(result.header.results.size(), 1);
  EXPECT_EQ(result.header.GetString(0), data::string("custom_header"));
}

TEST_F(AsyncRpcDispatcherTest, DispatchDynamicPayload) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{1};
  request.session_id = session_id_t{100};
  request.request_id = request_id_t{1000};
  request.AddParam(ParamMeta{ParamType::PRIMITIVE_INT32, PrimitiveValue(10)});
  request.AddParam(ParamMeta{ParamType::PRIMITIVE_INT32, PrimitiveValue(5)});

  auto sender = dispatcher_.Dispatch(std::move(request));
  auto result = unifex::sync_wait(std::move(sender)).value();

  ASSERT_FALSE(static_cast<bool>(result.header.status.value));
  // The result (int) should be in the header, not the payload variant
  ASSERT_EQ(result.header.results.size(), 1);
  EXPECT_EQ(result.header.GetPrimitive<int32_t>(0), 15);
  // The payload variant should exist and hold monostate
  EXPECT_TRUE(std::holds_alternative<std::monostate>(result.payload));
}

TEST_F(AsyncRpcDispatcherTest, DispatchDynamicWithPayload) {
  // Test dynamically dispatching a function that returns a context
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{8};

    auto sender = dispatcher_.Dispatch(std::move(request));
    auto result = unifex::sync_wait(std::move(sender)).value();

    ASSERT_TRUE(std::holds_alternative<ucxx::UcxBuffer>(result.payload));
    // Use std::visit to inspect the variant at runtime
    std::visit(
      [](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, ucxx::UcxBuffer>) {
          EXPECT_EQ(arg.size(), 1024);
        } else {
          FAIL() << "Expected UcxBuffer, but got another type.";
        }
      },
      result.payload);
  }

  // Test dynamically dispatching a function that returns monostate
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{4};

    auto sender = dispatcher_.Dispatch(std::move(request));
    auto result = unifex::sync_wait(std::move(sender)).value();

    EXPECT_TRUE(std::holds_alternative<std::monostate>(result.payload));
  }
}

TEST_F(AsyncRpcDispatcherTest, DispatchWithTupleReturn) {
  // Test case 1: Multiple serializable return values, no payload
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{14};

    auto sender = dispatcher_.Dispatch<std::monostate>(std::move(request));
    auto result = unifex::sync_wait(std::move(sender)).value();

    ASSERT_FALSE(static_cast<bool>(result.header.status.value));
    ASSERT_EQ(result.header.results.size(), 3);
    EXPECT_EQ(result.header.GetPrimitive<int>(0), 123);
    EXPECT_EQ(result.header.GetString(1), "hello");
    EXPECT_EQ(result.header.GetPrimitive<float>(2), 3.14f);
    EXPECT_EQ(result.payload, std::monostate{});
  }

  // Test case 2: Multiple return values with a payload
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{15};

    auto sender = dispatcher_.Dispatch<ucxx::UcxBufferVec>(std::move(request));
    auto result = unifex::sync_wait(std::move(sender)).value();

    ASSERT_FALSE(static_cast<bool>(result.header.status.value));
    ASSERT_EQ(result.header.results.size(), 1);
    EXPECT_EQ(result.header.GetPrimitive<int>(0), 456);
    ASSERT_TRUE(typeid(result.payload) == typeid(ucxx::UcxBufferVec));
    auto& payload = result.payload;
    EXPECT_EQ(payload.size(), 2);
    EXPECT_EQ(payload[0].size, 16);
    EXPECT_EQ(payload[1].size, 32);
  }
}

}  // namespace rpc
}  // namespace eux
