/* Copyright 2025 The MeepoEmbedding Authors. All Rights Reserved.

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

#include "gtest/gtest.h"

#include <map>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "rpc_core/rpc_dispatcher.hpp"
#include "ucx_context/ucx_context_data.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

namespace stdexe_ucx_runtime {
namespace rpc_core {

// Test fixture for RpcDispatcher
class RpcDispatcherTest : public ::testing::Test {
 protected:
  RpcDispatcherTest() : dispatcher("test_instance") {}
  RpcDispatcher dispatcher;
};

// =============================================================================
// Test Functions and Types
// =============================================================================

int add(int a, int b) { return a + b; }

void no_op() {}

data::string concat(const data::string& a, const data::string& b) {
  return data::string(
    std::string(a.data(), a.size()) + std::string(b.data(), b.size()));
}

int sum_vec(const data::vector<int32_t>& vec) {
  int32_t sum = 0;
  for (int32_t v : vec) {
    sum += v;
  }
  return sum;
}

// A test function that takes a UcxBufferVec by value (move) and processes it.
uint64_t process_buffer_vec(UcxBufferVec in_vec) {
  uint64_t total_size = 0;
  for (const auto& buf : in_vec) {
    total_size += buf.size;
  }
  return total_size;
}

struct MyContext {
  int value = 0;
};

struct MyOutputContext {
  data::string message;
};

int get_context_value(MyContext& ctx) { return ctx.value; }

class MyService {
 public:
  int multiply(int a, int b) { return a * b; }
};

// Returns a custom RpcResponseHeader
RpcResponseHeader custom_response_header_fn(session_id_t session_id) {
  RpcResponseHeader response{};
  response.session_id = session_id;
  ParamMeta result_meta{};
  result_meta.type = ParamType::STRING;
  result_meta.value.emplace<data::string>("custom_header");
  response.results.emplace_back(std::move(result_meta));
  return response;
}

// =============================================================================
// Test Cases
// =============================================================================

TEST_F(RpcDispatcherTest, Registration) {
  dispatcher.register_function(function_id_t{1}, &add);
  EXPECT_TRUE(dispatcher.is_registered(function_id_t{1}));
  EXPECT_FALSE(dispatcher.is_registered(function_id_t{2}));
  EXPECT_EQ(dispatcher.function_count(), 1);

  MyService service;
  dispatcher.register_function(function_id_t{2}, [&service](int a, int b) {
    return service.multiply(a, b);
  });
  EXPECT_TRUE(dispatcher.is_registered(function_id_t{2}));
  EXPECT_EQ(dispatcher.function_count(), 2);
}

TEST_F(RpcDispatcherTest, DispatchSimpleFunction) {
  dispatcher.register_function(function_id_t{1}, &add);

  RpcRequestHeader request{};
  request.function_id = function_id_t{1};
  request.session_id = session_id_t{100};

  ParamMeta p1{};
  p1.type = ParamType::PRIMITIVE_INT32;
  p1.value.emplace<PrimitiveValue>(5);
  request.add_param(std::move(p1));

  ParamMeta p2{};
  p2.type = ParamType::PRIMITIVE_INT32;
  p2.value.emplace<PrimitiveValue>(10);
  request.add_param(std::move(p2));

  auto request_buffer = cista::serialize(request);
  auto response_pair = dispatcher.dispatch(std::move(request_buffer));
  auto& response_header = response_pair.header;

  EXPECT_EQ(response_header.session_id.v_, 100);
  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.results[0].type, ParamType::PRIMITIVE_INT32);
  EXPECT_EQ(response_header.get_primitive<int32_t>(0), 15);
}

TEST_F(RpcDispatcherTest, DispatchVoidFunction) {
  dispatcher.register_function(function_id_t{2}, &no_op);

  RpcRequestHeader request{};
  request.function_id = function_id_t{2};
  request.session_id = session_id_t{101};

  auto request_buffer = cista::serialize(request);
  auto response_pair = dispatcher.dispatch(std::move(request_buffer));
  auto& response_header = response_pair.header;

  EXPECT_EQ(response_header.session_id.v_, 101);
  EXPECT_EQ(response_header.results.size(), 0);
  EXPECT_EQ(response_header.status.value, 0);
}

TEST_F(RpcDispatcherTest, DispatchStringFunction) {
  dispatcher.register_function(function_id_t{3}, &concat);

  RpcRequestHeader request{};
  request.function_id = function_id_t{3};
  request.session_id = session_id_t{102};

  ParamMeta p1{};
  p1.type = ParamType::STRING;
  p1.value.emplace<data::string>("hello ");
  request.add_param(std::move(p1));

  ParamMeta p2{};
  p2.type = ParamType::STRING;
  p2.value.emplace<data::string>("world");
  request.add_param(std::move(p2));

  auto request_buffer = cista::serialize(request);
  auto response_pair = dispatcher.dispatch(std::move(request_buffer));
  auto& response_header = response_pair.header;

  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.results[0].type, ParamType::STRING);
  EXPECT_EQ(response_header.get_string(0), data::string("hello world"));
}

TEST_F(RpcDispatcherTest, DispatchVectorFunction) {
  dispatcher.register_function(function_id_t{4}, &sum_vec);

  RpcRequestHeader request{};
  request.function_id = function_id_t{4};
  request.session_id = session_id_t{103};

  data::vector<int32_t> v;
  v.push_back(10);
  v.push_back(20);
  v.push_back(30);

  ParamMeta p{};
  p.type = ParamType::VECTOR_INT32;
  p.value.emplace<VectorValue>(std::move(v));
  request.add_param(std::move(p));

  auto request_buffer = cista::serialize(request);
  auto response_pair = dispatcher.dispatch(std::move(request_buffer));
  auto& response_header = response_pair.header;

  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.results[0].type, ParamType::PRIMITIVE_INT32);
  EXPECT_EQ(response_header.get_primitive<int32_t>(0), 60);
}

TEST_F(RpcDispatcherTest, DispatchMemberFunction) {
  MyService service;
  dispatcher.register_function(function_id_t{5}, [&service](int a, int b) {
    return service.multiply(a, b);
  });

  RpcRequestHeader request{};
  request.function_id = function_id_t{5};
  request.session_id = session_id_t{104};

  ParamMeta p1{};
  p1.type = ParamType::PRIMITIVE_INT32;
  p1.value.emplace<PrimitiveValue>(6);
  request.add_param(std::move(p1));

  ParamMeta p2{};
  p2.type = ParamType::PRIMITIVE_INT32;
  p2.value.emplace<PrimitiveValue>(7);
  request.add_param(std::move(p2));

  auto request_buffer = cista::serialize(request);
  auto response_pair = dispatcher.dispatch(std::move(request_buffer));
  auto& response_header = response_pair.header;

  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.get_primitive<int32_t>(0), 42);
}

// A test function that takes one context by value (move) and returns another.
MyOutputContext process_and_return_context(MyContext in_ctx) {
  return MyOutputContext{
    data::string("processed " + std::to_string(in_ctx.value))};
}

TEST_F(RpcDispatcherTest, DispatchWithMoveAndReturnContext) {
  dispatcher.register_function(function_id_t{7}, &process_and_return_context);

  MyContext ctx{};
  ctx.value = 123;

  RpcRequestHeader request{};
  request.function_id = function_id_t{7};
  request.session_id = session_id_t{106};

  auto request_buffer = cista::serialize(request);

  // Expect MyOutputContext to be returned.
  auto result = dispatcher.dispatch_move<MyOutputContext>(
    std::move(request_buffer), std::move(ctx));

  EXPECT_EQ(result.header.session_id.v_, 106);
  EXPECT_EQ(result.context.message, data::string("processed 123"));
}

TEST_F(RpcDispatcherTest, DispatchWithContext) {
  dispatcher.register_function(function_id_t{6}, &get_context_value);

  MyContext ctx{};
  ctx.value = 99;

  RpcRequestHeader request{};
  request.function_id = function_id_t{6};
  request.session_id = session_id_t{105};

  auto request_buffer = cista::serialize(request);
  auto response_pair = dispatcher.dispatch(std::move(request_buffer), ctx);

  auto& response_header = response_pair.header;
  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.get_primitive<int32_t>(0), 99);
}

TEST_F(RpcDispatcherTest, DispatchReturnRpcResponseHeader) {
  dispatcher.register_function(function_id_t{8}, &custom_response_header_fn);

  RpcRequestHeader request{};
  request.function_id = function_id_t{8};
  request.session_id = session_id_t{107};
  // The function takes session_id as an argument for this test
  ParamMeta p{};
  p.type = ParamType::PRIMITIVE_UINT64;
  p.value.emplace<PrimitiveValue>(request.session_id.v_);
  request.add_param(std::move(p));

  auto request_buffer = cista::serialize(request);
  auto response_pair = dispatcher.dispatch(std::move(request_buffer));
  auto& response_header = response_pair.header;

  EXPECT_EQ(response_header.session_id.v_, 107);
  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.get_string(0), data::string("custom_header"));
}

TEST_F(RpcDispatcherTest, FunctionNotFound) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{999};  // Not registered
  request.session_id = session_id_t{109};

  auto request_buffer = cista::serialize(request);
  auto response_pair = dispatcher.dispatch(std::move(request_buffer));
  auto& response_header = response_pair.header;

  EXPECT_EQ(response_header.session_id.v_, 109);
  EXPECT_EQ(
    response_header.status, std::make_error_code(std::errc::invalid_argument));
}

TEST_F(RpcDispatcherTest, TypeMismatch) {
  dispatcher.register_function(function_id_t{1}, &add);

  RpcRequestHeader request{};
  request.function_id = function_id_t{1};
  request.session_id = session_id_t{110};

  ParamMeta p1{};
  p1.type = ParamType::PRIMITIVE_INT32;
  p1.value.emplace<PrimitiveValue>(5);
  request.add_param(std::move(p1));

  ParamMeta p2{};
  p2.type = ParamType::STRING;  // Mismatch
  p2.value.emplace<data::string>("not a number");
  request.add_param(std::move(p2));

  auto request_buffer = cista::serialize(request);
  auto response_pair = dispatcher.dispatch(std::move(request_buffer));
  auto& response_header = response_pair.header;

  EXPECT_EQ(response_header.session_id.v_, 110);
  EXPECT_EQ(
    response_header.status, std::make_error_code(std::errc::invalid_argument));
}

TEST_F(RpcDispatcherTest, GetFunctionSignature) {
  dispatcher.register_function(
    function_id_t{1}, &add, data::string("add_func"));
  dispatcher.register_function(
    function_id_t{2}, &no_op, data::string("no_op_func"));
  dispatcher.register_function(
    function_id_t{6}, &get_context_value, data::string("get_context"));

  // Test signature for 'add'
  auto add_sig_opt = dispatcher.get_signature(function_id_t{1});
  ASSERT_TRUE(add_sig_opt.has_value());
  auto& add_sig = add_sig_opt.value();
  EXPECT_EQ(add_sig.id.v_, 1);
  EXPECT_EQ(add_sig.function_name, data::string("add_func"));
  EXPECT_EQ(add_sig.instance_name, data::string("test_instance"));
  EXPECT_FALSE(add_sig.takes_context);
  ASSERT_EQ(add_sig.param_types.size(), 2);
  EXPECT_EQ(add_sig.param_types[0], ParamType::PRIMITIVE_INT32);
  EXPECT_EQ(add_sig.param_types[1], ParamType::PRIMITIVE_INT32);
  EXPECT_EQ(add_sig.return_type, ParamType::PRIMITIVE_INT32);

  // Test signature for 'no_op'
  auto no_op_sig_opt = dispatcher.get_signature(function_id_t{2});
  ASSERT_TRUE(no_op_sig_opt.has_value());
  auto& no_op_sig = no_op_sig_opt.value();
  EXPECT_EQ(no_op_sig.id.v_, 2);
  EXPECT_FALSE(no_op_sig.takes_context);
  EXPECT_EQ(no_op_sig.param_types.size(), 0);
  EXPECT_EQ(no_op_sig.return_type, ParamType::VOID);

  // Test signature for 'get_context_value'
  auto ctx_sig_opt = dispatcher.get_signature(function_id_t{6});
  ASSERT_TRUE(ctx_sig_opt.has_value());
  auto& ctx_sig = ctx_sig_opt.value();
  EXPECT_EQ(ctx_sig.id.v_, 6);
  EXPECT_EQ(ctx_sig.instance_name, data::string("test_instance"));
  EXPECT_TRUE(ctx_sig.takes_context);
  EXPECT_EQ(ctx_sig.param_types.size(), 0);  // Context is not a parameter
  EXPECT_EQ(ctx_sig.return_type, ParamType::PRIMITIVE_INT32);

  // Test getting all signatures
  auto serialized_sigs = dispatcher.get_all_signatures();
  auto deserialized_sigs =
    cista::deserialize<data::vector<RpcFunctionSignature>>(serialized_sigs);

  ASSERT_NE(deserialized_sigs, nullptr);
  EXPECT_EQ(deserialized_sigs->size(), 3);

  // Verify one of the deserialized signatures
  bool found_add_sig = false;
  for (const auto& sig : *deserialized_sigs) {
    if (sig.id.v_ == 1) {
      found_add_sig = true;
      EXPECT_EQ(sig.function_name, data::string("add_func"));
      EXPECT_EQ(sig.instance_name, data::string("test_instance"));
      ASSERT_EQ(sig.param_types.size(), 2);
      EXPECT_EQ(sig.param_types[0], ParamType::PRIMITIVE_INT32);
      EXPECT_EQ(sig.return_type, ParamType::PRIMITIVE_INT32);
    }
  }
  EXPECT_TRUE(found_add_sig);
}

TEST_F(RpcDispatcherTest, DispatchWithUcxBufferVecContext) {
  // 1. Register the function that processes UcxBufferVec.
  dispatcher.register_function(function_id_t{10}, &process_buffer_vec);

  // 2. Prepare the context object (UcxBufferVec).
  // This requires a memory manager.
  auto mr = DefaultUcxMemoryResourceManager();
  std::vector<size_t> sizes = {1024, 2048, 4096};
  UcxBufferVec buffer_vec(mr, ucx_memory_type::HOST, sizes);

  // 3. Prepare and serialize the RPC request header.
  RpcRequestHeader request{};
  request.function_id = function_id_t{10};
  request.session_id = session_id_t{108};
  auto request_buffer = cista::serialize(request);

  // 4. Dispatch the call, moving the context object.
  auto response_pair =
    dispatcher.dispatch_move(std::move(request_buffer), std::move(buffer_vec));
  auto& response_header = response_pair.header;

  // 5. Verify the result.
  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(
    response_header.get_primitive<uint64_t>(0), 7168);  // 1024+2048+4096
  EXPECT_EQ(response_header.status.value, 0);
}

TEST_F(RpcDispatcherTest, EndToEndRpcWithRegistry) {
  // 1. A mock central registry for service discovery.
  // The key is the instance name, the value is the serialized signatures.
  std::map<std::string, cista::byte_buf> registry;

  // 2. Setup Client B (the service provider).
  // It creates a dispatcher, registers a function, and publishes its
  // signatures.
  {
    RpcDispatcher dispatcher_B("client_B_instance");
    dispatcher_B.register_function(
      function_id_t{100}, &add, data::string("add_func"));

    // Publish signatures to the central registry.
    registry["client_B_instance"] = dispatcher_B.get_all_signatures();

    // 3. Client A (the caller) discovers and calls the function.
    // --- Discovery Phase ---
    auto serialized_sigs_it = registry.find("client_B_instance");
    ASSERT_NE(serialized_sigs_it, registry.end());

    auto deserialized_sigs =
      cista::deserialize<data::vector<RpcFunctionSignature>>(
        serialized_sigs_it->second);
    ASSERT_NE(deserialized_sigs, nullptr);

    std::optional<function_id_t> target_function_id;
    for (const auto& sig : *deserialized_sigs) {
      if (sig.function_name == data::string("add_func")) {
        target_function_id = sig.id;
        break;
      }
    }
    ASSERT_TRUE(target_function_id.has_value());

    // --- RPC Call Phase ---
    RpcRequestHeader request{};
    request.function_id = target_function_id.value();
    request.session_id = session_id_t{2025};

    ParamMeta p1{};
    p1.type = ParamType::PRIMITIVE_INT32;
    p1.value.emplace<PrimitiveValue>(25);
    request.add_param(std::move(p1));

    ParamMeta p2{};
    p2.type = ParamType::PRIMITIVE_INT32;
    p2.value.emplace<PrimitiveValue>(17);
    request.add_param(std::move(p2));

    auto request_buffer = cista::serialize(request);

    // Simulate sending the request to Client B's dispatcher.
    auto response_pair = dispatcher_B.dispatch(std::move(request_buffer));
    auto& response_header = response_pair.header;

    // --- Verification Phase ---
    EXPECT_EQ(response_header.session_id.v_, 2025);
    ASSERT_EQ(response_header.results.size(), 1);
    EXPECT_EQ(response_header.get_primitive<int32_t>(0), 42);  // 25 + 17
    EXPECT_EQ(response_header.status.value, 0);
  }
}

}  // namespace rpc_core
}  // namespace stdexe_ucx_runtime
