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
#include "rpc_core/rpc_status.hpp"
#include "ucx_context/ucx_context_data.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

namespace eux {
namespace rpc {

// Test fixture for RpcDispatcher
class RpcDispatcherTest : public ::testing::Test {
 protected:
  RpcDispatcherTest() : dispatcher("test_instance") {}
  RpcDispatcher dispatcher;
};

// =============================================================================
// Test Functions and Types
// =============================================================================

ucxx::UcxBuffer return_ucx_buffer(ucxx::UcxMemoryResourceManager& mr) {
  return ucxx::UcxBuffer(mr, ucx_memory_type::HOST, 1024);
}

ucxx::UcxBufferVec return_ucx_buffer_vec(ucxx::UcxMemoryResourceManager& mr) {
  return ucxx::UcxBufferVec(mr, ucx_memory_type::HOST, {128, 256});
}

// A test function that takes mixed inputs and returns mixed outputs.
std::pair<RpcResponseHeader, ucxx::UcxBufferVec> process_mixed_request(
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

  return {std::move(resp), std::move(output_vec)};
}

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
uint64_t process_buffer_vec(ucxx::UcxBufferVec in_vec) {
  uint64_t total_size = 0;
  for (const auto& buf : in_vec) {
    total_size += buf.size;
  }
  return total_size;
}

// A test function that takes a UcxBufferVec by const ref.
uint64_t process_buffer_vec_ref(const eux::ucxx::UcxBufferVec& in_vec) {
  uint64_t total_size = 0;
  for (const auto& buf : in_vec) {
    total_size += buf.size;
  }
  return total_size;
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

TensorMeta mutate_tensor_meta(TensorMeta meta) {
  for (auto& dim : meta.shape) {
    dim += 1;
  }
  meta.byte_offset += 8;
  return meta;
}

int64_t tensor_meta_numel(const TensorMeta& meta) {
  int64_t total = 1;
  for (auto dim : meta.shape) {
    total *= dim;
  }
  return total;
}

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
  dispatcher.RegisterFunction(function_id_t{1}, &add);
  EXPECT_TRUE(dispatcher.IsRegistered(function_id_t{1}));
  EXPECT_FALSE(dispatcher.IsRegistered(function_id_t{2}));
  EXPECT_EQ(dispatcher.FunctionCount(), 1);

  MyService service;
  dispatcher.RegisterFunction(function_id_t{2}, [&service](int a, int b) {
    return service.multiply(a, b);
  });
  EXPECT_TRUE(dispatcher.IsRegistered(function_id_t{2}));
  EXPECT_EQ(dispatcher.FunctionCount(), 2);
}

TEST_F(RpcDispatcherTest, DispatchSimpleFunction) {
  dispatcher.RegisterFunction(function_id_t{1}, &add);

  RpcRequestHeader request{};
  request.function_id = function_id_t{1};
  request.session_id = session_id_t{100};

  ParamMeta p1{};
  p1.type = ParamType::PRIMITIVE_INT32;
  p1.value.emplace<PrimitiveValue>(5);
  request.AddParam(std::move(p1));

  ParamMeta p2{};
  p2.type = ParamType::PRIMITIVE_INT32;
  p2.value.emplace<PrimitiveValue>(10);
  request.AddParam(std::move(p2));

  auto request_buffer = cista::serialize(request);
  auto response_pair =
    dispatcher.Dispatch<std::monostate>(std::move(request_buffer));
  auto& response_header = response_pair.header;

  EXPECT_EQ(response_header.session_id.v_, 100);
  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.results[0].type, ParamType::PRIMITIVE_INT32);
  EXPECT_EQ(response_header.GetPrimitive<int32_t>(0), 15);
}

TEST_F(RpcDispatcherTest, DispatchSimpleFunctionWithHeader) {
  dispatcher.RegisterFunction(function_id_t{1}, &add);

  RpcRequestHeader request{};
  request.function_id = function_id_t{1};
  request.session_id = session_id_t{100};

  ParamMeta p1{};
  p1.type = ParamType::PRIMITIVE_INT32;
  p1.value.emplace<PrimitiveValue>(5);
  request.AddParam(std::move(p1));

  ParamMeta p2{};
  p2.type = ParamType::PRIMITIVE_INT32;
  p2.value.emplace<PrimitiveValue>(10);
  request.AddParam(std::move(p2));

  auto response_pair = dispatcher.Dispatch<std::monostate>(request);
  auto& response_header = response_pair.header;

  EXPECT_EQ(response_header.session_id.v_, 100);
  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.results[0].type, ParamType::PRIMITIVE_INT32);
  EXPECT_EQ(response_header.GetPrimitive<int32_t>(0), 15);
}

TEST_F(RpcDispatcherTest, DispatchVoidFunction) {
  dispatcher.RegisterFunction(function_id_t{2}, &no_op);

  RpcRequestHeader request{};
  request.function_id = function_id_t{2};
  request.session_id = session_id_t{101};

  auto request_buffer = cista::serialize(request);
  auto response_pair =
    dispatcher.Dispatch<std::monostate>(std::move(request_buffer));
  auto& response_header = response_pair.header;

  EXPECT_EQ(response_header.session_id.v_, 101);
  EXPECT_EQ(response_header.results.size(), 0);
  EXPECT_EQ(response_header.status.value, 0);
}

TEST_F(RpcDispatcherTest, DispatchStringFunction) {
  dispatcher.RegisterFunction(function_id_t{3}, &concat);

  RpcRequestHeader request{};
  request.function_id = function_id_t{3};
  request.session_id = session_id_t{102};

  ParamMeta p1{};
  p1.type = ParamType::STRING;
  p1.value.emplace<data::string>("hello ");
  request.AddParam(std::move(p1));

  ParamMeta p2{};
  p2.type = ParamType::STRING;
  p2.value.emplace<data::string>("world");
  request.AddParam(std::move(p2));

  auto request_buffer = cista::serialize(request);
  auto response_pair =
    dispatcher.Dispatch<std::monostate>(std::move(request_buffer));
  auto& response_header = response_pair.header;

  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.results[0].type, ParamType::STRING);
  EXPECT_EQ(response_header.GetString(0), data::string("hello world"));
}

TEST_F(RpcDispatcherTest, DispatchVectorFunction) {
  dispatcher.RegisterFunction(function_id_t{4}, &sum_vec);

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
  request.AddParam(std::move(p));

  auto request_buffer = cista::serialize(request);
  auto response_pair =
    dispatcher.Dispatch<std::monostate>(std::move(request_buffer));
  auto& response_header = response_pair.header;

  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.results[0].type, ParamType::PRIMITIVE_INT32);
  EXPECT_EQ(response_header.GetPrimitive<int32_t>(0), 60);
}

TEST_F(RpcDispatcherTest, DispatchMemberFunction) {
  MyService service;
  dispatcher.RegisterFunction(function_id_t{5}, [&service](int a, int b) {
    return service.multiply(a, b);
  });

  RpcRequestHeader request{};
  request.function_id = function_id_t{5};
  request.session_id = session_id_t{104};

  ParamMeta p1{};
  p1.type = ParamType::PRIMITIVE_INT32;
  p1.value.emplace<PrimitiveValue>(6);
  request.AddParam(std::move(p1));

  ParamMeta p2{};
  p2.type = ParamType::PRIMITIVE_INT32;
  p2.value.emplace<PrimitiveValue>(7);
  request.AddParam(std::move(p2));

  auto request_buffer = cista::serialize(request);
  auto response_pair =
    dispatcher.Dispatch<std::monostate>(std::move(request_buffer));
  auto& response_header = response_pair.header;

  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.GetPrimitive<int32_t>(0), 42);
}

TEST_F(RpcDispatcherTest, DispatchReturnRpcResponseHeader) {
  dispatcher.RegisterFunction(function_id_t{8}, &custom_response_header_fn);

  RpcRequestHeader request{};
  request.function_id = function_id_t{8};
  request.session_id = session_id_t{107};
  // The function takes session_id as an argument for this test
  ParamMeta p{};
  p.type = ParamType::PRIMITIVE_UINT32;
  p.value.emplace<PrimitiveValue>(request.session_id.v_);
  request.AddParam(std::move(p));

  auto request_buffer = cista::serialize(request);
  auto response_pair =
    dispatcher.Dispatch<std::monostate>(std::move(request_buffer));
  auto& response_header = response_pair.header;

  EXPECT_EQ(response_header.session_id.v_, 107);
  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.GetString(0), data::string("custom_header"));
}

TEST_F(RpcDispatcherTest, FunctionNotFound) {
  RpcRequestHeader request{};
  request.function_id = function_id_t{999};  // Not registered
  request.session_id = session_id_t{109};

  auto request_buffer = cista::serialize(request);
  auto response_pair =
    dispatcher.Dispatch<std::monostate>(std::move(request_buffer));
  auto& response_header = response_pair.header;

  EXPECT_EQ(response_header.session_id.v_, 109);
  EXPECT_EQ(
    response_header.status, std::make_error_code(std::errc::invalid_argument));
}

TEST_F(RpcDispatcherTest, TypeMismatch) {
  dispatcher.RegisterFunction(function_id_t{1}, &add);

  RpcRequestHeader request{};
  request.function_id = function_id_t{1};
  request.session_id = session_id_t{110};

  ParamMeta p1{};
  p1.type = ParamType::PRIMITIVE_INT32;
  p1.value.emplace<PrimitiveValue>(5);
  request.AddParam(std::move(p1));

  ParamMeta p2{};
  p2.type = ParamType::STRING;  // Mismatch
  p2.value.emplace<data::string>("not a number");
  request.AddParam(std::move(p2));

  auto request_buffer = cista::serialize(request);
  auto response_pair =
    dispatcher.Dispatch<std::monostate>(std::move(request_buffer));
  auto& response_header = response_pair.header;

  EXPECT_EQ(response_header.session_id.v_, 110);
  EXPECT_EQ(
    response_header.status, std::make_error_code(std::errc::invalid_argument));
}

int64_t tensor_meta_volume(const TensorMeta& meta) {
  return tensor_meta_numel(meta);
}

TEST_F(RpcDispatcherTest, GetFunctionSignature) {
  dispatcher.RegisterFunction(function_id_t{1}, &add, data::string("add_func"));
  dispatcher.RegisterFunction(
    function_id_t{2}, &no_op, data::string("no_op_func"));
  dispatcher.RegisterFunction(
    function_id_t{11}, &tensor_meta_volume, data::string("tensor_meta_volume"));

  ucxx::DefaultUcxMemoryResourceManager mr;
  dispatcher.RegisterFunction(
    function_id_t{12},
    [&mr]() { return return_ucx_buffer(mr); },
    data::string("return_ucx_buffer"));
  dispatcher.RegisterFunction(
    function_id_t{13},
    [&mr]() { return return_ucx_buffer_vec(mr); },
    data::string("return_ucx_buffer_vec"));

  // Test signature for 'add'
  auto add_sig_opt = dispatcher.GetSignature(function_id_t{1});
  ASSERT_TRUE(add_sig_opt.has_value());
  auto& add_sig = add_sig_opt.value();
  EXPECT_EQ(add_sig.id.v_, 1);
  EXPECT_EQ(add_sig.function_name, data::string("add_func"));
  EXPECT_EQ(add_sig.instance_name, data::string("test_instance"));
  EXPECT_FALSE(add_sig.takes_context);
  ASSERT_EQ(add_sig.param_types.size(), 2);
  EXPECT_EQ(add_sig.param_types[0], ParamType::PRIMITIVE_INT32);
  EXPECT_EQ(add_sig.param_types[1], ParamType::PRIMITIVE_INT32);
  ASSERT_EQ(add_sig.return_types.size(), 1);
  EXPECT_EQ(add_sig.return_types[0], ParamType::PRIMITIVE_INT32);

  // Test signature for 'no_op'
  auto no_op_sig_opt = dispatcher.GetSignature(function_id_t{2});
  ASSERT_TRUE(no_op_sig_opt.has_value());
  auto& no_op_sig = no_op_sig_opt.value();
  EXPECT_EQ(no_op_sig.id.v_, 2);
  EXPECT_FALSE(no_op_sig.takes_context);
  EXPECT_EQ(no_op_sig.param_types.size(), 0);
  EXPECT_TRUE(no_op_sig.return_types.empty());

  // Test signature for 'tensor_meta_volume'
  auto tensor_sig_opt = dispatcher.GetSignature(function_id_t{11});
  ASSERT_TRUE(tensor_sig_opt.has_value());
  auto& tensor_sig = tensor_sig_opt.value();
  EXPECT_EQ(tensor_sig.id.v_, 11);
  EXPECT_EQ(tensor_sig.function_name, data::string("tensor_meta_volume"));
  ASSERT_EQ(tensor_sig.param_types.size(), 1);
  EXPECT_EQ(tensor_sig.param_types[0], ParamType::TENSOR_META);
  ASSERT_EQ(tensor_sig.return_types.size(), 1);
  EXPECT_EQ(tensor_sig.return_types[0], ParamType::PRIMITIVE_INT64);
  EXPECT_EQ(tensor_sig.return_payload_type, PayloadType::MONOSTATE);

  // Test signature for 'return_ucx_buffer'
  auto buffer_sig_opt = dispatcher.GetSignature(function_id_t{12});
  ASSERT_TRUE(buffer_sig_opt.has_value());
  auto& buffer_sig = buffer_sig_opt.value();
  EXPECT_TRUE(buffer_sig.return_types.empty());
  EXPECT_EQ(buffer_sig.return_payload_type, PayloadType::UCX_BUFFER);

  // Test signature for 'return_ucx_buffer_vec'
  auto buffer_vec_sig_opt = dispatcher.GetSignature(function_id_t{13});
  ASSERT_TRUE(buffer_vec_sig_opt.has_value());
  auto& buffer_vec_sig = buffer_vec_sig_opt.value();
  EXPECT_TRUE(buffer_vec_sig.return_types.empty());
  EXPECT_EQ(buffer_vec_sig.return_payload_type, PayloadType::UCX_BUFFER_VEC);

  // Test getting all signatures
  auto serialized_sigs = dispatcher.GetAllSignatures();
  auto deserialized_sigs =
    cista::deserialize<data::vector<RpcFunctionSignature>, MODE>(
      serialized_sigs);

  ASSERT_NE(deserialized_sigs, nullptr);
  EXPECT_EQ(deserialized_sigs->size(), 5);

  // Verify one of the deserialized signatures
  bool found_add_sig = false;
  for (const auto& sig : *deserialized_sigs) {
    if (sig.id.v_ == 1) {
      found_add_sig = true;
      EXPECT_EQ(sig.function_name, data::string("add_func"));
      EXPECT_EQ(sig.instance_name, data::string("test_instance"));
      ASSERT_EQ(sig.param_types.size(), 2);
      EXPECT_EQ(sig.param_types[0], ParamType::PRIMITIVE_INT32);
      ASSERT_EQ(sig.return_types.size(), 1);
      EXPECT_EQ(sig.return_types[0], ParamType::PRIMITIVE_INT32);
    }
  }
  EXPECT_TRUE(found_add_sig);
}

TEST_F(RpcDispatcherTest, DispatchWithMixedInputsAndOutputs) {
  ucxx::DefaultUcxMemoryResourceManager mr;
  dispatcher.RegisterFunction(
    function_id_t{15},
    [&mr](
      const RpcRequestHeader& req, int multiplier, const data::string& tag,
      const ucxx::UcxBufferVec& input_vec) {
      return process_mixed_request(mr, req, multiplier, tag, input_vec);
    },
    data::string("process_mixed"));

  // 1. Prepare inputs
  // Serializable inputs for the header
  RpcRequestHeader request{};
  request.function_id = function_id_t{15};
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
    mr, ucx_memory_type::HOST, {10, 20});  // Total size 30

  // 2. Dispatch the call
  auto result = dispatcher.Dispatch<ucxx::UcxBufferVec>(request, input_vec);

  // 3. Verify outputs
  // Verify serializable outputs from the header
  EXPECT_EQ(result.header.session_id.v_, 400);
  ASSERT_EQ(result.header.results.size(), 2);
  EXPECT_EQ(result.header.GetPrimitive<uint64_t>(0), 300);  // 30 * 10
  EXPECT_EQ(result.header.GetString(1), data::string("processed"));

  // Verify non-serializable context output
  ASSERT_TRUE(result.context.has_value());
  auto& output_vec = result.context.value();
  EXPECT_EQ(output_vec.size(), 1);
  EXPECT_EQ(output_vec[0].size, 300);
}

TEST_F(RpcDispatcherTest, DispatchDynamic) {
  ucxx::DefaultUcxMemoryResourceManager mr;
  dispatcher.RegisterFunction(
    function_id_t(1), [&mr]() { return return_ucx_buffer(mr); });
  dispatcher.RegisterFunction(function_id_t(2), &no_op);

  // Test dynamically dispatching a function that returns a context
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{1};
    auto request_buffer = cista::serialize(request);

    // Call the non-template overload
    auto result = dispatcher.Dispatch(std::move(request_buffer));
    ASSERT_TRUE(result.context.has_value());

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
      result.context.value());
  }

  // Test dynamically dispatching a function that returns monostate
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{2};
    auto request_buffer = cista::serialize(request);

    // Call the non-template overload
    auto result = dispatcher.Dispatch(std::move(request_buffer));
    ASSERT_TRUE(result.context.has_value());
    ASSERT_TRUE(std::holds_alternative<std::monostate>(result.context.value()));
  }
}

TEST_F(RpcDispatcherTest, DispatchWithReturnContext) {
  ucxx::DefaultUcxMemoryResourceManager mr;
  dispatcher.RegisterFunction(
    function_id_t(1), [&mr]() { return return_ucx_buffer(mr); });
  dispatcher.RegisterFunction(
    function_id_t(2), [&mr]() { return return_ucx_buffer_vec(mr); });

  // Test returning UcxBuffer
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{1};
    auto request_buffer = cista::serialize(request);

    auto result =
      dispatcher.Dispatch<ucxx::UcxBuffer>(std::move(request_buffer));

    if (!result.context.has_value()) {
      if (
        !result.header.results.empty()
        && result.header.results[0].type == ParamType::STRING) {
        FAIL() << "RPC failed with message: "
               << result.header.GetString(0).str();
      } else {
        FAIL() << "RPC failed with no error message. Status: "
               << static_cast<std::error_code>(result.header.status).message();
      }
    }

    ASSERT_TRUE(result.context.has_value());
    auto& buffer = result.context.value();
    EXPECT_EQ(buffer.size(), 1024);
  }

  // Test returning UcxBufferVec
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{2};
    auto request_buffer = cista::serialize(request);

    auto result =
      dispatcher.Dispatch<ucxx::UcxBufferVec>(std::move(request_buffer));

    ASSERT_TRUE(result.context.has_value());
    auto& buffer_vec = result.context.value();
    EXPECT_EQ(buffer_vec.size(), 2);
    EXPECT_EQ(buffer_vec[0].size, 128);
    EXPECT_EQ(buffer_vec[1].size, 256);
  }

  // Test type mismatch (expecting context but getting monostate)
  dispatcher.RegisterFunction(function_id_t{3}, &no_op);
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{3};
    auto request_buffer = cista::serialize(request);
    EXPECT_THROW(
      dispatcher.Dispatch<ucxx::UcxBuffer>(std::move(request_buffer)),
      std::bad_variant_access);
  }

  // Test type mismatch (expecting monostate but getting context)
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{1};
    auto request_buffer = cista::serialize(request);
    EXPECT_THROW(
      dispatcher.Dispatch<std::monostate>(std::move(request_buffer)),
      std::bad_variant_access);
  }
}

TEST_F(RpcDispatcherTest, DispatchWithUcxBufferVecContext) {
  // 1. Register the function that processes UcxBufferVec.
  // NOTE: Due to the const RpcRequestHeader change, functions can no longer
  // take payload by value. We register a function that takes it by const ref.
  dispatcher.RegisterFunction(function_id_t{10}, &process_buffer_vec_ref);

  // 2. Prepare the context object (UcxBufferVec).
  // This requires a memory manager.
  auto mr = eux::ucxx::DefaultUcxMemoryResourceManager();
  std::vector<size_t> sizes = {1024, 2048, 4096};
  eux::ucxx::UcxBufferVec buffer_vec(mr, ucx_memory_type::HOST, sizes);

  // 3. Prepare and serialize the RPC request header.
  RpcRequestHeader request{};
  request.function_id = function_id_t{10};
  request.session_id = session_id_t{108};
  auto request_buffer = cista::serialize(request);

  // 4. Dispatch the call, moving the context object.
  auto response_pair =
    dispatcher.DispatchMove(std::move(request_buffer), std::move(buffer_vec));
  auto& response_header = response_pair.header;

  // 5. Verify the result.
  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.GetPrimitive<uint64_t>(0), 7168);  // 1024+2048+4096
  EXPECT_EQ(response_header.status.value, 0);
}

TensorMeta tensor_meta_identity(const TensorMeta& meta) { return meta; }

TensorMeta tensor_meta_mutator(TensorMeta meta) {
  return mutate_tensor_meta(std::move(meta));
}

TEST_F(RpcDispatcherTest, DispatchTensorMetadataParameter) {
  dispatcher.RegisterFunction(function_id_t{11}, &tensor_meta_volume);

  TensorMeta input_meta = make_test_tensor_meta();

  RpcRequestHeader request{};
  request.function_id = function_id_t{11};
  request.session_id = session_id_t{200};

  ParamMeta tensor_param{};
  tensor_param.type = ParamType::TENSOR_META;
  tensor_param.value.emplace<TensorMeta>(input_meta);
  request.AddParam(std::move(tensor_param));

  auto response_pair = dispatcher.Dispatch<std::monostate>(request);
  auto& response_header = response_pair.header;

  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.results[0].type, ParamType::PRIMITIVE_INT64);
  EXPECT_EQ(
    response_header.GetPrimitive<int64_t>(0), tensor_meta_numel(input_meta));
  EXPECT_EQ(response_header.status.value, 0);
}

TEST_F(RpcDispatcherTest, DispatchWithResponseBuilder) {
  bool builder_called = false;
  dispatcher.RegisterFunction(
    function_id_t{14},
    [&builder_called](
      const RpcRequestHeader& req, int a,
      const RpcResponseBuilder& builder) -> int {
      // Use the builder to prepare a dummy response. This doesn't affect the
      // actual return value but verifies the builder is functional.
      auto resp_header =
        builder.PrepareResponse(req.session_id, req.request_id, "test");
      EXPECT_EQ(resp_header.session_id, req.session_id);
      EXPECT_EQ(resp_header.request_id, req.request_id);
      EXPECT_EQ(resp_header.GetString(0), "test");
      builder_called = true;
      return a * 2;
    });

  RpcRequestHeader request{};
  request.function_id = function_id_t{14};
  request.session_id = session_id_t{300};
  request.request_id = request_id_t{500};
  ParamMeta p1{};
  p1.type = ParamType::PRIMITIVE_INT32;
  p1.value.emplace<PrimitiveValue>(50);
  request.AddParam(std::move(p1));

  auto response_pair = dispatcher.Dispatch<std::monostate>(request);
  auto& response_header = response_pair.header;

  EXPECT_TRUE(builder_called);
  ASSERT_EQ(response_header.results.size(), 1);
  EXPECT_EQ(response_header.GetPrimitive<int32_t>(0), 100);
  EXPECT_EQ(response_header.session_id.v_, 300);
  EXPECT_EQ(response_header.request_id.v_, 500);
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
    dispatcher_B.RegisterFunction(
      function_id_t{100}, &add, data::string("add_func"));

    // Publish signatures to the central registry.
    registry["client_B_instance"] = dispatcher_B.GetAllSignatures();

    // 3. Client A (the caller) discovers and calls the function.
    // --- Discovery Phase ---
    auto serialized_sigs_it = registry.find("client_B_instance");
    ASSERT_NE(serialized_sigs_it, registry.end());

    auto deserialized_sigs =
      cista::deserialize<data::vector<RpcFunctionSignature>, MODE>(
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
    request.AddParam(std::move(p1));

    ParamMeta p2{};
    p2.type = ParamType::PRIMITIVE_INT32;
    p2.value.emplace<PrimitiveValue>(17);
    request.AddParam(std::move(p2));

    auto request_buffer = cista::serialize(request);

    // Simulate sending the request to Client B's dispatcher.
    auto response_pair =
      dispatcher_B.Dispatch<std::monostate>(std::move(request_buffer));
    auto& response_header = response_pair.header;

    // --- Verification Phase ---
    EXPECT_EQ(response_header.session_id.v_, 2025);
    ASSERT_EQ(response_header.results.size(), 1);
    EXPECT_EQ(response_header.GetPrimitive<int32_t>(0), 42);  // 25 + 17
    EXPECT_EQ(response_header.status.value, 0);
  }
}

std::tuple<int, std::string, float> return_multiple_values() {
  return {123, "hello", 3.14f};
}

std::tuple<int, ucxx::UcxBufferVec> return_multiple_with_payload(
  ucxx::UcxMemoryResourceManager& mr) {
  return {456, ucxx::UcxBufferVec(mr, ucx_memory_type::HOST, {16, 32})};
}

TEST_F(RpcDispatcherTest, DispatchWithTupleReturn) {
  ucxx::DefaultUcxMemoryResourceManager mr;
  dispatcher.RegisterFunction(function_id_t{1}, &return_multiple_values);
  dispatcher.RegisterFunction(
    function_id_t{2}, [&mr]() { return return_multiple_with_payload(mr); });

  // Test case 1: Multiple serializable return values, no payload
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{1};
    auto result = dispatcher.Dispatch<std::monostate>(request);

    ASSERT_EQ(result.header.results.size(), 3);
    EXPECT_EQ(result.header.GetPrimitive<int>(0), 123);
    EXPECT_EQ(result.header.GetString(1), "hello");
    EXPECT_EQ(result.header.GetPrimitive<float>(2), 3.14f);
    EXPECT_FALSE(result.context.has_value());
  }

  // Test case 2: Multiple return values with a payload
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{2};
    auto result = dispatcher.Dispatch<ucxx::UcxBufferVec>(request);

    ASSERT_EQ(result.header.results.size(), 1);
    EXPECT_EQ(result.header.GetPrimitive<int>(0), 456);
    ASSERT_TRUE(result.context.has_value());
    auto& payload = result.context.value();
    EXPECT_EQ(payload.size(), 2);
    EXPECT_EQ(payload[0].size, 16);
    EXPECT_EQ(payload[1].size, 32);
  }
}

RpcResponseHeader return_manual_header() {
  RpcResponseHeader h;
  h.session_id = session_id_t{999};
  h.status = std::make_error_code(std::errc::function_not_supported);
  ParamMeta p;
  p.type = ParamType::STRING;
  p.value.emplace<data::string>("manual_header");
  h.results.push_back(std::move(p));
  return h;
}

std::pair<RpcResponseHeader, ucxx::UcxBuffer> return_manual_pair(
  ucxx::UcxMemoryResourceManager& mr) {
  RpcResponseHeader h;
  h.session_id = session_id_t{888};
  ParamMeta p;
  p.type = ParamType::PRIMITIVE_INT64;
  p.value.emplace<PrimitiveValue>(int64_t{1337});
  h.results.push_back(std::move(p));
  return {std::move(h), ucxx::UcxBuffer(mr, ucx_memory_type::HOST, 256)};
}

TEST_F(RpcDispatcherTest, DispatchWithManualResponse) {
  ucxx::DefaultUcxMemoryResourceManager mr;
  dispatcher.RegisterFunction(function_id_t{3}, &return_manual_header);
  dispatcher.RegisterFunction(
    function_id_t{4}, [&mr]() { return return_manual_pair(mr); });

  // Test case 1: Function returns RpcResponseHeader directly
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{3};
    auto result = dispatcher.Dispatch<std::monostate>(request);

    EXPECT_EQ(result.header.session_id.v_, 999);
    EXPECT_EQ(
      result.header.status,
      std::make_error_code(std::errc::function_not_supported));
    ASSERT_EQ(result.header.results.size(), 1);
    EXPECT_EQ(result.header.GetString(0), "manual_header");
    EXPECT_FALSE(result.context.has_value());
  }

  // Test case 2: Function returns std::pair<RpcResponseHeader, Payload>
  {
    RpcRequestHeader request{};
    request.function_id = function_id_t{4};
    auto result = dispatcher.Dispatch<ucxx::UcxBuffer>(request);

    EXPECT_EQ(result.header.session_id.v_, 888);
    ASSERT_EQ(result.header.results.size(), 1);
    EXPECT_EQ(result.header.GetPrimitive<int64_t>(0), 1337);
    ASSERT_TRUE(result.context.has_value());
    EXPECT_EQ(result.context.value().size(), 256);
  }
}

#ifdef EUX_RPC_ENABLE_NATURAL_CALL
TEST_F(RpcDispatcherTest, GetCallerNaturalSyntax) {
  ucxx::DefaultUcxMemoryResourceManager mr;

  // 1. Register functions for testing
  dispatcher.RegisterFunction(function_id_t{1}, &add, "add");
  dispatcher.RegisterFunction(function_id_t{2}, &no_op, "no_op");
  dispatcher.RegisterFunction(
    function_id_t{3}, &return_multiple_values, "return_multiple_values");
  dispatcher.RegisterFunction(
    function_id_t{4},
    [&mr]() { return return_ucx_buffer_vec(mr); },
    "return_ucx_buffer_vec");
  dispatcher.RegisterFunction(
    function_id_t{5}, &process_buffer_vec_ref, "process_buffer_vec_ref");
  dispatcher.RegisterFunction(
    function_id_t{6},
    [&mr]() { return return_multiple_with_payload(mr); },
    "return_multiple_with_payload");

  // 2. Test simple function: int add(int, int)
  {
    auto add_caller = dispatcher.GetCaller<int(int, int)>(function_id_t{1});
    int result = (*add_caller)(20, 22);
    EXPECT_EQ(result, 42);
  }

  // 3. Test void function: void no_op()
  {
    auto no_op_caller = dispatcher.GetCaller<void()>(function_id_t{2});
    (*no_op_caller)();  // Should not throw and complete successfully
  }

  // 4. Test tuple return: std::tuple<int, std::string, float>()
  {
    auto tuple_caller =
      dispatcher.GetCaller<std::tuple<int, std::string, float>()>(
        function_id_t{3});
    auto [i, s, f] = (*tuple_caller)();
    EXPECT_EQ(i, 123);
    EXPECT_EQ(s, "hello");
    EXPECT_FLOAT_EQ(f, 3.14f);
  }

  // 5. Test payload return: ucxx::UcxBufferVec()
  {
    auto payload_return_caller =
      dispatcher.GetCaller<ucxx::UcxBufferVec()>(function_id_t{4});
    auto result_vec = (*payload_return_caller)();
    EXPECT_EQ(result_vec.size(), 2);
    EXPECT_EQ(result_vec[0].size, 128);
    EXPECT_EQ(result_vec[1].size, 256);
  }

  // 6. Test payload argument: uint64_t(const ucxx::UcxBufferVec&)
  {
    auto payload_arg_caller =
      dispatcher.GetCaller<uint64_t(const ucxx::UcxBufferVec&)>(
        function_id_t{5});
    ucxx::UcxBufferVec input_vec(
      mr, ::ucx_memory_type_t::HOST, {100, 200, 300});
    uint64_t total_size = 0;
    for (const auto& buf : input_vec) {
      total_size += buf.size;
    }
    uint64_t result = (*payload_arg_caller)(input_vec);
    EXPECT_EQ(result, total_size);
    // The input_vec is passed by const reference and should not be modified.
    EXPECT_EQ(input_vec.size(), 3);
  }

  // 7. Test tuple with payload return: std::tuple<int, ucxx::UcxBufferVec>()
  {
    auto tuple_payload_caller =
      dispatcher.GetCaller<std::tuple<int, ucxx::UcxBufferVec>()>(
        function_id_t{6});
    auto [val, vec] = (*tuple_payload_caller)();
    EXPECT_EQ(val, 456);
    EXPECT_EQ(vec.size(), 2);
    EXPECT_EQ(vec[0].size, 16);
    EXPECT_EQ(vec[1].size, 32);
  }

  // 8. Test unregistered function throws exception
  {
    EXPECT_THROW(
      dispatcher.GetCaller<void()>(function_id_t{999}), std::runtime_error);
  }

  // 9. Test native call error propagation
  {
    dispatcher.RegisterFunction(function_id_t{7}, []() -> void {
      throw std::runtime_error("Something went wrong");
    });
    auto error_caller = dispatcher.GetCaller<void()>(function_id_t{7});
    try {
      (*error_caller)();
      FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error& e) {
      EXPECT_STREQ(e.what(), "Something went wrong");
    }
  }
}
#endif

// =============================================================================
// RpcStatus and RpcCategoryRegistry Test Cases
// =============================================================================

// 1. Define a mock third-party error system for testing extensibility.
enum class MyThirdPartyErrc {
  SUCCESS = 0,
  FATAL_ERROR = 1,
  NETWORK_TIMEOUT = 2
};

class MyThirdPartyErrorCategory : public std::error_category {
 public:
  const char* name() const noexcept override { return "MyThirdParty"; }
  std::string message(int ev) const override {
    switch (static_cast<MyThirdPartyErrc>(ev)) {
      case MyThirdPartyErrc::SUCCESS:
        return "Success";
      case MyThirdPartyErrc::FATAL_ERROR:
        return "Fatal error";
      case MyThirdPartyErrc::NETWORK_TIMEOUT:
        return "Network timeout";
      default:
        return "Unknown third-party error";
    }
  }
};

inline const MyThirdPartyErrorCategory& my_third_party_error_category() {
  static const MyThirdPartyErrorCategory category;
  return category;
}

}  // namespace rpc
}  // namespace eux

namespace std {
template <>
struct is_error_code_enum<eux::rpc::MyThirdPartyErrc> : public true_type {};
}  // namespace std

namespace eux {
namespace rpc {

inline std::error_code make_error_code(MyThirdPartyErrc e) {
  return {static_cast<int>(e), my_third_party_error_category()};
}

class RpcStatusTest : public ::testing::Test {};

TEST_F(RpcStatusTest, DefaultConstruction) {
  RpcStatus status;
  std::error_code ec = status;
  EXPECT_FALSE(ec);  // Default should be success (0)
  EXPECT_EQ(ec.value(), 0);
  EXPECT_EQ(ec, std::error_code(0, std::generic_category()));
  EXPECT_EQ(&ec.category(), &std::generic_category());
}

TEST_F(RpcStatusTest, StdErrorCodeConversion) {
  // Test with a generic error code
  std::error_code generic_ec =
    std::make_error_code(std::errc::invalid_argument);
  RpcStatus status1(generic_ec);
  std::error_code converted_generic_ec = status1;

  EXPECT_EQ(converted_generic_ec, generic_ec);
  EXPECT_EQ(
    converted_generic_ec.value(),
    static_cast<int>(std::errc::invalid_argument));
  EXPECT_EQ(&converted_generic_ec.category(), &std::generic_category());

  // Test with a custom RPC error code
  std::error_code rpc_ec = make_error_code(RpcErrc::UNAVAILABLE);
  RpcStatus status2(rpc_ec);
  std::error_code converted_rpc_ec = status2;

  EXPECT_EQ(converted_rpc_ec, rpc_ec);
  EXPECT_EQ(converted_rpc_ec.value(), static_cast<int>(RpcErrc::UNAVAILABLE));
  EXPECT_EQ(&converted_rpc_ec.category(), &rpc_error_category());
}

TEST_F(RpcStatusTest, ThirdPartyExtension) {
  // 1. Register the custom third-party category.
  // This would typically be done once during application startup.
  RpcCategoryRegistry::GetInstance().RegisterCategory(
    my_third_party_error_category());

  // 2. Create an error code using the third-party system.
  std::error_code original_ec =
    make_error_code(MyThirdPartyErrc::NETWORK_TIMEOUT);

  // 3. Construct an RpcStatus from it.
  RpcStatus status(original_ec);

  // 4. Convert it back to a std::error_code.
  std::error_code converted_ec = status;

  // 5. Verify that the converted error code is identical to the original.
  EXPECT_EQ(converted_ec, original_ec);
  EXPECT_EQ(
    converted_ec.value(), static_cast<int>(MyThirdPartyErrc::NETWORK_TIMEOUT));
  EXPECT_EQ(&converted_ec.category(), &my_third_party_error_category());
  EXPECT_EQ(converted_ec.message(), "Network timeout");
}

}  // namespace rpc
}  // namespace eux
