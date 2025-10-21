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

#pragma once

#ifndef RPC_CORE_RPC_TYPES_HPP_
#define RPC_CORE_RPC_TYPES_HPP_

#include <cista.h>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

#include "rpc_core/rpc_payload_types.hpp"
#include "rpc_core/rpc_status.hpp"
#include "rpc_core/utils/hybrid_logical_clock.hpp"
#include "rpc_core/utils/tensor_meta.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace rpc {

namespace data = cista::offset;

using utils::HybridLogicalClock;
using utils::TensorMeta;
using utils::workflow_id_t;

// Strong type definitions for type safety
using function_id_t = cista::strong<uint32_t, struct function_id_tag>;
using session_id_t = cista::strong<uint32_t, struct session_id_tag>;
using request_id_t = cista::strong<uint32_t, struct request_id_tag>;

// Parameter types for RPC calls
enum class ParamType : uint8_t {
  PRIMITIVE_BOOL = 0,
  PRIMITIVE_INT8 = 1,
  PRIMITIVE_INT16 = 2,
  PRIMITIVE_INT32 = 3,
  PRIMITIVE_INT64 = 4,
  PRIMITIVE_UINT8 = 5,
  PRIMITIVE_UINT16 = 6,
  PRIMITIVE_UINT32 = 7,
  PRIMITIVE_UINT64 = 8,
  PRIMITIVE_FLOAT32 = 9,
  PRIMITIVE_FLOAT64 = 10,
  VECTOR_BOOL = 11,
  VECTOR_INT8 = 12,
  VECTOR_INT16 = 13,
  VECTOR_INT32 = 14,
  VECTOR_INT64 = 15,
  VECTOR_UINT8 = 16,
  VECTOR_UINT16 = 17,
  VECTOR_UINT32 = 18,
  VECTOR_UINT64 = 19,
  VECTOR_FLOAT32 = 20,
  VECTOR_FLOAT64 = 21,
  STRING = 22,
  VOID = 23,
  TENSOR_META = 24,
  UNKNOWN = 255,
  LAST = UNKNOWN,
};

// Describes the type of context object returned by an RPC function.
// The set of possible payload types that can be part of an RPC call.

// --- Nested Variant Definitions to Avoid Cista Bug ---
// Cista's variant has a bug in its recursive template logic for variants with
// more than 16 types. We use nested variants to keep the number of types in
// each variant below this threshold.

// Variant for all primitive types
using PrimitiveValue = data::variant<
  bool,
  int8_t,
  int16_t,
  int32_t,
  int64_t,
  uint8_t,
  uint16_t,
  uint32_t,
  uint64_t,
  float,
  double>;

// Variant for all vector-of-primitive types
using VectorValue = data::variant<
  data::vector<bool>,
  data::vector<int8_t>,
  data::vector<int16_t>,
  data::vector<int32_t>,
  data::vector<int64_t>,
  data::vector<uint8_t>,
  data::vector<uint16_t>,
  data::vector<uint32_t>,
  data::vector<uint64_t>,
  data::vector<float>,
  data::vector<double>>;

// Top-level variant for parameter values
using ParamValue = data::variant<
  PrimitiveValue,
  VectorValue,
  data::string,
  std::nullptr_t,
  TensorMeta>;

// Optimized parameter metadata for RPC header
struct ParamMeta {
  ParamType type;     // Parameter type
  ParamValue value;   // Parameter value (variant storage)
  data::string name;  // Parameter name
  auto cista_members() const { return std::tie(type, value, name); }
};

// Helper to provide parameter type info at compile time
struct ParamInfo {
  ParamType type;
  std::string_view name;
};

// Describes the full signature of an RPC function.
struct RpcFunctionSignature {
  data::string instance_name;
  function_id_t id;
  data::string function_name;
  data::vector<ParamType> param_types;
  ParamType return_type;
  PayloadType return_payload_type;
  bool takes_context;

  auto cista_members() const {
    return std::tie(
      instance_name,
      id,
      function_name,
      param_types,
      return_type,
      return_payload_type,
      takes_context);
  }
};

// Constexpr function to get info for primitive types
template <typename T>
constexpr ParamInfo get_primitive_param_info() {
  if constexpr (std::is_same_v<T, bool>) {
    return {ParamType::PRIMITIVE_BOOL, "bool"};
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return {ParamType::PRIMITIVE_INT8, "int8_t"};
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return {ParamType::PRIMITIVE_INT16, "int16_t"};
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return {ParamType::PRIMITIVE_INT32, "int32_t"};
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return {ParamType::PRIMITIVE_INT64, "int64_t"};
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return {ParamType::PRIMITIVE_UINT8, "uint8_t"};
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return {ParamType::PRIMITIVE_UINT16, "uint16_t"};
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return {ParamType::PRIMITIVE_UINT32, "uint32_t"};
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return {ParamType::PRIMITIVE_UINT64, "uint64_t"};
  } else if constexpr (std::is_same_v<T, float>) {
    return {ParamType::PRIMITIVE_FLOAT32, "float"};
  } else if constexpr (std::is_same_v<T, double>) {
    return {ParamType::PRIMITIVE_FLOAT64, "double"};
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported primitive type");
  }
}

// Constexpr function to get info for vector types
template <typename T>
constexpr ParamInfo get_vector_param_info() {
  if constexpr (std::is_same_v<T, bool>) {
    return {ParamType::VECTOR_BOOL, "vector<bool>"};
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return {ParamType::VECTOR_INT8, "vector<int8_t>"};
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return {ParamType::VECTOR_INT16, "vector<int16_t>"};
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return {ParamType::VECTOR_INT32, "vector<int32_t>"};
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return {ParamType::VECTOR_INT64, "vector<int64_t>"};
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return {ParamType::VECTOR_UINT8, "vector<uint8_t>"};
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return {ParamType::VECTOR_UINT16, "vector<uint16_t>"};
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return {ParamType::VECTOR_UINT32, "vector<uint32_t>"};
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return {ParamType::VECTOR_UINT64, "vector<uint64_t>"};
  } else if constexpr (std::is_same_v<T, float>) {
    return {ParamType::VECTOR_FLOAT32, "vector<float>"};
  } else if constexpr (std::is_same_v<T, double>) {
    return {ParamType::VECTOR_FLOAT64, "vector<double>"};
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported vector element type");
  }
}

// CRTP base class to provide common accessors for RPC messages.
template <typename Derived>
struct RpcMessageAccessor {
 private:
  // Helpers to safely cast this to the derived class type.
  const Derived& derived() const { return static_cast<const Derived&>(*this); }
  Derived& derived() { return static_cast<Derived&>(*this); }

 public:
  // --- Const-ref accessors for zero-copy read ---

  template <typename T>
  T get_primitive(size_t index) const {
    const auto& container = derived().get_params_container();
    if (index >= container.size()) {
      throw std::out_of_range("Item index out of bounds");
    }
    const auto& item = container[index];
    constexpr auto expected_info = get_primitive_param_info<T>();
    if (item.type != expected_info.type) {
      std::string error_msg = "Item type mismatch for ";
      error_msg += expected_info.name;
      throw std::runtime_error(error_msg);
    }
    // Access the nested PrimitiveValue variant first
    return cista::get<T>(cista::get<PrimitiveValue>(item.value));
  }

  const data::string& get_string(size_t index) const {
    const auto& container = derived().get_params_container();
    if (
      index >= container.size() || container[index].type != ParamType::STRING) {
      throw std::runtime_error("Invalid item access for string");
    }
    // get_string now accesses the top-level variant directly
    return cista::get<data::string>(container[index].value);
  }

  template <typename T>
  const data::vector<T>& get_vector(size_t index) const {
    const auto& container = derived().get_params_container();
    if (index >= container.size()) {
      throw std::out_of_range("Item index out of bounds");
    }
    const auto& item = container[index];
    constexpr auto expected_info = get_vector_param_info<T>();
    if (item.type != expected_info.type) {
      std::string error_msg = "Item type mismatch for ";
      error_msg += expected_info.name;
      throw std::runtime_error(error_msg);
    }
    // Access the nested VectorValue variant first
    return cista::get<data::vector<T>>(cista::get<VectorValue>(item.value));
  }

  const TensorMeta& get_tensor(size_t index) const {
    const auto& container = derived().get_params_container();
    if (
      index >= container.size()
      || container[index].type != ParamType::TENSOR_META) {
      throw std::runtime_error("Invalid item access for tensor metadata");
    }
    return cista::get<TensorMeta>(container[index].value);
  }

  // --- Move-based accessors for taking ownership ---

  template <typename T>
  data::vector<T>&& move_vector(size_t index) {
    auto& container = derived().get_params_container();
    constexpr auto expected_info = get_vector_param_info<T>();
    if (
      index >= container.size()
      || container[index].type != expected_info.type) {
      std::string error_msg = "Item type mismatch for ";
      error_msg += expected_info.name;
      throw std::runtime_error(error_msg);
    }
    // Access the nested VectorValue variant first, then move from it
    return cista::get<data::vector<T>>(
      std::move(cista::get<VectorValue>(container[index].value)));
  }

  TensorMeta&& move_tensor(size_t index) {
    auto& container = derived().get_params_container();
    if (
      index >= container.size()
      || container[index].type != ParamType::TENSOR_META) {
      throw std::runtime_error("Invalid item access for tensor metadata");
    }
    return std::move(cista::get<TensorMeta>(container[index].value));
  }

  // --- Temporal metadata helpers
  // -------------------------------------------------

  void tick_local_hlc() noexcept { derived().get_hlc().tick_local(); }

  void merge_remote_hlc(uint64_t remote_raw_timestamp) noexcept {
    derived().get_hlc().merge(remote_raw_timestamp);
  }

  void merge_remote_hlc(const utils::HybridLogicalClock& remote) noexcept {
    derived().get_hlc().merge(remote);
  }

  void clear_workflow_id() noexcept {
    derived().get_workflow_id() = utils::workflow_id_t{};
  }

  void assign_workflow_id(utils::workflow_id_t new_workflow_id) noexcept {
    derived().get_workflow_id() = new_workflow_id;
  }
};

// RPC request header (contains all non-tensor parameters)
struct RpcRequestHeader : public RpcMessageAccessor<RpcRequestHeader> {
  function_id_t function_id;       // Target function identifier
  session_id_t session_id;         // RPC session identifier
  request_id_t request_id;         // Unique request identifier
  data::vector<ParamMeta> params;  // Parameter list
  utils::HybridLogicalClock hlc{};
  utils::workflow_id_t workflow_id{};

  RpcRequestHeader() = default;

  auto cista_members() const {
    return std::tie(
      function_id, session_id, request_id, params, hlc, workflow_id);
  }

  // Add a parameter to the request
  template <typename T>
  void add_param(T&& value) {
    params.emplace_back(std::forward<T>(value));
  }

  // CRTP interface method
  const data::vector<ParamMeta>& get_params_container() const { return params; }
  data::vector<ParamMeta>& get_params_container() { return params; }

  const utils::HybridLogicalClock& get_hlc() const { return hlc; }
  utils::HybridLogicalClock& get_hlc() { return hlc; }
  const utils::workflow_id_t& get_workflow_id() const { return workflow_id; }
  utils::workflow_id_t& get_workflow_id() { return workflow_id; }
};

// RPC response header
struct RpcResponseHeader : public RpcMessageAccessor<RpcResponseHeader> {
  session_id_t session_id;          // RPC session identifier
  request_id_t request_id;          // Unique request identifier
  data::vector<ParamMeta> results;  // List of result parameters
  RpcStatus status{};               // Response status
  utils::HybridLogicalClock hlc{};
  utils::workflow_id_t workflow_id{};

  RpcResponseHeader() = default;

  auto cista_members() const {
    return std::tie(session_id, request_id, results, status, hlc, workflow_id);
  }

  // Add a result to the response
  template <typename T>
  void add_result(T&& value) {
    results.emplace_back(std::forward<T>(value));
  }

  // CRTP interface method
  const data::vector<ParamMeta>& get_params_container() const {
    return results;
  }
  data::vector<ParamMeta>& get_params_container() { return results; }

  const utils::HybridLogicalClock& get_hlc() const { return hlc; }
  utils::HybridLogicalClock& get_hlc() { return hlc; }
  const utils::workflow_id_t& get_workflow_id() const { return workflow_id; }
  utils::workflow_id_t& get_workflow_id() { return workflow_id; }
};

}  // namespace rpc
}  // namespace eux

#endif  // MEEPO_EMBEDDING_FRAMEWORK_RPC_TYPES_HPP_
