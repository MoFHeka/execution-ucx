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

#ifndef RPC_CORE_RPC_STATUS_HPP_
#define RPC_CORE_RPC_STATUS_HPP_

#include <map>
#include <string>
#include <string_view>
#include <system_error>

#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace rpc {

namespace data = cista::offset;

// Define common RPC error codes, inspired by gRPC status codes.
enum class RpcErrc {
  OK = 0,
  CANCELLED = 1,
  UNKNOWN = 2,
  INVALID_ARGUMENT = 3,
  DEADLINE_EXCEEDED = 4,
  NOT_FOUND = 5,
  ALREADY_EXISTS = 6,
  PERMISSION_DENIED = 7,
  RESOURCE_EXHAUSTED = 8,
  FAILED_PRECONDITION = 9,
  ABORTED = 10,
  OUT_OF_RANGE = 11,
  UNIMPLEMENTED = 12,
  INTERNAL = 13,
  UNAVAILABLE = 14,
  DATA_LOSS = 15,
  UNAUTHENTICATED = 16,
};

// Error category for RPC status codes
class RpcErrorCategory : public std::error_category {
 public:
  const char* name() const noexcept override { return "RPC"; }

  std::string message(int ev) const override {
    switch (static_cast<RpcErrc>(ev)) {
      case RpcErrc::OK:
        return "OK";
      case RpcErrc::CANCELLED:
        return "Cancelled";
      case RpcErrc::UNKNOWN:
        return "Unknown";
      case RpcErrc::INVALID_ARGUMENT:
        return "Invalid argument";
      case RpcErrc::DEADLINE_EXCEEDED:
        return "Deadline exceeded";
      case RpcErrc::NOT_FOUND:
        return "Not found";
      case RpcErrc::ALREADY_EXISTS:
        return "Already exists";
      case RpcErrc::PERMISSION_DENIED:
        return "Permission denied";
      case RpcErrc::UNAUTHENTICATED:
        return "Unauthenticated";
      case RpcErrc::RESOURCE_EXHAUSTED:
        return "Resource exhausted";
      case RpcErrc::FAILED_PRECONDITION:
        return "Failed precondition";
      case RpcErrc::ABORTED:
        return "Aborted";
      case RpcErrc::OUT_OF_RANGE:
        return "Out of range";
      case RpcErrc::UNIMPLEMENTED:
        return "Unimplemented";
      case RpcErrc::INTERNAL:
        return "Internal";
      case RpcErrc::UNAVAILABLE:
        return "Unavailable";
      case RpcErrc::DATA_LOSS:
        return "Data loss";
      default:
        return "Unknown RPC error";
    }
  }
};

inline const RpcErrorCategory& rpc_error_category() noexcept {
  static const RpcErrorCategory category;
  return category;
}

inline std::error_code make_error_code(RpcErrc e) noexcept {
  return {static_cast<int>(e), rpc_error_category()};
}

/**
 * @class RpcCategoryRegistry
 * @brief A thread-safe singleton registry for std::error_category instances.
 *
 * This registry allows different parts of the application (including
 * third-party libraries) to register their custom error categories. The
 * RpcStatus struct uses this registry to dynamically look up the correct
 * category by name when converting back to a std::error_code, thus decoupling
 * the RPC core from specific error types like UCX errors.
 *
 * To extend RPC status handling with a custom error type:
 * 1. Define your error enum and std::error_category implementation.
 * 2. During your application's initialization, get the registry instance and
 *    register your category:
 *    RpcCategoryRegistry::get_instance().register_category(your_category());
 */
class RpcCategoryRegistry {
 public:
  static RpcCategoryRegistry& GetInstance() {
    static RpcCategoryRegistry instance;
    return instance;
  }

  void RegisterCategory(const std::error_category& category) {
    // Use string_view to avoid creating a string copy for lookup.
    categories_[std::string_view(category.name())] = &category;
  }

  const std::error_category* FindCategory(std::string_view name) const {
    auto it = categories_.find(name);
    if (it != categories_.end()) {
      return it->second;
    }
    return nullptr;
  }

 private:
  RpcCategoryRegistry() {
    // Pre-register the essential categories.
    RegisterCategory(std::generic_category());
    RegisterCategory(rpc_error_category());
  }
  ~RpcCategoryRegistry() = default;
  RpcCategoryRegistry(const RpcCategoryRegistry&) = delete;
  RpcCategoryRegistry& operator=(const RpcCategoryRegistry&) = delete;

  cista::raw::hash_map<std::string_view, const std::error_category*>
    categories_;
};

// Custom status structure for serialization, compatible with std::error_code.
struct RpcStatus {
  // Default constructor for success status.
  RpcStatus() = default;

  // Constructor from std::error_code.
  explicit RpcStatus(const std::error_code& ec)
    : value(ec.value()), category_name(ec.category().name()) {}

  // Assignment from std::error_code for convenience.
  RpcStatus& operator=(const std::error_code& ec) {
    value = ec.value();
    category_name = data::string{ec.category().name()};
    return *this;
  }

  // Conversion to std::error_code.
  operator std::error_code() const {
    auto& registry = RpcCategoryRegistry::GetInstance();
    // Use string_view to avoid allocation during lookup.
    std::string_view category_name_sv(
      category_name.data(), category_name.size());
    if (const auto* category = registry.FindCategory(category_name_sv)) {
      return {value, *category};
    }
    // Fallback to generic category for unknown categories.
    return {value, std::generic_category()};
  }

  bool operator==(const std::error_code& other) const {
    return value == other.value() && category_name == other.category().name();
  }

  int32_t value{0};
  data::string category_name{std::generic_category().name()};
  auto cista_members() const { return std::tie(value, category_name); }
};

}  // namespace rpc
}  // namespace eux

namespace std {
template <>
struct is_error_code_enum<eux::rpc::RpcErrc> : public true_type {};
}  // namespace std

#endif  // RPC_CORE_RPC_STATUS_HPP_
