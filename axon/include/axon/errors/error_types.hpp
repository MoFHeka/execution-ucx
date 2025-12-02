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

#ifndef AXON_CORE_ERRORS_ERROR_TYPES_HPP_
#define AXON_CORE_ERRORS_ERROR_TYPES_HPP_

#include <proxy/proxy.h>

#include <cstdint>
#include <exception>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

#include "rpc_core/rpc_status.hpp"
#include "rpc_core/utils/hybrid_logical_clock.hpp"

namespace eux {
namespace axon {
namespace errors {

enum class AxonErrc {
  Ok = 0,
  WorkerNotFound,
  WorkerStartFailed,
  CoordinatorError,
  ConnectError,
  SerializeError,
  StorageBackpressure,
  DeserializeError,
  RemoteRespondError,
};

std::error_code make_error_code(AxonErrc e);
const std::error_category& AxonErrcCategory();
std::string ToString(AxonErrc e);

struct AxonErrorContext {
  uint64_t conn_id{0};
  uint32_t session_id{0};
  uint32_t request_id{0};
  uint32_t function_id{0};
  rpc::RpcStatus status =
    rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::INTERNAL));
  std::string what = "Unknown error";
  rpc::utils::HybridLogicalClock hlc{};
  uint32_t workflow_id{0};
};

// Exception class that wraps AxonErrorContext
class AxonErrorException : public std::exception {
 public:
  explicit AxonErrorException(AxonErrorContext& context) : context_(context) {}

  explicit AxonErrorException(const AxonErrorContext& context)
    : context_(context) {}

  explicit AxonErrorException(AxonErrorContext&& context)
    : context_(std::move(context)) {}

  const char* what() const noexcept override { return context_.what.c_str(); }

  const AxonErrorContext& context() const noexcept { return context_; }
  AxonErrorContext& context() noexcept { return context_; }

 private:
  AxonErrorContext context_;
};

// Convert AxonErrorContext to std::exception_ptr
std::exception_ptr MakeExceptionPtr(AxonErrorContext& context);
std::exception_ptr MakeExceptionPtr(const AxonErrorContext& context);
std::exception_ptr MakeExceptionPtr(AxonErrorContext&& context);

PRO_DEF_MEM_DISPATCH(MemOnError, OnError);

struct ErrorObserverFacade
  : pro::facade_builder::add_convention<
      MemOnError, void(const AxonErrorContext&)>::build {};

using ErrorObserver = pro::proxy<ErrorObserverFacade>;

}  // namespace errors
}  // namespace axon
}  // namespace eux

namespace std {
template <>
struct is_error_code_enum<eux::axon::errors::AxonErrc> : true_type {};

// Enable std::make_error_code to work with AxonErrc
inline std::error_code make_error_code(eux::axon::errors::AxonErrc e) noexcept {
  return eux::axon::errors::make_error_code(e);
}
}  // namespace std

#endif  // AXON_CORE_ERRORS_ERROR_TYPES_HPP_
