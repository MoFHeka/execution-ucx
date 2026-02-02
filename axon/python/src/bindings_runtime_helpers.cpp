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

#include "axon/python/src/bindings_runtime_helpers.hpp"

#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>

#include <chrono>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include <unifex/any_sender_of.hpp>
#include <unifex/create.hpp>
#include <unifex/get_stop_token.hpp>
#include <unifex/inplace_stop_token.hpp>
#include <unifex/on.hpp>
#include <unifex/spawn_detached.hpp>
#include <unifex/then.hpp>
#include <unifex/upon_done.hpp>
#include <unifex/upon_error.hpp>
#include <unifex/v2/async_scope.hpp>
#include <unifex/with_query_value.hpp>

#include "axon/axon_runtime.hpp"
#include "axon/errors/error_types.hpp"
#include "axon/python/src/memory_policy_helpers.hpp"
#include "axon/python/src/python_helpers.hpp"
#include "axon/python/src/python_module.hpp"
#include "axon/python/src/python_wake_manager.hpp"
#include "rpc_core/rpc_types.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace axon {
namespace python {

// Build error message from AxonErrorContext
std::string BuildErrorMessageFromContext(
  const axon::errors::AxonErrorContext& ctx, const std::string& default_msg) {
  std::string_view error_msg_view = ctx.what;
  std::string error_msg;
  if (!error_msg_view.empty()) {
    error_msg = error_msg_view;
  } else {
    error_msg = ctx.status.GetErrorMessage();
    if (error_msg.empty()) {
      error_msg = default_msg;
    }
  }
  if (ctx.function_id != 0) {
    error_msg += std::format(" (function_id: {})", ctx.function_id);
  }
  if (ctx.session_id != 0) {
    error_msg += std::format(" (session_id: {})", ctx.session_id);
  }
  return error_msg;
}

// Set exception on Python Future
void SetFutureException(nb::handle future, const std::string& error_msg) {
  nb::module_ builtins = GetBuiltinsModule();
  nb::object runtime_error =
    builtins.attr("RuntimeError")(nb::str(error_msg.c_str(), error_msg.size()));
  future.attr("set_exception")(runtime_error);
}

// Throw Python RuntimeError exception
void ThrowRuntimeError(const std::string& error_msg) {
  nb::module_ builtins = GetBuiltinsModule();
  nb::object runtime_error = builtins.attr("RuntimeError")(error_msg);
  PyErr_SetObject(
    reinterpret_cast<PyObject*>(builtins.attr("RuntimeError").ptr()),
    runtime_error.ptr());
  throw nb::python_error();
}

// Helper function to extract error message from exception pointer
std::string ExtractErrorMessageFromExceptionPtr(std::exception_ptr error) {
  try {
    std::rethrow_exception(error);
  } catch (const axon::errors::AxonErrorException& e) {
    std::string error_msg =
      BuildErrorMessageFromContext(e.context(), "RPC invocation failed");
    return error_msg;
  } catch (const rpc::RpcException& e) {
    std::string_view error_msg_view = e.what();
    std::string error_msg;
    if (!error_msg_view.empty()) {
      error_msg = error_msg_view;
    } else {
      error_msg = "RPC error: " + e.code().message();
    }
    return error_msg;
  } catch (const std::exception& e) {
    std::string error_msg = e.what();
    return error_msg;
  } catch (...) {
    std::string error_msg = "Unknown error in RPC invocation";
    return error_msg;
  }
}

// Helper function to handle error and enqueue task
void HandleRpcError(
  SharedPyObject future, PythonWakeManager& manager,
  const std::string& error_msg) {
  manager.Enqueue(pro::make_proxy<TaskFacade>([future, error_msg]() mutable {
    if (!future) [[unlikely]] {
      std::cerr << "WARNING: Cannot set exception on future - object already "
                   "destroyed. "
                << "Error was: " << error_msg << std::endl;
      return;  // Exit early to prevent further error propagation
    }

    try {
      SetFutureException(future.get(), error_msg);
    } catch (...) {
      // Only catch exceptions from SetFutureException to prevent loop
      std::cerr
        << "CRITICAL: Exception in HandleRpcError while setting exception. "
        << "Original error: " << error_msg << std::endl;
    }
  }));
}

// Helper function to extract error message from response header
std::string ExtractErrorMessageFromResponseHeader(
  const rpc::RpcResponseHeader* response_header) {
  std::string error_msg = response_header->status.GetErrorMessage();
  if (
    !response_header->results.empty()
    && response_header->results[0].type == rpc::ParamType::STRING) {
    const auto& error_str =
      cista::get<cista::offset::string>(response_header->results[0].value);
    error_msg = std::string(error_str.data(), error_str.size());
  }
  return error_msg;
}

// Helper function to handle RPC done case
void HandleRpcDone(SharedPyObject future, PythonWakeManager& manager) {
  manager.Enqueue(pro::make_proxy<TaskFacade>([future]() mutable {
    // Check if the future is still valid before accessing it
    if (!future) [[unlikely]] {
      std::cerr
        << "WARNING: Cannot set exception on future - object already destroyed."
        << std::endl;
      return;  // Exit early to prevent segfault
    }

    nb::module_ asyncio = GetAsyncioModule();
    nb::object cancelled_error = asyncio.attr("CancelledError")();
    future.get().attr("set_exception")(cancelled_error);
  }));
}

// Detect payload type from Python object
PayloadTypeInfo DetectPayloadType(nb::object payload_obj) {
  PayloadTypeInfo info;
  if (payload_obj.is_none()) {
    info.type = PayloadTypeInfo::NONE;
    return info;
  }

  info.is_dlpack = nb::hasattr(payload_obj, "__dlpack__");
  info.is_ucx_buffer = nb::isinstance<ucxx::UcxBuffer>(payload_obj);
  info.is_ucx_buffer_vec = nb::isinstance<ucxx::UcxBufferVec>(payload_obj);

  if (info.is_ucx_buffer || info.is_dlpack) {
    info.type = PayloadTypeInfo::UCX_BUFFER;
  } else if (
    info.is_ucx_buffer_vec
    || (info.is_dlpack && nb::isinstance<nb::list>(payload_obj))) {
    info.type = PayloadTypeInfo::UCX_BUFFER_VEC;
  } else {
    info.type = PayloadTypeInfo::NONE;
  }

  return info;
}

// Helper function to register function with variant memory and lifecycle
// policies for UcxBuffer
template <typename BufferType>
void RegisterFunctionWithPoliciesImpl(
  axon::AxonRuntime& self, rpc::function_id_t function_id,
  const cista::offset::string& function_name,
  const cista::offset::vector<rpc::ParamType>& param_types,
  const cista::offset::vector<rpc::ParamType>& return_types,
  rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type,
  rpc::DynamicAsyncRpcFunction&& dynamic_func,
  axon::ReceiverMemoryPolicy<BufferType> mem_policy,
  axon::MessageLifecyclePolicy lc_policy) {
  std::visit(
    [&](auto&& mem_policy_v) {
      std::visit(
        [&](auto&& lc_policy_v) {
          using MemPolicyT = std::decay_t<decltype(mem_policy_v)>;
          using LcPolicyT = std::decay_t<decltype(lc_policy_v)>;
          self.RegisterFunction<BufferType, MemPolicyT, LcPolicyT>(
            function_id, function_name, param_types, return_types,
            input_payload_type, return_payload_type, std::move(dynamic_func),
            std::move(mem_policy_v), std::move(lc_policy_v));
        },
        std::move(lc_policy));
    },
    std::move(mem_policy));
}

// Helper function to register function with variant memory and lifecycle
// policies, automatically selecting buffer type based on payload type
void RegisterFunctionWithPolicies(
  axon::AxonRuntime& self, rpc::function_id_t function_id,
  const cista::offset::string& function_name,
  const cista::offset::vector<rpc::ParamType>& param_types,
  const cista::offset::vector<rpc::ParamType>& return_types,
  rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type,
  rpc::DynamicAsyncRpcFunction&& dynamic_func, nb::object memory_policy_factory,
  axon::MessageLifecyclePolicy lc_policy, axon::AxonRuntime& runtime) {
  if (input_payload_type == rpc::PayloadType::UCX_BUFFER_VEC) {
    auto mem_policy =
      create_memory_policy<ucxx::UcxBufferVec>(memory_policy_factory, runtime);
    RegisterFunctionWithPoliciesImpl<ucxx::UcxBufferVec>(
      self, function_id, function_name, param_types, return_types,
      input_payload_type, return_payload_type, std::move(dynamic_func),
      std::move(mem_policy), std::move(lc_policy));
  } else {
    auto mem_policy =
      create_memory_policy<ucxx::UcxBuffer>(memory_policy_factory, runtime);
    RegisterFunctionWithPoliciesImpl<ucxx::UcxBuffer>(
      self, function_id, function_name, param_types, return_types,
      input_payload_type, return_payload_type, std::move(dynamic_func),
      std::move(mem_policy), std::move(lc_policy));
  }
}

// Helper function to convert Python timeout to milliseconds
std::chrono::milliseconds ConvertTimeout(nb::object timeout_obj) {
  if (timeout_obj.is_none()) {
    return std::chrono::milliseconds(300);
  }

  try {
    int timeout_int = nb::cast<int>(timeout_obj);
    return std::chrono::milliseconds(timeout_int);
  } catch (...) {
    try {
      double timeout_float = nb::cast<double>(timeout_obj);
      return std::chrono::milliseconds(
        static_cast<int64_t>(timeout_float * 1000.0));
    } catch (...) {
      try {
        return nb::cast<std::chrono::milliseconds>(timeout_obj);
      } catch (...) {
        throw std::runtime_error(
          "timeout must be int (milliseconds), float (seconds), "
          "or timedelta");
      }
    }
  }
}

//
// Explicit template instantiations
//

// HandleRpcSuccessResult instantiations
template void HandleRpcSuccessResult<std::monostate>(
  SharedPyObject future, PythonWakeManager& manager,
  std::unique_ptr<
    const rpc::RpcResponseHeader, rpc::UcxDataDeleter<ucxx::UcxHeader>>
    response_header,
  std::monostate&& returned_payload, SharedPyObject from_dlpack_fn);

template void HandleRpcSuccessResult<ucxx::UcxBuffer>(
  SharedPyObject future, PythonWakeManager& manager,
  std::unique_ptr<
    const rpc::RpcResponseHeader, rpc::UcxDataDeleter<ucxx::UcxHeader>>
    response_header,
  ucxx::UcxBuffer&& returned_payload, SharedPyObject from_dlpack_fn);

template void HandleRpcSuccessResult<ucxx::UcxBufferVec>(
  SharedPyObject future, PythonWakeManager& manager,
  std::unique_ptr<
    const rpc::RpcResponseHeader, rpc::UcxDataDeleter<ucxx::UcxHeader>>
    response_header,
  ucxx::UcxBufferVec&& returned_payload, SharedPyObject from_dlpack_fn);

template void HandleRpcSuccessResult<rpc::PayloadVariant>(
  SharedPyObject future, PythonWakeManager& manager,
  std::unique_ptr<
    const rpc::RpcResponseHeader, rpc::UcxDataDeleter<ucxx::UcxHeader>>
    response_header,
  rpc::PayloadVariant&& returned_payload, SharedPyObject from_dlpack_fn);

// InvokeRpcWithMonostateSender instantiations
template auto
InvokeRpcWithMonostateSender<axon::AlwaysOnHostPolicy, std::monostate>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header,
  axon::AlwaysOnHostPolicy&& memory_policy);

template auto
InvokeRpcWithMonostateSender<axon::AlwaysOnHostPolicy, ucxx::UcxBuffer>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header,
  axon::AlwaysOnHostPolicy&& memory_policy);

template auto InvokeRpcWithMonostateSender<
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>, ucxx::UcxBuffer>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header,
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>&& memory_policy);

template auto
InvokeRpcWithMonostateSender<axon::AlwaysOnHostPolicy, ucxx::UcxBufferVec>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header,
  axon::AlwaysOnHostPolicy&& memory_policy);

template auto
InvokeRpcWithMonostateSender<axon::AlwaysOnHostPolicy, rpc::PayloadVariant>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header,
  axon::AlwaysOnHostPolicy&& memory_policy);

template auto InvokeRpcWithMonostateSender<
  axon::CustomMemoryPolicy<rpc::PayloadVariant>, rpc::PayloadVariant>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header,
  axon::CustomMemoryPolicy<rpc::PayloadVariant>&& memory_policy);

// InvokeRpcWithBufferSender instantiations
template auto
InvokeRpcWithBufferSender<axon::AlwaysOnHostPolicy, std::monostate>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::AlwaysOnHostPolicy&& memory_policy);

template auto InvokeRpcWithBufferSender<
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>, std::monostate>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>&& memory_policy);

template auto
InvokeRpcWithBufferSender<axon::AlwaysOnHostPolicy, ucxx::UcxBuffer>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::AlwaysOnHostPolicy&& memory_policy);

template auto InvokeRpcWithBufferSender<
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>, ucxx::UcxBuffer>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>&& memory_policy);

template auto
InvokeRpcWithBufferSender<axon::AlwaysOnHostPolicy, ucxx::UcxBufferVec>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::AlwaysOnHostPolicy&& memory_policy);

template auto
InvokeRpcWithBufferSender<axon::AlwaysOnHostPolicy, rpc::PayloadVariant>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::AlwaysOnHostPolicy&& memory_policy);

template auto InvokeRpcWithBufferSender<
  axon::CustomMemoryPolicy<rpc::PayloadVariant>, rpc::PayloadVariant>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::CustomMemoryPolicy<rpc::PayloadVariant>&& memory_policy);

// InvokeRpcWithBufferVecSender instantiations
template auto
InvokeRpcWithBufferVecSender<axon::AlwaysOnHostPolicy, std::monostate>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  axon::AlwaysOnHostPolicy&& memory_policy);

template auto InvokeRpcWithBufferVecSender<
  axon::CustomMemoryPolicy<ucxx::UcxBufferVec>, std::monostate>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  axon::CustomMemoryPolicy<ucxx::UcxBufferVec>&& memory_policy);

template auto
InvokeRpcWithBufferVecSender<axon::AlwaysOnHostPolicy, ucxx::UcxBuffer>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  axon::AlwaysOnHostPolicy&& memory_policy);

template auto
InvokeRpcWithBufferVecSender<axon::AlwaysOnHostPolicy, ucxx::UcxBufferVec>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  axon::AlwaysOnHostPolicy&& memory_policy);

template auto InvokeRpcWithBufferVecSender<
  axon::CustomMemoryPolicy<ucxx::UcxBufferVec>, ucxx::UcxBufferVec>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  axon::CustomMemoryPolicy<ucxx::UcxBufferVec>&& memory_policy);

template auto
InvokeRpcWithBufferVecSender<axon::AlwaysOnHostPolicy, rpc::PayloadVariant>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  axon::AlwaysOnHostPolicy&& memory_policy);

template auto InvokeRpcWithBufferVecSender<
  axon::CustomMemoryPolicy<rpc::PayloadVariant>, rpc::PayloadVariant>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  axon::CustomMemoryPolicy<rpc::PayloadVariant>&& memory_policy);

// Note: InvokeRpcWithCustomMemory and InvokeRpcWithHostPolicy cannot be
// explicitly instantiated because they have auto&& parameters (C++
// restriction). However, they internally call the instantiated functions above,
// so we still get compilation time benefits from those instantiations.

// RegisterFunctionWithPoliciesImpl instantiations
template void RegisterFunctionWithPoliciesImpl<ucxx::UcxBuffer>(
  axon::AxonRuntime& self, rpc::function_id_t function_id,
  const cista::offset::string& function_name,
  const cista::offset::vector<rpc::ParamType>& param_types,
  const cista::offset::vector<rpc::ParamType>& return_types,
  rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type,
  rpc::DynamicAsyncRpcFunction&& dynamic_func,
  axon::ReceiverMemoryPolicy<ucxx::UcxBuffer> mem_policy,
  axon::MessageLifecyclePolicy lc_policy);

template void RegisterFunctionWithPoliciesImpl<ucxx::UcxBufferVec>(
  axon::AxonRuntime& self, rpc::function_id_t function_id,
  const cista::offset::string& function_name,
  const cista::offset::vector<rpc::ParamType>& param_types,
  const cista::offset::vector<rpc::ParamType>& return_types,
  rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type,
  rpc::DynamicAsyncRpcFunction&& dynamic_func,
  axon::ReceiverMemoryPolicy<ucxx::UcxBufferVec> mem_policy,
  axon::MessageLifecyclePolicy lc_policy);

}  // namespace python
}  // namespace axon
}  // namespace eux
