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

#include <stdexcept>
#include <variant>
#include "rpc_core/rpc_payload_types.hpp"
#ifndef AXON_PYTHON_BINDINGS_RUNTIME_HELPERS_HPP_
#define AXON_PYTHON_BINDINGS_RUNTIME_HELPERS_HPP_

#include <nanobind/nanobind.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <unifex/any_sender_of.hpp>

#include "axon/axon_runtime.hpp"
#include "axon/errors/error_types.hpp"
#include "axon/memory_policy.hpp"
#include "axon/python/dlpack_helpers.hpp"
#include "axon/python/memory_policy_helpers.hpp"
#include "axon/python/param_conversion.hpp"
#include "axon/python/python_helpers.hpp"
#include "axon/python/python_wake_manager.hpp"
#include "rpc_core/rpc_types.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;
namespace rpc = eux::rpc;
namespace ucxx = eux::ucxx;
namespace axon = eux::axon;

// Build error message from AxonErrorContext
std::string BuildErrorMessageFromContext(
  const axon::errors::AxonErrorContext& ctx,
  const std::string& default_msg = "Operation failed");

// Set exception on Python Future (expects GIL to be held by caller)
void SetFutureException(nb::handle future, const std::string& error_msg);

// Throw Python RuntimeError exception
void ThrowRuntimeError(const std::string& error_msg);

// Helper function to extract error message from exception pointer
std::string ExtractErrorMessageFromExceptionPtr(std::exception_ptr error);

// Helper function to handle error and enqueue task
void HandleRpcError(
  SharedPyObject future, PythonWakeManager& manager,
  const std::string& error_msg);

// Helper function to extract error message from response header
std::string ExtractErrorMessageFromResponseHeader(
  const rpc::RpcResponseHeader* response_header);

// Helper function to handle RPC success result
template <typename PayloadT>
void HandleRpcSuccessResult(
  SharedPyObject future_guard, PythonWakeManager& manager,
  std::unique_ptr<
    const rpc::RpcResponseHeader, rpc::UcxDataDeleter<ucxx::UcxHeader>>
    response_header,
  PayloadT&& returned_payload,
  SharedPyObject from_dlpack_fn = SharedPyObject{}) {
  manager.Enqueue(pro::make_proxy<TaskFacade>(
    [future_guard, response_header = std::move(response_header),
     returned_payload = std::forward<PayloadT>(returned_payload),
     from_dlpack_fn]() mutable {
      nb::object future = future_guard.get();

      // Check if the future is valid (not none)
      if (future.is_none()) [[unlikely]] {
        std::cerr << "WARNING: Cannot set result on future - object is None."
                  << std::endl;
        return;
      }

      try {
        if (std::error_code(response_header->status)) {
          std::string error_msg_final =
            ExtractErrorMessageFromResponseHeader(response_header.get());
          SetFutureException(future, error_msg_final);
          return;
        }

        // Get from_dlpack_fn callable (GIL is held here in TaskFacade)
        nb::object from_dlpack_obj =
          from_dlpack_fn ? from_dlpack_fn.get() : nb::none();
        nb::object py_results = ResultsToPython<PayloadT>(
          response_header->results, std::move(returned_payload),
          std::move(from_dlpack_obj));
        future.attr("set_result")(py_results);
      } catch (const std::exception& e) {
        SetFutureException(future, e.what());
      } catch (...) {
        SetFutureException(
          future, "Unknown exception during RPC result handling");
      }
    }));
}

// Helper function to handle RPC done case
void HandleRpcDone(SharedPyObject future, PythonWakeManager& manager);

// Create error handler for RPC invocation results
struct CreateRpcErrorHandler {
  SharedPyObject future_;
  PythonWakeManager& manager_;
  mutable bool error_handled_{false};

  CreateRpcErrorHandler(SharedPyObject future, PythonWakeManager& manager)
    : future_(std::move(future)), manager_(manager) {}

  template <typename Error>
  void operator()(Error&& error) {
    // Prevent multiple error handling attempts for the same error handler
    // instance
    if (error_handled_) {
      // Error already being handled, skip to prevent loop
      std::cerr << "WARNING: CreateRpcErrorHandler called multiple times, "
                << "skipping duplicate error handling to prevent loop"
                << std::endl;
      return;
    }
    error_handled_ = true;

    using ErrorType = std::decay_t<decltype(error)>;
    constexpr bool is_axon_error_context =
      std::is_same_v<ErrorType, axon::errors::AxonErrorContext>;
    constexpr bool is_exception_ptr =
      std::is_same_v<ErrorType, std::exception_ptr>;

    std::string error_msg = "RPC invocation failed";

    if constexpr (is_axon_error_context) {
      error_msg = BuildErrorMessageFromContext(error, "RPC invocation failed");
    } else if constexpr (is_exception_ptr) {
      error_msg = ExtractErrorMessageFromExceptionPtr(error);
    }

    // HandleRpcError is already safe - it checks validity and handles
    // exceptions internally
    HandleRpcError(future_, manager_, error_msg);
  }
};

// Create result handler for RPC invocation
struct CreateRpcResultHandler {
  SharedPyObject future_;
  PythonWakeManager& manager_;
  SharedPyObject from_dlpack_;  // GIL-safe wrapper for from_dlpack_fn callable

  CreateRpcResultHandler(
    SharedPyObject future, PythonWakeManager& manager,
    nb::object from_dlpack_fn = nb::none())
    : future_(std::move(future)),
      manager_(manager),
      from_dlpack_(std::move(from_dlpack_fn)) {}

  template <typename Sender>
  auto operator()(Sender&& sender) {
    // Use SharedPyObject for both future and from_dlpack_fn to safely pass
    // across threads
    auto chained_sender =
      unifex::just(future_, from_dlpack_)
      | unifex::let_value(
        [&manager = manager_, sender = std::forward<Sender>(sender)](
          const SharedPyObject& future_in,
          const SharedPyObject& from_dlpack_in) mutable {
          SharedPyObject future = future_in;
          SharedPyObject from_dlpack_fn = from_dlpack_in;
          auto enhanced_sender =
            std::move(sender)
            | unifex::then(
              [future, &manager, from_dlpack_fn](auto&& result_pair) mutable {
                try {
                  auto& [response_header, returned_payload] = result_pair;
                  using PayloadType = std::decay_t<decltype(returned_payload)>;
                  if constexpr (std::is_same_v<PayloadType, std::monostate>) {
                    HandleRpcSuccessResult<std::monostate>(
                      future, manager, std::move(response_header),
                      std::move(returned_payload), from_dlpack_fn);
                  } else if constexpr (std::is_same_v<
                                         PayloadType, ucxx::UcxBuffer>) {
                    HandleRpcSuccessResult<ucxx::UcxBuffer>(
                      future, manager, std::move(response_header),
                      std::move(returned_payload), from_dlpack_fn);
                  } else if constexpr (std::is_same_v<
                                         PayloadType, ucxx::UcxBufferVec>) {
                    HandleRpcSuccessResult<ucxx::UcxBufferVec>(
                      future, manager, std::move(response_header),
                      std::move(returned_payload), from_dlpack_fn);
                  } else if constexpr (std::is_same_v<
                                         PayloadType, rpc::PayloadVariant>) {
                    HandleRpcSuccessResult<rpc::PayloadVariant>(
                      future, manager, std::move(response_header),
                      std::move(returned_payload), from_dlpack_fn);
                  }
                } catch (const std::exception& e) {
                  throw std::runtime_error(std::format(
                    "CRITICAL: Exception in CreateRpcResultHandler "
                    "success path: {}",
                    e.what()));
                } catch (...) {
                  throw std::runtime_error(
                    std::format("CRITICAL: Unknown exception in "
                                "CreateRpcResultHandler success path"));
                }
              })
            | unifex::upon_error(CreateRpcErrorHandler(future, manager))
            | unifex::upon_done(
              [future, &manager]() mutable { HandleRpcDone(future, manager); });
          return enhanced_sender;
        });
    return chained_sender;
  }
};

// Payload type detection result
struct PayloadTypeInfo {
  enum Type { NONE, UCX_BUFFER, UCX_BUFFER_VEC };
  Type type = NONE;
  bool is_dlpack = false;
  bool is_ucx_buffer = false;
  bool is_ucx_buffer_vec = false;
};

// Single-pass context for collecting invoke arguments.
// Caches DLManagedTensor* and owner to avoid repeated
// ExtractDlpackTensor calls when both TensorMeta and UcxBuffer are needed.
struct InvokeContext {
  std::vector<nb::object> tensor_objs;
  std::vector<rpc::utils::TensorMeta> tensor_metas;
  std::vector<DLManagedTensor*> dlm_ptrs;  // Cached DLManagedTensor pointers
  std::vector<nb::object> dlm_owners;      // Cached owners for lifetime
  rpc::RpcRequestHeader& header;

  explicit InvokeContext(rpc::RpcRequestHeader& h) : header(h) {
    // Reserve for typical case (1-4 tensors)
    tensor_objs.reserve(4);
    tensor_metas.reserve(4);
    dlm_ptrs.reserve(4);
    dlm_owners.reserve(4);
  }

  void add_tensor(nb::object obj) {
    // Extract dlpack tensor ONCE and cache both pointer and owner
    auto [dlm, owner] = ExtractDlpackTensor(obj);
    if (!dlm) {
      throw std::runtime_error(
        "Failed to extract DLManagedTensor from dlpack object");
    }
    tensor_objs.push_back(std::move(obj));
    dlm_ptrs.push_back(dlm);
    dlm_owners.push_back(std::move(owner));
    tensor_metas.push_back(ExtractTensorMetaFromDlm(dlm));
  }

  void add_non_tensor(nb::object obj) { header.AddParam(InferParamMeta(obj)); }

  // Finalize header by adding tensor meta(s) at the end.
  // This enables GetTensorMetas to find them quickly via reverse iteration.
  void finalize_header() {
    if (tensor_metas.size() == 1) {
      rpc::ParamMeta meta_param;
      meta_param.type = rpc::ParamType::TENSOR_META;
      meta_param.value.emplace<rpc::utils::TensorMeta>(tensor_metas[0]);
      header.AddParam(std::move(meta_param));
    } else if (tensor_metas.size() > 1) {
      rpc::ParamMeta meta_param;
      meta_param.type = rpc::ParamType::TENSOR_META_VEC;
      rpc::TensorMetaVec meta_vec;
      meta_vec.reserve(tensor_metas.size());
      for (auto& m : tensor_metas) {
        meta_vec.push_back(std::move(m));
      }
      meta_param.value.emplace<rpc::TensorMetaVec>(std::move(meta_vec));
      header.AddParam(std::move(meta_param));
    }
  }

  [[nodiscard]] size_t tensor_count() const noexcept {
    return tensor_objs.size();
  }

  // Create UcxBuffer from cached data (no repeated ExtractDlpackTensor call)
  [[nodiscard]] ucxx::UcxBuffer to_ucx_buffer(
    std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr) {
    if (dlm_ptrs.empty()) {
      throw std::runtime_error(
        "InvokeContext::to_ucx_buffer: no tensors to convert");
    }
    return DlpackToUcxBufferFromDlm(
      dlm_ptrs[0], std::move(dlm_owners[0]), tensor_metas[0], mr);
  }

  // Create UcxBufferVec from cached data (no repeated ExtractDlpackTensor
  // calls)
  [[nodiscard]] ucxx::UcxBufferVec to_ucx_buffer_vec(
    std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr) {
    return DlpackToUcxBufferVecFromDlm(dlm_ptrs, dlm_owners, tensor_metas, mr);
  }

  // Convert tensor_objs to nb::list for DlpackToUcxBufferVec
  [[nodiscard]] nb::list to_tensor_list() const {
    nb::list result;
    for (const auto& obj : tensor_objs) {
      result.append(obj);
    }
    return result;
  }
};

// Detect payload type from Python object
PayloadTypeInfo DetectPayloadType(nb::object payload_obj);

// Invoke RPC with monostate payload - returns sender
template <typename MemoryPolicy, typename ResultT>
auto InvokeRpcWithMonostateSender(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, MemoryPolicy&& memory_policy) {
  try {
    auto sender = self.InvokeRpc<std::monostate, ResultT, MemoryPolicy>(
      std::string(worker_name), std::move(request_header), std::monostate{},
      std::forward<MemoryPolicy>(memory_policy));
    return sender;
  } catch (...) {
    throw;
  }
}

// Invoke RPC with UcxBuffer payload - returns sender
template <typename MemoryPolicy, typename ResultT>
auto InvokeRpcWithBufferSender(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  MemoryPolicy&& memory_policy) {
  return self.InvokeRpc<ucxx::UcxBuffer, ResultT, MemoryPolicy>(
    std::string(worker_name), std::move(request_header), std::move(buffer),
    std::forward<MemoryPolicy>(memory_policy));
}

// Invoke RPC with UcxBufferVec payload - returns sender
template <typename MemoryPolicy, typename ResultT>
auto InvokeRpcWithBufferVecSender(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  MemoryPolicy&& memory_policy) {
  return self.InvokeRpc<ucxx::UcxBufferVec, ResultT, MemoryPolicy>(
    std::string(worker_name), std::move(request_header), std::move(buffer_vec),
    std::forward<MemoryPolicy>(memory_policy));
}

// Unified template for invoking RPC with different payload types
template <
  typename PayloadT, typename MemoryPolicy, typename ResultHandler,
  typename ResultT = rpc::PayloadVariant>
auto InvokeRpcWithPayload(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, PayloadT&& payload,
  MemoryPolicy&& memory_policy, ResultHandler&& result_handler) {
  constexpr bool is_monostate =
    std::is_same_v<std::decay_t<PayloadT>, std::monostate>;
  constexpr bool is_ucx_buffer =
    std::is_same_v<std::decay_t<PayloadT>, ucxx::UcxBuffer>;
  constexpr bool is_ucx_buffer_vec =
    std::is_same_v<std::decay_t<PayloadT>, ucxx::UcxBufferVec>;
  if constexpr (is_monostate) {
    return result_handler(InvokeRpcWithMonostateSender<MemoryPolicy, ResultT>(
      self, worker_name, std::move(request_header),
      std::forward<MemoryPolicy>(memory_policy)));
  } else if constexpr (is_ucx_buffer) {
    return result_handler(InvokeRpcWithBufferSender<MemoryPolicy, ResultT>(
      self, worker_name, std::move(request_header), std::move(payload),
      std::forward<MemoryPolicy>(memory_policy)));
  } else if constexpr (is_ucx_buffer_vec) {
    return result_handler(InvokeRpcWithBufferVecSender<MemoryPolicy, ResultT>(
      self, worker_name, std::move(request_header), std::move(payload),
      std::forward<MemoryPolicy>(memory_policy)));
  }
}

// Unified template for invoking RPC with custom memory policy
template <typename PayloadT>
auto InvokeRpcWithCustomMemory(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, PayloadT&& payload,
  nb::object memory_policy_factory, auto&& result_handler) {
  // Use PayloadVariant for dynamic API result policy, as ResultT defaults to
  // PayloadVariant in InvokeRpcWithPayload and memory factory is for result
  // allocation.
  using ResultType = rpc::PayloadVariant;
  auto custom_policy =
    create_custom_memory_policy<ResultType>(memory_policy_factory, self);
  return InvokeRpcWithPayload(
    self, worker_name, std::move(request_header),
    std::forward<PayloadT>(payload), std::move(custom_policy),
    std::forward<decltype(result_handler)>(result_handler));
}

// Unified template for invoking RPC with AlwaysOnHostPolicy
template <typename PayloadT>
auto InvokeRpcWithHostPolicy(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, PayloadT&& payload,
  auto&& result_handler) {
  return InvokeRpcWithPayload(
    self, worker_name, std::move(request_header),
    std::forward<PayloadT>(payload), axon::AlwaysOnHostPolicy{},
    std::forward<decltype(result_handler)>(result_handler));
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
  axon::MessageLifecyclePolicy lc_policy);

// Helper function to register function with variant memory and lifecycle
// policies, automatically selecting buffer type based on payload type
void RegisterFunctionWithPolicies(
  axon::AxonRuntime& self, rpc::function_id_t function_id,
  const cista::offset::string& function_name,
  const cista::offset::vector<rpc::ParamType>& param_types,
  const cista::offset::vector<rpc::ParamType>& return_types,
  rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type,
  rpc::DynamicAsyncRpcFunction&& dynamic_func, nb::object memory_policy_factory,
  axon::MessageLifecyclePolicy lc_policy, axon::AxonRuntime& runtime);

// Helper function to convert Python timeout to milliseconds
std::chrono::milliseconds ConvertTimeout(nb::object timeout_obj);

// Explicit template instantiation declarations
// These prevent template instantiation in each translation unit that includes
// this header, reducing compilation time

// HandleRpcSuccessResult instantiations
extern template void HandleRpcSuccessResult<ucxx::UcxBuffer>(
  SharedPyObject future, PythonWakeManager& manager,
  std::unique_ptr<
    const rpc::RpcResponseHeader, rpc::UcxDataDeleter<ucxx::UcxHeader>>
    response_header,
  ucxx::UcxBuffer&& returned_payload, SharedPyObject from_dlpack_fn);

extern template void HandleRpcSuccessResult<ucxx::UcxBufferVec>(
  SharedPyObject future, PythonWakeManager& manager,
  std::unique_ptr<
    const rpc::RpcResponseHeader, rpc::UcxDataDeleter<ucxx::UcxHeader>>
    response_header,
  ucxx::UcxBufferVec&& returned_payload, SharedPyObject from_dlpack_fn);

extern template void HandleRpcSuccessResult<rpc::PayloadVariant>(
  SharedPyObject future, PythonWakeManager& manager,
  std::unique_ptr<
    const rpc::RpcResponseHeader, rpc::UcxDataDeleter<ucxx::UcxHeader>>
    response_header,
  rpc::PayloadVariant&& returned_payload, SharedPyObject from_dlpack_fn);

// InvokeRpcWithMonostateSender instantiations
extern template auto
InvokeRpcWithMonostateSender<axon::AlwaysOnHostPolicy, std::monostate>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header,
  axon::AlwaysOnHostPolicy&& memory_policy);

extern template auto InvokeRpcWithMonostateSender<
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>, std::monostate>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header,
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>&& memory_policy);

extern template auto
InvokeRpcWithMonostateSender<axon::AlwaysOnHostPolicy, ucxx::UcxBuffer>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header,
  axon::AlwaysOnHostPolicy&& memory_policy);

extern template auto InvokeRpcWithMonostateSender<
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>, ucxx::UcxBuffer>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header,
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>&& memory_policy);

extern template auto
InvokeRpcWithMonostateSender<axon::AlwaysOnHostPolicy, ucxx::UcxBufferVec>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header,
  axon::AlwaysOnHostPolicy&& memory_policy);

// InvokeRpcWithBufferSender instantiations
extern template auto
InvokeRpcWithBufferSender<axon::AlwaysOnHostPolicy, std::monostate>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::AlwaysOnHostPolicy&& memory_policy);

extern template auto InvokeRpcWithBufferSender<
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>, std::monostate>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>&& memory_policy);

extern template auto
InvokeRpcWithBufferSender<axon::AlwaysOnHostPolicy, ucxx::UcxBuffer>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::AlwaysOnHostPolicy&& memory_policy);

extern template auto InvokeRpcWithBufferSender<
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>, ucxx::UcxBuffer>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::CustomMemoryPolicy<ucxx::UcxBuffer>&& memory_policy);

extern template auto
InvokeRpcWithBufferSender<axon::AlwaysOnHostPolicy, ucxx::UcxBufferVec>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::AlwaysOnHostPolicy&& memory_policy);

extern template auto InvokeRpcWithBufferSender<
  axon::CustomMemoryPolicy<ucxx::UcxBufferVec>, ucxx::UcxBufferVec>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBuffer&& buffer,
  axon::CustomMemoryPolicy<ucxx::UcxBufferVec>&& memory_policy);

// InvokeRpcWithBufferVecSender instantiations
extern template auto
InvokeRpcWithBufferVecSender<axon::AlwaysOnHostPolicy, ucxx::UcxBufferVec>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  axon::AlwaysOnHostPolicy&& memory_policy);

extern template auto InvokeRpcWithBufferVecSender<
  axon::CustomMemoryPolicy<ucxx::UcxBufferVec>, ucxx::UcxBufferVec>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  axon::CustomMemoryPolicy<ucxx::UcxBufferVec>&& memory_policy);

extern template auto
InvokeRpcWithBufferVecSender<axon::AlwaysOnHostPolicy, ucxx::UcxBuffer>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  axon::AlwaysOnHostPolicy&& memory_policy);

extern template auto
InvokeRpcWithBufferVecSender<axon::AlwaysOnHostPolicy, ucxx::UcxBufferVec>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  axon::AlwaysOnHostPolicy&& memory_policy);

extern template auto InvokeRpcWithBufferVecSender<
  axon::CustomMemoryPolicy<ucxx::UcxBufferVec>, ucxx::UcxBufferVec>(
  axon::AxonRuntime& self, std::string_view worker_name,
  rpc::RpcRequestHeader&& request_header, ucxx::UcxBufferVec&& buffer_vec,
  axon::CustomMemoryPolicy<ucxx::UcxBufferVec>&& memory_policy);

// Note: InvokeRpcWithCustomMemory and InvokeRpcWithHostPolicy cannot be
// explicitly instantiated because they have auto&& parameters. However, they
// internally call InvokeRpcWithPayload and the lower-level sender functions
// which are instantiated below, so we still get some benefit.

// RegisterFunctionWithPoliciesImpl instantiations
extern template void RegisterFunctionWithPoliciesImpl<ucxx::UcxBuffer>(
  axon::AxonRuntime& self, rpc::function_id_t function_id,
  const cista::offset::string& function_name,
  const cista::offset::vector<rpc::ParamType>& param_types,
  const cista::offset::vector<rpc::ParamType>& return_types,
  rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type,
  rpc::DynamicAsyncRpcFunction&& dynamic_func,
  axon::ReceiverMemoryPolicy<ucxx::UcxBuffer> mem_policy,
  axon::MessageLifecyclePolicy lc_policy);

extern template void RegisterFunctionWithPoliciesImpl<ucxx::UcxBufferVec>(
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

#endif  // AXON_PYTHON_BINDINGS_RUNTIME_HELPERS_HPP_
