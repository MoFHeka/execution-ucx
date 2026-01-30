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

#ifndef AXON_RUNTIME_CORE_AXON_RUNTIME_RUNTIME_HPP_
#define AXON_RUNTIME_CORE_AXON_RUNTIME_RUNTIME_HPP_

#include <cista.h>
#include <proxy/proxy.h>

#include <chrono>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "axon/axon_worker.hpp"
#include "axon/memory_policy.hpp"
#include "axon/message_lifecycle_policy.hpp"
#include "axon/metrics/metrics_observer.hpp"
#include "axon/storage/axon_storage.hpp"

namespace eux {
namespace axon {

namespace data = cista::offset;

/**
 * @brief High-level runtime interface for Axon RPC system.
 *
 * This class provides a simplified API on top of AxonWorker, designed for
 * language bindings (e.g., Python) and higher-level applications.
 */
class AxonRuntime {
 public:
  // --- Types ---
  using AxonRequestPtr = AxonWorker::AxonRequestPtr;
  using AxonRequestID = AxonWorker::AxonRequestID;
  using StorageRequestIt = AxonWorker::StorageRequestIt;
  using WorkerKey = AxonWorker::WorkerKey;
  using RequestVisitor = AxonWorker::RequestVisitor;
  using StorageIteratorVisitor = AxonWorker::StorageIteratorVisitor;

  /**
   * @brief Constructs an AxonRuntime instance.
   *
   * @param mr Memory resource manager shared pointer
   * @param worker_name Name identifier for this worker instance
   * @param thread_pool_size Size of the thread pool (default: auto-detect)
   * @param timeout Timeout for RPC operations (default: 300ms)
   * @param auto_device_context Optional device context
   */
  explicit AxonRuntime(
    std::shared_ptr<ucxx::UcxMemoryResourceManager> mr,
    const std::string& worker_name,
    size_t thread_pool_size =
      (std::thread::hardware_concurrency() < 16 ? 4 : 16),
    std::chrono::milliseconds timeout = std::chrono::milliseconds(300),
    std::unique_ptr<ucxx::UcxAutoDeviceContext> auto_device_context = nullptr);

  /**
   * @brief Constructs an AxonRuntime instance with default memory resource.
   *
   * This constructor creates a default memory resource manager internally.
   * For backward compatibility with existing code.
   *
   * @param worker_name Name identifier for this worker instance
   * @param thread_pool_size Size of the thread pool (default: auto-detect)
   * @param timeout Timeout for RPC operations (default: 300ms)
   */
  explicit AxonRuntime(
    const std::string& worker_name,
    size_t thread_pool_size =
      (std::thread::hardware_concurrency() < 16 ? 4 : 16),
    std::chrono::milliseconds timeout = std::chrono::milliseconds(300));

  ~AxonRuntime();

  // --- Lifecycle ---
  std::expected<void, errors::AxonErrorContext> StartServer();
  std::expected<void, errors::AxonErrorContext> StartClient();
  std::expected<void, errors::AxonErrorContext> Start();
  void StopServer();
  void StopClient();
  void Stop();

  // --- Metrics ---
  void ReportServerMetrics();
  void ReportClientMetrics();

  // --- Accessors ---
  storage::AxonStorage& GetStorage();
  rpc::AsyncRpcDispatcher& GetDispatcher();

  // --- Service Discovery & Connection APIs ---
  /**
   * @brief Get the local UCX address.
   *
   * @return The local UCX address
   */
  std::vector<std::byte> GetLocalAddress();

  /**
   * @brief Get the local signatures.
   *
   * @return The local signatures
   */
  cista::byte_buf GetLocalSignatures() const;

  /**
   * @brief Register the endpoint signatures.
   *
   * @param worker_name Name of the remote worker
   * @param signatures_blob Signatures blob
   * @return The worker key
   */
  std::expected<WorkerKey, errors::AxonErrorContext> RegisterEndpointSignatures(
    std::string_view worker_name, const cista::byte_buf& signatures_blob);

  /**
   * @brief Asynchronously register the endpoint signatures.
   *
   * @param worker_name Name of the remote worker
   * @param signatures_blob Signatures blob
   * @return The sender of the asynchronous operation
   */
  auto RegisterEndpointSignaturesAsync(
    std::string_view worker_name, const cista::byte_buf& signatures_blob);

  /**
   * @brief Connect to a remote endpoint.
   *
   * @param ucp_address UCX address of the remote endpoint
   * @param worker_name Name of the remote worker
   * @return The connection ID
   */
  std::expected<uint64_t, errors::AxonErrorContext> ConnectEndpoint(
    std::vector<std::byte> ucp_address, std::string_view worker_name);

  /**
   * @brief Asynchronously connect to a remote endpoint.
   *
   * @param ucp_address UCX address of the remote endpoint
   * @param worker_name Name of the remote worker
   * @return The sender of the asynchronous operation
   */
  auto ConnectEndpointAsync(
    std::vector<std::byte> ucp_address, std::string_view worker_name)
    -> decltype(std::declval<AxonWorker>().ConnectEndpointAsync(
      std::move(ucp_address), worker_name));

  template <typename Sender>
  void SpawnClientTask(Sender&& sender) {
    worker_->SpawnClientTask(std::forward<Sender>(sender));
  }

  // --- RPC Server API ---
  template <
    typename ReceivedBufferT, typename Fn,
    typename MemPolicyT = AlwaysOnHostPolicy,
    typename MsgLcPolicyT = TransientPolicy>
    requires(rpc::is_payload_v<ReceivedBufferT>
             || std::is_same_v<ReceivedBufferT, std::monostate>
             || std::is_same_v<ReceivedBufferT, rpc::PayloadVariant>)
            && is_receiver_memory_policy_v<MemPolicyT, ReceivedBufferT>
            && is_message_lifecycle_policy_v<MsgLcPolicyT>
  void RegisterFunction(
    rpc::function_id_t id, Fn&& fn,
    data::string&& function_name = data::string(),
    MemPolicyT mem_policy = AlwaysOnHostPolicy{},
    MsgLcPolicyT lc_policy = TransientPolicy{}) {
    worker_->RegisterFunction(
      id, fn, function_name, std::move(mem_policy), std::move(lc_policy));
  }

  template <
    typename ReceivedBufferT, typename MemPolicyT = AlwaysOnHostPolicy,
    typename MsgLcPolicyT = TransientPolicy>
    requires(rpc::is_payload_v<ReceivedBufferT>
             || std::is_same_v<ReceivedBufferT, std::monostate>
             || std::is_same_v<ReceivedBufferT, rpc::PayloadVariant>)
            && is_receiver_memory_policy_v<MemPolicyT, ReceivedBufferT>
            && is_message_lifecycle_policy_v<MsgLcPolicyT>
  void RegisterFunction(
    rpc::function_id_t id, const data::string& name,
    const data::vector<rpc::ParamType>& param_types,
    const data::vector<rpc::ParamType>& return_types,
    rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type,
    rpc::DynamicAsyncRpcFunction&& func, MemPolicyT mem_policy,
    MsgLcPolicyT lc_policy) {
    worker_->RegisterFunction<ReceivedBufferT, MemPolicyT, MsgLcPolicyT>(
      id, name, param_types, return_types, input_payload_type,
      return_payload_type, std::move(func), std::move(mem_policy),
      std::move(lc_policy));
  }

  // --- RPC Client API ---

  // High-level API using RpcRequestBuilder
  template <typename RespBufferT, typename MemPolicyT, typename... Args>
    requires(rpc::is_payload_v<RespBufferT>
             || std::is_same_v<RespBufferT, std::monostate>)
            && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
            && (!detail::first_arg_is_memory_policy<
                RespBufferT, Args...>::value)
            && detail::is_high_level_api<Args...>::value
  auto InvokeRpc(
    std::string_view worker_name, rpc::session_id_t session_id,
    rpc::function_id_t function_id, rpc::utils::workflow_id_t workflow_id = {},
    MemPolicyT mem_policy = AlwaysOnHostPolicy{}, Args&&... args)
    -> decltype(std::declval<AxonWorker>()
                  .InvokeRpc<RespBufferT, MemPolicyT, Args...>(
                    worker_name, session_id, function_id, workflow_id,
                    mem_policy, std::forward<Args>(args)...)) {
    worker_->InvokeRpc<RespBufferT, MemPolicyT, Args...>(
      worker_name, session_id, function_id, workflow_id, std::move(mem_policy),
      std::forward<Args>(args)...);
  }

  // Dynamic API
  template <
    typename PayloadT = std::monostate, typename RespBufferT = std::monostate,
    typename MemPolicyT = AlwaysOnHostPolicy>
    requires(rpc::is_payload_v<PayloadT>
             || std::is_same_v<PayloadT, std::monostate>)
            && (rpc::is_payload_v<RespBufferT> || std::is_same_v<RespBufferT, std::monostate> || std::is_same_v<RespBufferT, rpc::PayloadVariant>)
            && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
            && detail::is_dynamic_api<PayloadT>::value
  auto InvokeRpc(
    std::string_view worker_name, rpc::RpcRequestHeader&& request_header,
    std::optional<PayloadT>&& payload = std::nullopt,
    MemPolicyT mem_policy = AlwaysOnHostPolicy{})
    -> decltype(std::declval<AxonWorker>()
                  .InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(
                    worker_name, std::move(request_header), std::move(payload),
                    std::move(mem_policy))) {
    return worker_->InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(
      worker_name, std::move(request_header), std::move(payload),
      std::move(mem_policy));
  }

  template <
    typename PayloadT = std::monostate, typename RespBufferT = std::monostate,
    typename MemPolicyT = AlwaysOnHostPolicy>
    requires(rpc::is_payload_v<PayloadT>
             || std::is_same_v<PayloadT, std::monostate>)
            && (rpc::is_payload_v<RespBufferT> || std::is_same_v<RespBufferT, std::monostate> || std::is_same_v<RespBufferT, rpc::PayloadVariant>)
            && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
            && detail::is_dynamic_api<PayloadT>::value
            && (!std::is_same_v<
                std::decay_t<PayloadT>, std::optional<std::decay_t<PayloadT>>>)
  auto InvokeRpc(
    std::string_view worker_name, rpc::RpcRequestHeader&& request_header,
    PayloadT&& payload, MemPolicyT mem_policy = AlwaysOnHostPolicy{})
    -> decltype(std::declval<AxonWorker>()
                  .InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(
                    worker_name, std::move(request_header), std::move(payload),
                    std::move(mem_policy))) {
    return worker_->InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(
      worker_name, std::move(request_header), std::move(payload),
      std::move(mem_policy));
  }

  // --- Configuration ---
  void SetTimeout(std::chrono::milliseconds timeout);
  void SetRejectMessages(bool reject);
  template <typename Checker>
  void SetBypassRejectionFunction(Checker&& checker) {
    worker_->SetBypassRejectionFunction(std::forward<Checker>(checker));
  }
  void SetServerErrorObserver(errors::ErrorObserver obs);
  void SetClientErrorObserver(errors::ErrorObserver obs);
  void SetServerMetricsObserver(metrics::MetricsObserver obs);
  void SetClientMetricsObserver(metrics::MetricsObserver obs);

  // --- Accessors ---
  bool IsRejectingMessages() const noexcept;
  static AxonRequestID GetMessageID(
    const rpc::RpcRequestHeader& request_header);

  // --- Advanced API for Hybrid Policy ---
  std::expected<void, errors::AxonErrorContext> ProcessStoredRequests(
    AxonRequestID req_id, rpc::DynamicAsyncRpcFunctionView func,
    RequestVisitor visitor);
  template <typename Func>
  std::expected<void, errors::AxonErrorContext> ProcessStoredRequests(
    AxonRequestID req_id, Func&& func, RequestVisitor visitor) {
    return worker_->ProcessStoredRequests(
      req_id, std::forward<Func>(func), std::move(visitor));
  }
  std::expected<void, errors::AxonErrorContext> ProcessStoredRequests(
    AxonRequestID req_id, StorageIteratorVisitor visitor);
  template <typename Func>
    requires std::is_copy_constructible_v<Func>
  std::expected<void, errors::AxonErrorContext> ProcessStoredRequests(
    Func&& func, RequestVisitor visitor) {
    return worker_->ProcessStoredRequests(
      std::forward<Func>(func), std::move(visitor));
  }
  std::expected<void, errors::AxonErrorContext> ProcessStoredRequests(
    rpc::DynamicAsyncRpcFunctionView func, RequestVisitor visitor);
  std::expected<void, errors::AxonErrorContext> ProcessStoredRequests(
    StorageIteratorVisitor visitor);
  void EraseStoredRequest(StorageRequestIt&& it);
  std::optional<AxonRequestPtr> FindStoredRequest(AxonRequestID request_id);
  // Helper function to process a single stored request
  std::expected<void, errors::AxonErrorContext> ProcessSingleStoredRequest(
    StorageRequestIt storage_it, rpc::DynamicAsyncRpcFunctionView func,
    RequestVisitor& visitor);

  // --- Time Helper Functions ---
  auto GetTimeContextScheduler()
    -> decltype(std::declval<unifex::timed_single_thread_context>()
                  .get_scheduler());

  // --- Memory Policy Helper Functions ---
  static ucxx::UcxBufferVec DefaultMemoryProvider(
    std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr_,
    const utils::TensorMetaSpan tensor_metas);

  static ucxx::UcxBuffer DefaultMemoryProvider(
    std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr_,
    const rpc::utils::TensorMeta& tensor_meta);

  std::reference_wrapper<ucxx::UcxMemoryResourceManager>
  GetMemoryResourceManager() const;

  // --- Accessors ---
  AxonWorker& GetWorker() { return *worker_; }
  const AxonWorker& GetWorker() const { return *worker_; }

 private:
  std::shared_ptr<ucxx::UcxMemoryResourceManager> mr_;
  std::unique_ptr<AxonWorker> worker_;
};

#define AXON_RUNTIME_REGISTER_FUNCTION(PayloadT, MemPolicyT, LcPolicyT)        \
  template void                                                                \
  AxonRuntime::RegisterFunction<PayloadT, MemPolicyT, LcPolicyT>(              \
    rpc::function_id_t id, const data::string& name,                           \
    const data::vector<rpc::ParamType>& param_types,                           \
    const data::vector<rpc::ParamType>& return_types,                          \
    rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type, \
    rpc::DynamicAsyncRpcFunction&& func, MemPolicyT mem_policy,                \
    LcPolicyT lc_policy);

AXON_RUNTIME_REGISTER_FUNCTION(
  std::monostate, AlwaysOnHostPolicy, TransientPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  std::monostate, AlwaysOnHostPolicy, RetentionPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  ucxx::UcxBuffer, AlwaysOnHostPolicy, TransientPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>, TransientPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  ucxx::UcxBuffer, AlwaysOnHostPolicy, RetentionPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>, RetentionPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  ucxx::UcxBufferVec, AlwaysOnHostPolicy, TransientPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>, TransientPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  ucxx::UcxBufferVec, AlwaysOnHostPolicy, RetentionPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>, RetentionPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  rpc::PayloadVariant, AlwaysOnHostPolicy, TransientPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  rpc::PayloadVariant, CustomMemoryPolicy<rpc::PayloadVariant>, TransientPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  rpc::PayloadVariant, AlwaysOnHostPolicy, RetentionPolicy)
AXON_RUNTIME_REGISTER_FUNCTION(
  rpc::PayloadVariant, CustomMemoryPolicy<rpc::PayloadVariant>, RetentionPolicy)

#undef AXON_RUNTIME_REGISTER_FUNCTION

#define AXON_RUNTIME_NAME_INVOKE_RPC_OPT(PayloadT, RespBufferT, MemPolicyT)    \
  template auto AxonRuntime::InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(     \
    std::string_view worker_name, rpc::RpcRequestHeader && request_header,     \
    std::optional<PayloadT> && payload, MemPolicyT mem_policy)                 \
    ->decltype(std::declval<AxonWorker>()                                      \
                 .InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(                \
                   worker_name, std::move(request_header), std::move(payload), \
                   std::move(mem_policy)));

AXON_RUNTIME_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBuffer, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_RUNTIME_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBuffer, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_RUNTIME_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_RUNTIME_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>)
AXON_RUNTIME_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_RUNTIME_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_RUNTIME_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_RUNTIME_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec,
  CustomMemoryPolicy<ucxx::UcxBufferVec>)

#undef AXON_RUNTIME_NAME_INVOKE_RPC_OPT

#define AXON_RUNTIME_NAME_INVOKE_RPC(PayloadT, RespBufferT, MemPolicyT)        \
  template auto AxonRuntime::InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(     \
    std::string_view worker_name, rpc::RpcRequestHeader && request_header,     \
    PayloadT && payload, MemPolicyT mem_policy)                                \
    ->decltype(std::declval<AxonWorker>()                                      \
                 .InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(                \
                   worker_name, std::move(request_header), std::move(payload), \
                   std::move(mem_policy)));

AXON_RUNTIME_NAME_INVOKE_RPC(
  ucxx::UcxBuffer, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_RUNTIME_NAME_INVOKE_RPC(
  ucxx::UcxBuffer, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_RUNTIME_NAME_INVOKE_RPC(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_RUNTIME_NAME_INVOKE_RPC(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>)
AXON_RUNTIME_NAME_INVOKE_RPC(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_RUNTIME_NAME_INVOKE_RPC(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_RUNTIME_NAME_INVOKE_RPC(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_RUNTIME_NAME_INVOKE_RPC(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec,
  CustomMemoryPolicy<ucxx::UcxBufferVec>)

#undef AXON_RUNTIME_NAME_INVOKE_RPC

}  // namespace axon
}  // namespace eux

#endif  // AXON_RUNTIME_CORE_AXON_RUNTIME_RUNTIME_HPP_
