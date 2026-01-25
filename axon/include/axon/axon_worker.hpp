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

#ifndef AXON_CORE_AXON_WORKER_HPP_
#define AXON_CORE_AXON_WORKER_HPP_

#include <cista.h>
#include <proxy/proxy.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <unifex/any_sender_of.hpp>
#include <unifex/create.hpp>
#include <unifex/inplace_stop_token.hpp>
#include <unifex/just.hpp>
#include <unifex/just_error.hpp>
#include <unifex/just_from.hpp>
#include <unifex/let_done.hpp>
#include <unifex/let_error.hpp>
#include <unifex/let_value.hpp>
#include <unifex/let_value_with.hpp>
#include <unifex/on.hpp>
#include <unifex/sequence.hpp>
#include <unifex/spawn_detached.hpp>
#include <unifex/spawn_future.hpp>
#include <unifex/static_thread_pool.hpp>
#include <unifex/stop_when.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/task.hpp>
#include <unifex/then.hpp>
#include <unifex/timed_single_thread_context.hpp>
#include <unifex/upon_done.hpp>
#include <unifex/upon_error.hpp>
#include <unifex/v2/async_scope.hpp>
#include <unifex/variant_sender.hpp>
#include <unifex/when_all.hpp>

#include "axon/errors/error_types.hpp"
#include "axon/memory_policy.hpp"
#include "axon/message_lifecycle_policy.hpp"
#include "axon/metrics/metrics_observer.hpp"
#include "axon/storage/axon_storage.hpp"
#include "axon/utils/axon_message.hpp"
#include "axon/utils/hash.hpp"
#include "axon/utils/ring_buffer.hpp"
#include "axon/utils/slot_map.hpp"
#include "rpc_core/async_rpc_dispatcher.hpp"
#include "rpc_core/rpc_request_builder.hpp"
#include "rpc_core/rpc_status.hpp"
#include "rpc_core/rpc_types.hpp"
#include "rpc_core/utils/hybrid_logical_clock.hpp"
#include "ucx_context/ucx_am_context/ucx_am_context.hpp"
#include "ucx_context/ucx_context_data.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

namespace eux {
namespace axon {

namespace data = cista::offset;

namespace detail {

template <typename RespBufferT, typename... Args>
struct first_arg_is_memory_policy : std::false_type {};

template <typename RespBufferT, typename First, typename... Rest>
struct first_arg_is_memory_policy<RespBufferT, First, Rest...> {
  static constexpr bool value =
    is_receiver_memory_policy_v<std::decay_t<First>, RespBufferT>;
};

// Type traits to distinguish InvokeRpc overloads
// High-level API uses Args... parameter pack (variadic template)
template <typename... Args>
struct is_high_level_api : std::true_type {};

// Dynamic API uses PayloadT template parameter (non-variadic)
template <typename PayloadT>
struct is_dynamic_api : std::true_type {};

}  // namespace detail

// Forward declaration
class AxonRuntime;

class AxonWorker {
  friend class AxonRuntime;

 public:
  // --- Types ---
  using AxonRequestPtr = std::shared_ptr<utils::AxonRequest>;
  using AxonRequestID = utils::AxonMessageID;
  using StorageRequestIt = storage::AxonStorage::request_iterator;
  using WorkerKey = utils::SlotKey;

  // Storing remote worker and function signatures
  using FunctionSignaturesMap =
    cista::raw::hash_map<rpc::function_id_t, rpc::RpcFunctionSignature>;

  // A struct to store remote endpoint information
  struct RemoteEndpoint {
    data::string name;
    FunctionSignaturesMap signatures;
    std::optional<uint64_t> conn_id;
  };

  // A callable function to get normal function sender.
  struct ExecutorFacade
    : pro::facade_builder  //
      ::add_convention<
        pro::operator_dispatch<"()">,
        unifex::any_sender_of<rpc::DynamicFunctionReturnType>()>  //
      ::build {};
  using Executor = pro::proxy<ExecutorFacade>;

  // A visitor for iterating over stored requests.
  // Call Executor inside visit function result.
  struct RequestVisitorFacade : pro::facade_builder  //
                                ::add_convention<
                                  pro::operator_dispatch<"()">,
                                  LifecycleStatus(Executor)>  //
                                ::build {};
  using RequestVisitor = pro::proxy<RequestVisitorFacade>;

  struct StorageIteratorVisitorFacade
    : pro::facade_builder::add_convention<
        pro::operator_dispatch<"()">,
        LifecycleStatus(StorageRequestIt)>::build {};
  using StorageIteratorVisitor = pro::proxy<StorageIteratorVisitorFacade>;

  using ResponseExpected = std::expected<
    std::pair<rpc::RpcResponseHeader, rpc::ReturnedPayload>,
    errors::AxonErrorContext>;

 public:
  AxonWorker(
    std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr_,
    const std::string& worker_name,
    size_t thread_pool_size =
      (std::thread::hardware_concurrency() < 16 ? 4 : 16),
    std::chrono::milliseconds timeout = std::chrono::milliseconds(300),
    std::unique_ptr<ucxx::UcxAutoDeviceContext> auto_device_context = nullptr);

  ~AxonWorker();

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
  std::vector<std::byte> GetLocalAddress();

  cista::byte_buf GetLocalSignatures() const;

  std::expected<uint64_t, errors::AxonErrorContext> ConnectEndpoint(
    std::vector<std::byte> ucp_address, std::string_view worker_name);

  auto ConnectEndpointAsync(
    std::vector<std::byte> ucp_address, std::string_view worker_name) {
    auto sender =
      ucxx::connect_endpoint(
        client_ctx_->get_scheduler(), std::move(ucp_address))
      | unifex::then(
        [this, worker_name = std::string(worker_name)](uint64_t conn_id) {
          AssociateConnection(worker_name, conn_id);
          return std::expected<uint64_t, errors::AxonErrorContext>(conn_id);
        })
      | unifex::upon_error(
        [this](std::variant<std::error_code, std::exception_ptr>&& error)
          -> std::expected<uint64_t, errors::AxonErrorContext> {
          std::error_code ec;
          std::string what;
          if (std::holds_alternative<std::error_code>(error)) {
            ec = std::get<std::error_code>(error);
            what = std::format("ConnectEndpoint failed: {}", ec.message());
          } else {
            try {
              std::rethrow_exception(std::get<std::exception_ptr>(error));
            } catch (const std::exception& e) {
              ec = std::make_error_code(errors::AxonErrc::CoordinatorError);
              what = std::format("ConnectEndpoint failed: {}", e.what());
            }
          }
          errors::AxonErrorContext error_ctx{
            .status = rpc::RpcStatus(ec),
            .what = std::move(what),
            .hlc = hlc_,
          };
          ReportClientError_(error_ctx);
          return std::unexpected(std::move(error_ctx));
        });
    return sender;
  }

  template <typename Sender>
  void SpawnClientTask(Sender&& sender) {
    if (stopping_.load(std::memory_order_relaxed)) {
      return;
    }

    unifex::spawn_detached(
      unifex::on(client_ctx_->get_scheduler(), std::move(sender)),
      client_async_scope_);
  }

  void AssociateConnection(const std::string& worker_name, uint64_t conn_id);
  std::expected<WorkerKey, errors::AxonErrorContext> RegisterEndpointSignatures(
    std::string_view worker_name, const cista::byte_buf& signatures_blob);

  auto RegisterEndpointSignaturesAsync(
    std::string_view worker_name, const cista::byte_buf& signatures_blob) {
    return unifex::just_from(
      [this, worker_name = std::string(worker_name), signatures_blob]()
        -> std::expected<WorkerKey, errors::AxonErrorContext> {
        return RegisterEndpointSignatures(worker_name, signatures_blob);
      });
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
    MemPolicyT mem_policy = AlwaysOnHostPolicy{}, Args&&... args) {
    auto it = remote_workers_slot_.find(worker_name);
    WorkerKey worker_key{std::numeric_limits<uint32_t>::max(), 0};
    if (it != remote_workers_slot_.end()) {
      worker_key = it->second;
    }
    return InvokeRpc<RespBufferT>(
      worker_key, session_id, function_id, workflow_id, std::move(mem_policy),
      std::forward<Args>(args)...);
  }

  // Overload with default memory policy for convenience
  template <typename RespBufferT, typename... Args>
    requires(rpc::is_payload_v<RespBufferT>
             || std::is_same_v<RespBufferT, std::monostate>)
            && detail::is_high_level_api<Args...>::value
  auto InvokeRpc(
    std::string_view worker_name, rpc::session_id_t session_id,
    rpc::function_id_t function_id, rpc::utils::workflow_id_t workflow_id = {},
    Args&&... args) {
    return InvokeRpc<RespBufferT>(
      worker_name, session_id, function_id, workflow_id, AlwaysOnHostPolicy{},
      std::forward<Args>(args)...);
  }

  template <typename RespBufferT, typename MemPolicyT, typename... Args>
    requires(rpc::is_payload_v<RespBufferT>
             || std::is_same_v<RespBufferT, std::monostate>)
            && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
            && (!detail::first_arg_is_memory_policy<
                RespBufferT, Args...>::value)
            && detail::is_high_level_api<Args...>::value
  auto InvokeRpc(
    WorkerKey worker_key, rpc::session_id_t session_id,
    rpc::function_id_t function_id, rpc::utils::workflow_id_t workflow_id = {},
    MemPolicyT mem_policy = AlwaysOnHostPolicy{}, Args&&... args) {
    auto result = request_builder_.PrepareRequest(
      rpc::RpcRequestBuilderOptions{
        .session_id = session_id,
        .request_id = rpc::request_id_t{next_request_id_.fetch_add(1)},
        .function_id = function_id,
        .workflow_id = workflow_id,
      },
      std::forward<Args>(args)...);

    auto conn_id = GetConnectionId(worker_key);

    using ResultType = std::decay_t<decltype(result)>;
    if constexpr (std::is_same_v<ResultType, rpc::RpcRequestHeader>) {
      return InvokeRpcImpl<std::monostate, RespBufferT, MemPolicyT>(
        conn_id, std::move(result), std::monostate{}, std::move(mem_policy));
    } else {
      return InvokeRpcImpl<
        std::decay_t<decltype(result.second)>, RespBufferT, MemPolicyT>(
        conn_id, std::move(result.first), std::move(result.second),
        std::move(mem_policy));
    }
  }

  // Overload with default memory policy
  template <typename RespBufferT, typename... Args>
    requires(rpc::is_payload_v<RespBufferT>
             || std::is_same_v<RespBufferT, std::monostate>)
            && detail::is_high_level_api<Args...>::value
  auto InvokeRpc(
    WorkerKey worker_key, rpc::session_id_t session_id,
    rpc::function_id_t function_id, rpc::utils::workflow_id_t workflow_id = {},
    Args&&... args) {
    return InvokeRpc<RespBufferT>(
      worker_key, session_id, function_id, workflow_id, AlwaysOnHostPolicy{},
      std::forward<Args>(args)...);
  }

  // Dynamic API
  template <
    typename PayloadT = std::monostate, typename RespBufferT = std::monostate,
    typename MemPolicyT = AlwaysOnHostPolicy>
    requires(rpc::is_payload_v<PayloadT>
             || std::is_same_v<PayloadT, std::monostate>)
            && (rpc::is_payload_v<RespBufferT> || std::is_same_v<RespBufferT, std::monostate>)
            && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
            && detail::is_dynamic_api<PayloadT>::value
  auto InvokeRpc(
    std::string_view worker_name, rpc::RpcRequestHeader&& request_header,
    std::optional<PayloadT>&& payload = std::nullopt,
    MemPolicyT mem_policy = AlwaysOnHostPolicy{});

  template <
    typename PayloadT = std::monostate, typename RespBufferT = std::monostate,
    typename MemPolicyT = AlwaysOnHostPolicy>
    requires(rpc::is_payload_v<PayloadT>
             || std::is_same_v<PayloadT, std::monostate>)
            && (rpc::is_payload_v<RespBufferT> || std::is_same_v<RespBufferT, std::monostate>)
            && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
            && detail::is_dynamic_api<PayloadT>::value
            && (!std::is_same_v<
                std::decay_t<PayloadT>, std::optional<std::decay_t<PayloadT>>>)
  auto InvokeRpc(
    std::string_view worker_name, rpc::RpcRequestHeader&& request_header,
    PayloadT&& payload, MemPolicyT mem_policy = AlwaysOnHostPolicy{});

  template <
    typename PayloadT = std::monostate, typename RespBufferT = std::monostate,
    typename MemPolicyT = AlwaysOnHostPolicy>
    requires(rpc::is_payload_v<PayloadT>
             || std::is_same_v<PayloadT, std::monostate>)
            && (rpc::is_payload_v<RespBufferT> || std::is_same_v<RespBufferT, std::monostate>)
            && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
            && detail::is_dynamic_api<PayloadT>::value
  auto InvokeRpc(
    WorkerKey worker_key, rpc::RpcRequestHeader&& request_header,
    std::optional<PayloadT>&& payload = std::nullopt,
    MemPolicyT mem_policy = AlwaysOnHostPolicy{});

  template <
    typename PayloadT = std::monostate, typename RespBufferT = std::monostate,
    typename MemPolicyT = AlwaysOnHostPolicy>
    requires(rpc::is_payload_v<PayloadT>
             || std::is_same_v<PayloadT, std::monostate>)
            && (rpc::is_payload_v<RespBufferT> || std::is_same_v<RespBufferT, std::monostate>)
            && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
            && detail::is_dynamic_api<PayloadT>::value
            && (!std::is_same_v<
                std::decay_t<PayloadT>, std::optional<std::decay_t<PayloadT>>>)
  auto InvokeRpc(
    WorkerKey worker_key, rpc::RpcRequestHeader&& request_header,
    PayloadT&& payload, MemPolicyT mem_policy = AlwaysOnHostPolicy{});

  // --- RPC Server API ---
  template <
    typename ReceivedBufferT, typename Fn,
    typename MemPolicyT = AlwaysOnHostPolicy,
    typename MsgLcPolicyT = TransientPolicy>
    requires(((rpc::is_payload_v<ReceivedBufferT>
               || std::is_same_v<
                 ReceivedBufferT,
                 rpc::
                   PayloadVariant>)&&is_receiver_memory_policy_v<MemPolicyT, ReceivedBufferT>)
             || (std::is_same_v<ReceivedBufferT, std::monostate> && std::is_same_v<MemPolicyT, AlwaysOnHostPolicy>))
            && is_message_lifecycle_policy_v<MsgLcPolicyT>
  void RegisterFunction(
    rpc::function_id_t id, Fn&& fn,
    data::string&& function_name = data::string(),
    MemPolicyT mem_policy = AlwaysOnHostPolicy{},
    MsgLcPolicyT lc_policy = TransientPolicy{});

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
    MsgLcPolicyT lc_policy);

  template <typename ReceivedBufferT>
    requires(
      rpc::is_payload_v<ReceivedBufferT>
      || std::is_same_v<ReceivedBufferT, rpc::PayloadVariant>)
  void RegisterFunction(
    rpc::function_id_t id, const data::string& name,
    const data::vector<rpc::ParamType>& param_types,
    const data::vector<rpc::ParamType>& return_types,
    rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type,
    rpc::DynamicAsyncRpcFunction&& func) {
    RegisterFunction<ReceivedBufferT>(
      id, name, param_types, return_types, input_payload_type,
      return_payload_type, std::move(func), AlwaysOnHostPolicy{},
      TransientPolicy{});
  }

  template <typename ReceivedBufferT, typename MemPolicyT = AlwaysOnHostPolicy>
    requires(rpc::is_payload_v<ReceivedBufferT>
             || std::is_same_v<ReceivedBufferT, rpc::PayloadVariant>)
            && is_receiver_memory_policy_v<MemPolicyT, ReceivedBufferT>
  void RegisterFunction(
    rpc::function_id_t id, const data::string& name,
    const data::vector<rpc::ParamType>& param_types,
    const data::vector<rpc::ParamType>& return_types,
    rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type,
    rpc::DynamicAsyncRpcFunction&& func, MemPolicyT mem_policy) {
    RegisterFunction<ReceivedBufferT>(
      id, name, param_types, return_types, input_payload_type,
      return_payload_type, std::move(func), std::move(mem_policy),
      TransientPolicy{});
  }

  template <typename ReceivedBufferT, typename MsgLcPolicyT = TransientPolicy>
    requires(rpc::is_payload_v<ReceivedBufferT>
             || std::is_same_v<ReceivedBufferT, std::monostate>
             || std::is_same_v<ReceivedBufferT, rpc::PayloadVariant>)
            && is_message_lifecycle_policy_v<MsgLcPolicyT>
  void RegisterFunction(
    rpc::function_id_t id, const data::string& name,
    const data::vector<rpc::ParamType>& param_types,
    const data::vector<rpc::ParamType>& return_types,
    rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type,
    rpc::DynamicAsyncRpcFunction&& func, MsgLcPolicyT lc_policy) {
    RegisterFunction<ReceivedBufferT>(
      id, name, param_types, return_types, input_payload_type,
      return_payload_type, std::move(func), AlwaysOnHostPolicy{},
      std::move(lc_policy));
  }

  // --- Configuration ---
  void SetTimeout(std::chrono::milliseconds timeout);
  void SetRejectMessages(bool reject);
  template <typename Checker>
  void SetBypassRejectionFunction(Checker&& checker);
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
    AxonRequestID req_id, Func&& func, RequestVisitor visitor);
  std::expected<void, errors::AxonErrorContext> ProcessStoredRequests(
    AxonRequestID req_id, StorageIteratorVisitor visitor);
  template <typename Func>
    requires std::is_copy_constructible_v<Func>
  std::expected<void, errors::AxonErrorContext> ProcessStoredRequests(
    Func&& func, RequestVisitor visitor);
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
  static auto GetNowNanoseconds();
  static auto GetNowMicroseconds();
  static auto GetNowMilliseconds();

  // --- Memory Policy Helper Functions ---
  static ucxx::UcxBufferVec DefaultMemoryProvider(
    std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr_,
    const utils::TensorMetaSpan tensor_metas);

  static ucxx::UcxBuffer DefaultMemoryProvider(
    std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr_,
    const rpc::utils::TensorMeta& tensor_meta);

  std::reference_wrapper<ucxx::UcxMemoryResourceManager>
  GetMemoryResourceManager() const;

 private:
  using UcxSendCombinedSender = unifex::variant_sender<
    ucxx::ucx_am_context::send_sender_move,
    ucxx::ucx_am_context::send_iovec_sender_move,
    decltype(unifex::just_error(std::declval<std::error_code>()))>;
  using UcxRecvBufferCombinedSender = unifex::variant_sender<
    ucxx::ucx_am_context::recv_buffer_sender,
    ucxx::ucx_am_context::recv_iovec_buffer_sender>;
  using AnySender = unifex::any_sender_of<>;
  using RecvVariant = ucxx::recv_header_result_type;
  using BufferBundleVariant = std::variant<
    ucxx::active_message_buffer_bundle,
    ucxx::active_message_iovec_buffer_bundle>;
  using AnyBufferSender = unifex::any_sender_of<BufferBundleVariant>;
  using PayloadOrKey = std::variant<ucxx::UcxBuffer, uint64_t>;

  template <typename ErrorType>
  static constexpr bool kIsAxonErrorContext =
    std::is_same_v<ErrorType, errors::AxonErrorContext>;
  template <typename ErrorType>
  static constexpr bool kIsStdErrorCode =
    std::is_same_v<ErrorType, std::error_code>;
  template <typename ErrorType>
  static constexpr bool kIsStdExceptionPtr =
    std::is_same_v<ErrorType, std::exception_ptr>;

  // all-purpose error transformation helper function object
  struct TransformErrorToRpcException_ {
    template <typename ErrorT>
    auto operator()(ErrorT error) const noexcept
      -> decltype(unifex::just_error(std::declval<std::exception_ptr>()));
  };

  // ----------------------------------
  // ----------------------------------
  // ----- Facades for pro::proxy -----
  // ----------------------------------
  // ----------------------------------

  // ----------------------------------
  // --- Facades for Server Context ---
  // ----------------------------------

  struct BufferBundleProcessor_ {
    AxonWorker* worker;
    uint64_t conn_id;
    const rpc::RpcRequestHeader* req_header_ptr;
    MessageLifecyclePolicy lc_policy_variant;

    auto operator()(auto&& buffer_bundle) -> AnySender;
  };

  using RequestHandlerRndvReturnIovType =  //
    decltype(unifex::on(std::declval<ucxx::ucx_am_context::scheduler>(),  //
      std::declval<ucxx::ucx_am_context::recv_iovec_buffer_sender>())  //
             | unifex::let_value(std::declval<BufferBundleProcessor_>()));

  using RequestHandlerRndvReturnBufferType =   //
    decltype(unifex::on(std::declval<ucxx::ucx_am_context::scheduler>(),  //
      std::declval<ucxx::ucx_am_context::recv_buffer_sender>())  //
             | unifex::let_value(std::declval<BufferBundleProcessor_>()));

  using RequestHandlerEagerReturnType = decltype(std::declval<AnySender>());

  using RequestHandlerReturnType = unifex::variant_sender<
    RequestHandlerRndvReturnIovType, RequestHandlerRndvReturnBufferType,
    RequestHandlerEagerReturnType,
    decltype(unifex::just_error(std::declval<errors::AxonErrorContext>()))>;

  /*
  TODO(He Jia): Not sure variant_sender could be more efficient than
  any_sender_of<> version, comparing performance.
  */
  // template <typename MsgLcPolicyT>
  // struct BufferBundleProcessor_ {
  //   AxonWorker* worker;
  //   uint64_t conn_id;
  //   const rpc::RpcRequestHeader* req_header_ptr;
  //   MsgLcPolicyT lc_policy;

  //   template <typename BufferBundle>
  //   auto operator()(BufferBundle&& buffer_bundle) const {
  //     return worker->ServerProcessBufferBundle_(
  //       conn_id, req_header_ptr, std::move(buffer_bundle),
  //       std::move(lc_policy));
  //   }
  // };
  //
  // using RequestHandlerReturnType = AnySender;

  /**
   * @brief Handler signatures for full request dispatch.
   *
   * There are two primary handler signatures depending on the protocol:
   *
   * - Eager protocol: the payload is passed as a `ucxx::UcxBuffer`.
   * - RNDV protocol: the payload is passed as an `am_desc_key` (uint64_t).
   *
   * @param conn_id        The connection identifier.
   * @param req_header_ptr Pointer to the RPC request header.
   * @param payload        For Eager protocol: `ucxx::UcxBuffer` (the message
   * body). For RNDV protocol: `am_desc_key` (uint64_t).
   *
   * 1. <tt>AnySender(uint64_t conn_id, const rpc::RpcRequestHeader*
   * req_header_ptr, ucxx::UcxBuffer&& payload)</tt>
   *    - Eager protocol: payload is a ucxx::UcxBuffer.
   *
   * 2. <tt>AnySender(uint64_t conn_id, const rpc::RpcRequestHeader*
   * req_header_ptr, uint64_t am_desc_key)</tt>
   *    - RNDV protocol: payload is am_desc_key (uint64_t).
   */
  struct FullRequestHandlerFacade
    : pro::facade_builder                //
      ::add_skill<pro::skills::as_view>  //
      ::add_convention<
        pro::operator_dispatch<"()">,
        RequestHandlerReturnType(
          /*conn_id=*/uint64_t, const rpc::RpcRequestHeader*,
          ucxx::UcxBuffer&&)>  //
      ::add_convention<
        pro::operator_dispatch<"()">,
        RequestHandlerReturnType(
          /*conn_id=*/uint64_t, const rpc::RpcRequestHeader*,
          /*am_desc_key=*/size_t)>  //
      ::build {};

  using FullRequestHandler = pro::proxy<FullRequestHandlerFacade>;
  using FullRequestHandlerView = pro::proxy_view<FullRequestHandlerFacade>;

  // ----------------------------------
  // --- Facade for Bypass Backpressure Checker ---
  // ----------------------------------

  struct BypassRejectionCheckerFacade
    : pro::facade_builder  //
      ::add_convention<
        pro::operator_dispatch<"()">,
        bool(rpc::function_id_t)>  // NOLINT(readability/casting)
      ::build {};

  using BypassRejectionChecker = pro::proxy<BypassRejectionCheckerFacade>;

  // ----------------------------------
  // --- Facades for Client Context ---
  // ----------------------------------

  // Facade for the type-erased response handler.

  // // Build the reflector for the response buffer type.
  // class RespBufferTypeReflector {
  //  public:
  //   template <class T>
  //   constexpr explicit RespBufferTypeReflector(std::in_place_type_t<T>)
  //     : type_(ResolveType<T>()) {}

  //   template <class P, class R>
  //   struct accessor {
  //     rpc::PayloadType GetDataType() const noexcept {
  //       const auto& self = pro::proxy_reflect<R>(static_cast<const
  //       P&>(*this)); return self.type_;
  //     }
  //   };

  //  private:
  //   [[maybe_unused]] rpc::PayloadType type_;

  //   template <class T>
  //   static constexpr rpc::PayloadType ResolveType() {
  //     static constexpr bool kIsInvocable1 = std::is_invocable_v<
  //       T, rpc::ResponseHeaderUniquePtr&&, ucxx::UcxBuffer&&>;
  //     static constexpr bool kIsInvocable2 = std::is_invocable_v<
  //       T, rpc::ResponseHeaderUniquePtr&&, ucxx::UcxBufferVec&&>;
  //     static constexpr bool kIsMonostate = std::is_invocable_v<
  //       T, rpc::ResponseHeaderUniquePtr&&, std::monostate>;
  //     if constexpr (kIsInvocable1) {
  //       return rpc::PayloadType::UCX_BUFFER;
  //     } else if constexpr (kIsInvocable2) {
  //       return rpc::PayloadType::UCX_BUFFER_VEC;
  //     } else if constexpr (kIsMonostate) {
  //       return rpc::PayloadType::MONOSTATE;
  //     } else {
  //       return rpc::PayloadType::NO_PAYLOAD;
  //     }
  //   }
  // };

  // Build the facade for the response handler.
  using RpcResponseHandlerInputType =
    std::pair<rpc::ResponseHeaderUniquePtr, PayloadOrKey>;
  using RpcResponseHandlerExpectedType =
    std::expected<RpcResponseHandlerInputType, errors::AxonErrorContext>;
  struct RpcResponseHandlerFacade : pro::facade_builder                   //
                                    ::add_convention<                     //
                                      pro::operator_dispatch<"()">,       //
                                      void(                               //
                                        RpcResponseHandlerExpectedType)>  //
                                    ::build {};

  using RpcResponseHandler = pro::proxy<RpcResponseHandlerFacade>;
  using RpcResponseHandlerView = pro::proxy_view<RpcResponseHandlerFacade>;

  // High-performance ring buffer for pending RPCs (size must be power of 2)
  // Uses request_id as direct index, session_id for validation
  static constexpr size_t kPendingRpcBufferSize = 2048;
  using PendingRpcBuffer = utils::RingBuffer<
    rpc::session_id_t, RpcResponseHandler, kPendingRpcBufferSize>;

 private:
  // ╔════════════════════════════════════════════════════════════════════╗
  // ║                       Receiving Context Loop                       ║
  // ╚════════════════════════════════════════════════════════════════════╝

  // --- ServerMessageHandlerSender Type---
  using WorkerScheduler =
    decltype(std::declval<ucxx::ucx_am_context>().get_scheduler());

  template <typename PayloadT>
  struct _ProcessValidMessageHelper;

  struct ReturnTimeoutErrorContextHelper_ {
    mutable errors::AxonErrorContext error_ctx;
    auto operator()() const noexcept
      -> decltype(unifex::just_error(std::declval<errors::AxonErrorContext>()));

    auto operator()(auto&& /* ignored_error */) const noexcept
      -> decltype(unifex::just_error(std::declval<errors::AxonErrorContext>()));
  };

  struct RethrowErrorContextHelper_ {
    mutable errors::AxonErrorContext error_ctx;

    template <typename ErrorT>
    auto operator()(ErrorT error) const noexcept
      -> decltype(unifex::just_error(std::declval<errors::AxonErrorContext>()));
  };

  using ServerProcessValidMessageTimeoutSenderType =                         //
    decltype(unifex::stop_when(                                              //
      unifex::on(                                                            //
        std::declval<unifex::static_thread_pool>().get_scheduler(),          //
        std::declval<RequestHandlerReturnType>()                             //
          | unifex::let_error(std::declval<RethrowErrorContextHelper_>())),  //
      unifex::on(                                                            //
        std::declval<unifex::timed_single_thread_context>().get_scheduler(),
        unifex::schedule_after(std::declval<std::chrono::milliseconds>()))));

  using ServerProcessValidMessageSenderType =  //
    decltype(std::declval<ServerProcessValidMessageTimeoutSenderType>()  //
    | unifex::let_done(std::declval<ReturnTimeoutErrorContextHelper_>()));

  // Helper functions for error reporting
  void ReportServerError_(const errors::AxonErrorContext& ctx) noexcept;
  void ReportClientError_(const errors::AxonErrorContext& ctx) noexcept;

  // Helper functions for ProcessStoredRequests
  template <typename Func>
  Executor ServerMakeProcessStorageExecutor_(
    StorageRequestIt storage_it, Func&& func);
  std::expected<void, errors::AxonErrorContext> ProcessLifecycleStatus_(
    StorageRequestIt storage_it, LifecycleStatus lifecycle_status);

  UcxSendCombinedSender ServerHandleSendResponse_(
    uint64_t conn_id, const rpc::RpcResponseHeader& resp_header,
    const rpc::ReturnedPayload& resp_payload);
  auto ServerHandleSendErrorResponse_(
    uint64_t conn_id, const rpc::RpcRequestHeader& request_header,
    std::error_code ec);
  auto ServerHandleSendErrorResponse_(
    const errors::AxonErrorContext& error_ctx);

  template <typename PayloadT, typename MsgLcPolicyT>
    requires(std::is_same_v<PayloadT, rpc::PayloadVariant>
             || rpc::is_payload_v<PayloadT>)
            && is_message_lifecycle_policy_v<MsgLcPolicyT>
  AnySender ServerDispatchAndManageLifecycle_(
    uint64_t conn_id, const rpc::RpcRequestHeader* req_header_ptr,
    PayloadT&& payload, MsgLcPolicyT lc_policy);

  // Helper functions for processing buffer bundles
  template <typename MsgLcPolicyT>
  AnySender ServerHandleVariantBufferBundle_(
    uint64_t conn_id, const rpc::RpcRequestHeader* req_header_ptr,
    BufferBundleVariant&& buffer_bundle, MsgLcPolicyT lc_policy);

  template <typename BundleType, typename MsgLcPolicyT>
  AnySender ServerHandleConcreteBufferBundle_(
    uint64_t conn_id, const rpc::RpcRequestHeader* req_header_ptr,
    BundleType&& buffer_bundle, MsgLcPolicyT lc_policy);

  template <typename BundleType, typename MsgLcPolicyT>
  AnySender ServerProcessBufferBundle_(
    uint64_t conn_id, const rpc::RpcRequestHeader* req_header_ptr,
    BundleType&& buffer_bundle, MsgLcPolicyT lc_policy);

  template <typename RequestHandlerInputT>
  auto ServerProcessValidMessage_(
    FullRequestHandlerView handler, uint64_t conn_id,
    const rpc::RpcRequestHeader* req_header_ptr,
    RequestHandlerInputT payload_or_key) -> ServerProcessValidMessageSenderType;

  using ServerMessageHandlerSender = unifex::variant_sender<
    ServerProcessValidMessageSenderType,
    decltype(unifex::just_error(std::declval<errors::AxonErrorContext>()))>;

  template <typename ReceivedBufferT, typename MemPolicyT>
    requires std::is_same_v<ReceivedBufferT, rpc::PayloadVariant>
             && is_receiver_memory_policy_v<MemPolicyT, ReceivedBufferT>
  UcxRecvBufferCombinedSender ProcessRndvBuffer_(
    WorkerScheduler scheduler, uint64_t am_desc_key, MemPolicyT mem_policy,
    const utils::TensorMetaSpan tensor_metas);

  template <typename ReceivedBufferT, typename MemPolicyT>
    requires std::is_same_v<ReceivedBufferT, ucxx::UcxBuffer>
             && (is_receiver_memory_policy_v<MemPolicyT, ReceivedBufferT> || is_receiver_memory_policy_v<MemPolicyT, rpc::PayloadVariant>)
  auto ProcessRndvBuffer_(
    WorkerScheduler scheduler, uint64_t am_desc_key, MemPolicyT mem_policy,
    const utils::TensorMetaSpan tensor_metas)
    -> ucxx::ucx_am_context::recv_buffer_sender;

  template <typename ReceivedBufferT, typename MemPolicyT>
    requires std::is_same_v<ReceivedBufferT, ucxx::UcxBufferVec>
             && (is_receiver_memory_policy_v<MemPolicyT, ReceivedBufferT> || is_receiver_memory_policy_v<MemPolicyT, rpc::PayloadVariant>)
  auto ProcessRndvBuffer_(
    WorkerScheduler scheduler, uint64_t am_desc_key, MemPolicyT mem_policy,
    const utils::TensorMetaSpan tensor_metas)
    -> ucxx::ucx_am_context::recv_iovec_buffer_sender;

  template <
    typename ReceivedBufferT, typename MemPolicyT, typename MsgLcPolicyT>
  struct ServerRequestHandlerVisitor_;

  template <
    typename ReceivedBufferT, typename MemPolicyT, typename MsgLcPolicyT>
  pro::proxy<FullRequestHandlerFacade> ServerMakeFullRequestHandler_(
    MemPolicyT mem_policy, MsgLcPolicyT lc_policy);

  std::expected<
    std::pair<const rpc::RpcRequestHeader*, AxonWorker::FullRequestHandlerView>,
    errors::AxonErrorContext>
  ServerProcessHeader_(uint64_t conn_id, std::string_view header_bytes);

  auto ServerProcessMessage_(const RecvVariant& message) noexcept;

  auto ServerCtxLoop_(
    std::reference_wrapper<unifex::inplace_stop_source> stop_source);
  auto ServerCtxLoopImpl_(
    std::reference_wrapper<unifex::inplace_stop_source> stop_source);
  void ServerCtxLoopSpawnImpl_(
    std::reference_wrapper<unifex::inplace_stop_source> stop_source,
    RecvVariant&& header_or_bundle);

  // ╔════════════════════════════════════════════════════════════════════╗
  // ║                        Sending Context Loop                        ║
  // ╚════════════════════════════════════════════════════════════════════╝

  template <typename PayloadT>
  UcxSendCombinedSender ClientSendRequest_(
    std::expected<uint64_t, std::error_code> conn_id,
    const cista::byte_buf& header_buf, const PayloadT& payload);

  template <typename RespBufferT, typename MemPolicyT>
  struct ClientResponseHandlerVisitor_;

  template <typename RespBufferT, typename MemPolicyT>
  struct ResponseHandlerRndvReturnType {
    mutable rpc::ResponseHeaderUniquePtr resp_header_ptr;
    auto operator()(RespBufferT&& payload) const
      -> std::pair<rpc::ResponseHeaderUniquePtr, RespBufferT> {
      return std::make_pair(std::move(resp_header_ptr), std::move(payload));
    }
  };

  template <typename RespBufferT>
  struct MoveBundleBuffer {
    auto operator()(
      ucxx::active_message_buffer_bundle_t<RespBufferT>&& recv_bundle) const
      -> RespBufferT {
      return recv_bundle.move_buffer();
    }
  };  // namespace axon

  template <typename RespBufferT, typename MemPolicyT>
 using ResponseHandlerRndvProcessReturnType = std::conditional_t< //
  std::is_same_v<RespBufferT, std::monostate>, //
  decltype(unifex::just_error(std::declval<errors::AxonErrorContext>())), //
  decltype(unifex::on( //
      std::declval<ucxx::ucx_am_context::scheduler>(), //
      std::declval<ucxx::ucx_am_context::recv_buffer_sender_t<RespBufferT>>()) //
    | unifex::then(std::declval<MoveBundleBuffer<RespBufferT>>()) //
    | unifex::then(std::declval<ResponseHandlerRndvReturnType<RespBufferT, MemPolicyT>>()))>;

  template <typename RespBufferT, typename MemPolicyT>
  using ResponseHandlerReturnType = unifex::variant_sender<
    ResponseHandlerRndvProcessReturnType<RespBufferT, MemPolicyT>,
    decltype(unifex::just(
      std::declval<std::pair<rpc::ResponseHeaderUniquePtr, RespBufferT>>())),
    decltype(unifex::just_error(std::declval<errors::AxonErrorContext>()))>;

  void ClientProcessExpectedResponseMessage_(
    RpcResponseHandler&& resp_handler,
    RpcResponseHandlerExpectedType&& resp_handler_expected);

  std::optional<RpcResponseHandler> PopPendingRpcResponseHandler_(
    rpc::request_id_t request_id, rpc::session_id_t session_id);

  std::expected<
    std::pair<rpc::ResponseHeaderUniquePtr, RpcResponseHandler>,
    errors::AxonErrorContext>
  ClientProcessHeader_(ucxx::UcxHeader&& header);

  auto ClientProcessResponseMessage_(RecvVariant&& message) noexcept;

  auto ClientCtxLoop_(
    std::reference_wrapper<unifex::inplace_stop_source> stop_source);
  auto ClientCtxLoopImpl_();

 private:
  // Helper for detecting optional
  template <typename T>
  struct IsOptional : std::false_type {};
  template <typename T>
  struct IsOptional<std::optional<T>> : std::true_type {};
  template <typename T>
  struct UnwrapOptional {
    using type = T;
  };
  template <typename T>
  struct UnwrapOptional<std::optional<T>> {
    using type = T;
  };

  [[nodiscard]] std::expected<uint64_t, std::error_code> GetConnectionId(
    WorkerKey worker_key) const noexcept {
    const auto* worker_info = remote_workers_.access_lockless(worker_key);
    if (!worker_info) [[unlikely]] {
      return std::unexpected(
        std::make_error_code(errors::AxonErrc::WorkerNotFound));
    }
    if (!worker_info->conn_id.has_value()) [[unlikely]] {
      return std::unexpected(
        std::make_error_code(errors::AxonErrc::ConnectError));
    }
    return worker_info->conn_id.value();
  }

  // --- Sender Implementation ---
  template <typename PayloadT, typename RespBufferT, typename MemPolicyT>
    requires(rpc::is_payload_v<PayloadT>
             || std::is_same_v<PayloadT, std::monostate>)
            && (rpc::is_payload_v<RespBufferT>  //
             || std::is_same_v<RespBufferT, std::monostate>)
            && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
  inline auto InvokeRpcImpl(
    std::expected<uint64_t, std::error_code> conn_id,
    rpc::RpcRequestHeader&& request_header,
    PayloadT&& payload,
    MemPolicyT mem_policy) {
    // Setup receiver
    auto recv_callback_sender =
      unifex::create<rpc::ResponseHeaderUniquePtr, PayloadOrKey>(
        [this, conn_id, request_id = request_header.request_id,
         session_id = request_header.session_id,
         function_id = request_header.function_id,
         workflow_id = request_header.workflow_id](auto& receiver) {
          // Check if stop is requested before registering callback
          const bool is_stop_requested =
            unifex::get_stop_token(receiver).stop_requested();
          if (is_stop_requested) {
            receiver.set_done();
            return;
          }

          auto callback =
            [&receiver](
              RpcResponseHandlerExpectedType&& resp_handler_expected) mutable {
              const bool is_stop =
                unifex::get_stop_token(receiver).stop_requested();
              if (is_stop) [[unlikely]] {
                receiver.set_done();
                return;
              }
              if (!resp_handler_expected.has_value()) {
                receiver.set_error(errors::MakeExceptionPtr(
                  static_cast<errors::AxonErrorContext&&>(
                    resp_handler_expected.error())));
                return;
              }
              auto&& [resp_header_ptr, payload_or_key] =
                std::move(resp_handler_expected.value());
              if (std::error_code(resp_header_ptr->status)) [[unlikely]] {
                // Extract error message from response results if available
                std::string error_msg =
                  resp_header_ptr->status.GetErrorMessage();
                if (
                  !resp_header_ptr->results.empty()
                  && resp_header_ptr->results[0].type
                       == rpc::ParamType::STRING) {
                  const auto& error_str =
                    cista::get<data::string>(resp_header_ptr->results[0].value);
                  if (!error_str.empty()) {
                    error_msg = std::string(error_str.data(), error_str.size());
                  }
                }
                receiver.set_error(std::make_exception_ptr(rpc::RpcException(
                  std::error_code(resp_header_ptr->status), error_msg)));
                return;
              }
              receiver.set_value(
                std::move(resp_header_ptr), std::move(payload_or_key));
            };

          bool inserted = this->pending_rpcs_.emplace(
            cista::to_idx(request_id), session_id,
            pro::make_proxy<RpcResponseHandlerFacade>(std::move(callback)));
          if (!inserted) {
            receiver.set_error(
              errors::MakeExceptionPtr(errors::AxonErrorContext{
                .conn_id = conn_id.value_or(0),
                .session_id = cista::to_idx(session_id),
                .request_id = cista::to_idx(request_id),
                .function_id = cista::to_idx(function_id),
                .status = rpc::RpcStatus(
                  std::make_error_code(rpc::RpcErrc::RESOURCE_EXHAUSTED)),
                .what = "Client pending RPC ring buffer is full",
                .hlc = this->hlc_,
                .workflow_id = cista::to_idx(workflow_id),
              }));
            return;
          }
        })  //
      | unifex::let_value(
        [this, mem_policy = std::move(mem_policy)](
          auto&& resp_header_ptr, auto&& payload_or_key) mutable {
          return std::visit(
            ClientResponseHandlerVisitor_<RespBufferT, MemPolicyT>{
              this, std::move(mem_policy), std::move(resp_header_ptr)},
            std::move(payload_or_key));
        });

    auto error_ctx = errors::AxonErrorContext{
      .conn_id = conn_id.value_or(0),
      .session_id = cista::to_idx(request_header.session_id),
      .request_id = cista::to_idx(request_header.request_id),
      .function_id = cista::to_idx(request_header.function_id),
      .status =
        rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::DEADLINE_EXCEEDED)),
      .what = std::format(
        "Deadline exceeded {} milliseconds   when executing "
        "registered function: {}",
        timeout_.count(), cista::to_idx(request_header.function_id)),
      .hlc = this->hlc_,
      .workflow_id = cista::to_idx(request_header.workflow_id),
    };

    // Create timeout sender that produces an error when it completes
    auto timeout_sender =
      unifex::on(GetTimeContextScheduler(), unifex::schedule_after(timeout_))
      | unifex::then([this, error_ctx, request_id = request_header.request_id,
                      session_id = request_header.session_id]() {
          auto resp_handler_opt =
            this->PopPendingRpcResponseHandler_(request_id, session_id);
          if (resp_handler_opt.has_value()) {
            (*resp_handler_opt.value())(std::unexpected(std::move(error_ctx)));
          }
          return;
        });

    // Setup sending sender
    auto send_sender =
      unifex::just(std::move(request_header), std::move(payload))
      | unifex::then([](auto&& request_header, auto&& payload) {
          auto header_buf =
            cista::serialize<rpc::utils::SerializerMode>(request_header);
          return std::tuple(std::move(header_buf), std::move(payload));
        })
      | unifex::let_value(
        [this, conn_id](const auto& header_and_buf_and_payload) {
          const auto& [header_buf, payload] = header_and_buf_and_payload;
          auto send_sender =
            this->ClientSendRequest_<PayloadT>(conn_id, header_buf, payload);
          // Capture header_buf_size by value to avoid dangling reference
          return std::move(send_sender)
                 | unifex::then([&header_buf, &payload](auto&&... args) {
                     return unifex::just(std::forward<decltype(args)>(args)...);
                   })
                 | unifex::let_error([](auto&& error) {
                     return unifex::just_error(
                       std::forward<decltype(error)>(error));
                   });
        });

    return unifex::stop_when(
             unifex::when_all(
               std::move(send_sender),
               std::move(recv_callback_sender))  //
             ,
             std::move(timeout_sender))
           | unifex::then([](auto&& send_result, auto&& recv_result) {
               return std::get<0>(std::get<0>(std::move(recv_result)));
             })
           | unifex::let_error([error_ctx](auto&& error) {
               return RethrowErrorContextHelper_{.error_ctx = error_ctx}(
                 std::forward<decltype(error)>(error));
             })
           | unifex::let_done([error_ctx](auto&&...) {
               return ReturnTimeoutErrorContextHelper_{
                 .error_ctx = error_ctx}();
             });
  }

 private:
  // A map to store response receivers for pending RPC calls
  // Lock-free: uses request_id as direct index, different request_ids
  // map to different slots, so no locking needed
  PendingRpcBuffer pending_rpcs_;
  std::atomic<uint32_t> next_request_id_{1};  // Never 0
  // If true, reject the messages by returning Backpressure error response
  std::atomic<bool> reject_messages_{false};
  BypassRejectionChecker bypass_backpressure_checker_;

  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr_;
  std::string worker_name_;
  utils::hash_t worker_id_;
  std::chrono::milliseconds timeout_;

  rpc::utils::HybridLogicalClock hlc_;

  std::unique_ptr<ucxx::ucx_am_context> server_ctx_;
  std::unique_ptr<ucxx::ucx_am_context> client_ctx_;
  ucp_context_h common_ucp_context_{nullptr};
  std::unique_ptr<ucxx::UcxAutoDeviceContext> auto_device_context_;
  std::unique_ptr<storage::AxonStorage> storage_;

  rpc::AsyncRpcDispatcher dispatcher_;
  rpc::RpcRequestBuilder request_builder_;
  rpc::RpcResponseBuilder response_builder_;

  std::thread server_thread_;
  std::thread client_thread_;
  unifex::static_thread_pool thread_pool_;
  unifex::timed_single_thread_context time_ctx_;
  unifex::inplace_stop_source server_stop_source_;
  unifex::inplace_stop_source client_stop_source_;
  unifex::inplace_stop_source server_context_stop_source_;
  unifex::inplace_stop_source client_context_stop_source_;
  unifex::v2::async_scope server_async_scope_;
  unifex::v2::async_scope client_async_scope_;
  std::atomic<bool> server_running_{false};
  std::atomic<bool> client_running_{false};
  std::atomic<bool> stopping_{false};
  std::mutex stop_mutex_;
  std::atomic<bool> server_stopped_{false};
  std::atomic<bool> client_stopped_{false};
  std::optional<unifex::v2::future<unifex::v2::async_scope>> server_future_;
  std::optional<unifex::v2::future<unifex::v2::async_scope>> client_future_;

  errors::ErrorObserver server_err_observer_{};
  errors::ErrorObserver client_err_observer_{};
  metrics::MetricsObserver server_metrics_observer_{};
  metrics::MetricsObserver client_metrics_observer_{};
  size_t client_pipe_failed_count_{0};
  size_t server_pipe_invoked_count_{0};
  size_t server_pipe_failed_count_{0};

  cista::raw::hash_map<rpc::function_id_t, FullRequestHandler>
    registered_handlers_;
  cista::raw::hash_map<AxonRequestID, StorageRequestIt> request_id_to_iterator_;

  utils::ThreadSafeSlotMap<RemoteEndpoint>
    remote_workers_;  // remote_workers_[worker_key] = worker info
  cista::raw::hash_map<std::string, WorkerKey> remote_workers_slot_;
};  // namespace eux

template <typename PayloadT, typename RespBufferT, typename MemPolicyT>
  requires(rpc::is_payload_v<PayloadT>
           || std::is_same_v<PayloadT, std::monostate>)
          && (rpc::is_payload_v<RespBufferT> || std::is_same_v<RespBufferT, std::monostate>)
          && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
          && detail::is_dynamic_api<PayloadT>::value
auto AxonWorker::InvokeRpc(
  std::string_view worker_name, rpc::RpcRequestHeader&& request_header,
  std::optional<PayloadT>&& payload, MemPolicyT mem_policy) {
  auto it = remote_workers_slot_.find(worker_name);
  if (it == remote_workers_slot_.end()) {
    return InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(
      WorkerKey{}, std::move(request_header), std::move(payload),
      std::move(mem_policy));
  }
  return InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(
    it->second, std::move(request_header), std::move(payload),
    std::move(mem_policy));
}

template <typename PayloadT, typename RespBufferT, typename MemPolicyT>
  requires(rpc::is_payload_v<PayloadT>
           || std::is_same_v<PayloadT, std::monostate>)
          && (rpc::is_payload_v<RespBufferT> || std::is_same_v<RespBufferT, std::monostate>)
          && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
          && detail::is_dynamic_api<PayloadT>::value
          && (!std::is_same_v<
              std::decay_t<PayloadT>, std::optional<std::decay_t<PayloadT>>>)
auto AxonWorker::InvokeRpc(
  std::string_view worker_name, rpc::RpcRequestHeader&& request_header,
  PayloadT&& payload, MemPolicyT mem_policy) {
  auto it = remote_workers_slot_.find(worker_name);
  if (it == remote_workers_slot_.end()) {
    return InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(
      WorkerKey{}, std::move(request_header), std::move(payload),
      std::move(mem_policy));
  }
  return InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(
    it->second, std::move(request_header), std::move(payload),
    std::move(mem_policy));
}

template <typename PayloadT, typename RespBufferT, typename MemPolicyT>
  requires(rpc::is_payload_v<PayloadT>
           || std::is_same_v<PayloadT, std::monostate>)
          && (rpc::is_payload_v<RespBufferT> || std::is_same_v<RespBufferT, std::monostate>)
          && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
          && detail::is_dynamic_api<PayloadT>::value
auto AxonWorker::InvokeRpc(
  WorkerKey worker_key, rpc::RpcRequestHeader&& request_header,
  std::optional<PayloadT>&& payload, MemPolicyT mem_policy) {
  auto conn_id = GetConnectionId(worker_key);

  request_header.request_id = rpc::request_id_t{next_request_id_.fetch_add(1)};

  if constexpr (std::is_same_v<PayloadT, ucxx::UcxBuffer>) {
    return InvokeRpcImpl<PayloadT, RespBufferT, MemPolicyT>(
      conn_id, std::move(request_header),
      std::move(payload).value_or(PayloadT{mr_, ucx_memory_type::HOST, 0}),
      std::move(mem_policy));
  } else if constexpr (std::is_same_v<PayloadT, ucxx::UcxBufferVec>) {
    return InvokeRpcImpl<PayloadT, RespBufferT, MemPolicyT>(
      conn_id, std::move(request_header),
      std::move(payload).value_or(
        PayloadT{mr_, ucx_memory_type::HOST, std::vector<size_t>{0}}),
      std::move(mem_policy));
  } else {
    return InvokeRpcImpl<std::monostate, RespBufferT, MemPolicyT>(
      conn_id, std::move(request_header), std::monostate{},
      std::move(mem_policy));
  }
}

template <typename PayloadT, typename RespBufferT, typename MemPolicyT>
  requires(rpc::is_payload_v<PayloadT>
           || std::is_same_v<PayloadT, std::monostate>)
          && (rpc::is_payload_v<RespBufferT> || std::is_same_v<RespBufferT, std::monostate>)
          && is_receiver_memory_policy_v<MemPolicyT, RespBufferT>
          && detail::is_dynamic_api<PayloadT>::value
          && (!std::is_same_v<
              std::decay_t<PayloadT>, std::optional<std::decay_t<PayloadT>>>)
auto AxonWorker::InvokeRpc(
  WorkerKey worker_key, rpc::RpcRequestHeader&& request_header,
  PayloadT&& payload, MemPolicyT mem_policy) {
  auto conn_id = GetConnectionId(worker_key);

  request_header.request_id = rpc::request_id_t{next_request_id_.fetch_add(1)};

  return InvokeRpcImpl<PayloadT, RespBufferT, MemPolicyT>(
    conn_id, std::move(request_header), std::move(payload),
    std::move(mem_policy));
}

template <typename ReceivedBufferT, typename MemPolicyT, typename MsgLcPolicyT>
pro::proxy<AxonWorker::FullRequestHandlerFacade>
AxonWorker::ServerMakeFullRequestHandler_(
  MemPolicyT mem_policy, MsgLcPolicyT lc_policy) {
  return pro::make_proxy<FullRequestHandlerFacade>(
    ServerRequestHandlerVisitor_<ReceivedBufferT, MemPolicyT, MsgLcPolicyT>{
      this, std::move(mem_policy), std::move(lc_policy)});
}

template <typename ReceivedBufferT, typename MemPolicyT, typename MsgLcPolicyT>
struct AxonWorker::ServerRequestHandlerVisitor_ {
  AxonWorker* worker;
  mutable MemPolicyT mem_policy;
  mutable MsgLcPolicyT lc_policy;

  auto operator()(
    uint64_t conn_id, const rpc::RpcRequestHeader* req_header_ptr,
    size_t am_desc_key) const -> RequestHandlerReturnType {
    // RNDV Path
    auto tensor_metas = utils::GetTensorMetas(req_header_ptr->params);
    if (tensor_metas.empty()) {
      if (
        auto am_desc_opt =
          worker->server_ctx_->view_pending_am_desc(am_desc_key)) {
        auto& am_desc = am_desc_opt.value().get();
        if (am_desc.data_length > 0) {
          // Fallback: if no tensor metas, but data is expected, create a
          // single tensor meta for the whole data (thread_local for safety).
          thread_local rpc::TensorMeta fallback_meta;
          fallback_meta.dtype = DLDataType{
            .code = DLDataTypeCode::kDLOpaqueHandle, .bits = 8, .lanes = 1};
          fallback_meta.shape = {static_cast<int64_t>(am_desc.data_length)};
          tensor_metas = utils::TensorMetaSpan{&fallback_meta, 1};
        }
      }
    }

    if (tensor_metas.empty()) [[unlikely]] {
      return unifex::just_error(errors::AxonErrorContext{
        .conn_id = conn_id,
        .session_id = cista::to_idx(req_header_ptr->session_id),
        .request_id = cista::to_idx(req_header_ptr->request_id),
        .function_id = cista::to_idx(req_header_ptr->function_id),
        .status = rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::INTERNAL)),
        .what = "RNDV request received but no tensor metas found in request "
                "header and no data expected.",
        .hlc = worker->hlc_,
        .workflow_id = cista::to_idx(req_header_ptr->workflow_id),
      });
    }

    auto run_pipeline = [&]<typename BufferT>() {
      return unifex::on(
               worker->server_ctx_->get_scheduler(),
               worker->ProcessRndvBuffer_<BufferT, MemPolicyT>(
                 worker->server_ctx_->get_scheduler(), am_desc_key,
                 std::move(mem_policy), tensor_metas))
             | unifex::let_value(BufferBundleProcessor_{
               worker, conn_id, req_header_ptr, std::move(lc_policy)});
    };

    // std::monostate is not a valid buffer type when RNDV Path
    if constexpr (std::is_same_v<ReceivedBufferT, std::monostate>) {
      return unifex::just_error(errors::AxonErrorContext{
        .conn_id = conn_id,
        .session_id = cista::to_idx(req_header_ptr->session_id),
        .request_id = cista::to_idx(req_header_ptr->request_id),
        .function_id = cista::to_idx(req_header_ptr->function_id),
        .status =
          rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::INVALID_ARGUMENT)),
        .what = "Received RNDV request for function registered with "
                "std::monostate (no payload)",
        .hlc = worker->hlc_,
        .workflow_id = cista::to_idx(req_header_ptr->workflow_id),
      });
    } else if constexpr (std::is_same_v<ReceivedBufferT, rpc::PayloadVariant>) {
      if (tensor_metas.size() == 1) {
        return run_pipeline.template operator()<ucxx::UcxBuffer>();
      } else {
        return run_pipeline.template operator()<ucxx::UcxBufferVec>();
      }
    } else {
      return run_pipeline.template operator()<ReceivedBufferT>();
    }
  }

  auto operator()(
    uint64_t conn_id, const rpc::RpcRequestHeader* req_header_ptr,
    ucxx::UcxBuffer&& payload) const -> RequestHandlerReturnType {
    // Eager Path
    if constexpr (std::is_same_v<ReceivedBufferT, std::monostate>) {
      return unifex::just_error(errors::AxonErrorContext{
        .conn_id = conn_id,
        .session_id = cista::to_idx(req_header_ptr->session_id),
        .request_id = cista::to_idx(req_header_ptr->request_id),
        .function_id = cista::to_idx(req_header_ptr->function_id),
        .status =
          rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::INVALID_ARGUMENT)),
        .what = "Received Eager request for function registered with "
                "std::monostate (no payload)",
        .hlc = worker->hlc_,
        .workflow_id = cista::to_idx(req_header_ptr->workflow_id),
      });
    } else if constexpr (std::is_same_v<ReceivedBufferT, ucxx::UcxBufferVec>) {
      auto tensor_metas = utils::GetTensorMetas(req_header_ptr->params);
      if (tensor_metas.empty()) {
        // If tensor_metas is empty, create UcxBufferVec directly with single
        // segment instead of using to_buffer_vec
        auto ucx_buffer_vec =
          std::move(payload).to_buffer_vec({payload.size()});
        return worker
          ->ServerDispatchAndManageLifecycle_<ReceivedBufferT, MsgLcPolicyT>(
            conn_id, req_header_ptr, std::move(ucx_buffer_vec),
            std::move(lc_policy));
      } else {
        std::vector<size_t> sizes;
        sizes.reserve(tensor_metas.size());
        std::ranges::transform(
          tensor_metas, std::back_inserter(sizes), [](const auto& meta) {
            return rpc::utils::CalculateTensorSize(meta);
          });
        auto ucx_buffer = std::move(payload).to_buffer_vec(sizes);
        return worker
          ->ServerDispatchAndManageLifecycle_<ReceivedBufferT, MsgLcPolicyT>(
            conn_id, req_header_ptr, std::move(ucx_buffer),
            std::move(lc_policy));
      }
    } else {
      return worker
        ->ServerDispatchAndManageLifecycle_<ReceivedBufferT, MsgLcPolicyT>(
          conn_id, req_header_ptr, std::move(payload), std::move(lc_policy));
    }
  }
};

template <
  typename ReceivedBufferT, typename Fn, typename MemPolicyT,
  typename MsgLcPolicyT>
  requires(((rpc::is_payload_v<ReceivedBufferT>
             || std::is_same_v<
               ReceivedBufferT,
               rpc::
                 PayloadVariant>)&&is_receiver_memory_policy_v<MemPolicyT, ReceivedBufferT>)
           || (std::is_same_v<ReceivedBufferT, std::monostate> && std::is_same_v<MemPolicyT, AlwaysOnHostPolicy>))
          && is_message_lifecycle_policy_v<MsgLcPolicyT>
void AxonWorker::RegisterFunction(
  rpc::function_id_t id, Fn&& fn, data::string&& function_name,
  MemPolicyT mem_policy, MsgLcPolicyT lc_policy) {
  dispatcher_.RegisterFunction(
    id, std::forward<Fn>(fn), std::move(function_name));

  registered_handlers_.emplace(
    id,
    ServerMakeFullRequestHandler_<ReceivedBufferT, MemPolicyT, MsgLcPolicyT>(
      std::move(mem_policy), std::move(lc_policy)));
}

template <typename Checker>
void AxonWorker::SetBypassRejectionFunction(Checker&& checker) {
  bypass_backpressure_checker_ =
    pro::make_proxy<BypassRejectionCheckerFacade>(std::move(checker));
}

template <typename Func>
std::expected<void, errors::AxonErrorContext> AxonWorker::ProcessStoredRequests(
  AxonRequestID req_id, Func&& func, RequestVisitor visitor) {
  auto it = request_id_to_iterator_.find(req_id);
  if (it == request_id_to_iterator_.end()) {
    // If request not found, we can't extract header info, use default values
    return std::unexpected(errors::AxonErrorContext{
      .request_id = static_cast<uint32_t>(req_id),
      .status = rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::NOT_FOUND)),
      .what = "Request not found in storage",
      .hlc = hlc_,
    });
  }
  auto storage_it = it->second;
  auto executor =
    ServerMakeProcessStorageExecutor_(storage_it, std::forward<Func>(func));
  auto lifecycle_status = (*visitor)(std::move(executor));
  return ProcessLifecycleStatus_(storage_it, lifecycle_status);
}

template <typename Func>
  requires std::is_copy_constructible_v<Func>
std::expected<void, errors::AxonErrorContext> AxonWorker::ProcessStoredRequests(
  Func&& func, RequestVisitor visitor) {
  for (auto it = storage_->begin(); it != storage_->end();) {
    auto executor = ServerMakeProcessStorageExecutor_(it, func);
    auto lifecycle_status = (*visitor)(std::move(executor));
    auto result = ProcessLifecycleStatus_(it, lifecycle_status);
    if (!result) {
      return result;
    }
    if (
      lifecycle_status != LifecycleStatus::Discard
      && lifecycle_status != LifecycleStatus::Error) {
      ++it;
    }
  }
  return std::expected<void, errors::AxonErrorContext>(std::in_place);
}

// Helper function implementations
template <typename Func>
AxonWorker::Executor AxonWorker::ServerMakeProcessStorageExecutor_(
  StorageRequestIt storage_it, Func&& func) {
  return pro::make_proxy<ExecutorFacade>(
    [this, storage_it, func = std::forward<Func>(func)]() mutable {
      auto& req_ptr = *storage_it;
      return unifex::just_from([&]() mutable {
        rpc::RpcContextPtr context_ptr(
          std::holds_alternative<std::monostate>(req_ptr->payload)
            ? nullptr
            : &req_ptr->payload,
          [](void*) {});
        return dispatcher_.DispatchAdhoc(
          std::move(func), req_ptr->header, std::move(context_ptr),
          response_builder_);
      });
    });
}

inline std::expected<void, errors::AxonErrorContext>
AxonWorker::ProcessLifecycleStatus_(
  StorageRequestIt storage_it, LifecycleStatus lifecycle_status) {
  if (
    lifecycle_status == LifecycleStatus::Discard
    || lifecycle_status == LifecycleStatus::Error) {
    auto axon_req_id = GetMessageID((*storage_it)->header);
    const auto& req_header = (*storage_it)->header;
    request_id_to_iterator_.erase(axon_req_id);
    storage_->erase(storage_it);
    if (lifecycle_status == LifecycleStatus::Error) {
      return std::unexpected(errors::AxonErrorContext{
        .session_id = cista::to_idx(req_header.session_id),
        .request_id = cista::to_idx(req_header.request_id),
        .function_id = cista::to_idx(req_header.function_id),
        .status = rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::INTERNAL)),
        .what = "Error processing stored request",
        .hlc = hlc_,
        .workflow_id = cista::to_idx(req_header.workflow_id),
      });
    }
  }
  return std::expected<void, errors::AxonErrorContext>(std::in_place);
}

template <typename RespBufferT, typename MemPolicyT>
struct AxonWorker::ClientResponseHandlerVisitor_ {
  AxonWorker* worker;
  mutable MemPolicyT mem_policy;
  mutable rpc::ResponseHeaderUniquePtr resp_header_ptr;

  auto operator()(size_t am_desc_key)
    -> ResponseHandlerReturnType<RespBufferT, MemPolicyT> {
    // RNDV Path
    auto tensor_metas = utils::GetTensorMetas(resp_header_ptr->results);
    if (tensor_metas.empty()) [[unlikely]] {
      if (
        auto am_desc_opt =
          worker->client_ctx_->view_pending_am_desc(am_desc_key)) {
        auto& am_desc = am_desc_opt.value().get();
        if (am_desc.data_length > 0) {
          // Fallback: if no tensor metas, but data is expected, create a
          // single tensor meta for the whole data (thread_local for safety).
          thread_local rpc::TensorMeta fallback_meta;
          fallback_meta.dtype = DLDataType{
            .code = DLDataTypeCode::kDLOpaqueHandle, .bits = 8, .lanes = 1};
          fallback_meta.shape = {static_cast<int64_t>(am_desc.data_length)};
          tensor_metas = utils::TensorMetaSpan{&fallback_meta, 1};
        }
      }
    }

    if (tensor_metas.empty()) [[unlikely]] {
      return unifex::just_error(errors::AxonErrorContext{
        .conn_id = 0,  // TODO(He Jia): Add conn_id
        .session_id = cista::to_idx(resp_header_ptr->session_id),
        .request_id = cista::to_idx(resp_header_ptr->request_id),
        .status = rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::INTERNAL)),
        .what = "RNDV response received but no tensor metas found in response "
                "header and no data expected.",
        .hlc = worker->hlc_,
        .workflow_id = cista::to_idx(resp_header_ptr->workflow_id),
      });
    }

    auto run_pipeline = [&]<typename BufferT>() {
      return unifex::on(
               worker->client_ctx_->get_scheduler(),
               worker->ProcessRndvBuffer_<BufferT, MemPolicyT>(
                 worker->client_ctx_->get_scheduler(), am_desc_key,
                 std::move(mem_policy), tensor_metas))
             | unifex::then(MoveBundleBuffer<BufferT>{})
             | unifex::then(
               AxonWorker::ResponseHandlerRndvReturnType<BufferT, MemPolicyT>{
                 std::move(resp_header_ptr)});
    };

    if constexpr (std::is_same_v<RespBufferT, std::monostate>) {
      return unifex::just_error(errors::AxonErrorContext{
        .conn_id = 0,
        .session_id = cista::to_idx(resp_header_ptr->session_id),
        .request_id = cista::to_idx(resp_header_ptr->request_id),
        .status = rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::INTERNAL)),
        .what = "std::monostate is not a valid response buffer type.",
        .hlc = worker->hlc_,
        .workflow_id = cista::to_idx(resp_header_ptr->workflow_id),
      });
    } else if constexpr (std::is_same_v<RespBufferT, rpc::PayloadVariant>) {
      if (tensor_metas.size() == 1) {
        return run_pipeline.template operator()<ucxx::UcxBuffer>();
      } else {
        return run_pipeline.template operator()<ucxx::UcxBufferVec>();
      }
    } else {
      return run_pipeline.template operator()<RespBufferT>();
    }
  }

  auto operator()(ucxx::UcxBuffer&& payload)
    -> ResponseHandlerReturnType<RespBufferT, MemPolicyT> {
    // Eager Path
    if constexpr (std::is_same_v<RespBufferT, ucxx::UcxBufferVec>) {
      auto tensor_metas = utils::GetTensorMetas(resp_header_ptr->results);
      if (tensor_metas.empty()) {
        // If tensor_metas is empty, create UcxBufferVec directly with single
        // segment instead of using to_buffer_vec
        auto ucx_buffer_vec =
          std::move(payload).to_buffer_vec({payload.size()});
        return unifex::just(std::make_pair(
          std::move(resp_header_ptr), std::move(ucx_buffer_vec)));
      } else {
        std::vector<size_t> sizes;
        sizes.reserve(tensor_metas.size());
        std::ranges::transform(
          tensor_metas, std::back_inserter(sizes), [](const auto& meta) {
            return rpc::utils::CalculateTensorSize(meta);
          });
        auto ucx_buffer = std::move(payload).to_buffer_vec(sizes);
        return unifex::just(
          std::make_pair(std::move(resp_header_ptr), std::move(ucx_buffer)));
      }
    } else {
      // UcxBuffer or rpc::PayloadVariant
      return unifex::just(
        std::make_pair(std::move(resp_header_ptr), std::move(payload)));
    }
  }
};

// --- External Template Instances ---
#define AXON_SEND_REQUEST_EXTERN(PayloadT)            \
  extern template AxonWorker::UcxSendCombinedSender   \
  AxonWorker::ClientSendRequest_<PayloadT>(           \
    std::expected<uint64_t, std::error_code> conn_id, \
    const cista::byte_buf& header_buf, const PayloadT& payload);

AXON_SEND_REQUEST_EXTERN(rpc::PayloadVariant)
AXON_SEND_REQUEST_EXTERN(ucxx::UcxBufferVec)
AXON_SEND_REQUEST_EXTERN(ucxx::UcxBuffer)

#undef AXON_SEND_REQUEST_EXTERN

#define AXON_REGISTER_FUNCTION_EXTERN(PayloadT, MemPolicyT, LcPolicyT)         \
  extern template void                                                         \
  AxonWorker::RegisterFunction<PayloadT, MemPolicyT, LcPolicyT>(               \
    rpc::function_id_t id, const data::string& name,                           \
    const data::vector<rpc::ParamType>& param_types,                           \
    const data::vector<rpc::ParamType>& return_types,                          \
    rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type, \
    rpc::DynamicAsyncRpcFunction&& func, MemPolicyT mem_policy,                \
    LcPolicyT lc_policy);

AXON_REGISTER_FUNCTION_EXTERN(
  std::monostate, AlwaysOnHostPolicy, TransientPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  std::monostate, AlwaysOnHostPolicy, RetentionPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  ucxx::UcxBuffer, AlwaysOnHostPolicy, TransientPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>, TransientPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  ucxx::UcxBuffer, AlwaysOnHostPolicy, RetentionPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>, RetentionPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  ucxx::UcxBufferVec, AlwaysOnHostPolicy, TransientPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>, TransientPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  ucxx::UcxBufferVec, AlwaysOnHostPolicy, RetentionPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>, RetentionPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  rpc::PayloadVariant, AlwaysOnHostPolicy, TransientPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  rpc::PayloadVariant, CustomMemoryPolicy<rpc::PayloadVariant>, TransientPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  rpc::PayloadVariant, AlwaysOnHostPolicy, RetentionPolicy)
AXON_REGISTER_FUNCTION_EXTERN(
  rpc::PayloadVariant, CustomMemoryPolicy<rpc::PayloadVariant>, RetentionPolicy)

#undef AXON_REGISTER_FUNCTION_EXTERN

#define AXON_DISPATCH_AND_MANAGE_LIFECYCLE_EXTERN(PayloadT, LcPolicyT) \
  extern template AxonWorker::AnySender                                \
  AxonWorker::ServerDispatchAndManageLifecycle_<PayloadT, LcPolicyT>(  \
    uint64_t conn_id, const rpc::RpcRequestHeader* req_header_ptr,     \
    PayloadT&& payload, LcPolicyT lc_policy);

AXON_DISPATCH_AND_MANAGE_LIFECYCLE_EXTERN(rpc::PayloadVariant, TransientPolicy)
AXON_DISPATCH_AND_MANAGE_LIFECYCLE_EXTERN(rpc::PayloadVariant, RetentionPolicy)
AXON_DISPATCH_AND_MANAGE_LIFECYCLE_EXTERN(ucxx::UcxBuffer, RetentionPolicy)
AXON_DISPATCH_AND_MANAGE_LIFECYCLE_EXTERN(ucxx::UcxBuffer, TransientPolicy)
AXON_DISPATCH_AND_MANAGE_LIFECYCLE_EXTERN(ucxx::UcxBuffer, RetentionPolicy)
AXON_DISPATCH_AND_MANAGE_LIFECYCLE_EXTERN(ucxx::UcxBufferVec, TransientPolicy)
AXON_DISPATCH_AND_MANAGE_LIFECYCLE_EXTERN(ucxx::UcxBufferVec, RetentionPolicy)

#undef AXON_DISPATCH_AND_MANAGE_LIFECYCLE_EXTERN

#define AXON_NAME_INVOKE_RPC_OPT_EXTERN(PayloadT, RespBufferT, MemPolicyT) \
  extern template auto                                                     \
  AxonWorker::InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(                \
    std::string_view worker_name, rpc::RpcRequestHeader && request_header, \
    std::optional<PayloadT> && payload, MemPolicyT mem_policy);

AXON_NAME_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_NAME_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>)
AXON_NAME_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_NAME_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec,
  CustomMemoryPolicy<ucxx::UcxBufferVec>)

#undef AXON_NAME_INVOKE_RPC_OPT_EXTERN

#define AXON_KEY_INVOKE_RPC_OPT_EXTERN(PayloadT, RespBufferT, MemPolicyT) \
  extern template auto                                                    \
  AxonWorker::InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(               \
    WorkerKey worker_key, rpc::RpcRequestHeader && request_header,        \
    std::optional<PayloadT> && payload, MemPolicyT mem_policy);

AXON_KEY_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_KEY_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_KEY_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>)
AXON_KEY_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_KEY_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_KEY_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_KEY_INVOKE_RPC_OPT_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec,
  CustomMemoryPolicy<ucxx::UcxBufferVec>)

#undef AXON_KEY_INVOKE_RPC_OPT_EXTERN

#define AXON_NAME_INVOKE_RPC_EXTERN(PayloadT, RespBufferT, MemPolicyT)     \
  extern template auto                                                     \
  AxonWorker::InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(                \
    std::string_view worker_name, rpc::RpcRequestHeader && request_header, \
    PayloadT && payload, MemPolicyT mem_policy);

AXON_NAME_INVOKE_RPC_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_NAME_INVOKE_RPC_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>)
AXON_NAME_INVOKE_RPC_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_NAME_INVOKE_RPC_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec,
  CustomMemoryPolicy<ucxx::UcxBufferVec>)

#undef AXON_NAME_INVOKE_RPC_EXTERN

#define AXON_KEY_INVOKE_RPC_EXTERN(PayloadT, RespBufferT, MemPolicyT) \
  extern template auto                                                \
  AxonWorker::InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(           \
    WorkerKey worker_key, rpc::RpcRequestHeader && request_header,    \
    PayloadT && payload, MemPolicyT mem_policy);

AXON_KEY_INVOKE_RPC_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_KEY_INVOKE_RPC_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_KEY_INVOKE_RPC_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>)
AXON_KEY_INVOKE_RPC_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_KEY_INVOKE_RPC_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_KEY_INVOKE_RPC_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_KEY_INVOKE_RPC_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec,
  CustomMemoryPolicy<ucxx::UcxBufferVec>)

#undef AXON_KEY_INVOKE_RPC_EXTERN

#define AXON_RPC_INVOKE_IMPL_EXTERN(PayloadT, RespBufferT, MemPolicyT) \
  extern template auto                                                 \
  AxonWorker::InvokeRpcImpl<PayloadT, RespBufferT, MemPolicyT>(        \
    std::expected<uint64_t, std::error_code> conn_id,                  \
    rpc::RpcRequestHeader && request_header, PayloadT && payload,      \
    MemPolicyT mem_policy);

AXON_RPC_INVOKE_IMPL_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_RPC_INVOKE_IMPL_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_RPC_INVOKE_IMPL_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_RPC_INVOKE_IMPL_EXTERN(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>)
AXON_RPC_INVOKE_IMPL_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_RPC_INVOKE_IMPL_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_RPC_INVOKE_IMPL_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_RPC_INVOKE_IMPL_EXTERN(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec,
  CustomMemoryPolicy<ucxx::UcxBufferVec>)

#undef AXON_RPC_INVOKE_IMPL_EXTERN

// Explicit template instantiation declarations for RethrowErrorContextHelper_
#define AXON_RETHROW_ERROR_CONTEXT_HELPER_EXTERN(ErrorType)                \
  extern template auto                                                     \
  AxonWorker::RethrowErrorContextHelper_::operator()<ErrorType>(ErrorType) \
    const noexcept                                                         \
    -> decltype(unifex::just_error(std::declval<errors::AxonErrorContext>()));

AXON_RETHROW_ERROR_CONTEXT_HELPER_EXTERN(std::error_code)
AXON_RETHROW_ERROR_CONTEXT_HELPER_EXTERN(std::exception_ptr)
AXON_RETHROW_ERROR_CONTEXT_HELPER_EXTERN(errors::AxonErrorContext)
AXON_RETHROW_ERROR_CONTEXT_HELPER_EXTERN(errors::AxonErrorContext&)
AXON_RETHROW_ERROR_CONTEXT_HELPER_EXTERN(const errors::AxonErrorContext&)
AXON_RETHROW_ERROR_CONTEXT_HELPER_EXTERN(errors::AxonErrorContext&&)

#undef AXON_RETHROW_ERROR_CONTEXT_HELPER_EXTERN

// Explicit template instantiation declarations for
// TransformErrorToRpcException_
#define AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION_EXTERN(ErrorType)               \
  extern template auto                                                        \
  AxonWorker::TransformErrorToRpcException_::operator()<ErrorType>(ErrorType) \
    const noexcept                                                            \
    -> decltype(unifex::just_error(std::declval<std::exception_ptr>()));

AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION_EXTERN(std::error_code)
AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION_EXTERN(std::exception_ptr)
AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION_EXTERN(errors::AxonErrorContext)
AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION_EXTERN(errors::AxonErrorContext&)
AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION_EXTERN(const errors::AxonErrorContext&)
AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION_EXTERN(errors::AxonErrorContext&&)

#undef AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION_EXTERN

// Helper function to convert DLDevice to ucx_memory_type
constexpr ucx_memory_type GetMemoryType(DLDevice device);

}  // namespace axon
}  // namespace eux

#endif  // AXON_CORE_AXON_WORKER_HPP_
