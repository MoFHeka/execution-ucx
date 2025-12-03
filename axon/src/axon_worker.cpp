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

#include "axon/axon_worker.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <expected>
#include <format>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <unifex/create.hpp>
#include <unifex/defer.hpp>
#include <unifex/get_stop_token.hpp>
#include <unifex/into_variant.hpp>
#include <unifex/just_error.hpp>
#include <unifex/just_from.hpp>
#include <unifex/let_done.hpp>
#include <unifex/let_value.hpp>
#include <unifex/let_value_with.hpp>
#include <unifex/on.hpp>
#include <unifex/repeat_effect_until.hpp>
#include <unifex/spawn_detached.hpp>
#include <unifex/spawn_future.hpp>
#include <unifex/stop_when.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/then.hpp>
#include <unifex/upon_error.hpp>
#include <unifex/with_query_value.hpp>

#include "axon/errors/error_types.hpp"
#include "axon/utils/axon_message.hpp"
#include "axon/utils/hash.hpp"
#include "rpc_core/rpc_dispatcher.hpp"
#include "rpc_core/rpc_payload_types.hpp"
#include "rpc_core/rpc_status.hpp"
#include "ucx_context/ucx_am_context/ucx_am_context.hpp"
#include "ucx_context/ucx_context_data.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

namespace eux {
namespace axon {

AxonWorker::AxonWorker(
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr,
  const std::string& worker_name, size_t thread_pool_size,
  std::chrono::milliseconds timeout,
  std::unique_ptr<ucxx::UcxAutoDeviceContext> auto_device_context)
  : mr_(mr),
    worker_name_(worker_name),
    worker_id_(utils::StringHash(worker_name_)),
    timeout_(timeout),
    hlc_(rpc::utils::HybridLogicalClock::Now()),
    auto_device_context_(std::move(auto_device_context)),
    dispatcher_(cista::offset::string(worker_name_)),
    thread_pool_(thread_pool_size) {
  if (common_ucp_context_ == nullptr) {
    auto ec = ucxx::ucx_am_context::init_ucp_context(
      worker_name_, &common_ucp_context_,
      /*mtWorkersShared=*/true, /*printConfig=*/false);
    if (ec) {
      throw std::runtime_error(
        "UCX Context initialization failed: " + ec.message());
    }
  }
  storage_ = std::make_unique<storage::AxonStorage>();
  client_ctx_ = std::make_unique<ucxx::ucx_am_context>(
    mr_, common_ucp_context_, worker_name_, timeout_, /*handleErr*/ false,
    /*clientId=*/worker_id_.high(), auto_device_context_->clone());
  server_ctx_ = std::make_unique<ucxx::ucx_am_context>(
    mr_, common_ucp_context_, worker_name_, timeout_, /*handleErr*/ false,
    /*clientId=*/worker_id_.low(), auto_device_context_->clone());
}

AxonWorker::~AxonWorker() {
  Stop();
  // Ensure contexts using common_ucp_context_ are destroyed first
  server_ctx_.reset();
  client_ctx_.reset();

  if (common_ucp_context_) {
    ucxx::ucx_am_context::destroy_ucp_context(common_ucp_context_);
    common_ucp_context_ = nullptr;
  }
}

std::error_code AxonWorker::Start() {
  if (StartServer() != std::error_code{}) {
    return std::make_error_code(errors::AxonErrc::WorkerStartFailed);
  }
  if (StartClient() != std::error_code{}) {
    return std::make_error_code(errors::AxonErrc::WorkerStartFailed);
  }
  return {};
}

void AxonWorker::StopServer() {
  bool was_running = server_running_.exchange(false);
  if (!was_running) {
    unifex::sync_wait(server_async_scope_.join());
    return;
  }

  server_stop_source_.request_stop();

  if (server_future_.has_value()) {
    unifex::sync_wait(std::move(server_future_.value()));
  }
  unifex::sync_wait(server_async_scope_.join());

  server_context_stop_source_.request_stop();
  if (server_thread_.joinable()) {
    server_thread_.join();
  }
}

void AxonWorker::StopClient() {
  bool was_running = client_running_.exchange(false);
  if (!was_running) {
    unifex::sync_wait(client_async_scope_.join());
    return;
  }

  client_stop_source_.request_stop();
  if (client_future_.has_value()) {
    unifex::sync_wait(std::move(client_future_.value()));
  }
  unifex::sync_wait(client_async_scope_.join());

  client_context_stop_source_.request_stop();
  if (client_thread_.joinable()) {
    client_thread_.join();
  }
}

void AxonWorker::Stop() {
  StopServer();
  StopClient();
}

std::vector<std::byte> AxonWorker::GetLocalAddress() {
  std::vector<std::byte> addr;
  if (auto ec = server_ctx_->get_ucp_address(addr)) {
    return {};
  }
  return addr;
}

cista::byte_buf AxonWorker::GetLocalSignatures() const {
  return dispatcher_.GetAllSignatures();
}

auto AxonWorker::ConnectEndpointSender_(
  const std::vector<std::byte>& ucp_address, const std::string& worker_name) {
  return ucxx::connect_endpoint(client_ctx_->get_scheduler(), ucp_address)
         | unifex::then([this, worker_name](uint64_t conn_id) {
             AssociateConnection(worker_name, conn_id);
             return std::expected<uint64_t, std::error_code>(conn_id);
           })
         | unifex::upon_error(
           [this](std::variant<std::error_code, std::exception_ptr>&& error)
             -> std::expected<uint64_t, std::error_code> {
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
             ReportClientError_({
               .conn_id = 0,
               .session_id = 0,
               .request_id = 0,
               .function_id = 0,
               .status = rpc::RpcStatus(ec),
               .what = std::move(what),
               .hlc = hlc_,
               .workflow_id = 0,
             });
             return std::unexpected(ec);
           });
}

std::expected<uint64_t, std::error_code> AxonWorker::ConnectEndpoint(
  const std::vector<std::byte>& ucp_address, const std::string& worker_name) {
  auto res =
    ConnectEndpointSender_(ucp_address, worker_name) | unifex::sync_wait();
  return res.value();
}

auto AxonWorker::ConnectEndpointAsync(
  const std::vector<std::byte>& ucp_address, const std::string& worker_name) {
  return ConnectEndpointSender_(ucp_address, worker_name);
}

void AxonWorker::AssociateConnection(
  const std::string& worker_name, uint64_t conn_id) {
  auto key_it = remote_workers_slot_.find(worker_name);
  if (key_it != remote_workers_slot_.end()) {
    if (auto* worker_info = remote_workers_.access(key_it->second)) {
      worker_info->conn_id = conn_id;
    }
    return;
  }
  auto key =
    remote_workers_.emplace(worker_name, FunctionSignaturesMap({}), conn_id);
  remote_workers_slot_[worker_name] = key;
}

#define AXON_NAME_INVOKE_RPC_OPT(PayloadT, RespBufferT, MemPolicyT)         \
  template auto AxonWorker::InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(   \
    const std::string& worker_name, rpc::RpcRequestHeader&& request_header, \
    std::optional<PayloadT>&& payload, MemPolicyT mem_policy);

AXON_NAME_INVOKE_RPC_OPT(ucxx::UcxBuffer, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBuffer, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>)
AXON_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC_OPT(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec,
  CustomMemoryPolicy<ucxx::UcxBufferVec>)

#undef AXON_NAME_INVOKE_RPC_OPT_EXTERN

#define AXON_KEY_INVOKE_RPC_OPT(PayloadT, RespBufferT, MemPolicyT)        \
  template auto AxonWorker::InvokeRpc<PayloadT, RespBufferT, MemPolicyT>( \
    WorkerKey worker_key, rpc::RpcRequestHeader && request_header,        \
    std::optional<PayloadT> && payload, MemPolicyT mem_policy);

AXON_KEY_INVOKE_RPC_OPT(
  ucxx::UcxBuffer, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_KEY_INVOKE_RPC_OPT(ucxx::UcxBuffer, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_KEY_INVOKE_RPC_OPT(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>)
AXON_KEY_INVOKE_RPC_OPT(ucxx::UcxBufferVec, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_KEY_INVOKE_RPC_OPT(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_KEY_INVOKE_RPC_OPT(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_KEY_INVOKE_RPC_OPT(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec,
  CustomMemoryPolicy<ucxx::UcxBufferVec>)

#undef AXON_KEY_INVOKE_RPC_OPT_EXTERN

#define AXON_NAME_INVOKE_RPC(PayloadT, RespBufferT, MemPolicyT)             \
  template auto AxonWorker::InvokeRpc<PayloadT, RespBufferT, MemPolicyT>(   \
    const std::string& worker_name, rpc::RpcRequestHeader&& request_header, \
    PayloadT&& payload, MemPolicyT mem_policy);

AXON_NAME_INVOKE_RPC(ucxx::UcxBuffer, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC(
  ucxx::UcxBuffer, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_NAME_INVOKE_RPC(ucxx::UcxBuffer, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>)
AXON_NAME_INVOKE_RPC(ucxx::UcxBufferVec, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_NAME_INVOKE_RPC(ucxx::UcxBufferVec, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_NAME_INVOKE_RPC(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec,
  CustomMemoryPolicy<ucxx::UcxBufferVec>)

#undef AXON_NAME_INVOKE_RPC_EXTERN

#define AXON_KEY_INVOKE_RPC(PayloadT, RespBufferT, MemPolicyT)            \
  template auto AxonWorker::InvokeRpc<PayloadT, RespBufferT, MemPolicyT>( \
    WorkerKey worker_key, rpc::RpcRequestHeader && request_header,        \
    PayloadT && payload, MemPolicyT mem_policy);

AXON_KEY_INVOKE_RPC(
  ucxx::UcxBuffer, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_KEY_INVOKE_RPC(ucxx::UcxBuffer, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_KEY_INVOKE_RPC(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>)
AXON_KEY_INVOKE_RPC(ucxx::UcxBufferVec, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_KEY_INVOKE_RPC(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_KEY_INVOKE_RPC(ucxx::UcxBufferVec, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_KEY_INVOKE_RPC(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec,
  CustomMemoryPolicy<ucxx::UcxBufferVec>)

#undef AXON_KEY_INVOKE_RPC_EXTERN

#define AXON_INVOKE_RPC_IMPL(PayloadT, RespBufferT, MemPolicyT)               \
  template auto AxonWorker::InvokeRpcImpl<PayloadT, RespBufferT, MemPolicyT>( \
    std::expected<uint64_t, std::error_code> conn_id,                         \
    rpc::RpcRequestHeader && request_header, PayloadT && payload,             \
    MemPolicyT mem_policy);

AXON_INVOKE_RPC_IMPL(ucxx::UcxBuffer, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_INVOKE_RPC_IMPL(
  ucxx::UcxBuffer, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_INVOKE_RPC_IMPL(ucxx::UcxBuffer, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_INVOKE_RPC_IMPL(
  ucxx::UcxBuffer, ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>)
AXON_INVOKE_RPC_IMPL(ucxx::UcxBufferVec, ucxx::UcxBuffer, AlwaysOnHostPolicy)
AXON_INVOKE_RPC_IMPL(
  ucxx::UcxBufferVec, ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>)
AXON_INVOKE_RPC_IMPL(ucxx::UcxBufferVec, ucxx::UcxBufferVec, AlwaysOnHostPolicy)
AXON_INVOKE_RPC_IMPL(
  ucxx::UcxBufferVec, ucxx::UcxBufferVec,
  CustomMemoryPolicy<ucxx::UcxBufferVec>)

#undef AXON_INVOKE_RPC_IMPL

auto AxonWorker::RegisterEndpointSignaturesSender_(
  const std::string& worker_name, const cista::byte_buf& signatures_blob) {
  return unifex::just_from([this, worker_name, signatures_blob]() {
    // Try to deserialize the remote worker signatures. Any failure (including
    // overflow) is treated as INVALID_ARGUMENT.
    const data::vector<rpc::RpcFunctionSignature>* signatures = nullptr;
    try {
      signatures = cista::deserialize<
        data::vector<rpc::RpcFunctionSignature>, rpc::utils::SerializerMode>(
        signatures_blob);
    } catch (const std::exception&) {
      // TODO(He Jia): maybe report error
      return std::expected<WorkerKey, std::error_code>(
        std::unexpected(std::make_error_code(rpc::RpcErrc::INVALID_ARGUMENT)));
    }

    if (signatures == nullptr) {
      // TODO(He Jia): maybe report error
      return std::expected<WorkerKey, std::error_code>(
        std::unexpected(std::make_error_code(rpc::RpcErrc::INVALID_ARGUMENT)));
    }

    auto key_it = remote_workers_slot_.find(worker_name);
    if (key_it == remote_workers_slot_.end()) {
      return std::expected<WorkerKey, std::error_code>(std::unexpected(
        std::make_error_code(errors::AxonErrc::WorkerNotFound)));
    }

    if (auto* worker_info = remote_workers_.access(key_it->second)) {
      for (auto& sig : *signatures) {
        worker_info->signatures[sig.id] = sig;
      }
    } else {
      return std::expected<WorkerKey, std::error_code>(std::unexpected(
        std::make_error_code(errors::AxonErrc::WorkerNotFound)));
    }

    return std::expected<WorkerKey, std::error_code>(key_it->second);
  });
}

std::expected<AxonWorker::WorkerKey, std::error_code>
AxonWorker::RegisterEndpointSignatures(
  const std::string& worker_name, const cista::byte_buf& signatures_blob) {
  auto res = RegisterEndpointSignaturesSender_(worker_name, signatures_blob)
             | unifex::sync_wait();
  return res.value();
}

auto AxonWorker::RegisterEndpointSignaturesAsync(
  const std::string& worker_name, const cista::byte_buf& signatures_blob) {
  return RegisterEndpointSignaturesSender_(worker_name, signatures_blob);
}

template <typename ReceivedBufferT, typename MemPolicy, typename MsgLcPolicy>
  requires rpc::is_payload_v<ReceivedBufferT>
           && is_receiver_memory_policy_v<MemPolicy, ReceivedBufferT>
           && is_message_lifecycle_policy_v<MsgLcPolicy>
void AxonWorker::RegisterFunction(
  rpc::function_id_t id, const data::string& name,
  const data::vector<rpc::ParamType>& param_types,
  const data::vector<rpc::ParamType>& return_types,
  rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type,
  rpc::DynamicAsyncRpcFunction&& func, MemPolicy mem_policy,
  MsgLcPolicy lc_policy) {
  dispatcher_.RegisterFunction(
    id, name, param_types, return_types, input_payload_type,
    return_payload_type, std::forward<rpc::DynamicAsyncRpcFunction>(func));

  registered_handlers_.emplace(
    id, ServerMakeFullRequestHandler_<ReceivedBufferT, MemPolicy, MsgLcPolicy>(
          mem_policy, lc_policy));
}

#define AXON_REGISTER_FUNCTION_TEMPLATE(PayloadT, MemPolicyT, LcPolicyT)       \
  template void AxonWorker::RegisterFunction<PayloadT, MemPolicyT, LcPolicyT>( \
    rpc::function_id_t id, const data::string& name,                           \
    const data::vector<rpc::ParamType>& param_types,                           \
    const data::vector<rpc::ParamType>& return_types,                          \
    rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type, \
    rpc::DynamicAsyncRpcFunction&& func, MemPolicyT mem_policy,                \
    LcPolicyT lc_policy);

AXON_REGISTER_FUNCTION_TEMPLATE(
  ucxx::UcxBuffer, AlwaysOnHostPolicy, TransientPolicy)
AXON_REGISTER_FUNCTION_TEMPLATE(
  ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>, TransientPolicy)
AXON_REGISTER_FUNCTION_TEMPLATE(
  ucxx::UcxBuffer, AlwaysOnHostPolicy, RetentionPolicy)
AXON_REGISTER_FUNCTION_TEMPLATE(
  ucxx::UcxBuffer, CustomMemoryPolicy<ucxx::UcxBuffer>, RetentionPolicy)
AXON_REGISTER_FUNCTION_TEMPLATE(
  ucxx::UcxBufferVec, AlwaysOnHostPolicy, TransientPolicy)
AXON_REGISTER_FUNCTION_TEMPLATE(
  ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>, TransientPolicy)
AXON_REGISTER_FUNCTION_TEMPLATE(
  ucxx::UcxBufferVec, AlwaysOnHostPolicy, RetentionPolicy)
AXON_REGISTER_FUNCTION_TEMPLATE(
  ucxx::UcxBufferVec, CustomMemoryPolicy<ucxx::UcxBufferVec>, RetentionPolicy)

#undef AXON_REGISTER_FUNCTION_TEMPLATE

void AxonWorker::SetTimeout(std::chrono::milliseconds timeout) {
  timeout_ = timeout;
}

void AxonWorker::SetRejectMessages(bool reject) {
  reject_messages_.store(reject, std::memory_order_release);
}

void AxonWorker::SetServerErrorObserver(errors::ErrorObserver obs) {
  server_err_observer_ = std::move(obs);
}

void AxonWorker::SetClientErrorObserver(errors::ErrorObserver obs) {
  client_err_observer_ = std::move(obs);
}

void AxonWorker::SetServerMetricsObserver(metrics::MetricsObserver obs) {
  server_metrics_observer_ = std::move(obs);
}

void AxonWorker::SetClientMetricsObserver(metrics::MetricsObserver obs) {
  client_metrics_observer_ = std::move(obs);
}

bool AxonWorker::IsRejectingMessages() const noexcept {
  return reject_messages_.load(std::memory_order_acquire);
}

storage::AxonStorage& AxonWorker::GetStorage() { return *storage_; }

rpc::AsyncRpcDispatcher& AxonWorker::GetDispatcher() { return dispatcher_; }

AxonWorker::UcxSendCombinedSender AxonWorker::ServerHandleSendResponse_(
  uint64_t conn_id,
  const rpc::RpcResponseHeader& resp_header,
  const rpc::ReturnedPayload& resp_payload) {
  const size_t resp_buf_required_size =
    rpc::utils::GetSerializedSize(resp_header);
  auto&& send_sched = server_ctx_->get_scheduler();

  auto serialize_fn = [&](std::byte* base) {
    rpc::utils::FixedBufferWriter writer(base, resp_buf_required_size);
    rpc::RpcDispatcher::SerializeResponse(resp_header, writer);
  };

  UcxSendCombinedSender send_sender = std::visit(
    [&](const auto& payload) -> UcxSendCombinedSender {
      using T = std::decay_t<decltype(payload)>;
      // TODO(He Jia): Add a function to use mem_h(ucp_mem_h)
      if constexpr (std::is_same_v<T, ucxx::UcxBufferVec>) {
        // Collect buffer sizes from payload
        const auto& src_bufs = payload.buffers();
        std::vector<size_t> buffer_sizes;
        buffer_sizes.reserve(src_bufs.size());
        for (const auto& buf : src_bufs) {
          buffer_sizes.push_back(buf.size);
        }
        ucxx::UcxAmIovec iov_data(
          mr_, resp_buf_required_size, buffer_sizes, payload.type(),
          /*own_header=*/true, /*own_buffer=*/false);
        serialize_fn(static_cast<std::byte*>(iov_data.get()->header.data));
        if (payload.size() > 0 && iov_data.get()->buffer_vec != nullptr) {
          auto& dest_vec = iov_data.get()->buffer_vec;
          const size_t n = payload.size();
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#elif defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
          for (size_t i = 0; i < n; ++i) {
            dest_vec[i].data = src_bufs[i].data;
            dest_vec[i].size = src_bufs[i].size;
          }
        }
        return ucxx::connection_send(send_sched, conn_id, std::move(iov_data));
      } else if constexpr (std::is_same_v<T, ucxx::UcxBuffer>) {
        const size_t payload_size = payload.size();
        const void* payload_data = payload.get()->data;
        ucxx::UcxAmData data(
          mr_, resp_buf_required_size, payload_size, payload.type(),
          /*own_header=*/true, /*own_buffer=*/false);
        serialize_fn(static_cast<std::byte*>(data.get()->header.data));
        data.get()->buffer.data = const_cast<void*>(payload_data);
        return ucxx::connection_send(send_sched, conn_id, std::move(data));
      } else {  // monostate
        ucxx::UcxAmData data(
          mr_, resp_buf_required_size, 0, ucx_memory_type::HOST,
          /*own_header=*/true, /*own_buffer=*/false);
        serialize_fn(static_cast<std::byte*>(data.get()->header.data));
        return ucxx::connection_send(send_sched, conn_id, std::move(data));
      }
    },
    resp_payload);

  return send_sender;
}

void AxonWorker::ReportServerError_(
  const errors::AxonErrorContext& ctx) noexcept {
  if (server_err_observer_) {
    server_err_observer_->OnError(std::move(ctx));
  }
}

void AxonWorker::ReportClientError_(
  const errors::AxonErrorContext& ctx) noexcept {
  if (client_err_observer_) [[unlikely]] {
    client_err_observer_->OnError(std::move(ctx));
  }
}

auto AxonWorker::ServerHandleSendErrorResponse_(
  uint64_t conn_id, const rpc::RpcRequestHeader& request_header,
  std::error_code ec) {
  return unifex::let_value([this, &request_header, ec, conn_id]() {
    // TODO(He Jia): Make metrics report more accurate
    ++client_pipe_failed_count_;
    // Set HLC for error response
    hlc_.TickLocal();
    auto header = rpc::RpcResponseHeader{
      /*.session_id = */ request_header.session_id,
      /*.request_id = */ request_header.request_id,
      /*.hlc = */ hlc_,
      /*.workflow_id = */ request_header.workflow_id,
      /*.status = */ rpc::RpcStatus(ec),
      /*.results = */ {},
    };
    auto header_buffer_size = rpc::utils::GetSerializedSize(header);
    ucxx::UcxAmData am_data(mr_, header_buffer_size, 0, ucx_memory_type::HOST);
    auto writer = rpc::utils::FixedBufferWriter(
      static_cast<std::byte*>(am_data.get()->header.data), header_buffer_size);
    rpc::RpcDispatcher::SerializeResponse(header, writer);
    return ucxx::connection_send(
      server_ctx_->get_scheduler(), conn_id, std::move(am_data));
  });
}

auto AxonWorker::ServerHandleSendErrorResponse_(
  const errors::AxonErrorContext& error_ctx) {
  // TODO(He Jia): Make metrics report more accurate
  ++client_pipe_failed_count_;

  // Set HLC for error response
  hlc_.TickLocal();
  auto header = rpc::RpcResponseHeader{
    /*.session_id = */ rpc::session_id_t(error_ctx.session_id),
    /*.request_id = */ rpc::request_id_t(error_ctx.request_id),
    /*.hlc = */ hlc_,
    /*.workflow_id = */ rpc::workflow_id_t(error_ctx.workflow_id),
    /*.status = */ std::move(error_ctx.status),
    /*.results = */ {},
  };
  auto header_buffer_size = rpc::utils::GetSerializedSize(header);
  ucxx::UcxAmData am_data(mr_, header_buffer_size, 0, ucx_memory_type::HOST);
  auto writer = rpc::utils::FixedBufferWriter(
    static_cast<std::byte*>(am_data.get()->header.data), header_buffer_size);
  rpc::RpcDispatcher::SerializeResponse(header, writer);
  return ucxx::connection_send(
    server_ctx_->get_scheduler(), error_ctx.conn_id, std::move(am_data));
}

template <typename PayloadT, typename MsgLcPolicy>
  requires(std::is_same_v<PayloadT, rpc::PayloadVariant>
           || rpc::is_payload_v<PayloadT>)
          && is_message_lifecycle_policy_v<MsgLcPolicy>
AxonWorker::AnySender AxonWorker::ServerDispatchAndManageLifecycle_(
  uint64_t conn_id, const rpc::RpcRequestHeader* req_header_ptr,
  PayloadT&& payload, MsgLcPolicy lc_policy) {
  std::shared_ptr<std::string> func_name_ptr;
  if (server_metrics_observer_) {
    auto func_sig_opt = dispatcher_.GetSignature(req_header_ptr->function_id);
    if (func_sig_opt) {
      func_name_ptr = std::make_shared<std::string>(
        func_sig_opt->function_name.data(), func_sig_opt->function_name.size());
    }
  }

  auto handle_result_sender = unifex::let_value(
    [this, conn_id, req_header_ptr,
     func_name_ptr](const rpc::RpcInvokeResult<rpc::ReturnedPayload>& result) {
      if (server_metrics_observer_) {
        server_metrics_observer_->OnDispatchComplete(
          metrics::RpcMetricsContext{
            .conn_id = conn_id,
            .session_id = cista::to_idx(req_header_ptr->session_id),
            .request_id = cista::to_idx(req_header_ptr->request_id),
            .function_id = cista::to_idx(req_header_ptr->function_id),
            .function_name = func_name_ptr ? *func_name_ptr : "",
            .hlc = req_header_ptr->hlc,
            .workflow_id = cista::to_idx(req_header_ptr->workflow_id),
          },
          std::chrono::steady_clock::now());
      }
      return ServerHandleSendResponse_(conn_id, result.header, result.payload);
    });

  auto server_metrics_fn = [this, conn_id, req_header_ptr, func_name_ptr]() {
    if (server_metrics_observer_) {
      server_metrics_observer_->OnDispatchStart(
        metrics::RpcMetricsContext{
          .conn_id = conn_id,
          .session_id = cista::to_idx(req_header_ptr->session_id),
          .request_id = cista::to_idx(req_header_ptr->request_id),
          .function_id = cista::to_idx(req_header_ptr->function_id),
          .function_name = func_name_ptr ? *func_name_ptr : "",
          .hlc = req_header_ptr->hlc,
          .workflow_id = cista::to_idx(req_header_ptr->workflow_id),
        },
        std::chrono::steady_clock::now());
    }
  };

  if constexpr (std::is_same_v<MsgLcPolicy, RetentionPolicy>) {
    auto dispatch_sender_fn =
      [this, req_header_ptr](
        std::reference_wrapper<rpc::PayloadVariant> payload) mutable {
        if constexpr (rpc::is_payload_v<PayloadT>) {
          return dispatcher_.Dispatch(
            *req_header_ptr, std::get<PayloadT>(payload.get()),
            response_builder_);
        } else {
          return dispatcher_.Dispatch(
            *req_header_ptr, payload.get(), response_builder_);
        }
      };
    auto retention_fn = [this, req_header_ptr, lc_policy,
                         payload = std::move(payload)]() mutable {
      // Use HLC as the request ID for now.
      // It must be obtained before the header is moved.
      const utils::AxonMessageID req_id = GetMessageID(*req_header_ptr);
      // Convert payload to PayloadVariant for AxonRequest construction
      rpc::PayloadVariant payload_variant;
      if constexpr (std::is_same_v<PayloadT, rpc::PayloadVariant>) {
        payload_variant = std::move(payload);
      } else {
        payload_variant = rpc::PayloadVariant(std::move(payload));
      }
      auto req_ptr = std::make_shared<utils::AxonRequest>(
        std::move(const_cast<rpc::RpcRequestHeader&>(*req_header_ptr)),
        std::move(payload_variant));
      auto status = (*lc_policy)(req_ptr, req_id);
      if (status == LifecycleStatus::Preserve) {
        auto it = storage_->emplace(req_ptr);
        request_id_to_iterator_.emplace(req_id, it);
      }
      return std::ref(req_ptr->GetPayload());
    };
    if (server_metrics_observer_) {
      return unifex::just_from(
               [server_metrics_fn = std::move(server_metrics_fn),
                retention_fn = std::move(retention_fn)]() mutable {
                 server_metrics_fn();
                 return retention_fn();
               })
             | unifex::let_value(std::move(dispatch_sender_fn))
             | std::move(handle_result_sender)
             | unifex::let_error(TransformErrorToRpcException_{});
    } else {
      return unifex::just_from(std::move(retention_fn))
             | unifex::let_value(std::move(dispatch_sender_fn))
             | std::move(handle_result_sender)
             | unifex::let_error(TransformErrorToRpcException_{});
    }
  } else {
    auto dispatch_sender_fn = [this, req_header_ptr,
                               payload = std::move(payload)]() mutable {
      return dispatcher_.Dispatch(
        *req_header_ptr, std::move(payload), response_builder_);
    };
    if (server_metrics_observer_) {
      return unifex::just_from(std::move(server_metrics_fn))
             | unifex::let_value(std::move(dispatch_sender_fn))
             | std::move(handle_result_sender)
             | unifex::let_error(TransformErrorToRpcException_{});
    } else {
      return dispatch_sender_fn() | std::move(handle_result_sender)
             | unifex::let_error(TransformErrorToRpcException_{});
    }
  }
}

#define AXON_DISPATCH_AND_MANAGE_LIFECYCLE_TEMPLATE(PayloadT, LcPolicyT) \
  template AxonWorker::AnySender                                         \
  AxonWorker::ServerDispatchAndManageLifecycle_<PayloadT, LcPolicyT>(    \
    uint64_t conn_id, const rpc::RpcRequestHeader* req_header_ptr,       \
    PayloadT&& payload, LcPolicyT lc_policy);

AXON_DISPATCH_AND_MANAGE_LIFECYCLE_TEMPLATE(
  rpc::PayloadVariant, TransientPolicy)
AXON_DISPATCH_AND_MANAGE_LIFECYCLE_TEMPLATE(
  rpc::PayloadVariant, RetentionPolicy)
AXON_DISPATCH_AND_MANAGE_LIFECYCLE_TEMPLATE(ucxx::UcxBuffer, TransientPolicy)
AXON_DISPATCH_AND_MANAGE_LIFECYCLE_TEMPLATE(ucxx::UcxBuffer, RetentionPolicy)
AXON_DISPATCH_AND_MANAGE_LIFECYCLE_TEMPLATE(ucxx::UcxBufferVec, TransientPolicy)
AXON_DISPATCH_AND_MANAGE_LIFECYCLE_TEMPLATE(ucxx::UcxBufferVec, RetentionPolicy)

#undef AXON_DISPATCH_AND_MANAGE_LIFECYCLE_TEMPLATE

// Helper function to handle variant buffer bundle
template <typename MsgLcPolicy>
AxonWorker::AnySender AxonWorker::ServerHandleVariantBufferBundle_(
  uint64_t conn_id, const rpc::RpcRequestHeader* req_header_ptr,
  BufferBundleVariant&& buffer_bundle, MsgLcPolicy lc_policy) {
  return std::visit(
    [this, conn_id, req_header_ptr,
     lc_policy](auto&& bundle) mutable -> AnySender {
      using BundleType = std::decay_t<decltype(bundle)>;
      constexpr bool IsUcxBufferBundle =
        std::is_same_v<BundleType, ucxx::active_message_buffer_bundle>;
      constexpr bool IsUcxBufferVecBundle =
        std::is_same_v<BundleType, ucxx::active_message_iovec_buffer_bundle>;

      if constexpr (IsUcxBufferBundle) {
        return ServerDispatchAndManageLifecycle_<ucxx::UcxBuffer, MsgLcPolicy>(
          conn_id, req_header_ptr, std::move(bundle.move_buffer()), lc_policy);
      } else if constexpr (IsUcxBufferVecBundle) {
        return ServerDispatchAndManageLifecycle_<
          ucxx::UcxBufferVec, MsgLcPolicy>(
          conn_id, req_header_ptr, std::move(bundle.move_buffer()), lc_policy);
      } else {
        static_assert(
          !(IsUcxBufferBundle || IsUcxBufferVecBundle),
          "Invalid buffer bundle type in variant");
        return unifex::just_error(std::make_exception_ptr(
          rpc::RpcException(rpc::RpcErrc::INVALID_ARGUMENT)));
      }
    },
    std::forward<BufferBundleVariant>(buffer_bundle));
}

// Helper function to handle concrete buffer bundle
template <typename BundleType, typename MsgLcPolicy>
AxonWorker::AnySender AxonWorker::ServerHandleConcreteBufferBundle_(
  uint64_t conn_id, const rpc::RpcRequestHeader* req_header_ptr,
  BundleType&& buffer_bundle, MsgLcPolicy lc_policy) {
  constexpr bool IsUcxBufferBundle = std::is_same_v<
    std::decay_t<BundleType>, ucxx::active_message_buffer_bundle>;
  constexpr bool IsUcxBufferVecBundle = std::is_same_v<
    std::decay_t<BundleType>, ucxx::active_message_iovec_buffer_bundle>;

  if constexpr (IsUcxBufferBundle) {
    return ServerDispatchAndManageLifecycle_<ucxx::UcxBuffer, MsgLcPolicy>(
      conn_id, req_header_ptr, std::move(buffer_bundle.move_buffer()),
      lc_policy);
  } else if constexpr (IsUcxBufferVecBundle) {
    return ServerDispatchAndManageLifecycle_<ucxx::UcxBufferVec, MsgLcPolicy>(
      conn_id, req_header_ptr, std::move(buffer_bundle.move_buffer()),
      lc_policy);
  } else {
    // Always false, but dependent on template parameter
    static_assert(
      std::is_same_v<BundleType, void>, "Invalid buffer bundle type");
    return unifex::just_error(std::make_exception_ptr(
      rpc::RpcException(rpc::RpcErrc::INVALID_ARGUMENT)));
  }
}

// Unified entry point for processing buffer bundles
template <typename BundleType, typename MsgLcPolicy>
AxonWorker::AnySender AxonWorker::ServerProcessBufferBundle_(
  uint64_t conn_id, const rpc::RpcRequestHeader* req_header_ptr,
  BundleType&& buffer_bundle, MsgLcPolicy lc_policy) {
  using DecayedType = std::decay_t<BundleType>;
  constexpr bool IsVariant = std::is_same_v<DecayedType, BufferBundleVariant>;

  if constexpr (IsVariant) {
    return ServerHandleVariantBufferBundle_(
      conn_id, req_header_ptr, std::forward<BundleType>(buffer_bundle),
      lc_policy);
  } else {
    return ServerHandleConcreteBufferBundle_(
      conn_id, req_header_ptr, std::forward<BundleType>(buffer_bundle),
      lc_policy);
  }
}

template <typename ErrorT>
auto AxonWorker::TransformErrorToRpcException_::operator()(
  ErrorT error) const noexcept
  -> decltype(unifex::just_error(std::declval<std::exception_ptr>())) {
  using DecayedErrorT = std::decay_t<ErrorT>;
  if constexpr (kIsStdErrorCode<DecayedErrorT>) {
    return unifex::just_error(
      std::make_exception_ptr(rpc::RpcException(std::forward<ErrorT>(error))));
  } else if constexpr (kIsAxonErrorContext<DecayedErrorT>) {
    return unifex::just_error(
      errors::MakeExceptionPtr(std::forward<ErrorT>(error)));
  } else {
    return unifex::just_error(std::forward<ErrorT>(error));
  }
}

// Explicit template instantiations for
// TransformErrorToRpcException_::operator()
#define AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION(ErrorType)                      \
  template auto                                                               \
  AxonWorker::TransformErrorToRpcException_::operator()<ErrorType>(ErrorType) \
    const noexcept                                                            \
    -> decltype(unifex::just_error(std::declval<std::exception_ptr>()));

AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION(std::error_code)
AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION(std::exception_ptr)
AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION(errors::AxonErrorContext)
AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION(errors::AxonErrorContext&)
AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION(const errors::AxonErrorContext&)
AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION(errors::AxonErrorContext&&)

#undef AXON_TRANSFORM_ERROR_TO_RPC_EXCEPTION

auto AxonWorker::BufferBundleProcessor_::operator()(auto&& buffer_bundle)
  -> AxonWorker::AnySender {
  return std::visit(
    [this,                                      //
     buffer_bundle = std::move(buffer_bundle)]  //
    (auto&& lc_policy) mutable {
      using BundleT = std::decay_t<decltype(buffer_bundle)>;
      using LcPolicyT = std::decay_t<decltype(lc_policy)>;
      return worker->ServerProcessBufferBundle_<BundleT, LcPolicyT>(
        conn_id, req_header_ptr, std::move(buffer_bundle), lc_policy);
    },
    lc_policy_variant);
}

auto AxonWorker::ReturnTimeoutErrorContextHelper_::operator()() const noexcept
  -> decltype(unifex::just_error(std::declval<errors::AxonErrorContext>())) {
  return unifex::just_error(std::move(error_ctx));
}

auto AxonWorker::ReturnTimeoutErrorContextHelper_::operator()(
  auto&& /* ignored_error */) const noexcept
  -> decltype(unifex::just_error(std::declval<errors::AxonErrorContext>())) {
  return unifex::just_error(std::move(error_ctx));
}

template <typename ErrorT>
auto AxonWorker::RethrowErrorContextHelper_::operator()(
  ErrorT error) const noexcept
  -> decltype(unifex::just_error(std::declval<errors::AxonErrorContext>())) {
  using ErrorType = std::decay_t<ErrorT>;
  if constexpr (kIsStdErrorCode<ErrorType>) {
    error_ctx.status = rpc::RpcStatus(std::forward<ErrorT>(error));
    error_ctx.what = error.message();
    return unifex::just_error(std::move(error_ctx));
  } else if constexpr (kIsStdExceptionPtr<ErrorType>) {
    try {
      std::rethrow_exception(std::forward<ErrorT>(error));
    } catch (const rpc::RpcException& e) {
      error_ctx.status = rpc::RpcStatus(e.code());
      error_ctx.what = e.what();
      return unifex::just_error(std::move(error_ctx));
    } catch (const errors::AxonErrorException& e) {
      return unifex::just_error(e.context());
    } catch (const std::exception& e) {
      error_ctx.status =
        rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::INTERNAL));
      error_ctx.what = e.what();
      return unifex::just_error(std::move(error_ctx));
    } catch (...) {
      error_ctx.status =
        rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::INTERNAL));
      error_ctx.what = "Unknown error in RethrowErrorContextHelper_";
      return unifex::just_error(std::move(error_ctx));
    }
  } else if constexpr (kIsAxonErrorContext<ErrorType>) {
    return unifex::just_error(std::forward<ErrorT>(error));
  } else {
    return unifex::just_error(std::move(error_ctx));
  }
}

// Explicit template instantiations for RethrowErrorContextHelper_::operator()
#define AXON_RETHROW_ERROR_CONTEXT_HELPER(ErrorType)                           \
  template auto AxonWorker::RethrowErrorContextHelper_::operator()<ErrorType>( \
    ErrorType) const noexcept                                                  \
    -> decltype(unifex::just_error(std::declval<errors::AxonErrorContext>()));

AXON_RETHROW_ERROR_CONTEXT_HELPER(std::error_code)
AXON_RETHROW_ERROR_CONTEXT_HELPER(std::exception_ptr)
AXON_RETHROW_ERROR_CONTEXT_HELPER(errors::AxonErrorContext)
AXON_RETHROW_ERROR_CONTEXT_HELPER(errors::AxonErrorContext&)
AXON_RETHROW_ERROR_CONTEXT_HELPER(const errors::AxonErrorContext&)
AXON_RETHROW_ERROR_CONTEXT_HELPER(errors::AxonErrorContext&&)

#undef AXON_RETHROW_ERROR_CONTEXT_HELPER

template <typename RequestHandlerInputT>
auto AxonWorker::ServerProcessValidMessage_(
  FullRequestHandlerView handler, uint64_t conn_id,
  const rpc::RpcRequestHeader* req_header_ptr,
  RequestHandlerInputT payload_or_key) -> ServerProcessValidMessageSenderType {
  auto handle_sender =
    (*handler)(
      conn_id, req_header_ptr,
      std::forward<RequestHandlerInputT>(payload_or_key))
    | unifex::let_error(RethrowErrorContextHelper_{.error_ctx{
      .conn_id = conn_id,
      .session_id = cista::to_idx(req_header_ptr->session_id),
      .request_id = cista::to_idx(req_header_ptr->request_id),
      .function_id = cista::to_idx(req_header_ptr->function_id),
      .hlc = hlc_,
      .workflow_id = cista::to_idx(req_header_ptr->workflow_id)}});

  auto process_sender =
    unifex::stop_when(
      unifex::on(thread_pool_.get_scheduler(), std::move(handle_sender)),
      unifex::on(GetTimeContextScheduler(), unifex::schedule_after(timeout_)))
    | unifex::let_done(ReturnTimeoutErrorContextHelper_{.error_ctx{
      .conn_id = conn_id,
      .session_id = cista::to_idx(req_header_ptr->session_id),
      .request_id = cista::to_idx(req_header_ptr->request_id),
      .function_id = cista::to_idx(req_header_ptr->function_id),
      .status =
        rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::DEADLINE_EXCEEDED)),
      .what = std::format(
        "Deadline exceeded {} seconds when executing "
        "registered function: {}",
        timeout_.count(),
        cista::to_idx(req_header_ptr->function_id)),
      .hlc = hlc_,
      .workflow_id = cista::to_idx(req_header_ptr->workflow_id),
    }});

  return process_sender;
}

template auto AxonWorker::ServerProcessValidMessage_<ucxx::UcxBuffer&&>(
  FullRequestHandlerView handler, uint64_t conn_id,
  const rpc::RpcRequestHeader* req_header_ptr, ucxx::UcxBuffer&& payload_or_key)
  -> ServerProcessValidMessageSenderType;

template auto AxonWorker::ServerProcessValidMessage_<size_t>(
  FullRequestHandlerView handler, uint64_t conn_id,
  const rpc::RpcRequestHeader* req_header_ptr, size_t payload_or_key)
  -> ServerProcessValidMessageSenderType;

std::expected<
  std::pair<const rpc::RpcRequestHeader*, AxonWorker::FullRequestHandlerView>,
  errors::AxonErrorContext>
AxonWorker::ServerProcessHeader_(
  uint64_t conn_id, std::string_view header_bytes) {
  const rpc::RpcRequestHeader* req_header_ptr;
  try {
    req_header_ptr =
      cista::deserialize<rpc::RpcRequestHeader, rpc::utils::SerializerMode>(
        header_bytes);
  } catch (const std::exception& e) {
    return std::unexpected(errors::AxonErrorContext{
      .conn_id = conn_id,
      .status = rpc::RpcStatus(
        std::make_error_code(errors::AxonErrc::DeserializeError)),
      .what = std::format("Invalid request header: {}", e.what()),
      .hlc = hlc_,
    });
  }

  // Update Hybrid Logical Clock
  hlc_.Merge(req_header_ptr->hlc);

  bool is_rejecting = IsRejectingMessages();

  // Check if this function_id is allowed to bypass backpressure
  bool can_bypass = false;
  if (is_rejecting && bypass_backpressure_checker_.has_value()) {
    can_bypass = (*bypass_backpressure_checker_)(
      rpc::function_id_t{req_header_ptr->function_id});
  }

  if (is_rejecting && !can_bypass) [[unlikely]] {
    return std::unexpected(errors::AxonErrorContext{
      .conn_id = conn_id,
      .session_id = cista::to_idx(req_header_ptr->session_id),
      .request_id = cista::to_idx(req_header_ptr->request_id),
      .function_id = cista::to_idx(req_header_ptr->function_id),
      .status = rpc::RpcStatus(
        std::make_error_code(errors::AxonErrc::StorageBackpressure)),
      .what = "Message rejected by server context",
      .hlc = hlc_,
      .workflow_id = cista::to_idx(req_header_ptr->workflow_id),
    });
  }

  auto handler_it = registered_handlers_.find(req_header_ptr->function_id);
  if (handler_it == registered_handlers_.end()) {
    return std::unexpected(errors::AxonErrorContext{
      .conn_id = conn_id,
      .session_id = cista::to_idx(req_header_ptr->session_id),
      .request_id = cista::to_idx(req_header_ptr->request_id),
      .function_id = cista::to_idx(req_header_ptr->function_id),
      .status = rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::NOT_FOUND)),
      .what = "Function not registered",
      .hlc = hlc_,
      .workflow_id = cista::to_idx(req_header_ptr->workflow_id),
    });
  }

  return std::make_pair(
    req_header_ptr, AxonWorker::FullRequestHandlerView(handler_it->second));
}

template <typename ReceivedBufferT, typename MemPolicy>
  requires std::is_same_v<ReceivedBufferT, rpc::PayloadVariant>
           && is_receiver_memory_policy_v<MemPolicy, ReceivedBufferT>
AxonWorker::UcxRecvBufferCombinedSender AxonWorker::ProcessRndvBuffer_(
  WorkerScheduler scheduler, uint64_t am_desc_key, MemPolicy mem_policy,
  const TensorMetaRefVec& tensor_metas) {
  constexpr const bool IsOnHost = std::is_same_v<MemPolicy, AlwaysOnHostPolicy>;
  constexpr const bool IsCustomMemory =
    std::is_same_v<MemPolicy, CustomMemoryPolicy<ReceivedBufferT>>;
  auto recv_sender_fn = [&]() {
    if constexpr (IsOnHost) {
      return ucxx::connection_recv_buffer(
        scheduler, am_desc_key, ucx_memory_type::HOST);
    } else if constexpr (IsCustomMemory) {
      auto payload_buffer = (*mem_policy)(tensor_metas);
      return std::visit(
        [this, scheduler, am_desc_key](auto&& arg) {
          return ucxx::connection_recv_buffer(
            scheduler, am_desc_key, std::move(arg));
        },
        std::move(payload_buffer));
    }
  };
  return recv_sender_fn();
}

namespace {
template <typename SenderT, typename MemPolicy, typename BufferReceiverFn>
auto ProcessRndvBufferImpl_(
  BufferReceiverFn&& buffer_receiver_fn,
  uint64_t am_desc_key,
  MemPolicy mem_policy,
  const TensorMetaRefVec& tensor_metas) {
  constexpr const bool IsOnHost = std::is_same_v<MemPolicy, AlwaysOnHostPolicy>;
  constexpr const bool IsCustomMemory =
    std::is_same_v<MemPolicy, CustomMemoryPolicy<SenderT>>;
  if constexpr (IsOnHost) {
    return buffer_receiver_fn(ucx_memory_type::HOST);
  } else if constexpr (IsCustomMemory) {
    auto payload_buffer = (*mem_policy)(tensor_metas);
    return buffer_receiver_fn(std::move(payload_buffer));
  }
}
}  // namespace

template <typename ReceivedBufferT, typename MemPolicy>
  requires std::is_same_v<ReceivedBufferT, ucxx::UcxBuffer>
           && is_receiver_memory_policy_v<MemPolicy, ReceivedBufferT>
auto AxonWorker::ProcessRndvBuffer_(
  WorkerScheduler scheduler, uint64_t am_desc_key, MemPolicy mem_policy,
  const TensorMetaRefVec& tensor_metas)
  -> ucxx::ucx_am_context::recv_buffer_sender {
  return ProcessRndvBufferImpl_<ucxx::UcxBuffer>(
    [&](auto&& arg) {
      return ucxx::connection_recv_buffer(
        scheduler, am_desc_key, std::forward<decltype(arg)>(arg));
    },
    am_desc_key, mem_policy, tensor_metas);
}

template <typename ReceivedBufferT, typename MemPolicy>
  requires std::is_same_v<ReceivedBufferT, ucxx::UcxBufferVec>
           && is_receiver_memory_policy_v<MemPolicy, ReceivedBufferT>
auto AxonWorker::ProcessRndvBuffer_(
  WorkerScheduler scheduler, uint64_t am_desc_key, MemPolicy mem_policy,
  const TensorMetaRefVec& tensor_metas)
  -> ucxx::ucx_am_context::recv_iovec_buffer_sender {
  return ProcessRndvBufferImpl_<ucxx::UcxBufferVec>(
    [&](auto&& arg) {
      using ArgType = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<ArgType, ucx_memory_type>) {
        // For UcxBufferVec with AlwaysOnHostPolicy, create an default empty
        // UcxBufferVec
        auto empty_buffer_vec = DefaultMemoryProvider(mr_, tensor_metas);
        return ucxx::connection_recv_buffer(
          scheduler, am_desc_key, std::move(empty_buffer_vec));
      } else {
        return ucxx::connection_recv_buffer(
          scheduler, am_desc_key, std::forward<decltype(arg)>(arg));
      }
    },
    am_desc_key, mem_policy, tensor_metas);
}

auto AxonWorker::ServerProcessMessage_(const RecvVariant& message) noexcept {
  auto handler = std::visit(
    [this](const auto& message) -> ServerMessageHandlerSender {
      constexpr bool IsRndvPath = std::is_same_v<
        std::decay_t<decltype(message)>, std::pair<size_t, ucxx::UcxHeader>>;
      constexpr bool IsEagerPath = std::is_same_v<
        std::decay_t<decltype(message)>,
        ucxx::active_message_header_buffer_bundle>;
      if constexpr (IsRndvPath) {
        auto& [am_desc_key, header] = message;
        std::string_view header_bytes{
          reinterpret_cast<const char*>(header.data()), header.size()};
        uint64_t conn_id = 0;
        auto am_desc_opt = server_ctx_->view_pending_am_desc(am_desc_key);
        if (am_desc_opt.has_value()) {
          conn_id = am_desc_opt.value().get().conn_id;
        } else {
          return unifex::just_error(errors::AxonErrorContext{
            .status = rpc::RpcStatus(rpc::RpcErrc::NOT_FOUND),
            .what = "AM descriptor not found",
            .hlc = hlc_,
          });
        }
        const auto req_header_and_handler =
          ServerProcessHeader_(conn_id, header_bytes);
        if (req_header_and_handler.has_value()) {
          auto [req_header_ptr, handler] = req_header_and_handler.value();
          return ServerProcessValidMessage_<size_t>(
            handler, conn_id, req_header_ptr, am_desc_key);
        } else {
          // When rejecting RNDV message, we need to release the AM descriptor
          // to avoid resource leak and deadlock
          auto popped_desc = server_ctx_->pop_pending_am_desc(am_desc_key);
          if (popped_desc.has_value()) {
            // Release the RNDV AM descriptor (header and descriptor)
            // Header ownership is transferred to the caller, so we don't need
            // to release the header
            server_ctx_->release_rndv_am_desc(
              std::move(popped_desc.value()), mr_, /*release_header=*/false);
          }
          return unifex::just_error(std::move(req_header_and_handler.error()));
        }
      } else if constexpr (IsEagerPath) {
        const auto& header = message.header();
        const auto conn_id = message.connection().id();
        std::string_view header_bytes{
          reinterpret_cast<const char*>(header.data()), header.size()};
        const auto req_header_and_handler =
          ServerProcessHeader_(conn_id, header_bytes);
        if (req_header_and_handler.has_value()) {
          auto& [req_header_ptr, handler] = req_header_and_handler.value();
          return ServerProcessValidMessage_<ucxx::UcxBuffer&&>(
            handler, conn_id, req_header_ptr,
            std::move(
              const_cast<ucxx::active_message_header_buffer_bundle&>(message)
                .move_buffer()));
        } else {
          return unifex::just_error(std::move(req_header_and_handler.error()));
        }
      } else {
        return unifex::just_error(errors::AxonErrorContext{
          .status = rpc::RpcStatus(rpc::RpcErrc::INVALID_ARGUMENT),
          .what = "Invalid received message type",
          .hlc = hlc_,
        });
      }
    },
    message);

  return handler;
}

inline void AxonWorker::ServerCtxLoopSpawnImpl_(
  std::reference_wrapper<unifex::inplace_stop_source> stop_source,
  RecvVariant&& header_or_bundle) {
  auto stop_token = stop_source.get().get_token();
  auto header_or_bundle_ptr =
    std::make_shared<RecvVariant>(std::move(header_or_bundle));
  auto process_sender = unifex::let_value_with(
    [header_or_bundle_ptr]() { return std::move(*header_or_bundle_ptr); },
    [this](const RecvVariant& header_or_bundle) noexcept {
      auto message_processor_sender =
        ServerProcessMessage_(header_or_bundle)
        | unifex::let_error([this](auto&& error) noexcept {
            using ErrorT = std::decay_t<decltype(error)>;
            if constexpr (kIsAxonErrorContext<ErrorT>) {
              ReportServerError_(error);
              return ServerHandleSendErrorResponse_(error);
            } else {
              return unifex::just_error(std::forward<ErrorT>(error));
            }
          })
        | unifex::upon_error(
          [this](std::variant<
                 std::error_code, std::exception_ptr,
                 errors::AxonErrorContext>&& error) noexcept {
            std::visit(
              [this](auto&& error) {
                using ErrorType = std::decay_t<decltype(error)>;
                if constexpr (kIsAxonErrorContext<ErrorType>) {
                  ReportServerError_(error);
                } else {
                  auto error_ctx = errors::AxonErrorContext{
                    .hlc = this->hlc_, .workflow_id = 0};
                  if constexpr (kIsStdErrorCode<ErrorType>) {
                    error_ctx.status = rpc::RpcStatus(std::move(error));
                    error_ctx.what = std::format(
                      "ServerCtxLoopImpl_ failed: {}", error.message());
                  } else if constexpr (kIsStdExceptionPtr<ErrorType>) {
                    try {
                      std::rethrow_exception(error);
                    } catch (const rpc::RpcException& e) {
                      error_ctx.status = rpc::RpcStatus(e.code());
                      error_ctx.what = e.what();
                    } catch (const std::exception& e) {
                      error_ctx.status = rpc::RpcStatus(
                        std::make_error_code(rpc::RpcErrc::INTERNAL));
                      error_ctx.what =
                        std::format("ServerCtxLoopImpl_ failed: {}", e.what());
                    } catch (...) {
                      error_ctx.status = rpc::RpcStatus(
                        std::make_error_code(rpc::RpcErrc::INTERNAL));
                      error_ctx.what = "Unknown error in ServerCtxLoopImpl_";
                    }
                  }
                  ReportServerError_(error_ctx);
                }
              },
              std::move(error));
          });
      return message_processor_sender;
    });

  // TIP(He Jia): if use &stop_token here may cause deadlock. I don't know why.
  unifex::spawn_detached(
    unifex::with_query_value(
      std::move(process_sender), unifex::get_stop_token, std::move(stop_token)),
    server_async_scope_);
}

auto AxonWorker::ServerCtxLoopImpl_(
  std::reference_wrapper<unifex::inplace_stop_source> stop_source) {
  return ucxx::connection_recv_header(server_ctx_->get_scheduler())
         | unifex::then([this, stop_source](auto&& header_or_bundle) {
             ++server_pipe_invoked_count_;
             ServerCtxLoopSpawnImpl_(stop_source, std::move(header_or_bundle));
           });
}

auto AxonWorker::ServerCtxLoop_(
  std::reference_wrapper<unifex::inplace_stop_source> stop_source) {
  auto stop_token = stop_source.get().get_token();
  auto defer_fn = unifex::defer([this, stop_source]() {
    return ServerCtxLoopImpl_(stop_source)
           | unifex::upon_error(
             [this](std::variant<std::error_code, std::exception_ptr>&&
                      error) mutable {
               std::visit(
                 [this](auto&& error) {
                   using ErrorType = std::decay_t<decltype(error)>;
                   std::error_code ec_value;
                   std::string what;
                   if constexpr (kIsStdErrorCode<ErrorType>) {
                     ec_value = std::move(error);
                     what = std::format(
                       "ServerCtxLoop_ failed: {}", ec_value.message());
                   } else if constexpr (kIsStdExceptionPtr<ErrorType>) {
                     try {
                       std::rethrow_exception(error);
                     } catch (const std::exception& e) {
                       ec_value = std::make_error_code(rpc::RpcErrc::INTERNAL);
                       what =
                         std::format("ServerCtxLoop_ failed: {}", e.what());
                     } catch (...) {
                       ec_value = std::make_error_code(rpc::RpcErrc::INTERNAL);
                       what = "Unknown error in ServerCtxLoop_";
                     }
                   }

                   ++server_pipe_failed_count_;
                   ReportServerError_(errors::AxonErrorContext{
                     .status = rpc::RpcStatus(ec_value),
                     .what = std::move(what),
                     .hlc = this->hlc_});
                 },
                 std::move(error));
             });
  });
  return unifex::repeat_effect_until(
    unifex::with_query_value(
      std::move(defer_fn), unifex::get_stop_token, std::move(stop_token)),
    [stop_source]() { return stop_source.get().stop_requested(); });
}

namespace {
template <typename T>
struct is_variant : std::false_type {};

template <typename... Args>
struct is_variant<std::variant<Args...>> : std::true_type {};
}  // namespace

template <typename PayloadT>
AxonWorker::UcxSendCombinedSender AxonWorker::ClientSendRequest_(
  std::expected<uint64_t, std::error_code> conn_id,
  const cista::byte_buf& header_buf,
  const PayloadT& payload) {
  if (!conn_id.has_value()) {
    return unifex::just_error(conn_id.error());
  }
  auto send_sched = client_ctx_->get_scheduler();
  auto visitor = [&](auto&& payload) -> UcxSendCombinedSender {
    using T = std::decay_t<decltype(payload)>;
    if constexpr (std::is_same_v<T, ucxx::UcxBufferVec>) {
      ucx_am_iovec_t iov_data{
        .header{
          .data = const_cast<unsigned char*>(header_buf.data()),
          .size = header_buf.size(),
        },
        .buffer_vec = const_cast<ucx_buffer_t*>(payload.buffers().data()),
        .buffer_count = payload.size(),
        .buffer_type = payload.type(),
        .mem_h = nullptr,
      };
      return eux::ucxx::connection_send(
        send_sched, *conn_id,
        ucxx::UcxAmIovec(
          mr_, std::move(iov_data), /*own_header=*/false,
          /*own_buffer=*/false));
    } else if constexpr (std::is_same_v<T, ucxx::UcxBuffer>) {
      ucx_am_data send_data{
        .header{
          .data = const_cast<unsigned char*>(header_buf.data()),
          .size = header_buf.size(),
        },
        .buffer = *payload.get(),
        .buffer_type = payload.type(),
        .mem_h = nullptr,
      };
      return eux::ucxx::connection_send(
        send_sched, *conn_id,
        ucxx::UcxAmData(
          mr_, std::move(send_data), /*own_header=*/false,
          /*own_buffer=*/false));
    } else {
      ucx_am_data send_data{
        .header{
          .data = const_cast<unsigned char*>(header_buf.data()),
          .size = header_buf.size(),
        },
        .buffer{
          .data = nullptr,
          .size = 0,
        },
        .buffer_type = ucx_memory_type::HOST,
        .mem_h = nullptr,
      };
      return eux::ucxx::connection_send(
        send_sched, *conn_id,
        ucxx::UcxAmData(
          mr_, std::move(send_data), /*own_header=*/false,
          /*own_buffer=*/false));
    }
  };
  if constexpr (is_variant<PayloadT>::value) {
    return std::visit(visitor, payload);
  } else {
    return visitor(payload);
  }
}

#define AXON_SEND_REQUEST_TEMPLATE(PayloadT)          \
  template AxonWorker::UcxSendCombinedSender          \
  AxonWorker::ClientSendRequest_<PayloadT>(           \
    std::expected<uint64_t, std::error_code> conn_id, \
    const cista::byte_buf& header_buf, const PayloadT& payload);

AXON_SEND_REQUEST_TEMPLATE(rpc::PayloadVariant)
AXON_SEND_REQUEST_TEMPLATE(ucxx::UcxBufferVec)
AXON_SEND_REQUEST_TEMPLATE(ucxx::UcxBuffer)

#undef AXON_SEND_REQUEST_TEMPLATE

std::expected<
  std::pair<rpc::ResponseHeaderUniquePtr, AxonWorker::RpcResponseHandler>,
  errors::AxonErrorContext>
AxonWorker::ClientProcessHeader_(ucxx::UcxHeader&& header) {
  rpc::ResponseHeaderUniquePtr response_ptr;

  try {
    response_ptr = rpc::RpcDispatcher::DeerializeResponse(std::move(header));
  } catch (const std::exception& e) {
    return std::unexpected(errors::AxonErrorContext{
      .status = rpc::RpcStatus(
        std::make_error_code(errors::AxonErrc::DeserializeError)),
      .what = std::format("Invalid response header: {}", e.what()),
      .hlc = hlc_,
    });
  }

  // Update Hybrid Logical Clock
  hlc_.Merge(response_ptr->hlc);

  size_t resp_handler_idx = pending_rpcs_.find(
    cista::to_idx(response_ptr->request_id), response_ptr->session_id);

  if (resp_handler_idx == kPendingRpcBufferSize) {
    return std::unexpected(errors::AxonErrorContext{
      .session_id = cista::to_idx(response_ptr->session_id),
      .request_id = cista::to_idx(response_ptr->request_id),
      .status = rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::NOT_FOUND)),
      .what = "Response handler not found",
      .hlc = hlc_,
      .workflow_id = cista::to_idx(response_ptr->workflow_id),
    });
  }

  auto resp_handler = std::move(pending_rpcs_.at(resp_handler_idx));
  pending_rpcs_.erase(cista::to_idx(response_ptr->request_id));
  return std::make_pair(std::move(response_ptr), std::move(resp_handler));
}

inline void AxonWorker::ClientProcessExpectedResponseMessage_(
  RpcResponseHandler&& resp_handler,
  RpcResponseHandlerExpectedType&& resp_handler_expected) {
  (*resp_handler)(std::move(resp_handler_expected));
}

auto AxonWorker::ClientProcessResponseMessage_(RecvVariant&& message) noexcept {
  return std::visit(
    [this](auto& message) {
      constexpr bool IsRndvPath = std::is_same_v<
        std::decay_t<decltype(message)>, std::pair<size_t, ucxx::UcxHeader>>;
      constexpr bool IsEagerPath = std::is_same_v<
        std::decay_t<decltype(message)>,
        ucxx::active_message_header_buffer_bundle>;

      if constexpr (IsRndvPath) {
        auto&& [am_desc_key, header] = std::move(message);
        auto resp_header_ptr_and_handler =
          ClientProcessHeader_(std::move(header));
        if (!resp_header_ptr_and_handler.has_value()) [[unlikely]] {
          ReportClientError_(resp_header_ptr_and_handler.error());
          return;
        }
        auto&& [resp_header_ptr, resp_handler] =
          std::move(resp_header_ptr_and_handler.value());
        return ClientProcessExpectedResponseMessage_(
          std::move(resp_handler),
          RpcResponseHandlerExpectedType(
            std::in_place,
            std::make_pair(std::move(resp_header_ptr), am_desc_key)));
      } else if constexpr (IsEagerPath) {
        auto&& header = message.move_header();
        auto resp_header_ptr_and_handler =
          ClientProcessHeader_(std::move(header));
        if (!resp_header_ptr_and_handler.has_value()) [[unlikely]] {
          ReportClientError_(resp_header_ptr_and_handler.error());
          return;
        }
        auto&& [resp_header_ptr, resp_handler] =
          std::move(resp_header_ptr_and_handler.value());
        return ClientProcessExpectedResponseMessage_(
          std::move(resp_handler),
          RpcResponseHandlerExpectedType(
            std::in_place,
            std::make_pair(std::move(resp_header_ptr), message.move_buffer())));
      } else {
        ReportClientError_(errors::AxonErrorContext{
          .status = rpc::RpcStatus(rpc::RpcErrc::INVALID_ARGUMENT),
          .what = "Invalid received response message type",
          .hlc = hlc_,
        });
        return;
      }
    },
    message);
}

auto AxonWorker::ClientCtxLoopImpl_() {
  return ucxx::connection_recv_header(client_ctx_->get_scheduler())
         | unifex::then([this](auto&& header_or_bundle) -> void {
             ClientProcessResponseMessage_(std::move(header_or_bundle));
           });
}

auto AxonWorker::ClientCtxLoop_(
  std::reference_wrapper<unifex::inplace_stop_source> stop_source) {
  auto stop_token = stop_source.get().get_token();
  auto defer_fn = unifex::defer([this]() {
    return ClientCtxLoopImpl_()
           | unifex::upon_error(
             [this](std::variant<std::error_code, std::exception_ptr>&&
                      error) mutable {
               std::visit(
                 [this](auto&& error) {
                   using ErrorType = std::decay_t<decltype(error)>;
                   std::error_code ec_value;
                   std::string what;
                   if constexpr (kIsStdErrorCode<ErrorType>) {
                     ec_value = std::move(error);
                     what = std::format(
                       "ClientCtxLoop_ failed: {}", ec_value.message());
                   } else if constexpr (kIsStdExceptionPtr<ErrorType>) {
                     try {
                       std::rethrow_exception(error);
                     } catch (const std::exception& e) {
                       ec_value = std::make_error_code(rpc::RpcErrc::INTERNAL);
                       what =
                         std::format("ClientCtxLoop_ failed: {}", e.what());
                     } catch (...) {
                       ec_value = std::make_error_code(rpc::RpcErrc::INTERNAL);
                       what = "Unknown error in ClientCtxLoop_";
                     }
                   }

                   ++client_pipe_failed_count_;
                   ReportClientError_(errors::AxonErrorContext{
                     .status = rpc::RpcStatus(ec_value),
                     .what = std::move(what),
                     .hlc = this->hlc_});
                 },
                 std::move(error));
             });
  });
  return unifex::repeat_effect_until(
    unifex::with_query_value(
      std::move(defer_fn), unifex::get_stop_token, std::move(stop_token)),
    [stop_source]() { return stop_source.get().stop_requested(); });
}

std::error_code AxonWorker::StartServer() {
  if (server_running_.exchange(true)) {
    return {};
  }
  try {
    server_thread_ = std::thread(
      [this]() { server_ctx_->run(server_context_stop_source_.get_token()); });

    // The recv_loop is started in the recv_thread
    server_future_ = unifex::spawn_future(
      unifex::on(
        server_ctx_->get_scheduler(), ServerCtxLoop_(server_stop_source_)),
      server_async_scope_);
  } catch (const std::exception& e) {
    std::cerr << "StartServer failed: " << e.what() << std::endl;
    server_running_.exchange(false);
    return std::make_error_code(errors::AxonErrc::WorkerStartFailed);
  }

  return {};
}

std::error_code AxonWorker::StartClient() {
  if (client_running_.exchange(true)) {
    return {};
  }
  try {
    client_thread_ = std::thread(
      [this]() { client_ctx_->run(client_context_stop_source_.get_token()); });

    // The send_loop is started in the send_thread
    client_future_ = unifex::spawn_future(
      unifex::on(
        client_ctx_->get_scheduler(), ClientCtxLoop_(client_stop_source_)),
      client_async_scope_);
  } catch (const std::exception& e) {
    std::cerr << "StartClient failed: " << e.what() << std::endl;
    client_running_.exchange(false);
    return std::make_error_code(errors::AxonErrc::WorkerStartFailed);
  }

  return {};
}

std::expected<void, std::error_code> AxonWorker::ProcessSingleStoredRequest(
  StorageRequestIt storage_it, rpc::DynamicAsyncRpcFunctionView func,
  RequestVisitor& visitor) {
  auto executor = pro::make_proxy<ExecutorFacade>(
    [this, storage_it,
     &func]() mutable -> unifex::any_sender_of<rpc::DynamicFunctionReturnType> {
      auto& req_ptr = *storage_it;
      auto sig_opt = dispatcher_.GetSignature(req_ptr->header.function_id);
      if (!sig_opt) {
        return unifex::just_error(std::make_exception_ptr(rpc::RpcException(
          std::make_error_code(rpc::RpcErrc::NOT_FOUND),
          "Function not found")));
      }

      rpc::RpcContextPtr context_ptr(
        std::holds_alternative<std::monostate>(req_ptr->payload)
          ? nullptr
          : &req_ptr->payload,
        [](void*) {});

      return dispatcher_.DispatchAdhoc(
               func, sig_opt->input_payload_type, req_ptr->header,
               std::move(context_ptr), response_builder_)
             | unifex::then([&](auto&& result) {
                 auto& [response_header, payload] = result;
                 return std::make_pair(
                   std::move(response_header.results), std::move(payload));
               });
    });
  auto lifecycle_status = (*visitor)(std::move(executor));
  if (
    lifecycle_status == LifecycleStatus::Discard
    || lifecycle_status == LifecycleStatus::Error) {
    auto axon_req_id = GetMessageID((*storage_it)->header);
    request_id_to_iterator_.erase(axon_req_id);
    storage_->erase(storage_it);
    if (lifecycle_status == LifecycleStatus::Error) {
      return std::unexpected(std::make_error_code(rpc::RpcErrc::INTERNAL));
    }
  }
  return std::expected<void, std::error_code>(std::in_place);
}

std::expected<void, std::error_code> AxonWorker::ProcessStoredRequests(
  AxonRequestID req_id, rpc::DynamicAsyncRpcFunctionView func,
  RequestVisitor visitor) {
  auto it = request_id_to_iterator_.find(req_id);
  if (it == request_id_to_iterator_.end()) {
    return std::unexpected(std::make_error_code(rpc::RpcErrc::NOT_FOUND));
  }
  return ProcessSingleStoredRequest(it->second, func, visitor);
}

std::expected<void, std::error_code> AxonWorker::ProcessStoredRequests(
  rpc::DynamicAsyncRpcFunctionView func, RequestVisitor visitor) {
  for (auto it = storage_->begin(); it != storage_->end(); ++it) {
    auto result = ProcessSingleStoredRequest(it, func, visitor);
    if (!result) {
      return result;
    }
  }
  return std::expected<void, std::error_code>(std::in_place);
}

void AxonWorker::EraseStoredRequest(StorageRequestIt&& it) {
  request_id_to_iterator_.erase(GetMessageID((*it)->header));
  storage_->erase(it);
}

std::optional<AxonWorker::AxonRequestPtr> AxonWorker::FindStoredRequest(
  AxonWorker::AxonRequestID request_id) {
  auto it = request_id_to_iterator_.find(request_id);
  if (it != request_id_to_iterator_.end()) {
    return *(it->second);
  }
  return std::nullopt;
}

void AxonWorker::ReportServerMetrics() {
  if (server_metrics_observer_) {
    size_t pending_count = pending_rpcs_.size();
    server_metrics_observer_->OnWorkerStats(metrics::WorkerMetrics{
      .pending_rpcs_count = pending_count,
      .client_pipe_failed_count = client_pipe_failed_count_,
      .server_pipe_failed_count = server_pipe_failed_count_,
    });
  }
}

void AxonWorker::ReportClientMetrics() {
  if (client_metrics_observer_) {
    size_t pending_count = pending_rpcs_.size();
    client_metrics_observer_->OnWorkerStats(metrics::WorkerMetrics{
      .pending_rpcs_count = pending_count,
      .client_pipe_failed_count = client_pipe_failed_count_,
      .server_pipe_failed_count = server_pipe_failed_count_,
    });
  }
}

AxonWorker::AxonRequestID AxonWorker::GetMessageID(
  const rpc::RpcRequestHeader& request_header) {
  return (request_header.hlc).Raw();
}

auto AxonWorker::GetTimeContextScheduler()
  -> decltype(std::declval<unifex::timed_single_thread_context>()
                .get_scheduler()) {
  return time_ctx_.get_scheduler();
}

inline auto AxonWorker::GetNowNanoseconds() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch());
}
inline auto AxonWorker::GetNowMicroseconds() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch());
}
inline auto AxonWorker::GetNowMilliseconds() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch());
}

/*
.get() means it's a std::reference_wrapper<ucxx::UcxBuffer>
If function was registered with ucxx::UcxBufferVec, we need to
convert it to ucxx::UcxBufferVec
*/
ucxx::UcxBufferVec AxonWorker::DefaultMemoryProvider(
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr,
  const TensorMetaRefVec& tensor_metas) {
  std::vector<size_t> sizes;
  sizes.reserve(tensor_metas.size());
  ucx_memory_type_t memory_type = ucx_memory_type::HOST;
  for (const auto& tensor_meta_ref : tensor_metas) {
    const auto& meta = tensor_meta_ref.get();
    sizes.push_back(rpc::utils::CalculateTensorSize(meta));
  }
  return ucxx::UcxBufferVec(mr, memory_type, sizes);
}

ucxx::UcxBuffer AxonWorker::DefaultMemoryProvider(
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr,
  TensorMetaRef tensor_meta) {
  const auto& meta = tensor_meta.get();
  size_t size = rpc::utils::CalculateTensorSize(meta);
  ucx_memory_type_t memory_type = GetMemoryType(meta.device);
  return ucxx::UcxBuffer(mr, memory_type, size);
}

std::reference_wrapper<ucxx::UcxMemoryResourceManager>
AxonWorker::GetMemoryResourceManager() const {
  return mr_;
}

// Helper function to convert DLDevice to ucx_memory_type
constexpr inline ucx_memory_type GetMemoryType(DLDevice device) {
  switch (device.device_type) {
    case kDLCPU:
      return ucx_memory_type::HOST;
    case kDLCUDA:
    case kDLCUDAHost:
      return ucx_memory_type::CUDA;
    case kDLROCM:
      return ucx_memory_type::ROCM;
    default:
      // Default to HOST for unknown device types
      return ucx_memory_type::HOST;
  }
}

namespace {
// Helper struct to register the UCX error category at static initialization
// time.
struct UcxErrorCategoryRegistrar {
  UcxErrorCategoryRegistrar() {
    rpc::RpcCategoryRegistry::GetInstance().RegisterCategory(
      ucxx::ucx_error_category());
  }
};
static UcxErrorCategoryRegistrar ucx_category_registrar_;

// Helper struct to register the std::system_error category at static
// initialization time.
struct SystemErrorCategoryRegistrar {
  SystemErrorCategoryRegistrar() {
    rpc::RpcCategoryRegistry::GetInstance().RegisterCategory(
      std::generic_category());
  }
};
static SystemErrorCategoryRegistrar system_error_category_registrar_;
}  // namespace

}  // namespace axon
}  // namespace eux
