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

#include "axon/axon_runtime.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace eux {
namespace axon {

AxonRuntime::AxonRuntime(
  std::shared_ptr<ucxx::UcxMemoryResourceManager> mr,
  const std::string& worker_name, size_t thread_pool_size,
  std::chrono::milliseconds timeout,
  std::unique_ptr<ucxx::UcxAutoDeviceContext> auto_device_context)
  : mr_(mr),
    worker_(std::make_unique<AxonWorker>(
      std::ref(*mr_), worker_name, thread_pool_size, timeout,
      auto_device_context
        ? std::move(auto_device_context)
        : std::make_unique<ucxx::UcxAutoDefaultDeviceContext>())) {}

AxonRuntime::AxonRuntime(
  const std::string& worker_name, size_t thread_pool_size,
  std::chrono::milliseconds timeout,
  std::unique_ptr<ucxx::UcxAutoDeviceContext> auto_device_context)
  : mr_(std::make_shared<ucxx::DefaultUcxMemoryResourceManager>()),
    worker_(std::make_unique<AxonWorker>(
      std::ref(*mr_), worker_name, thread_pool_size, timeout,
      auto_device_context
        ? std::move(auto_device_context)
        : std::make_unique<ucxx::UcxAutoDefaultDeviceContext>())) {}

AxonRuntime::~AxonRuntime() {
  if (worker_) {
    worker_->Stop();
  }
}

std::expected<void, errors::AxonErrorContext> AxonRuntime::StartServer() {
  return worker_->StartServer();
}

std::expected<void, errors::AxonErrorContext> AxonRuntime::StartClient() {
  return worker_->StartClient();
}

std::expected<void, errors::AxonErrorContext> AxonRuntime::Start() {
  return worker_->Start();
}

void AxonRuntime::Stop() {
  if (worker_) {
    worker_->Stop();
  }
}

void AxonRuntime::StopServer() { worker_->StopServer(); }

void AxonRuntime::StopClient() { worker_->StopClient(); }

void AxonRuntime::ReportServerMetrics() { worker_->ReportServerMetrics(); }

void AxonRuntime::ReportClientMetrics() { worker_->ReportClientMetrics(); }

storage::AxonStorage& AxonRuntime::GetStorage() {
  return worker_->GetStorage();
}

rpc::AsyncRpcDispatcher& AxonRuntime::GetDispatcher() {
  return worker_->GetDispatcher();
}

std::vector<std::byte> AxonRuntime::GetLocalAddress() {
  return worker_->GetLocalAddress();
}

cista::byte_buf AxonRuntime::GetLocalSignatures() const {
  return worker_->GetLocalSignatures();
}

std::expected<AxonRuntime::WorkerKey, errors::AxonErrorContext>
AxonRuntime::RegisterEndpointSignatures(
  std::string_view worker_name, const cista::byte_buf& signatures_blob) {
  return worker_->RegisterEndpointSignatures(worker_name, signatures_blob);
}

auto AxonRuntime::RegisterEndpointSignaturesAsync(
  std::string_view worker_name, const cista::byte_buf& signatures_blob) {
  return worker_->RegisterEndpointSignaturesAsync(worker_name, signatures_blob);
}

std::expected<uint64_t, errors::AxonErrorContext> AxonRuntime::ConnectEndpoint(
  std::vector<std::byte> ucp_address, std::string_view worker_name) {
  return worker_->ConnectEndpoint(
    std::move(ucp_address), std::move(worker_name));
}

auto AxonRuntime::ConnectEndpointAsync(
  std::vector<std::byte> ucp_address, std::string_view worker_name)
  -> decltype(std::declval<AxonWorker>().ConnectEndpointAsync(
    std::move(ucp_address), worker_name)) {
  return worker_->ConnectEndpointAsync(std::move(ucp_address), worker_name);
}

void AxonRuntime::SetTimeout(std::chrono::milliseconds timeout) {
  worker_->SetTimeout(timeout);
}

void AxonRuntime::SetRejectMessages(bool reject) {
  worker_->SetRejectMessages(reject);
}

void AxonRuntime::SetServerErrorObserver(errors::ErrorObserver obs) {
  worker_->SetServerErrorObserver(std::move(obs));
}

void AxonRuntime::SetClientErrorObserver(errors::ErrorObserver obs) {
  worker_->SetClientErrorObserver(std::move(obs));
}

void AxonRuntime::SetServerMetricsObserver(metrics::MetricsObserver obs) {
  worker_->SetServerMetricsObserver(std::move(obs));
}

void AxonRuntime::SetClientMetricsObserver(metrics::MetricsObserver obs) {
  worker_->SetClientMetricsObserver(std::move(obs));
}

bool AxonRuntime::IsRejectingMessages() const noexcept {
  return worker_->IsRejectingMessages();
}

AxonRuntime::AxonRequestID AxonRuntime::GetMessageID(
  const rpc::RpcRequestHeader& request_header) {
  return AxonWorker::GetMessageID(request_header);
}

std::expected<void, errors::AxonErrorContext>
AxonRuntime::ProcessStoredRequests(
  AxonRequestID req_id, rpc::DynamicAsyncRpcFunctionView func,
  RequestVisitor visitor) {
  return worker_->ProcessStoredRequests(req_id, func, std::move(visitor));
}

std::expected<void, errors::AxonErrorContext>
AxonRuntime::ProcessStoredRequests(
  AxonRequestID req_id, StorageIteratorVisitor visitor) {
  return worker_->ProcessStoredRequests(req_id, std::move(visitor));
}

std::expected<void, errors::AxonErrorContext>
AxonRuntime::ProcessStoredRequests(
  rpc::DynamicAsyncRpcFunctionView func, RequestVisitor visitor) {
  return worker_->ProcessStoredRequests(func, std::move(visitor));
}

std::expected<void, errors::AxonErrorContext>
AxonRuntime::ProcessStoredRequests(StorageIteratorVisitor visitor) {
  return worker_->ProcessStoredRequests(std::move(visitor));
}

void AxonRuntime::EraseStoredRequest(StorageRequestIt&& it) {
  worker_->EraseStoredRequest(std::move(it));
}

std::optional<AxonRuntime::AxonRequestPtr> AxonRuntime::FindStoredRequest(
  AxonRequestID request_id) {
  return worker_->FindStoredRequest(request_id);
}

std::expected<void, errors::AxonErrorContext>
AxonRuntime::ProcessSingleStoredRequest(
  StorageRequestIt storage_it, rpc::DynamicAsyncRpcFunctionView func,
  RequestVisitor& visitor) {
  return worker_->ProcessSingleStoredRequest(
    std::move(storage_it), func, visitor);
}

auto AxonRuntime::GetTimeContextScheduler()
  -> decltype(std::declval<unifex::timed_single_thread_context>()
                .get_scheduler()) {
  return worker_->GetTimeContextScheduler();
}

ucxx::UcxBufferVec AxonRuntime::DefaultMemoryProvider(
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr_,
  const utils::TensorMetaSpan tensor_metas) {
  return AxonWorker::DefaultMemoryProvider(mr_, tensor_metas);
}

ucxx::UcxBuffer AxonRuntime::DefaultMemoryProvider(
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr_,
  const rpc::utils::TensorMeta& tensor_meta) {
  return AxonWorker::DefaultMemoryProvider(mr_, tensor_meta);
}

std::reference_wrapper<ucxx::UcxMemoryResourceManager>
AxonRuntime::GetMemoryResourceManager() const {
  return worker_->GetMemoryResourceManager();
}

}  // namespace axon
}  // namespace eux
