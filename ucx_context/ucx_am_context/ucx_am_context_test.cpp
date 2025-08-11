/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.

Licensed under the Apache License Version 2.0 with LLVM Exceptions
(the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    https://llvm.org/LICENSE.txt

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "ucx_context/ucx_am_context/ucx_am_context.hpp"

#include <arpa/inet.h>
#if CUDA_ENABLED
#include <cuda_runtime.h>
#endif
#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unifex/defer.hpp>
#include <unifex/for_each.hpp>
#include <unifex/get_stop_token.hpp>
#include <unifex/just.hpp>
#include <unifex/just_done.hpp>
#include <unifex/just_error.hpp>
#include <unifex/just_from.hpp>
#include <unifex/let_done.hpp>
#include <unifex/let_error.hpp>
#include <unifex/on.hpp>
#include <unifex/overload.hpp>
#include <unifex/range_stream.hpp>
#include <unifex/repeat_effect_until.hpp>
#include <unifex/scope_guard.hpp>
#include <unifex/sequence.hpp>
#include <unifex/single.hpp>
#include <unifex/spawn_detached.hpp>
#include <unifex/spawn_future.hpp>
#include <unifex/static_thread_pool.hpp>
#include <unifex/stop_if_requested.hpp>
#include <unifex/stop_immediately.hpp>
#include <unifex/stop_on_request.hpp>
#include <unifex/stop_when.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/take_until.hpp>
#include <unifex/task.hpp>
#include <unifex/then.hpp>
#include <unifex/v2/async_scope.hpp>
#include <unifex/when_all.hpp>
#include <unifex/with_query_value.hpp>

#include "ucx_context/ucx_connection.hpp"
#include "ucx_context/ucx_context_def.h"
#include "ucx_context/ucx_memory_resource.hpp"
#include "ucx_context/ucx_status.hpp"

#if CUDA_ENABLED
#include "ucx_context/cuda/ucx_cuda_context.hpp"
#include "ucx_context/cuda/ucx_cuda_memory_manager.hpp"
#endif
#include "ucx_context/ucx_am_context/ucx_am_context_test_helper.h"

using stdexe_ucx_runtime::accept_endpoint;
using stdexe_ucx_runtime::connect_endpoint;
using stdexe_ucx_runtime::connection_recv;
using stdexe_ucx_runtime::connection_send;
using stdexe_ucx_runtime::DefaultUcxMemoryResourceManager;
#if CUDA_ENABLED
using stdexe_ucx_runtime::UcxCudaMemoryResourceManager;
#endif
using stdexe_ucx_runtime::active_message_bundle;
using stdexe_ucx_runtime::handle_error_connection;
using stdexe_ucx_runtime::ucx_am_context;
using stdexe_ucx_runtime::UcxAmDesc;
using stdexe_ucx_runtime::UcxConnection;
using stdexe_ucx_runtime::UcxMemoryResourceManager;

using unifex::current_scheduler;
using unifex::defer;
using unifex::for_each;
using unifex::get_stop_token;
using unifex::inplace_stop_source;
using unifex::just;
using unifex::just_done;
using unifex::just_error;
using unifex::just_from;
using unifex::let_done;
using unifex::let_error;
using unifex::on;
using unifex::range_stream;
using unifex::repeat_effect_until;
using unifex::schedule_after;
using unifex::sequence;
using unifex::single;
using unifex::spawn_detached;
using unifex::spawn_future;
using unifex::static_thread_pool;
using unifex::stop_if_requested;
using unifex::stop_immediately;
using unifex::stop_on_request;
using unifex::stop_when;
using unifex::sync_wait;
using unifex::take_until;
using unifex::task;
using unifex::then;
using unifex::when_all;
using unifex::with_query_value;
using unifex::v2::async_scope;

static constexpr size_t kUcxRndvThreshold = 8192;

// Helper class to manage UCX context and thread
class UcxContextRunner {
 public:
  explicit UcxContextRunner(
    std::string name, std::chrono::seconds timeout = std::chrono::seconds(300))
    : name_(name), timeout_(timeout) {}

  virtual ~UcxContextRunner() { cleanup(); }

  void cleanup() {
    if (!isCleanedUp_) {
      stopSource_.request_stop();
      thread_.join();
      isCleanedUp_ = true;
    }
  }

  ucx_am_context& get_context() { return *context_; }

  const std::unique_ptr<UcxMemoryResourceManager>& get_memory_resource() const {
    return memoryResource_;
  }

 protected:
  virtual void init() = 0;

  std::unique_ptr<UcxMemoryResourceManager> memoryResource_;
  std::unique_ptr<ucx_am_context> context_;
  inplace_stop_source stopSource_;
  std::thread thread_;
  std::string name_;
  std::chrono::seconds timeout_;
  bool isCleanedUp_ = false;
};

class UcxContextHostRunner : public UcxContextRunner {
 public:
  UcxContextHostRunner(
    std::string name, std::chrono::seconds timeout = std::chrono::seconds(300))
    : UcxContextRunner(name, timeout) {
    init();
  }

 protected:
  void init() override {
    memoryResource_.reset(new DefaultUcxMemoryResourceManager());
    context_.reset(new ucx_am_context(
      memoryResource_, name_, timeout_, /*connectionHandleError=*/true));
    thread_ = std::thread([this] { context_->run(stopSource_.get_token()); });
    // Wait for context to initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
};

#if CUDA_ENABLED

static ucp_context_h common_gpu_ucp_context_ = nullptr;
class UcxContextCUDARunner : public UcxContextRunner {
 public:
  UcxContextCUDARunner(
    std::string name, bool use_ucp_address,
    std::chrono::seconds timeout = std::chrono::seconds(300))
    : UcxContextRunner(name, timeout), use_ucp_address_(use_ucp_address) {
    init();
  }

  ~UcxContextCUDARunner() override {
    cleanup();
    ucx_am_context::destroy_ucp_context(ucp_context_);
  }

 protected:
  void init() override {
    if (ucp_context_ == nullptr) {
      auto ec = ucx_am_context::init_ucp_context(
        name_, &ucp_context_,
        /*mtWorkersShared=*/true, /*printConfig=*/false);
      if (ec) {
        throw std::runtime_error(
          "UCX Context initialization failed: " + ec.message());
      }
    }
    if (!use_ucp_address_) {
      // Only test with UcxAutoCudaDeviceContext when use_ucp_address_ is false.
      CUcontext context;
      ASSERT_EQ(cuCtxGetCurrent(&context), CUDA_SUCCESS);
      auto_cuda_context_ = std::make_unique<UcxAutoCudaDeviceContext>(context);
    }
    memoryResource_.reset(new UcxCudaMemoryResourceManager());
    context_.reset(new ucx_am_context(
      memoryResource_, ucp_context_, timeout_,
      /*connectionHandleError=*/!use_ucp_address_,
      /*clientId=*/stdexe_ucx_runtime::CLIENT_ID_UNDEFINED,
      std::move(auto_cuda_context_)));
    thread_ = std::thread([this] {
      if (use_ucp_address_) {
        // You can use cudaSetDevice or UcxAutoCudaDeviceContext to set CUDA
        // context.
        cudaSetDevice(0);
      }
      context_->run(stopSource_.get_token());
    });
    // Wait for context to initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ucp_context_h ucp_context_ = common_gpu_ucp_context_;
  std::unique_ptr<UcxAutoDeviceContext> auto_cuda_context_ = nullptr;
  bool use_ucp_address_ = false;
};
#endif

// Test fixture for UCX AM tests
class UcxAmTest : public ::testing::Test {
 protected:
  unsigned int seed = static_cast<unsigned int>(time(nullptr));

  void SetUp() override {
    // Set UCX environment variables
    setenv(
      "UCX_RNDV_THRESH", std::to_string(kUcxRndvThreshold).c_str(),
      1);  // Set RNDV threshold to 8KB
    setenv("UCX_RNDV_SCHEME", "get_zcopy", 1);
    setenv("UCX_TCP_MAX_CONN_RETRIES", "1", 1);
  }

  void TearDown() override {
    unsetenv("UCX_RNDV_THRESH");
    unsetenv("UCX_RNDV_SCHEME");
    unsetenv("UCX_TCP_MAX_CONN_RETRIES");
  }

  // Helper to create a socket address
  static std::unique_ptr<sockaddr> create_server_socket(uint16_t port = 0) {
    sockaddr_in* addr = new sockaddr_in{
      .sin_family = AF_INET,
      .sin_port = htons(port),
      .sin_addr = {.s_addr = htonl(INADDR_ANY)}};
    return std::unique_ptr<sockaddr>(reinterpret_cast<sockaddr*>(addr));
  }

  static std::unique_ptr<sockaddr> create_client_socket(uint16_t port = 0) {
    sockaddr_in* addr = new sockaddr_in{
      .sin_family = AF_INET,
      .sin_port = htons(port),
      .sin_addr = {.s_addr = htonl(INADDR_LOOPBACK)}};
    return std::unique_ptr<sockaddr>(reinterpret_cast<sockaddr*>(addr));
  }

  std::vector<char> create_test_data(size_t size) {
    std::vector<char> data(size);
    for (size_t i = 0; i < size; i++) {
      data[i] = static_cast<char>(i % 256);
    }
    return data;
  }

  bool verify_test_data(const void* data, size_t size) {
    if (data == nullptr || size <= 0) {
      return false;
    }
    const char* ptr = static_cast<const char*>(data);
    for (size_t i = 0; i < size; i++) {
      if (ptr[i] != static_cast<char>(i % 256)) {
        return false;
      }
    }
    return true;
  }

  task<void> serverRecvTask(
    ucx_am_context::scheduler& scheduler,
    ucx_am_data& recvData,
    std::optional<std::reference_wrapper<const UcxConnection>>& conn,
    std::atomic<bool>& messageReceived,
    inplace_stop_source& stopSource) {
    auto active_message_bundle = co_await connection_recv(scheduler, recvData);
    if (active_message_bundle.connection().is_established()) {
      conn = std::ref(active_message_bundle.connection());
      messageReceived.store(true);
    }
    stopSource.request_stop();
    co_await stop_if_requested();
  }

  task<void> serverListenTask(
    ucx_am_context::scheduler& scheduler, uint16_t port, ucx_am_data& recvData,
    std::optional<std::reference_wrapper<const UcxConnection>>& conn,
    std::atomic<bool>& messageReceived, inplace_stop_source& stopSource) {
    async_scope scope;
    auto mainThread = co_await current_scheduler();
    auto serverSocket = create_server_socket(port);
    co_await for_each(
      take_until(
        accept_endpoint(
          scheduler, std::move(serverSocket), sizeof(sockaddr_in)),
        single(stop_on_request(stopSource.get_token()))),
      [&](std::vector<std::pair<std::uint64_t, std::error_code>>&&
            conn_id_status_vector) {
        if (conn_id_status_vector.size() > 0) {
          for (auto [conn_id, status] : conn_id_status_vector) {
            EXPECT_NE(conn_id, std::uintptr_t(nullptr));
            EXPECT_TRUE(!status);
          }
        }
        spawn_detached(
          on(
            mainThread,
            serverRecvTask(
              scheduler, recvData, conn, messageReceived, stopSource)),
          scope);
      });
    co_await scope.join();
  }

  task<void> clientSendTask(
    ucx_am_context::scheduler& scheduler,
    std::uint64_t conn_id,
    ucx_am_data& sendData,
    std::atomic<bool>& sendSuccess) {
    co_await connection_send(scheduler, conn_id, sendData);
    sendSuccess.store(true);
  }

  task<void> clientConnectTask(
    ucx_am_context::scheduler& scheduler, uint16_t port, ucx_am_data& sendData,
    std::atomic<bool>& sendSuccess) {
    async_scope scope;
    auto mainThread = co_await current_scheduler();
    auto clientSocket = create_client_socket(port);
    auto conn_id = co_await connect_endpoint(
      scheduler, nullptr, std::move(clientSocket), sizeof(sockaddr_in));
    spawn_detached(
      on(mainThread, clientSendTask(scheduler, conn_id, sendData, sendSuccess)),
      scope);
    co_await scope.join();
  }

  std::vector<float> create_float_test_data(size_t size) {
    std::vector<float> data(size);
    for (size_t i = 0; i < size; i++) {
      data[i] = static_cast<float>(i % 256);
    }
    return data;
  }

  bool verify_processed_float_test_data(const void* data, size_t data_length) {
    if (data == nullptr || data_length <= 0) {
      return false;
    }
    const float* ptr = static_cast<const float*>(data);
    size_t size = data_length / sizeof(float);
    for (size_t i = 0; i < size; i++) {
      if (ptr[i] != (static_cast<float>(i % 256) / 2)) {
        return false;
      }
    }
    return true;
  }

  task<void> launchProcessTask(
    static_thread_pool::scheduler& scheduler, ucx_am_data_t& recvData) {
    co_await on(scheduler, defer([&]() noexcept {
                  switch (recvData.data_type) {
                    case ucx_memory_type::HOST:
                      processRecvDataHost(recvData);
                      break;
#if CUDA_ENABLED
                    case ucx_memory_type::CUDA:
                      processRecvDataCuda(recvData);
                      break;
#endif
                    default:
                      break;
                  }
                  return unifex::just();
                }));
    co_return;
  }

  task<void> biDiServerRecvSendTaskImpl(
    ucx_am_context::scheduler& ucxScheduler,
    static_thread_pool::scheduler& processScheduler,
    active_message_bundle& data) {
    auto conn_id = data.connection().id();
    auto recvData = data.get_data();
    co_await launchProcessTask(processScheduler, recvData);
    co_await connection_send(ucxScheduler, conn_id, recvData);
    co_return;
  }

  task<void> biDiServerRecvSendTask(
    ucx_am_context::scheduler& ucxScheduler,
    static_thread_pool::scheduler& processScheduler,
    ucx_memory_type recvDataType) {
    auto active_message_bundle =
      co_await connection_recv(ucxScheduler, recvDataType);
    co_await biDiServerRecvSendTaskImpl(
      ucxScheduler, processScheduler, active_message_bundle);
  }

  task<void> biDiServerStart(
    ucx_am_context::scheduler& scheduler,
    static_thread_pool::scheduler& processScheduler, uint16_t port,
    ucx_memory_type recvDataType, inplace_stop_source& stopSource) {
    async_scope scope;
    auto workerThread = processScheduler;
    auto serverSocket = create_server_socket(port);
    co_await for_each(
      take_until(
        accept_endpoint(
          scheduler, std::move(serverSocket), sizeof(sockaddr_in)),
        single(stop_on_request(stopSource.get_token()))),
      [&](std::vector<std::pair<std::uint64_t, std::error_code>>&&
            conn_id_status_vector) {
        if (conn_id_status_vector.size() > 0) {
          for (auto [conn_id, status] : conn_id_status_vector) {
            EXPECT_NE(conn_id, std::uintptr_t(nullptr));
            EXPECT_TRUE(!status);
          }
        }
        spawn_detached(
          with_query_value(
            on(
              workerThread,
              biDiServerRecvSendTask(
                scheduler, processScheduler, recvDataType)),
            get_stop_token,
            stopSource.get_token()),
          scope);
      });
    co_await scope.join();
    co_return;
  }

  task<void> biDiServerRepeatRecvSendTask(
    ucx_am_context::scheduler& ucxScheduler,
    static_thread_pool::scheduler& processScheduler,
    ucx_memory_type recvDataType, inplace_stop_source& stopSource) {
    co_await repeat_effect_until(
      with_query_value(
        defer([&]() {
          return biDiServerRecvSendTask(
            ucxScheduler, processScheduler, recvDataType);
        }),
        get_stop_token,
        stopSource.get_token()),
      [&]() {
        assert(stopSource.stop_requested() == false);
        return stopSource.stop_requested();
      });
    co_return;
  }

  task<void> biDiServerStart(
    ucx_am_context::scheduler& scheduler,
    static_thread_pool::scheduler& processScheduler,
    std::vector<std::byte>& client_ucp_address, ucx_memory_type recvDataType,
    inplace_stop_source& stopSource) {
    async_scope scope;
    auto workerThread = processScheduler;
    [[maybe_unused]] auto conn_id =
      co_await connect_endpoint(scheduler, client_ucp_address);
    spawn_detached(
      on(
        workerThread,
        biDiServerRepeatRecvSendTask(
          scheduler, processScheduler, recvDataType, stopSource)),
      scope);
    co_await scope.join();
    co_return;
  }

  task<ucx_am_data> biDiClientSendRecvTask(
    ucx_am_context::scheduler& ucxScheduler,
    static_thread_pool::scheduler& processScheduler,
    std::uint64_t conn_id,
    ucx_am_data& sendData) {
    co_await connection_send(ucxScheduler, conn_id, sendData);
    auto recvBundle =
      co_await connection_recv(ucxScheduler, sendData.data_type);
    co_return recvBundle.get_data();
  }

  task<ucx_am_data> biDiClientSpawnTask(
    ucx_am_context::scheduler& ucxScheduler,
    static_thread_pool::scheduler& workerThread,
    async_scope& scope,
    inplace_stop_source& stopSource,
    std::uint64_t conn_id,
    ucx_am_data& sendData) {
    auto recvData = co_await let_error(
      spawn_future(
        on(
          workerThread,
          biDiClientSendRecvTask(
            ucxScheduler, workerThread, conn_id, sendData)),
        scope),
      [&](std::exception_ptr ep) -> task<ucx_am_data> {
        try {
          std::rethrow_exception(ep);
        } catch (const std::exception& e) {
          std::cerr << "Error in biDiClientSendRecvTask: " << e.what()
                    << std::endl;
        }
        co_return ucx_am_data{};
      });

    co_await scope.join();
    stopSource.request_stop();
    co_await stop_if_requested();
    co_return recvData;
  }

  task<ucx_am_data> biDiClientStart(
    ucx_am_context::scheduler& ucxScheduler,
    static_thread_pool::scheduler& processScheduler, uint16_t port,
    ucx_am_data& sendData, inplace_stop_source& stopSource) {
    async_scope scope;
    auto workerThread = processScheduler;
    auto clientSocket = create_client_socket(port);
    auto conn_id = co_await connect_endpoint(
      ucxScheduler, nullptr, std::move(clientSocket), sizeof(sockaddr_in));
    co_return co_await biDiClientSpawnTask(
      ucxScheduler, workerThread, scope, stopSource, conn_id, sendData);
  }

  task<ucx_am_data> biDiClientStart(
    ucx_am_context::scheduler& ucxScheduler,
    static_thread_pool::scheduler& processScheduler,
    std::vector<std::byte>& server_ucp_address, ucx_am_data& sendData,
    inplace_stop_source& stopSource) {
    async_scope scope;
    auto workerThread = processScheduler;
    auto conn_id = co_await connect_endpoint(ucxScheduler, server_ucp_address);
    co_return co_await biDiClientSpawnTask(
      ucxScheduler, workerThread, scope, stopSource, conn_id, sendData);
  }

  // Helper function for bidirectional transfer tests
  void runBidirectionalTransferTestLogic(
    size_t floatDataSize, ucx_memory_type test_memory_type,
    bool use_ucp_address = false) {
#if !CUDA_ENABLED
    if (test_memory_type == ucx_memory_type::CUDA) {
      GTEST_SKIP()
        << "CUDA not enabled, skipping CUDA memory type test variant.";
      return;
    }
#endif

    std::unique_ptr<UcxContextRunner> server_runner_ptr;
    std::unique_ptr<UcxContextRunner> client_runner_ptr;

    switch (test_memory_type) {
      case ucx_memory_type::HOST:
        server_runner_ptr =
          std::make_unique<UcxContextHostRunner>("server_host");
        client_runner_ptr =
          std::make_unique<UcxContextHostRunner>("client_host");
        break;
#if CUDA_ENABLED
      case ucx_memory_type::CUDA:
        server_runner_ptr = std::make_unique<UcxContextCUDARunner>(
          "server_cuda", use_ucp_address);
        client_runner_ptr = std::make_unique<UcxContextCUDARunner>(
          "client_cuda", use_ucp_address);
        break;
#endif
      case ucx_memory_type::CUDA_MANAGED:
      case ucx_memory_type::ROCM:
      case ucx_memory_type::ROCM_MANAGED:
      case ucx_memory_type::RDMA:
      case ucx_memory_type::ZE_HOST:
      case ucx_memory_type::ZE_DEVICE:
      case ucx_memory_type::ZE_MANAGED:
      case ucx_memory_type::UNKNOWN:
      default:
        FAIL() << "Unsupported memory type ("
               << static_cast<int>(test_memory_type)
               << ") for UcxContextRunner setup, or CUDA features not enabled "
                  "for this type.";
        return;
    }

    UcxContextRunner& server = *server_runner_ptr;
    UcxContextRunner& client = *client_runner_ptr;

    uint16_t port =
      static_cast<uint16_t>(1024 + (rand_r(&seed) % (65535 - 1024)));

    auto serverScheduler = server.get_context().get_scheduler();
    auto clientScheduler = client.get_context().get_scheduler();

    static_thread_pool tpContext{2};
    auto processScheduler = tpContext.get_scheduler();

    const size_t headerFixedSize = 1024;
    auto headerData = create_test_data(headerFixedSize);
    auto testFloatVec = create_float_test_data(floatDataSize);

    ucx_am_data sendData{}, recvData{};
    sendData.header = headerData.data();
    sendData.header_length = headerData.size();
    sendData.data = testFloatVec.data();
    sendData.data_length = testFloatVec.size() * sizeof(float);
    sendData.data_type = test_memory_type;

    UcxMemoryResourceManager* base_manager_ptr =
      client.get_memory_resource().get();

#if CUDA_ENABLED
    if (sendData.data_type == ucx_memory_type::CUDA) {
      UcxCudaMemoryResourceManager* clientCudaMemoryResource =
        dynamic_cast<UcxCudaMemoryResourceManager*>(base_manager_ptr);
      ASSERT_NE(clientCudaMemoryResource, nullptr)
        << "Failed to dynamic_cast UcxMemoryResourceManager to "
           "UcxCudaMemoryResourceManager for CUDA type.";
      auto dev_ptr = clientCudaMemoryResource->allocate(
        ucx_memory_type::CUDA, sendData.data_length);
      clientCudaMemoryResource->memcpy(
        ucx_memory_type::CUDA, dev_ptr, ucx_memory_type::HOST, sendData.data,
        sendData.data_length);
      sendData.data = dev_ptr;
      cudaPointerAttributes attributes;
      ASSERT_EQ(
        cudaPointerGetAttributes(&attributes, sendData.data), cudaSuccess);
      //
      CUmemorytype cuda_mem_type = CU_MEMORYTYPE_HOST;
      uint32_t is_managed = 0;
      CUdevice cuda_device = -1;
      CUcontext cuda_mem_ctx = NULL;
      CUpointer_attribute attr_type[4];
      void* attr_data[4];
      attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
      attr_data[0] = &cuda_mem_type;
      attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
      attr_data[1] = &is_managed;
      attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
      attr_data[2] = &cuda_device;
      attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
      attr_data[3] = &cuda_mem_ctx;
      CUresult result = cuPointerGetAttributes(
        4, attr_type, attr_data, (CUdeviceptr)sendData.data);
      ASSERT_EQ(result, CUDA_SUCCESS);
      ASSERT_EQ(cuda_mem_type, CU_MEMORYTYPE_DEVICE);
      ASSERT_EQ(is_managed, 0);
      ASSERT_EQ(cuda_device, 0);
      ASSERT_NE(cuda_mem_ctx, nullptr);
      //
      ASSERT_EQ(attributes.type, cudaMemoryTypeDevice);
    }
#endif

    inplace_stop_source stopSource;

    std::optional<task<void>> server_task;
    std::optional<task<ucx_am_data>> client_task;
    std::vector<std::byte> server_ucp_address;
    std::vector<std::byte> client_ucp_address;

    if (use_ucp_address) {
      ASSERT_EQ(
        server.get_context().get_ucp_address(server_ucp_address).value(), 0);
      ASSERT_EQ(
        client.get_context().get_ucp_address(client_ucp_address).value(), 0);
      auto server_task_impl = biDiServerStart(
        serverScheduler, processScheduler, client_ucp_address,
        sendData.data_type, stopSource);
      auto client_task_impl = biDiClientStart(
        clientScheduler, processScheduler, server_ucp_address, sendData,
        stopSource);
      server_task.emplace(std::move(server_task_impl));
      client_task.emplace(std::move(client_task_impl));
    } else {
      auto server_task_impl = biDiServerStart(
        serverScheduler, processScheduler, port, sendData.data_type,
        stopSource);
      auto client_task_impl = biDiClientStart(
        clientScheduler, processScheduler, port, sendData, stopSource);
      server_task.emplace(std::move(server_task_impl));
      client_task.emplace(std::move(client_task_impl));
    }

    auto combined_tasks =
      when_all(std::move(*server_task), std::move(*client_task));

    sync_wait(
      std::move(combined_tasks)
      | then([&](
               [[maybe_unused]] std::variant<std::tuple<>> server_result,
               std::variant<std::tuple<ucx_am_data>>
                 client_result_variant) {
          if (std::holds_alternative<std::tuple<ucx_am_data>>(
                client_result_variant)) {
            recvData = std::get<0>(
              std::get<std::tuple<ucx_am_data>>(client_result_variant));
          } else {
            FAIL() << "Client task did not return ucx_am_data as expected.";
          }
        }));

    EXPECT_TRUE(verify_test_data(recvData.header, headerData.size()));

    auto recv_data_host = recvData.data;
    std::vector<float> hostRecvVec;

#if CUDA_ENABLED
    if (sendData.data_type == ucx_memory_type::CUDA) {
      UcxCudaMemoryResourceManager* clientCudaMemoryResource =
        dynamic_cast<UcxCudaMemoryResourceManager*>(base_manager_ptr);
      hostRecvVec = create_float_test_data(floatDataSize);
      recv_data_host = hostRecvVec.data();
      // recvData.data_type may be ucx_memory_type::HOST when Eager Protocol
      clientCudaMemoryResource->memcpy(
        ucx_memory_type::HOST, recv_data_host, recvData.data_type,
        recvData.data, recvData.data_length);
    }
#endif

    EXPECT_TRUE(
      verify_processed_float_test_data(recv_data_host, recvData.data_length));

    switch (test_memory_type) {
      case ucx_memory_type::HOST: {
        DefaultUcxMemoryResourceManager* clientMemoryResource =
          dynamic_cast<DefaultUcxMemoryResourceManager*>(base_manager_ptr);
        ASSERT_NE(clientMemoryResource, nullptr)
          << "Failed to dynamic_cast UcxMemoryResourceManager to "
             "DefaultUcxMemoryResourceManager for HOST type.";
        if (clientMemoryResource) {
          clientMemoryResource->deallocate(
            ucx_memory_type::HOST, recvData.header, recvData.header_length);
          clientMemoryResource->deallocate(
            recvData.data_type, recvData.data, recvData.data_length);
        }
      } break;
#if CUDA_ENABLED
      case ucx_memory_type::CUDA: {
        UcxCudaMemoryResourceManager* clientCudaMemoryResource =
          dynamic_cast<UcxCudaMemoryResourceManager*>(base_manager_ptr);
        if (clientCudaMemoryResource) {
          clientCudaMemoryResource->deallocate(
            ucx_memory_type::HOST,
            recvData.header,  // Header is always host memory
            recvData.header_length);
          clientCudaMemoryResource->deallocate(
            recvData.data_type, recvData.data,  // Data is CUDA memory
            recvData.data_length);
        }
      } break;
#endif
      // Explicitly list other known enumeration values.
      // For these, deallocation logic is currently not implemented.
      case ucx_memory_type::CUDA_MANAGED:
      case ucx_memory_type::ROCM:
      case ucx_memory_type::ROCM_MANAGED:
      case ucx_memory_type::RDMA:
      case ucx_memory_type::ZE_HOST:
      case ucx_memory_type::ZE_DEVICE:
      case ucx_memory_type::ZE_MANAGED:
      case ucx_memory_type::UNKNOWN:
      default:
        // If control reaches here, it means memory was potentially allocated
        // for a type for which deallocation logic is not implemented in this
        // switch. This is a critical issue, so the test should fail.
        FAIL() << "Deallocation logic not implemented for memory type: "
               << static_cast<int>(test_memory_type)
               << ". Potential memory leak for header_length="
               << recvData.header_length
               << " and data_length=" << recvData.data_length;
        break;
    }
  }
};

template <typename T>
std::unique_ptr<T> copy_unique(const std::unique_ptr<T>& source) {
  return source ? std::make_unique<T>(*source) : nullptr;
}

// Test connection establishment with stop token
TEST_F(UcxAmTest, ConnectionEstablishmentWithStopToken) {
  UcxContextHostRunner server("server");

  uint16_t port =
    static_cast<uint16_t>(1024 + (rand_r(&seed) % (65535 - 1024)));
  auto serverSocket = create_server_socket(port);
  auto scheduler = server.get_context().get_scheduler();
  bool stoped = false;

  sync_wait(then(
    stop_when(
      for_each(
        accept_endpoint(
          scheduler, std::move(serverSocket), sizeof(sockaddr_in)),
        [&stoped](auto&&) { stoped = false; }),
      schedule_after(scheduler, std::chrono::milliseconds{1000})),
    [&stoped]() { stoped = true; }));
  EXPECT_TRUE(stoped);
}

// Test connection establishment
TEST_F(UcxAmTest, ConnectionEstablishment) {
  UcxContextHostRunner server("server");
  UcxContextHostRunner client("client");

  // Generate random port number between 1024 and 65535
  uint16_t port =
    static_cast<uint16_t>(1024 + (rand_r(&seed) % (65535 - 1024)));
  auto serverSocket = create_server_socket(port);
  auto clientDstSocket = create_client_socket(port);
  auto serverScheduler = server.get_context().get_scheduler();
  auto clientScheduler = client.get_context().get_scheduler();
  bool listened = false;
  bool connected = false;

  sync_wait(when_all(
    for_each(
      take_until(
        accept_endpoint(
          serverScheduler, std::move(serverSocket), sizeof(sockaddr_in)),
        single(
          schedule_after(serverScheduler, std::chrono::milliseconds{1000}))),
      [&listened](auto&&) { listened = true; }),
    then(
      connect_endpoint(
        clientScheduler, nullptr, std::move(clientDstSocket),
        sizeof(sockaddr_in)),
      [&connected](auto&& res) {
        if (res > 0) {
          connected = true;
        }
      })));
  EXPECT_TRUE(listened && connected);
}

// Test small message transfer (eager protocol)
TEST_F(UcxAmTest, SmallMessageTransfer) {
  UcxContextHostRunner server("server");
  UcxContextHostRunner client("client");

  uint16_t port =
    static_cast<uint16_t>(1024 + (rand_r(&seed) % (65535 - 1024)));
  auto serverScheduler = server.get_context().get_scheduler();
  auto clientScheduler = client.get_context().get_scheduler();
  const size_t messageSize = 1024;  // 1KB - should use eager protocol
  auto testData = create_test_data(messageSize);

  std::atomic<bool> messageReceived = false;
  std::atomic<bool> sendSuccess = false;
  ucx_am_data sendData{}, recvData{};
  sendData.header = testData.data();
  sendData.header_length = testData.size();
  sendData.data = testData.data();
  sendData.data_length = testData.size();
  sendData.data_type = ucx_memory_type::HOST;
  recvData.data = server.get_memory_resource()->allocate(
    ucx_memory_type::HOST, messageSize * 2);
  recvData.data_length = messageSize * 2;
  recvData.data_type = ucx_memory_type::HOST;

  std::optional<std::reference_wrapper<const UcxConnection>> conn;

  inplace_stop_source stopSource;

  sync_wait(when_all(
    with_query_value(
      serverListenTask(
        serverScheduler, port, recvData, conn, messageReceived, stopSource),
      get_stop_token,
      stopSource.get_token()),
    clientConnectTask(clientScheduler, port, sendData, sendSuccess)));

  EXPECT_TRUE(messageReceived.load() && sendSuccess.load());
  EXPECT_EQ(recvData.data_length, messageSize);
  EXPECT_TRUE(verify_test_data(recvData.header, messageSize));
  EXPECT_TRUE(verify_test_data(recvData.data, messageSize));
}

// Test large message transfer (RNDV protocol)
TEST_F(UcxAmTest, LargeMessageTransfer) {
  UcxContextHostRunner server("server");
  UcxContextHostRunner client("client");

  uint16_t port =
    static_cast<uint16_t>(1024 + (rand_r(&seed) % (65535 - 1024)));
  auto serverScheduler = server.get_context().get_scheduler();
  auto clientScheduler = client.get_context().get_scheduler();
  const size_t headerSize = 1024;
  auto headerData = create_test_data(headerSize);
  const size_t messageSize = 1024 * 1024;  // 1MB - should use RNDV protocol
  auto testData = create_test_data(messageSize);

  std::atomic<bool> messageReceived = false;
  std::atomic<bool> sendSuccess = false;
  ucx_am_data sendData{}, recvData{};
  sendData.header = headerData.data();
  sendData.header_length = headerData.size();
  sendData.data = testData.data();
  sendData.data_length = testData.size();
  sendData.data_type = ucx_memory_type::HOST;
  recvData.data = server.get_memory_resource()->allocate(
    ucx_memory_type::HOST, messageSize / 2);
  recvData.data_length = messageSize / 2;
  recvData.data_type = ucx_memory_type::HOST;

  std::optional<std::reference_wrapper<const UcxConnection>> conn;

  inplace_stop_source stopSource;

  sync_wait(when_all(
    with_query_value(
      serverListenTask(
        serverScheduler, port, recvData, conn, messageReceived, stopSource),
      get_stop_token,
      stopSource.get_token()),
    clientConnectTask(clientScheduler, port, sendData, sendSuccess)));

  EXPECT_TRUE(messageReceived.load() && sendSuccess.load());
  EXPECT_EQ(recvData.data_length, messageSize);
  EXPECT_TRUE(verify_test_data(recvData.header, headerSize));
  EXPECT_TRUE(verify_test_data(recvData.data, messageSize));
}

// Test error handling
TEST_F(UcxAmTest, ErrorHandling) {
  UcxContextHostRunner client("client");

  uint16_t port =
    static_cast<uint16_t>(1024 + (rand_r(&seed) % (65535 - 1024)));
  auto clientScheduler = client.get_context().get_scheduler();
  sockaddr_in* addr = new sockaddr_in{
    .sin_family = AF_INET,
    .sin_port = htons(port),
    // .sin_addr = {.s_addr = inet_addr("192.0.2.1")}};
    .sin_addr = {.s_addr = inet_addr("192.0.2.1")}};
  auto clientSocket =
    std::unique_ptr<sockaddr>(reinterpret_cast<sockaddr*>(addr));
  auto testData = create_test_data(1024);
  ucx_am_data sendData{};
  sendData.header = testData.data();
  sendData.header_length = testData.size();
  sendData.data = testData.data();
  sendData.data_length = testData.size();
  sendData.data_type = ucx_memory_type::HOST;

  std::atomic<bool> errorHandled = false;
  std::atomic<bool> sendSuccess = false;

  auto conn_id = sync_wait(connect_endpoint(
    clientScheduler, nullptr, std::move(clientSocket), sizeof(sockaddr_in)));

  sync_wait(let_error(
    connection_send(clientScheduler, conn_id.value(), sendData)
      | then([&]() { sendSuccess.store(true); }),
    [&](std::variant<std::error_code, std::exception_ptr>&& error_variant) {
      return sequence(
        then(
          handle_error_connection(
            clientScheduler,
            [&](std::uint64_t conn_id, ucs_status_t status) {
              EXPECT_LT(status, 0);
              return true;
            }),
          [&]() { errorHandled.store(true); }),
        just_done());
    }));

  EXPECT_FALSE(sendSuccess.load());
  EXPECT_TRUE(errorHandled.load());
}

// Test bidirectional small message transfer and processing (eager protocol)
TEST_F(UcxAmTest, BidirectionalSmallMessageTransfer) {
  runBidirectionalTransferTestLogic(1024, ucx_memory_type::HOST);
}

// Test bidirectional large message transfer and processing (RNDV protocol)
TEST_F(UcxAmTest, BidirectionalLargeMessageTransfer) {
  runBidirectionalTransferTestLogic(1024 * 1024, ucx_memory_type::HOST);
}

#if CUDA_ENABLED
// Test bidirectional small message transfer and processing (eager protocol)
// with CUDA
TEST_F(UcxAmTest, BidirectionalSmallMessageCUDATransfer) {
  CUdevice device;
  CUcontext context;
  ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
  ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
  ASSERT_EQ(cuCtxCreate(&context, 0, device), CUDA_SUCCESS);
  ASSERT_EQ(cuCtxSetCurrent(context), CUDA_SUCCESS);
  runBidirectionalTransferTestLogic(1024, ucx_memory_type::CUDA);
  ASSERT_EQ(cuCtxDestroy(context), CUDA_SUCCESS);
}

// Test bidirectional  large message transfer and processing (RNDV protocol)
// with CUDA
TEST_F(UcxAmTest, BidirectionalLargeMessageCUDATransfer) {
  CUdevice device;
  CUcontext context;
  ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
  ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
  ASSERT_EQ(cuCtxCreate(&context, 0, device), CUDA_SUCCESS);
  ASSERT_EQ(cuCtxSetCurrent(context), CUDA_SUCCESS);
  runBidirectionalTransferTestLogic(1024 * 1024, ucx_memory_type::CUDA);
  ASSERT_EQ(cuCtxDestroy(context), CUDA_SUCCESS);
}

// Test bidirectional small message transfer and processing (eager protocol)
// with CUDA and ucp address
TEST_F(UcxAmTest, BidirectionalSmallMessageCUDATransferWithUcpAddress) {
  cudaSetDevice(0);
  runBidirectionalTransferTestLogic(1024, ucx_memory_type::CUDA, true);
}

// Test bidirectional  large message transfer and processing (RNDV protocol)
// with CUDA and ucp address
TEST_F(UcxAmTest, BidirectionalLargeMessageCUDATransferWithUcpAddress) {
  cudaSetDevice(0);
  runBidirectionalTransferTestLogic(1024 * 1024, ucx_memory_type::CUDA, true);
}
#endif  // CUDA_ENABLED

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
