/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.
 *
 *Licensed under the Apache License, Version 2.0 (the "License");
 *you may not use this file except in compliance with the License.
 *You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 *==============================================================================*/

#include "axon/axon_worker.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

#include "ucx_context/ucx_device_context.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

namespace eux::axon {

class AxonWorkerBasicTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mr_ = std::make_unique<ucxx::DefaultUcxMemoryResourceManager>();
  }

  std::unique_ptr<ucxx::DefaultUcxMemoryResourceManager> mr_;
};

TEST_F(AxonWorkerBasicTest, StartStop) {
  // Basic start/stop lifecycle.
  auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
  AxonWorker worker(
    *mr_, "test_worker", 2, std::chrono::milliseconds(10),
    std::move(device_ctx));

  EXPECT_TRUE(worker.Start().has_value());
  // Start() is idempotent.
  EXPECT_TRUE(worker.Start().has_value());

  worker.Stop();
  // Stop() is also idempotent and should not crash.
  worker.Stop();
}

TEST_F(AxonWorkerBasicTest, StartServerClientIdempotent) {
  // Explicit server/client start/stop paths.
  auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
  AxonWorker worker(
    *mr_, "test_worker_idempotent", 2, std::chrono::milliseconds(10),
    std::move(device_ctx));

  // Server start is idempotent.
  EXPECT_TRUE(worker.StartServer().has_value());
  EXPECT_TRUE(worker.StartServer().has_value());

  // Client start is idempotent.
  EXPECT_TRUE(worker.StartClient().has_value());
  EXPECT_TRUE(worker.StartClient().has_value());

  worker.StopServer();
  worker.StopClient();

  // Repeated stop should be safe.
  worker.StopServer();
  worker.StopClient();
}

TEST_F(AxonWorkerBasicTest, RejectMessagesFlagAndInfo) {
  // Verify reject-messages flag toggling and info query.
  auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
  AxonWorker worker(
    *mr_, "reject_flag_worker", 2, std::chrono::milliseconds(10),
    std::move(device_ctx));

  // Initially should not be rejecting messages.
  EXPECT_FALSE(worker.IsRejectingMessages());

  worker.SetRejectMessages(true);
  EXPECT_TRUE(worker.IsRejectingMessages());

  worker.SetRejectMessages(false);
  EXPECT_FALSE(worker.IsRejectingMessages());
}

TEST_F(AxonWorkerBasicTest, RegisterEndpointSignaturesInvalidBlob) {
  // Invalid signatures blob should trigger INVALID_ARGUMENT.
  auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
  AxonWorker worker(
    *mr_, "signatures_worker_invalid", 2, std::chrono::milliseconds(10),
    std::move(device_ctx));

  cista::byte_buf invalid_blob;
  invalid_blob.resize(1);
  const std::byte b{0xFF};
  std::memcpy(invalid_blob.data(), &b, 1);

  auto result =
    worker.RegisterEndpointSignatures("remote_worker_invalid", invalid_blob);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(
    std::error_code(result.error().status),
    std::make_error_code(rpc::RpcErrc::INVALID_ARGUMENT));
}

TEST_F(AxonWorkerBasicTest, RegisterEndpointSignaturesWorkerAlreadyRegistered) {
  // When the worker name is already registered, the current implementation
  // should successfully attach signatures to the existing worker.
  auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
  AxonWorker worker(
    *mr_, "signatures_worker_existing", 2, std::chrono::milliseconds(10),
    std::move(device_ctx));

  // First, associate a connection so that the worker name is present.
  worker.AssociateConnection("remote_worker_existing", /*conn_id=*/1234);

  // Serialize an empty signatures vector, which should be deserializable.
  cista::offset::vector<rpc::RpcFunctionSignature> signatures;
  auto blob = cista::serialize<rpc::utils::SerializerMode>(signatures);

  auto result =
    worker.RegisterEndpointSignatures("remote_worker_existing", blob);

  // Now the registration should succeed and return the worker key.
  ASSERT_TRUE(result.has_value());
}

TEST_F(AxonWorkerBasicTest, TimeoutConfiguration) {
  // Verify that timeout can be configured without crashing.
  auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
  AxonWorker worker(
    *mr_, "timeout_worker", 2, std::chrono::milliseconds(10),
    std::move(device_ctx));

  worker.SetTimeout(std::chrono::milliseconds(1));
  worker.SetTimeout(std::chrono::milliseconds(5));
}

TEST_F(AxonWorkerBasicTest, MetricsObserversNoCrash) {
  // Install dummy metrics observers; just ensure callbacks can be set.
  auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
  AxonWorker worker(
    *mr_, "metrics_worker", 2, std::chrono::milliseconds(10),
    std::move(device_ctx));

  struct DummyMetricsObserverImpl {
    void OnDispatchStart(
      const metrics::RpcMetricsContext&,
      std::chrono::steady_clock::time_point) {
      // no-op
    }

    void OnDispatchComplete(
      const metrics::RpcMetricsContext&,
      std::chrono::steady_clock::time_point) {
      // no-op
    }

    void OnWorkerStats(const metrics::WorkerMetrics&) {
      // no-op
    }
  };

  metrics::MetricsObserver observer =
    pro::make_proxy<metrics::MetricsObserverFacade>(DummyMetricsObserverImpl{});

  worker.SetServerMetricsObserver(std::move(observer));

  // Create a new observer for client side to avoid relying on copyability.
  metrics::MetricsObserver client_observer =
    pro::make_proxy<metrics::MetricsObserverFacade>(DummyMetricsObserverImpl{});
  worker.SetClientMetricsObserver(std::move(client_observer));

  // Without starting UCX, just calling report should not crash.
  worker.ReportServerMetrics();
  worker.ReportClientMetrics();
}

TEST_F(AxonWorkerBasicTest, ErrorObserversNoCrash) {
  // Install dummy error observers and trigger a simple error path by trying to
  // connect with an empty address (expected failure).
  auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
  AxonWorker worker(
    *mr_, "error_worker", 2, std::chrono::milliseconds(10),
    std::move(device_ctx));

  struct DummyErrorObserverImpl {
    void OnError(const errors::AxonErrorContext&) {
      // no-op
    }
  };

  errors::ErrorObserver observer =
    pro::make_proxy<errors::ErrorObserverFacade>(DummyErrorObserverImpl{});
  worker.SetServerErrorObserver(std::move(observer));

  // Create a separate observer for client side for clarity.
  errors::ErrorObserver client_observer =
    pro::make_proxy<errors::ErrorObserverFacade>(DummyErrorObserverImpl{});
  worker.SetClientErrorObserver(std::move(client_observer));

  // Start only client side; connect with invalid address to hit error path.
  EXPECT_TRUE(worker.StartClient().has_value());
  std::vector<std::byte> empty_addr;

  // The assertion failure in ucx_am_context::connect_sender will trigger
  auto res = worker.ConnectEndpoint(empty_addr, "non_exist_worker");
  ASSERT_FALSE(res.has_value());

  worker.StopClient();
}

}  // namespace eux::axon
