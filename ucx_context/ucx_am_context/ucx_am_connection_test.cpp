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

#include "ucx_context/ucx_connection.hpp"

#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <ucp/api/ucp.h>
#include <ucs/datastruct/list.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "ucx_context/ucx_context_logger.hpp"

static constexpr size_t RNDV_THRESHOLD = 8192;

namespace stdexe_ucx_runtime {
namespace test {

// Test fixture for UcxConnection tests
class UcxConnectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("UCX_RNDV_THRESH", std::to_string(RNDV_THRESHOLD).c_str(), 1);
    setenv("UCX_RNDV_SCHEME", "get_zcopy", 1);

    // Initialize UCX context and worker
    ucp_params_t ucp_params;
    ucp_config_t* config;
    ucs_status_t status;

    status = ucp_config_read(nullptr, nullptr, &config);
    ASSERT_EQ(status, UCS_OK);

    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features = UCP_FEATURE_AM | UCP_FEATURE_RMA;

    status = ucp_init(&ucp_params, config, &ucp_context_);
    ucp_config_release(config);
    ASSERT_EQ(status, UCS_OK);

    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

    status = ucp_worker_create(ucp_context_, &worker_params, &ucp_worker_);
    ASSERT_EQ(status, UCS_OK);

    seed = time(nullptr);
  }

  void TearDown() override {
    if (ucp_worker_) {
      ucp_worker_destroy(ucp_worker_);
    }
    if (ucp_context_) {
      ucp_cleanup(ucp_context_);
    }
    unsetenv("UCX_RNDV_THRESH");
    unsetenv("UCX_RNDV_SCHEME");
  }

  ucp_context_h ucp_context_ = nullptr;
  ucp_worker_h ucp_worker_ = nullptr;
  unsigned int seed;
};

// Mock callback for testing
class MockCallback : public UcxCallback {
 public:
  explicit MockCallback(std::function<void(ucs_status_t)> callback)
    : callback_(std::move(callback)) {}

  void operator()(ucs_status_t status) override { callback_(status); }

 private:
  std::function<void(ucs_status_t)> callback_;
};

// Test basic connection creation
TEST_F(UcxConnectionTest, CreateConnection) {
  std::atomic<bool> callback_called{false};
  auto callback =
    std::make_unique<MockCallback>([&callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      callback_called = true;
    });

  auto conn = std::make_unique<UcxConnection>(ucp_worker_, std::move(callback));
  EXPECT_NE(conn, nullptr);
  EXPECT_EQ(conn->id(), 1);
  // First connection should have ID 1
}

// Test connection establishment
TEST_F(UcxConnectionTest, Connect) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  std::atomic<bool> callback_called{false};
  auto callback =
    std::make_unique<MockCallback>([&callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      callback_called = true;
    });

  auto conn = std::make_unique<UcxConnection>(ucp_worker_, std::move(callback));

  // Create socket addresses for connection
  struct sockaddr_in src_addr, dst_addr;
  memset(&src_addr, 0, sizeof(src_addr));
  memset(&dst_addr, 0, sizeof(dst_addr));

  src_addr.sin_family = AF_INET;
  src_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  src_addr.sin_port = htons(0);  // Let the system choose a port

  dst_addr.sin_family = AF_INET;
  dst_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  dst_addr.sin_port = htons(12345);  // Use a specific port for testing

  // Create a connect callback
  std::atomic<bool> connect_callback_called{false};
  auto connect_callback = std::make_unique<MockCallback>(
    [&connect_callback_called](ucs_status_t status) {
      // Connection might fail in test environment, so we don't assert on
      // status
      connect_callback_called = true;
    });

  // Attempt to connect
  conn->connect(
    (const struct sockaddr*)&src_addr, (const struct sockaddr*)&dst_addr,
    sizeof(src_addr), std::move(connect_callback));

  // In a real test environment, we would need a server to accept the connection
  // For now, we just check that the callback was called
  EXPECT_TRUE(connect_callback_called);
  auto reset_fn = [&conn]() { conn.reset(nullptr); };
  EXPECT_DEATH(reset_fn(), ".*");
  conn->disconnect();
}

// Test sending and receiving active messages
TEST_F(UcxConnectionTest, SendRecvAmData) {
  // This test requires a more complex setup with a server and client
  // We'll implement a more comprehensive test in the multi-process test
  GTEST_SKIP() << "Skipping single-process AM test, will be covered in "
                  "multi-process test";
}

// Server connection handler callback function
static void server_conn_handle_cb(ucp_conn_request_h conn_request, void* arg) {
  ucp_conn_request_attr_t attr;
  ucs_status_t status;

  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
  status = ucp_conn_request_query(conn_request, &attr);
  ASSERT_EQ(status, UCS_OK);

  // Save the connection request to the arg parameter
  auto* conn_request_ptr = static_cast<ucp_conn_request_h*>(arg);
  *conn_request_ptr = conn_request;
}

void request_init(void* request) {
  UcxRequest* r = reinterpret_cast<UcxRequest*>(request);
  r->status = UCS_INPROGRESS;
}

// Server-side run function
void RunServer(
  int port, std::promise<bool>& server_ready,
  std::promise<std::vector<uint8_t>>& received_data,
  std::shared_future<void>& test_complete) {
  // Initialize UCX context and worker
  ucp_params_t ucp_params;
  ucp_config_t* config;
  ucs_status_t status;

  status = ucp_config_read(nullptr, nullptr, &config);
  ASSERT_EQ(status, UCS_OK);

  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES
                          | UCP_PARAM_FIELD_REQUEST_INIT
                          | UCP_PARAM_FIELD_REQUEST_SIZE | UCP_PARAM_FIELD_NAME;
  ucp_params.features = UCP_FEATURE_AM;
  ucp_params.request_init = UcxConnection::request_init;
  ucp_params.request_size = sizeof(UcxRequest);
  ucp_params.name = "server";

  ucp_context_h ucp_context;
  status = ucp_init(&ucp_params, config, &ucp_context);
  ucp_config_release(config);
  ASSERT_EQ(status, UCS_OK);

  ucp_worker_params_t worker_params;
  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

  ucp_worker_h ucp_worker;
  status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);
  ASSERT_EQ(status, UCS_OK);

  struct ucx_am_recv_data_callback_arg {
    std::atomic<bool>& recv_data_called;
    std::promise<std::vector<uint8_t>>& received_data;
    std::unique_ptr<UcxAmDesc> data_desc;
  };
  std::atomic<bool> recv_data_called{false};
  ucx_am_recv_data_callback_arg arg{recv_data_called, received_data, nullptr};
  // Register AM callback
  ucp_am_handler_param_t param;
  param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID
                     | UCP_AM_HANDLER_PARAM_FIELD_CB
                     | UCP_AM_HANDLER_PARAM_FIELD_ARG;
  param.id = DEFAULT_AM_MSG_ID;
  param.cb = [](
               void* arg, const void* header, size_t header_length, void* data,
               size_t length,
               const ucp_am_recv_param_t* param) -> ucs_status_t {
    if (length >= RNDV_THRESHOLD) {
      EXPECT_TRUE((param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    } else {
      EXPECT_FALSE((param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    }
    EXPECT_EQ(header_length, 1024);
    auto* header_ptr = static_cast<uint8_t*>(const_cast<void*>(header));
    for (size_t i = 0; i < header_length; ++i) {
      EXPECT_EQ(header_ptr[i], static_cast<uint8_t>(i % 256));
    }
    auto* arg_data = static_cast<ucx_am_recv_data_callback_arg*>(arg);
    if (!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
      auto* data_ptr = static_cast<uint8_t*>(const_cast<void*>(data));
      std::vector<uint8_t> received_data(data_ptr, data_ptr + length);
      arg_data->received_data.set_value(std::move(received_data));
    } else {
      arg_data->data_desc = std::make_unique<UcxAmDesc>(
        nullptr, 0, data, length, 0,
        static_cast<ucp_am_recv_attr_t>(param->recv_attr));
    }
    arg_data->recv_data_called = true;
    return UCS_INPROGRESS;
  };
  param.arg = &arg;

  status = ucp_worker_set_am_recv_handler(ucp_worker, &param);
  ASSERT_EQ(status, UCS_OK);

  // Create connection callback
  std::atomic<bool> callback_called{false};
  auto callback =
    std::make_unique<MockCallback>([&callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      callback_called = true;
    });

  auto conn = std::make_shared<UcxConnection>(ucp_worker, std::move(callback));

  // Create listening address
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);

  // Create listener
  ucp_listener_params_t listener_params;
  memset(&listener_params, 0, sizeof(listener_params));
  listener_params.field_mask =
    UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
  listener_params.sockaddr.addr = (struct sockaddr*)&addr;
  listener_params.sockaddr.addrlen = sizeof(addr);

  // Variable to store the connection request
  ucp_conn_request_h conn_request = nullptr;
  listener_params.conn_handler.cb = server_conn_handle_cb;
  listener_params.conn_handler.arg = &conn_request;

  ucp_listener_h listener;
  status = ucp_listener_create(ucp_worker, &listener_params, &listener);
  ASSERT_EQ(status, UCS_OK);

  // Notify that server is ready
  server_ready.set_value(true);

  // Wait for connection request
  std::atomic<bool> accept_callback_called{false};
  auto accept_callback = std::make_unique<MockCallback>(
    [&accept_callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      accept_callback_called = true;
    });

  // Wait for connection request to arrive
  while (conn_request == nullptr) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Accept connection
  conn->accept(conn_request, std::move(accept_callback));

  // Wait for connection establishment
  while (!accept_callback_called) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  while (!recv_data_called) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (
    arg.data_desc && (arg.data_desc->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
    std::atomic<bool> recv_nbx_called{false};
    class recv_am_nbx_callback : public UcxCallback {
     public:
      explicit recv_am_nbx_callback(std::atomic<bool>& recv_nbx_called)
        : recv_nbx_called_(recv_nbx_called) {}
      void operator()(ucs_status_t status) override {
        EXPECT_EQ(status, UCS_OK);
        recv_nbx_called_ = true;
      };

     private:
      std::atomic<bool>& recv_nbx_called_;
    };

    ASSERT_TRUE(arg.data_desc->desc != nullptr);
    std::vector<uint8_t> recv_data(arg.data_desc->data_length);
    auto [status, request] = conn->recv_am_data(
      recv_data.data(), recv_data.size(), nullptr, std::move(*arg.data_desc),
      std::make_unique<recv_am_nbx_callback>(recv_nbx_called));
    ASSERT_EQ(status, UCS_OK);
    ASSERT_EQ(request->type, UcxRequestType::Recv);
    ASSERT_TRUE(request->status == UCS_OK || request->status == UCS_INPROGRESS);
    while (!recv_nbx_called) {
      ucp_worker_progress(ucp_worker);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(recv_nbx_called);
    received_data.set_value(std::move(recv_data));
  }

  // Wait for test completion signal using shared_future
  test_complete.wait();

  // Clean up resources
  conn->disconnect();
  ucp_listener_destroy(listener);
  ucp_worker_destroy(ucp_worker);
  ucp_cleanup(ucp_context);
}

// Client-side run function
void RunClient(
  int port, std::vector<uint8_t>& send_data, std::promise<bool>& client_ready,
  std::shared_future<void>& test_complete) {
  // Initialize UCX context and worker
  ucp_params_t ucp_params;
  ucp_config_t* config;
  ucs_status_t status;

  status = ucp_config_read(nullptr, nullptr, &config);
  ASSERT_EQ(status, UCS_OK);

  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES
                          | UCP_PARAM_FIELD_REQUEST_INIT
                          | UCP_PARAM_FIELD_REQUEST_SIZE | UCP_PARAM_FIELD_NAME;
  ucp_params.features = UCP_FEATURE_AM;
  ucp_params.request_init = UcxConnection::request_init;
  ucp_params.request_size = sizeof(UcxRequest);
  ucp_params.name = "client";

  ucp_context_h ucp_context;
  status = ucp_init(&ucp_params, config, &ucp_context);
  ucp_config_release(config);
  ASSERT_EQ(status, UCS_OK);

  ucp_worker_params_t worker_params;
  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

  ucp_worker_h ucp_worker;
  status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);
  ASSERT_EQ(status, UCS_OK);

  // Create connection callback
  std::atomic<bool> callback_called{false};
  auto callback =
    std::make_unique<MockCallback>([&callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      callback_called = true;
    });

  auto conn = std::make_shared<UcxConnection>(ucp_worker, std::move(callback));

  // Create connection address
  struct sockaddr_in src_addr, dst_addr;
  memset(&src_addr, 0, sizeof(src_addr));
  memset(&dst_addr, 0, sizeof(dst_addr));

  src_addr.sin_family = AF_INET;
  src_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  src_addr.sin_port = htons(0);  // Let the system choose a port

  dst_addr.sin_family = AF_INET;
  dst_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  dst_addr.sin_port = htons(port);

  // Create connection callback
  std::atomic<bool> connect_callback_called{false};
  auto connect_callback = std::make_unique<MockCallback>(
    [&connect_callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      connect_callback_called = true;
    });

  // Connect to server
  conn->connect(
    (const struct sockaddr*)&src_addr, (const struct sockaddr*)&dst_addr,
    sizeof(src_addr), std::move(connect_callback));

  // Wait for connection establishment
  while (!connect_callback_called) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Notify that client is ready
  client_ready.set_value(true);

  // Prepare data to send
  std::vector<uint8_t> header_data(1024);
  for (size_t i = 0; i < header_data.size(); ++i) {
    header_data[i] = static_cast<uint8_t>(i % 256);
  }

  // Create send callback
  std::atomic<bool> send_callback_called{false};
  auto send_callback = std::make_unique<MockCallback>(
    [&send_callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      send_callback_called = true;
    });

  // Send data
  auto [status_send, request] = conn->send_am_data(
    header_data.data(), header_data.size(), send_data.data(), send_data.size(),
    nullptr, std::move(send_callback));
  ASSERT_EQ(status_send, UCS_OK);
  ASSERT_EQ(request->type, UcxRequestType::Send);
  ASSERT_TRUE(request->status == UCS_OK || request->status == UCS_INPROGRESS);

  // Wait for data sending to complete
  while (!send_callback_called) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Wait for test completion signal using shared_future
  test_complete.wait();

  // Clean up resources
  conn->disconnect();
  ucp_worker_destroy(ucp_worker);
  ucp_cleanup(ucp_context);
}

// Test AM eager data sending and receiving between client and server
TEST_F(UcxConnectionTest, SendRecvAmSmallData) {
  const int port =
    10000 + (rand_r(&seed) % 55535);  // Random port between 10000-65535
  std::promise<bool> server_ready, client_ready;
  std::promise<void> test_complete_promise;
  auto test_complete = test_complete_promise.get_future().share();
  std::promise<std::vector<uint8_t>> received_data;

  // Prepare data to send
  std::vector<uint8_t> send_data(1024);
  for (size_t i = 0; i < send_data.size(); ++i) {
    send_data[i] = static_cast<uint8_t>(i % 256);
  }

  // Start server thread
  std::thread server_thread(
    RunServer, port, std::ref(server_ready), std::ref(received_data),
    std::ref(test_complete));

  // Wait for server to be ready
  server_ready.get_future().wait();

  // Start client thread
  std::thread client_thread(
    RunClient, port, std::ref(send_data), std::ref(client_ready),
    std::ref(test_complete));

  // Wait for client to be ready
  client_ready.get_future().wait();

  // Wait for data reception to complete
  auto received = received_data.get_future().get();

  // Verify that received data matches sent data
  ASSERT_EQ(received.size(), 1024);
  for (size_t i = 0; i < received.size(); ++i) {
    ASSERT_EQ(received[i], static_cast<uint8_t>(i % 256));
  }

  // Send test completion signal using promise
  test_complete_promise.set_value();

  // Wait for threads to complete
  server_thread.join();
  client_thread.join();
}

// Test AM RNDV data sending and receiving between client and server
TEST_F(UcxConnectionTest, SendRecvAmLargeData) {
  const int port =
    10000 + (rand_r(&seed) % 55535);  // Random port between 10000-65535
  std::promise<bool> server_ready, client_ready;
  std::promise<void> test_complete_promise;
  auto test_complete = test_complete_promise.get_future().share();
  std::promise<std::vector<uint8_t>> received_data;

  // Prepare data to send
  std::vector<uint8_t> send_data(1024 * 1024);
  for (size_t i = 0; i < send_data.size(); ++i) {
    send_data[i] = static_cast<uint8_t>(i % 256);
  }

  // Start server thread
  std::thread server_thread(
    RunServer, port, std::ref(server_ready), std::ref(received_data),
    std::ref(test_complete));

  // Wait for server to be ready
  server_ready.get_future().wait();

  // Start client thread
  std::thread client_thread(
    RunClient, port, std::ref(send_data), std::ref(client_ready),
    std::ref(test_complete));

  // Wait for client to be ready
  client_ready.get_future().wait();

  // Wait for data reception to complete
  auto received = received_data.get_future().get();

  // Verify that received data matches sent data
  ASSERT_EQ(received.size(), 1024 * 1024);
  for (size_t i = 0; i < received.size(); ++i) {
    ASSERT_EQ(received[i], static_cast<uint8_t>(i % 256));
  }

  // Send test completion signal using promise
  test_complete_promise.set_value();

  // Wait for threads to complete
  server_thread.join();
  client_thread.join();
}

}  // namespace test
}  // namespace stdexe_ucx_runtime
