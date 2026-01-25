/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.

Licensed under the Apache License Version 2.0 with LLVM Exceptions
(the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    https://llvm.org/LICENSE.txt

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUTHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ucx_context/ucx_context_logger.hpp"

static constexpr size_t RNDV_THRESHOLD = 8192;

namespace eux {
namespace ucxx {
namespace test {

bool isPortAvailable(int port) {
  int sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) {
    return false;
  }

  struct sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);

  int optval = 1;
  setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

  bool available = (bind(sockfd, (struct sockaddr*)&addr, sizeof(addr)) == 0);
  close(sockfd);
  return available;
}

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
  std::atomic<bool> callback_called(false);
  auto callback =
    std::make_unique<MockCallback>([&callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      callback_called = true;
    });

  auto conn = std::make_unique<UcxConnection>(ucp_worker_, std::move(callback));
  EXPECT_NE(conn, nullptr);
  // should be nullptr because ep is nullptr
  EXPECT_EQ(conn->id(), reinterpret_cast<std::uintptr_t>(nullptr));
}

// Test connection establishment
TEST_F(UcxConnectionTest, Connect) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  std::atomic<bool> callback_called(false);
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
  std::atomic<bool> connect_callback_called(false);
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
  testing::internal::CaptureStdout();
  conn.reset(nullptr);
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

// ================= Test Infrastructure helper functions ======================

static void setup_peer(
  const char* name, ucp_context_h& ucp_context, ucp_worker_h& ucp_worker) {
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
  ucp_params.name = name;

  status = ucp_init(&ucp_params, config, &ucp_context);
  ucp_config_release(config);
  ASSERT_EQ(status, UCS_OK);

  ucp_worker_params_t worker_params;
  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

  status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);
  ASSERT_EQ(status, UCS_OK);
}

static ucp_listener_h create_listener(
  ucp_worker_h ucp_worker, int port, ucp_conn_request_h* conn_request) {
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);

  ucp_listener_params_t listener_params;
  memset(&listener_params, 0, sizeof(listener_params));
  listener_params.field_mask =
    UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
  listener_params.sockaddr.addr = (struct sockaddr*)&addr;
  listener_params.sockaddr.addrlen = sizeof(addr);
  listener_params.conn_handler.cb = server_conn_handle_cb;
  listener_params.conn_handler.arg = conn_request;

  ucp_listener_h listener;
  ucs_status_t status =
    ucp_listener_create(ucp_worker, &listener_params, &listener);
  EXPECT_EQ(status, UCS_OK);
  return listener;
}

static void accept_connection(
  std::shared_ptr<UcxConnection>& conn, ucp_worker_h ucp_worker,
  ucp_conn_request_h& conn_request) {
  while (conn_request == nullptr) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::atomic<bool> accept_callback_called{false};
  auto accept_callback = std::make_unique<MockCallback>(
    [&accept_callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      accept_callback_called = true;
    });

  conn->accept(conn_request, std::move(accept_callback));

  while (!accept_callback_called) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

static void connect_to_server(
  std::shared_ptr<UcxConnection>& conn, ucp_worker_h ucp_worker, int port) {
  struct sockaddr_in src_addr, dst_addr;
  memset(&src_addr, 0, sizeof(src_addr));
  memset(&dst_addr, 0, sizeof(dst_addr));

  src_addr.sin_family = AF_INET;
  src_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  src_addr.sin_port = htons(0);

  dst_addr.sin_family = AF_INET;
  dst_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  dst_addr.sin_port = htons(port);

  std::atomic<bool> connect_callback_called{false};
  auto connect_callback = std::make_unique<MockCallback>(
    [&connect_callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      connect_callback_called = true;
    });

  conn->connect(
    (const struct sockaddr*)&src_addr, (const struct sockaddr*)&dst_addr,
    sizeof(src_addr), std::move(connect_callback));

  while (!connect_callback_called) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

// Server-side run function
void RunServer(
  int port, std::promise<bool>& server_ready,
  std::promise<std::vector<uint8_t>>& received_data,
  std::shared_future<void>& test_complete) {
  ucp_context_h ucp_context;
  ucp_worker_h ucp_worker;
  setup_peer("server", ucp_context, ucp_worker);

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
      std::vector<uint8_t> received_data_vec(data_ptr, data_ptr + length);
      arg_data->received_data.set_value(std::move(received_data_vec));
      arg_data->recv_data_called = true;
      return UCS_OK;  // Eager path must return OK, not INPROGRESS
    } else {
      arg_data->data_desc = std::make_unique<UcxAmDesc>(
        nullptr, 0, data, length, 0,
        static_cast<ucp_am_recv_attr_t>(param->recv_attr));
    }
    arg_data->recv_data_called = true;
    return UCS_INPROGRESS;
  };
  param.arg = &arg;

  ucs_status_t status = ucp_worker_set_am_recv_handler(ucp_worker, &param);
  ASSERT_EQ(status, UCS_OK);

  // Create connection callback
  std::atomic<bool> callback_called{false};
  auto callback =
    std::make_unique<MockCallback>([&callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      callback_called = true;
    });

  auto conn = std::make_shared<UcxConnection>(ucp_worker, std::move(callback));

  ucp_conn_request_h conn_request = nullptr;
  ucp_listener_h listener = create_listener(ucp_worker, port, &conn_request);

  // Notify that server is ready
  server_ready.set_value(true);

  accept_connection(conn, ucp_worker, conn_request);

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
    auto [req_status, request] = conn->recv_am_data(
      recv_data.data(), recv_data.size(), nullptr, std::move(*arg.data_desc),
      UCS_MEMORY_TYPE_HOST,
      std::make_unique<recv_am_nbx_callback>(recv_nbx_called));
    ASSERT_EQ(req_status, UCS_OK);
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
  ucp_context_h ucp_context;
  ucp_worker_h ucp_worker;
  setup_peer("client", ucp_context, ucp_worker);

  // Create connection callback
  std::atomic<bool> callback_called{false};
  auto callback =
    std::make_unique<MockCallback>([&callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      callback_called = true;
    });

  auto conn = std::make_shared<UcxConnection>(ucp_worker, std::move(callback));

  connect_to_server(conn, ucp_worker, port);

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
    nullptr, UCS_MEMORY_TYPE_HOST, std::move(send_callback));
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
  unsigned int seed = static_cast<unsigned int>(time(nullptr));
  int port =
    10000 + (rand_r(&seed) % 55535);  // Random port between 10000-65535
  while (!isPortAvailable(port)) {
    port = 10000 + (rand_r(&seed) % 55535);
  }
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

// ============================ IOV tests helpers ============================

// Server-side run function for IOV AM
void RunServerIov(
  int port, std::promise<bool>& server_ready,
  std::promise<std::vector<uint8_t>>& received_data,
  std::shared_future<void>& test_complete,
  const std::vector<size_t>& expected_lengths) {
  ucp_context_h ucp_context;
  ucp_worker_h ucp_worker;
  setup_peer("server-iov", ucp_context, ucp_worker);

  struct ucx_am_recv_iov_callback_arg {
    std::atomic<bool>& recv_called;
    std::promise<std::vector<uint8_t>>& received_data;
    std::unique_ptr<UcxAmDesc> data_desc;
  };

  std::atomic<bool> recv_called{false};
  ucx_am_recv_iov_callback_arg arg{recv_called, received_data, nullptr};

  // Register AM callback (same AM id as IOV uses)
  ucp_am_handler_param_t param;
  param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID
                     | UCP_AM_HANDLER_PARAM_FIELD_CB
                     | UCP_AM_HANDLER_PARAM_FIELD_ARG;
  param.id = IOVEC_AM_MSG_ID;  // same id as default in code
  param.cb = [](
               void* arg, const void* header, size_t header_length, void* data,
               size_t length,
               const ucp_am_recv_param_t* param) -> ucs_status_t {
    auto* a = static_cast<ucx_am_recv_iov_callback_arg*>(arg);
    // For IOV path we always force RNDV in send
    EXPECT_TRUE((param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    // Verify header {1,2,3,4}
    EXPECT_EQ(header_length, 4u);
    if (header_length == 4u) {
      const uint8_t* h = static_cast<const uint8_t*>(header);
      EXPECT_EQ(h[0], 1);
      EXPECT_EQ(h[1], 2);
      EXPECT_EQ(h[2], 3);
      EXPECT_EQ(h[3], 4);
    }
    a->data_desc = std::make_unique<UcxAmDesc>(
      const_cast<void*>(header), header_length, data, length, 0,
      static_cast<ucp_am_recv_attr_t>(param->recv_attr));
    a->recv_called = true;
    return UCS_INPROGRESS;
  };
  param.arg = &arg;

  ucs_status_t status = ucp_worker_set_am_recv_handler(ucp_worker, &param);
  ASSERT_EQ(status, UCS_OK);

  // Create connection callback
  std::atomic<bool> callback_called{false};
  auto callback =
    std::make_unique<MockCallback>([&callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      callback_called = true;
    });

  auto conn = std::make_shared<UcxConnection>(ucp_worker, std::move(callback));

  ucp_conn_request_h conn_request = nullptr;
  ucp_listener_h listener = create_listener(ucp_worker, port, &conn_request);

  // Notify that server is ready
  server_ready.set_value(true);

  accept_connection(conn, ucp_worker, conn_request);

  // Wait AM arrival
  while (!recv_called) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Use expected lengths provided by test
  ASSERT_TRUE(arg.data_desc != nullptr);
  // arg.data_desc->data_length is total data length
  const size_t iov_count = expected_lengths.size();

  // Prepare receive IOVs
  std::vector<std::vector<uint8_t>> recv_segs(iov_count);
  std::vector<ucp_dt_iov_t> recv_iov(iov_count);
  for (size_t i = 0; i < iov_count; ++i) {
    recv_segs[i].resize(expected_lengths[i]);
    recv_iov[i].buffer = recv_segs[i].data();
    recv_iov[i].length = recv_segs[i].size();
  }

  // Receive via NBX IOV
  std::atomic<bool> recv_nbx_called{false};
  class recv_am_iov_callback : public UcxCallback {
   public:
    explicit recv_am_iov_callback(std::atomic<bool>& flag) : flag_(flag) {}
    void operator()(ucs_status_t status) override {
      EXPECT_EQ(status, UCS_OK);
      flag_ = true;
    }

   private:
    std::atomic<bool>& flag_;
  };

  auto [recv_status, recv_req] = conn->recv_am_iov_data(
    recv_iov.data(), recv_iov.size(), nullptr, std::move(*arg.data_desc),
    UCS_MEMORY_TYPE_HOST,
    std::make_unique<recv_am_iov_callback>(recv_nbx_called));
  ASSERT_EQ(recv_status, UCS_OK);
  ASSERT_TRUE(
    recv_req == nullptr || recv_req->status == UCS_INPROGRESS
    || recv_req->status == UCS_OK);

  while (!recv_nbx_called) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Flatten to a single vector for assertion convenience
  size_t total = 0;
  for (size_t i = 0; i < iov_count; ++i) total += recv_segs[i].size();
  std::vector<uint8_t> flat;
  flat.reserve(total);
  for (size_t i = 0; i < iov_count; ++i) {
    flat.insert(flat.end(), recv_segs[i].begin(), recv_segs[i].end());
  }
  received_data.set_value(std::move(flat));

  // Wait for test completion
  test_complete.wait();

  // Clean up
  conn->disconnect();
  ucp_listener_destroy(listener);
  ucp_worker_destroy(ucp_worker);
  ucp_cleanup(ucp_context);
}

// Client-side run function for IOV AM
void RunClientIov(
  int port, const std::vector<std::vector<uint8_t>>& segs,
  std::promise<bool>& client_ready, std::shared_future<void>& test_complete) {
  ucp_context_h ucp_context;
  ucp_worker_h ucp_worker;
  setup_peer("client-iov", ucp_context, ucp_worker);

  // Create connection callback
  std::atomic<bool> callback_called{false};
  auto callback =
    std::make_unique<MockCallback>([&callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      callback_called = true;
    });

  auto conn = std::make_shared<UcxConnection>(ucp_worker, std::move(callback));

  connect_to_server(conn, ucp_worker, port);

  // Notify client ready
  client_ready.set_value(true);

  // Prepare simple header {1,2,3,4}
  std::vector<uint8_t> header_bytes{1, 2, 3, 4};

  // Build IOV list
  std::vector<ucp_dt_iov_t> iov(segs.size());
  for (size_t i = 0; i < segs.size(); ++i) {
    iov[i].buffer = const_cast<uint8_t*>(segs[i].data());
    iov[i].length = segs[i].size();
  }

  std::atomic<bool> send_callback_called{false};
  auto send_callback = std::make_unique<MockCallback>(
    [&send_callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      send_callback_called = true;
    });

  auto [status_send, request] = conn->send_am_iov_data(
    header_bytes.data(), header_bytes.size(), iov.data(), iov.size(), nullptr,
    UCS_MEMORY_TYPE_HOST, std::move(send_callback));
  ASSERT_EQ(status_send, UCS_OK);
  ASSERT_TRUE(
    request == nullptr || request->status == UCS_INPROGRESS
    || request->status == UCS_OK);

  while (!send_callback_called) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Wait for test completion
  test_complete.wait();

  // Clean up
  conn->disconnect();
  ucp_worker_destroy(ucp_worker);
  ucp_cleanup(ucp_context);
}

// Server-side run function for IOV AM sending with contiguous receive
void RunServerIovRecvContig(
  int port, std::promise<bool>& server_ready,
  std::promise<std::vector<uint8_t>>& received_data,
  std::shared_future<void>& test_complete) {
  ucp_context_h ucp_context;
  ucp_worker_h ucp_worker;
  setup_peer("server-iov-contig", ucp_context, ucp_worker);

  struct ucx_am_recv_iov_to_contig_callback_arg {
    std::atomic<bool>& recv_called;
    std::promise<std::vector<uint8_t>>& received_data;
    std::unique_ptr<UcxAmDesc> data_desc;
  };

  std::atomic<bool> recv_called{false};
  ucx_am_recv_iov_to_contig_callback_arg arg{
    recv_called, received_data, nullptr};

  // Register AM callback (same AM id as IOV uses)
  ucp_am_handler_param_t param;
  param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID
                     | UCP_AM_HANDLER_PARAM_FIELD_CB
                     | UCP_AM_HANDLER_PARAM_FIELD_ARG;
  param.id = IOVEC_AM_MSG_ID;
  param.cb = [](
               void* arg, const void* header, size_t header_length, void* data,
               size_t length,
               const ucp_am_recv_param_t* param) -> ucs_status_t {
    auto* a = static_cast<ucx_am_recv_iov_to_contig_callback_arg*>(arg);
    // For IOV path we always force RNDV in send
    EXPECT_TRUE((param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    // Verify header {1,2,3,4}
    EXPECT_EQ(header_length, 4u);
    if (header_length == 4u) {
      const uint8_t* h = static_cast<const uint8_t*>(header);
      EXPECT_EQ(h[0], 1);
      EXPECT_EQ(h[1], 2);
      EXPECT_EQ(h[2], 3);
      EXPECT_EQ(h[3], 4);
    }
    a->data_desc = std::make_unique<UcxAmDesc>(
      const_cast<void*>(header), header_length, data, length, 0,
      static_cast<ucp_am_recv_attr_t>(param->recv_attr));
    a->recv_called = true;
    return UCS_INPROGRESS;
  };
  param.arg = &arg;

  ucs_status_t status = ucp_worker_set_am_recv_handler(ucp_worker, &param);
  ASSERT_EQ(status, UCS_OK);

  // Create connection callback
  std::atomic<bool> callback_called{false};
  auto callback =
    std::make_unique<MockCallback>([&callback_called](ucs_status_t status) {
      EXPECT_EQ(status, UCS_OK);
      callback_called = true;
    });

  auto conn = std::make_shared<UcxConnection>(ucp_worker, std::move(callback));

  ucp_conn_request_h conn_request = nullptr;
  ucp_listener_h listener = create_listener(ucp_worker, port, &conn_request);

  // Notify that server is ready
  server_ready.set_value(true);

  accept_connection(conn, ucp_worker, conn_request);

  // Wait AM arrival
  while (!recv_called) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Receive into contiguous buffer
  ASSERT_TRUE(arg.data_desc != nullptr);
  std::vector<uint8_t> recv_data(arg.data_desc->data_length);

  std::atomic<bool> recv_nbx_called{false};
  class recv_am_nbx_callback : public UcxCallback {
   public:
    explicit recv_am_nbx_callback(std::atomic<bool>& flag) : flag_(flag) {}
    void operator()(ucs_status_t status) override {
      EXPECT_EQ(status, UCS_OK);
      flag_ = true;
    }

   private:
    std::atomic<bool>& flag_;
  };

  auto [recv_status, recv_req] = conn->recv_am_data(
    recv_data.data(), recv_data.size(), nullptr, std::move(*arg.data_desc),
    UCS_MEMORY_TYPE_HOST,
    std::make_unique<recv_am_nbx_callback>(recv_nbx_called));
  ASSERT_EQ(recv_status, UCS_OK);
  ASSERT_TRUE(
    recv_req == nullptr || recv_req->status == UCS_INPROGRESS
    || recv_req->status == UCS_OK);

  while (!recv_nbx_called) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  received_data.set_value(std::move(recv_data));

  // Wait for test completion
  test_complete.wait();

  // Clean up
  conn->disconnect();
  ucp_listener_destroy(listener);
  ucp_worker_destroy(ucp_worker);
  ucp_cleanup(ucp_context);
}

// Test AM RNDV data sending and receiving between client and server
TEST_F(UcxConnectionTest, SendRecvAmLargeData) {
  unsigned int seed = static_cast<unsigned int>(time(nullptr));
  int port =
    10000 + (rand_r(&seed) % 55535);  // Random port between 10000-65535
  while (!isPortAvailable(port)) {
    port = 10000 + (rand_r(&seed) % 55535);
  }
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

// ============================ IOV tests ============================

TEST_F(UcxConnectionTest, SendRecvAmIovSmallData) {
  unsigned int seed = static_cast<unsigned int>(time(nullptr));
  int port =
    10000 + (rand_r(&seed) % 55535);  // Random port between 10000-65535
  while (!isPortAvailable(port)) {
    port = 10000 + (rand_r(&seed) % 55535);
  }
  std::promise<bool> server_ready, client_ready;
  std::promise<void> test_complete_promise;
  auto test_complete = test_complete_promise.get_future().share();
  std::promise<std::vector<uint8_t>> received_data;

  // Prepare small IOV segments
  std::vector<std::vector<uint8_t>> segs;
  segs.emplace_back(256);
  segs.emplace_back(512);
  segs.emplace_back(300);
  // Fill pattern
  size_t cursor = 0;
  for (auto& s : segs) {
    for (size_t i = 0; i < s.size(); ++i, ++cursor)
      s[i] = static_cast<uint8_t>(cursor % 256);
  }

  // Expected flat
  std::vector<uint8_t> expected;
  expected.reserve(256 + 512 + 300);
  for (auto& s : segs) expected.insert(expected.end(), s.begin(), s.end());

  // Start server
  std::vector<size_t> lengths;
  lengths.reserve(segs.size());
  for (auto& s : segs) lengths.push_back(s.size());
  std::thread server_thread(
    RunServerIov, port, std::ref(server_ready), std::ref(received_data),
    std::ref(test_complete), std::ref(lengths));

  // Wait server ready
  server_ready.get_future().wait();

  // Start client
  std::thread client_thread(
    RunClientIov, port, std::cref(segs), std::ref(client_ready),
    std::ref(test_complete));

  // Wait client ready
  client_ready.get_future().wait();

  // Wait receive done
  auto received = received_data.get_future().get();

  ASSERT_EQ(received.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(received[i], expected[i]);
  }

  // Signal finish
  test_complete_promise.set_value();

  server_thread.join();
  client_thread.join();
}

TEST_F(UcxConnectionTest, SendRecvAmIovLargeData) {
  unsigned int seed = static_cast<unsigned int>(time(nullptr));
  int port =
    10000 + (rand_r(&seed) % 55535);  // Random port between 10000-65535
  while (!isPortAvailable(port)) {
    port = 10000 + (rand_r(&seed) % 55535);
  }
  std::promise<bool> server_ready, client_ready;
  std::promise<void> test_complete_promise;
  auto test_complete = test_complete_promise.get_future().share();
  std::promise<std::vector<uint8_t>> received_data;

  // Prepare large IOV segments (total ~1.5MB)
  std::vector<std::vector<uint8_t>> segs;
  segs.emplace_back(512 * 1024);
  segs.emplace_back(256 * 1024);
  segs.emplace_back(800 * 1024);
  size_t cursor = 0;
  for (auto& s : segs) {
    for (size_t i = 0; i < s.size(); ++i, ++cursor)
      s[i] = static_cast<uint8_t>(cursor % 256);
  }

  std::vector<uint8_t> expected;
  expected.reserve(512 * 1024 + 256 * 1024 + 800 * 1024);
  for (auto& s : segs) expected.insert(expected.end(), s.begin(), s.end());

  std::vector<size_t> lengths2;
  lengths2.reserve(segs.size());
  for (auto& s : segs) lengths2.push_back(s.size());
  std::thread server_thread(
    RunServerIov, port, std::ref(server_ready), std::ref(received_data),
    std::ref(test_complete), std::ref(lengths2));

  server_ready.get_future().wait();

  std::thread client_thread(
    RunClientIov, port, std::cref(segs), std::ref(client_ready),
    std::ref(test_complete));

  client_ready.get_future().wait();

  auto received = received_data.get_future().get();

  ASSERT_EQ(received.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(received[i], expected[i]);
  }

  test_complete_promise.set_value();

  server_thread.join();
  client_thread.join();
}

TEST_F(UcxConnectionTest, SendIovRecvAmLargeData) {
  unsigned int seed = static_cast<unsigned int>(time(nullptr));
  int port =
    10000 + (rand_r(&seed) % 55535);  // Random port between 10000-65535
  while (!isPortAvailable(port)) {
    port = 10000 + (rand_r(&seed) % 55535);
  }
  std::promise<bool> server_ready, client_ready;
  std::promise<void> test_complete_promise;
  auto test_complete = test_complete_promise.get_future().share();
  std::promise<std::vector<uint8_t>> received_data;

  // Prepare large IOV segments (total ~1.5MB)
  std::vector<std::vector<uint8_t>> segs;
  segs.emplace_back(512 * 1024);
  segs.emplace_back(256 * 1024);
  segs.emplace_back(800 * 1024);
  size_t cursor = 0;
  for (auto& s : segs) {
    for (size_t i = 0; i < s.size(); ++i, ++cursor)
      s[i] = static_cast<uint8_t>(cursor % 256);
  }

  // Expected flat
  std::vector<uint8_t> expected;
  expected.reserve(512 * 1024 + 256 * 1024 + 800 * 1024);
  for (auto& s : segs) expected.insert(expected.end(), s.begin(), s.end());

  // Start server
  std::thread server_thread(
    RunServerIovRecvContig, port, std::ref(server_ready),
    std::ref(received_data), std::ref(test_complete));

  // Wait server ready
  server_ready.get_future().wait();

  // Start client
  std::thread client_thread(
    RunClientIov, port, std::cref(segs), std::ref(client_ready),
    std::ref(test_complete));

  // Wait client ready
  client_ready.get_future().wait();

  // Wait receive done
  auto received = received_data.get_future().get();

  ASSERT_EQ(received.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(received[i], expected[i]);
  }

  // Signal finish
  test_complete_promise.set_value();

  server_thread.join();
  client_thread.join();
}

}  // namespace test
}  // namespace ucxx
}  // namespace eux
