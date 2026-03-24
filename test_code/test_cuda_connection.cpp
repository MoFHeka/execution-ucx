/**
 * @file test_cuda_connection.cpp
 * @author Hejia Zhang (hejia.zhang@gmail.com)
 * @brief Test CUDA connection
 * @version 0.1
 * @date 2025-08-07
 *
 * @copyright Copyright (c) 2025
 */

#include <cuda_runtime.h>
#include <ucp/api/ucp.h>
#include <ucs/datastruct/list.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "ucx_context/ucx_connection.hpp"

static constexpr size_t RNDV_THRESHOLD = 8192;

namespace eux {

// Mock callback for testing
class MockCallback : public UcxCallback {
 public:
  explicit MockCallback(std::function<void(ucs_status_t)> callback)
    : callback_(std::move(callback)) {}

  void operator()(ucs_status_t status) override { callback_(status); }

 private:
  std::function<void(ucs_status_t)> callback_;
};

void request_init(void* request) {
  UcxRequest* r = reinterpret_cast<UcxRequest*>(request);
  r->status = UCS_INPROGRESS;
}

// Server-side run function
void RunServer(
  ucp_worker_h ucp_worker, std::vector<std::byte>& client_ucp_address,
  std::promise<bool>& server_ready,
  std::promise<std::vector<uint8_t>>& received_data,
  std::shared_future<void>& test_complete) {
  struct ucx_am_recv_data_callback_arg {
    std::atomic<bool>& recv_data_called;
    std::promise<std::vector<uint8_t>>& received_data;
    std::unique_ptr<UcxAmDesc> data_desc;
  };
  cudaSetDevice(0);
  auto conn = std::make_shared<UcxConnection>(ucp_worker, nullptr);

  // Create connection callback
  std::atomic<bool> connect_callback_called{false};
  auto connect_callback = std::make_unique<MockCallback>(
    [&connect_callback_called](ucs_status_t status) {
      connect_callback_called = true;
    });

  // Connect to client
  conn->connect(
    reinterpret_cast<const ucp_address_t*>(client_ucp_address.data()),
    std::move(connect_callback));

  // Wait for connection establishment
  while (!connect_callback_called) {
    ucp_worker_progress(ucp_worker);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

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

  ucp_worker_set_am_recv_handler(ucp_worker, &param);

  // Notify that server is ready
  server_ready.set_value(true);

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
        recv_nbx_called_ = true;
      };

     private:
      std::atomic<bool>& recv_nbx_called_;
    };

    std::vector<uint8_t> recv_data(arg.data_desc->data_length);
    void* recv_data_device = nullptr;
    cudaError_t cuda_status = cudaMalloc(&recv_data_device, recv_data.size());
    assert(
      cuda_status == cudaSuccess && "cudaMalloc failed for recv_data_device");
    conn->recv_am_data(
      recv_data_device, recv_data.size(), nullptr, std::move(*arg.data_desc),
      UCS_MEMORY_TYPE_CUDA,
      std::make_unique<recv_am_nbx_callback>(recv_nbx_called));
    while (!recv_nbx_called) {
      ucp_worker_progress(ucp_worker);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    cuda_status = cudaMemcpy(
      recv_data.data(), recv_data_device, recv_data.size(),
      cudaMemcpyDeviceToHost);
    assert(
      cuda_status == cudaSuccess && "cudaMemcpy failed for recv_data_device");
    received_data.set_value(std::move(recv_data));
  }

  // Wait for test completion signal using shared_future
  test_complete.wait();

  // Clean up resources
  conn->disconnect();
  ucp_worker_destroy(ucp_worker);
}

// Client-side run function
void RunClient(
  ucp_worker_h ucp_worker, std::vector<std::byte>& server_ucp_address,
  void* send_data, size_t send_data_size, std::promise<bool>& client_ready,
  std::shared_future<void>& test_complete) {
  cudaSetDevice(0);
  auto conn = std::make_shared<UcxConnection>(ucp_worker, nullptr);

  // Create connection callback
  std::atomic<bool> connect_callback_called{false};
  auto connect_callback = std::make_unique<MockCallback>(
    [&connect_callback_called](ucs_status_t status) {
      connect_callback_called = true;
    });

  // Connect to server
  conn->connect(
    reinterpret_cast<const ucp_address_t*>(server_ucp_address.data()),
    std::move(connect_callback));

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
      send_callback_called = true;
    });

  // Send data
  conn->send_am_data(
    header_data.data(), header_data.size(), send_data, send_data_size, nullptr,
    UCS_MEMORY_TYPE_CUDA, std::move(send_callback));

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
}
}  // namespace eux

int main() {
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Device Name: " << prop.name << std::endl;

  // Initialize UCX context and worker
  ucp_params_t ucp_params;
  ucp_config_t* config;
  ucs_status_t status;

  status = ucp_config_read(nullptr, nullptr, &config);

  memset(&ucp_params, 0, sizeof(ucp_params));
  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES
                          | UCP_PARAM_FIELD_REQUEST_INIT
                          | UCP_PARAM_FIELD_REQUEST_SIZE | UCP_PARAM_FIELD_NAME;
  ucp_params.features = UCP_FEATURE_AM | UCP_FEATURE_TAG | UCP_FEATURE_RMA;
  ucp_params.request_init = eux::ucxx::UcxConnection::request_init;
  ucp_params.request_size = sizeof(eux::ucxx::UcxRequest);
  ucp_params.name = "server";

  ucp_context_h ucp_context;
  status = ucp_init(&ucp_params, config, &ucp_context);

  // ucp_config_print(
  //   config, stdout, "UCX Configuration:", UCS_CONFIG_PRINT_CONFIG);

  ucp_config_release(config);

  // Create server UCX context and worker

  ucp_worker_params_t server_worker_params;
  memset(&server_worker_params, 0, sizeof(server_worker_params));
  server_worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  server_worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  ucp_worker_h server_ucp_worker;
  status =
    ucp_worker_create(ucp_context, &server_worker_params, &server_ucp_worker);

  std::vector<std::byte> server_address_buffer;
  ucp_worker_attr_t server_ucp_worker_attr;
  server_ucp_worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS;
  status = ucp_worker_query(server_ucp_worker, &server_ucp_worker_attr);
  auto address =
    reinterpret_cast<const std::byte*>(server_ucp_worker_attr.address);
  server_address_buffer.assign(
    address, address + server_ucp_worker_attr.address_length);
  std::cout << "Server UCX address: " << server_address_buffer.size()
            << std::endl;
  assert(status == UCS_OK);

  // Initialize UCX worker
  ucp_worker_params_t client_worker_params;
  memset(&client_worker_params, 0, sizeof(client_worker_params));
  client_worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  client_worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  ucp_worker_h client_ucp_worker;
  status =
    ucp_worker_create(ucp_context, &client_worker_params, &client_ucp_worker);

  std::vector<std::byte> client_address_buffer;
  ucp_worker_attr_t ucp_worker_attr;
  ucp_worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS;
  status = ucp_worker_query(client_ucp_worker, &ucp_worker_attr);
  auto client_address =
    reinterpret_cast<const std::byte*>(ucp_worker_attr.address);
  client_address_buffer.assign(
    client_address, client_address + ucp_worker_attr.address_length);
  std::cout << "Client UCX address: " << client_address_buffer.size()
            << std::endl;
  assert(status == UCS_OK);

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
    eux::RunServer, server_ucp_worker, std::ref(client_address_buffer),
    std::ref(server_ready), std::ref(received_data), std::ref(test_complete));

  // Wait for server to be ready
  server_ready.get_future().wait();

  // Allocate device memory for send_data and copy from host to device
  void* send_data_device = nullptr;
  cudaError_t cuda_status = cudaMalloc(&send_data_device, send_data.size());
  assert(
    cuda_status == cudaSuccess && "cudaMalloc failed for send_data_device");
  cuda_status = cudaMemcpy(
    send_data_device, send_data.data(), send_data.size(),
    cudaMemcpyHostToDevice);
  assert(
    cuda_status == cudaSuccess && "cudaMemcpy failed for send_data_device");

  // Start client thread
  std::thread client_thread(
    eux::RunClient, client_ucp_worker, std::ref(server_address_buffer),
    send_data_device, send_data.size(), std::ref(client_ready),
    std::ref(test_complete));

  // Wait for client to be ready
  client_ready.get_future().wait();

  // Wait for data reception to complete
  auto received = received_data.get_future().get();

  std::cout << "Received data size: " << received.size() << std::endl;
  std::cout << "First 16 bytes of received data: ";
  for (size_t i = 0; i < std::min<size_t>(16, received.size()); ++i) {
    std::cout << static_cast<int>(received[i]) << " ";
  }
  std::cout << std::endl;

  test_complete_promise.set_value();

  server_thread.join();
  client_thread.join();

  ucp_cleanup(ucp_context);

  return 0;
}