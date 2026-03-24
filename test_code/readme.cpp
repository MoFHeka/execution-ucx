#include <netinet/in.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <unifex/for_each.hpp>
#include <unifex/inplace_stop_token.hpp>
#include <unifex/on.hpp>
#include <unifex/single.hpp>
#include <unifex/spawn_detached.hpp>
#include <unifex/stop_if_requested.hpp>
#include <unifex/stop_on_request.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/take_until.hpp>
#include <unifex/task.hpp>
#include <unifex/v2/async_scope.hpp>
#include <unifex/when_all.hpp>

#include "ucx_context/ucx_am_context/ucx_am_context.hpp"
#include "ucx_context/ucx_context_def.h"
#include "ucx_context/ucx_memory_resource.hpp"

// Using declarations for clarity
using eux::ucxx::accept_endpoint;
using eux::ucxx::active_message_bundle;
using eux::ucxx::connect_endpoint;
using eux::ucxx::connection_recv;
using eux::ucxx::connection_send;
using eux::ucxx::DefaultUcxMemoryResourceManager;
using eux::ucxx::ucx_am_context;
using eux::ucxx::UcxMemoryResourceManager;
using unifex::task;

// Helper to create a socket address
static std::unique_ptr<sockaddr> create_socket_address(
  uint16_t port, bool is_server) {
  sockaddr_in* addr = new sockaddr_in{
    .sin_family = AF_INET,
    .sin_port = htons(port),
    .sin_addr = {.s_addr = htonl(is_server ? INADDR_ANY : INADDR_LOOPBACK)}};
  return std::unique_ptr<sockaddr>(reinterpret_cast<sockaddr*>(addr));
}

int main() {
  // 1. Setup server and client contexts, and run them in separate threads.
  std::unique_ptr<UcxMemoryResourceManager> server_mem_res;
  server_mem_res.reset(new DefaultUcxMemoryResourceManager());
  auto server_context =
    std::make_shared<ucx_am_context>(*server_mem_res, "server");
  unifex::inplace_stop_source server_stop_source;
  std::thread server_thread{
    [&] { server_context->run(server_stop_source.get_token()); }};

  std::unique_ptr<UcxMemoryResourceManager> client_mem_res;
  client_mem_res.reset(new DefaultUcxMemoryResourceManager());
  auto client_context =
    std::make_shared<ucx_am_context>(*client_mem_res, "client");
  unifex::inplace_stop_source client_stop_source;
  std::thread client_thread{
    [&] { client_context->run(client_stop_source.get_token()); }};

  // Allow contexts to initialize
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto server_scheduler = server_context->get_scheduler();
  auto client_scheduler = client_context->get_scheduler();
  unsigned int seed = static_cast<unsigned int>(time(nullptr));
  uint16_t port =
    static_cast<uint16_t>(1024 + (rand_r(&seed) % (65535 - 1024 + 1)));

  // 2. Prepare data for transfer.
  const size_t message_size = 1024;
  std::vector<char> test_data(message_size);
  for (size_t i = 0; i < message_size; ++i) {
    test_data[i] = static_cast<char>(i);
  }

  ucx_am_data send_data{};
  send_data.header.data = test_data.data();
  send_data.header.size = test_data.size();
  send_data.buffer.data = test_data.data();
  send_data.buffer.size = test_data.size();
  send_data.buffer_type = ucx_memory_type::HOST;

  ucx_am_data
    recv_data{};  // An empty descriptor, to be filled by the receiver.
  std::atomic<bool> message_received = false;
  std::atomic<bool> send_success = false;

  // 3. Define the server and client logic as unifex tasks.
  auto server_recv_logic =
    [&](std::vector<std::pair<std::uint64_t, std::error_code>>&&
          conn_id_status_vector) -> task<void> {
    // This call will populate the 'recv_data' struct upon message arrival.
    active_message_bundle bundle =
      co_await connection_recv(server_scheduler, recv_data);
    if (bundle.connection().is_established()) {
      message_received.store(true);
      server_stop_source
        .request_stop();  // Stop the server after receiving one message
      co_await unifex::stop_if_requested();
    }
  };

  auto server_logic = [&]() -> task<void> {
    // take_first from the stream of incoming connections.
    unifex::v2::async_scope scope;
    co_await unifex::for_each(
      unifex::take_until(
        accept_endpoint(
          server_scheduler, create_socket_address(port, true),
          sizeof(sockaddr_in)),
        unifex::single(
          unifex::stop_on_request(server_stop_source.get_token()))),
      [&](std::vector<std::pair<std::uint64_t, std::error_code>>&&
            conn_id_status_vector) {
        // Only spawn_detached is available in a not-coroutine function
        unifex::spawn_detached(
          unifex::on(
            server_scheduler,
            server_recv_logic(std::move(conn_id_status_vector))),
          scope);
      });
    co_await scope.join();
  };

  auto client_logic = [&]() -> task<void> {
    auto conn_id = co_await connect_endpoint(
      client_scheduler, nullptr, create_socket_address(port, false),
      sizeof(sockaddr_in));
    assert(conn_id != 0);

    co_await connection_send(client_scheduler, conn_id, send_data);
  };

  // 4. Run tasks concurrently and wait for them to complete.
  unifex::sync_wait(unifex::when_all(server_logic(), client_logic()));

  // 5. Verify results.
  assert(message_received.load());
  assert(send_success.load());
  assert(recv_data.header.size == message_size);
  assert(memcmp(recv_data.header.data, test_data.data(), message_size) == 0);
  assert(recv_data.buffer.size == message_size);
  assert(memcmp(recv_data.buffer.data, test_data.data(), message_size) == 0);
  std::cout << "Successfully transferred " << message_size << " bytes."
            << std::endl;

  // 6. Shutdown.
  client_stop_source.request_stop();
  server_thread.join();
  client_thread.join();

  return 0;
}
