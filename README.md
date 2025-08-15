[简体中文](README_Chinese.md)

# execution-ucx

`execution-ucx` is a full-featured OpenUCX runtime context based on the C++26 `P2300` `std::execution` proposal. It is designed to unify all UCX communication tasks (both control plane and data plane) within the `ucx_context`'s worker thread. This approach eliminates lock contention and reduces thread switching, achieving ultimate performance.

This project currently uses Meta's [libunifex](https://github.com/facebookexperimental/libunifex) as the implementation for `std::execution`, but it can be easily switched to NVIDIA's [stdexec](https://github.com/NVIDIA/stdexec) or other compatible implementations.

Its design goal is to provide an efficient, flexible, and composable asynchronous RDMA communication backend for modern C++ applications.

![Architecture](doc/ucx_context.png)

## Core Features

*   **Based on `std::execution`**: Utilizes the latest C++ asynchronous model, offering an expressive and composable API.
*   **High Performance**: All operations are executed in a dedicated `ucx_context` thread, avoiding multithreading synchronization overhead and maximizing UCX performance.
*   **Active Message Support**: Built-in efficient Active Message (`ucx_am_context`) implementation, supporting zero-copy and callback handling.
*   **Connection Management**: Automated connection establishment, caching, and management (`ucx_connection_manager`).
*   **Memory Management**: Integrated UCX memory registration/deregistration (`ucx_memory_resource`), simplifying RDMA operations.
*   **CUDA Support**: Seamless support for CUDA device memory, enabling GPU-Direct RDMA (GDR).
*   **Extensibility**: Modular design allows for easy extension to support new protocols or hardware.

## Core Concepts

*   `ucx_context`: The core component that encapsulates `ucp_worker_h` and drives all asynchronous operations. It has its own thread for polling UCX events and executing tasks.
*   `ucx_am_context`: The Active Message context, providing an interface for sending and receiving Active Messages.
*   `ucx_connection`: Encapsulates `ucp_ep_h`, representing a connection to a remote peer.
*   `ucx_connection_manager`: Manages and reuses `ucx_connection`, handling connection establishment and teardown.
*   `ucx_memory_resource`: A C++ PMR-style memory resource for allocating registered memory that can be used directly by UCX for RDMA operations.

## Dependencies

*   **Bazel**: >= 7.0.0
*   **C++ Compiler**: C++17 support
*   **OpenUCX**: v1.18.1 or later
*   **libunifex**: (default) As the `std::execution` implementation
*   **liburing**: A dependency for `libunifex`
*   **Googletest**: For unit tests
*   **(Optional) CUDA Toolkit**: For building with GPU support
*   **(Optional) Various Communication Interface**: Please modify the [OpenUCX BUILD file](third_party/openucx/BUILD.bazel) to select your preferred options. It is recommended to install the [Nvidia HPC SDK](https://developer.nvidia.com/hpc-sdk)

## Build and Test

The project is built using Bazel.

1.  **Build the project**:
    ```bash
    bazel build //ucx_context:ucx_am_context
    ```

2.  **Run tests (CPU)**:
    ```bash
    bazel test //ucx_context:ucx_am_context_test
    ```

3.  **Run tests (CUDA)**:
    Requires a local installation of the CUDA Toolkit. Bazel will automatically detect it and enable CUDA support.
    ```bash
    bazel test //ucx_context:ucx_am_context_test --@rules_cuda//cuda:enable=True
    ```

## Usage Example

The following is a simplified example demonstrating how to send and receive an Active Message using `ucx_am_context`. It is based on the logic from the `TEST_F(UcxAmTest, SmallMessageTransfer)` test case.
For more details, please refer to the [test code](ucx_context/ucx_am_context_test.cpp)

#### test.cpp
```cpp
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
using stdexe_ucx_runtime::accept_endpoint;
using stdexe_ucx_runtime::active_message_bundle;
using stdexe_ucx_runtime::connect_endpoint;
using stdexe_ucx_runtime::connection_recv;
using stdexe_ucx_runtime::connection_send;
using stdexe_ucx_runtime::DefaultUcxMemoryResourceManager;
using stdexe_ucx_runtime::ucx_am_context;
using stdexe_ucx_runtime::UcxMemoryResourceManager;
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
    std::make_shared<ucx_am_context>(server_mem_res, "server");
  unifex::inplace_stop_source server_stop_source;
  std::thread server_thread{
    [&] { server_context->run(server_stop_source.get_token()); }};

  std::unique_ptr<UcxMemoryResourceManager> client_mem_res;
  client_mem_res.reset(new DefaultUcxMemoryResourceManager());
  auto client_context =
    std::make_shared<ucx_am_context>(client_mem_res, "client");
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
    [&](std::vector<std::pair<std::uint64_t, ucs_status_t>>&&
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
          server_scheduler,
          create_socket_address(port, true),
          sizeof(sockaddr_in)),
        unifex::single(
          unifex::stop_on_request(server_stop_source.get_token()))),
      [&](std::vector<std::pair<std::uint64_t, ucs_status_t>>&&
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
      client_scheduler,
      nullptr,
      create_socket_address(port, false),
      sizeof(sockaddr_in));
    assert(conn_id != 0);

    co_await connection_send(client_scheduler, conn_id, send_data);
    send_success.store(true);
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
```

#### BUILD.bazel
```python
cc_binary(
    name = "readme",
    srcs = ["readme.cpp"],
    copts = [
        "-std=c++17",
        "-fcoroutines",
    ],
    linkstatic = False,  # Important for OpenUCX specific library linking
    deps = [
        "@execution-ucx//ucx_context:ucx_am_context",
        "@unifex",
    ],
)
```

## License

This project is licensed under the [Apache License 2.0](LICENSE) license.

