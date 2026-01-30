[English](README.md)

# execution-ucx

`execution-ucx` 是一个基于 C++26 `P2300` `std::execution` 提案实现的全功能 OpenUCX 运行时上下文。它旨在将所有 UCX 通信任务（包括控制面和数据面）统一在 `ucx_context` 的工作线程内，通过消除锁竞争和减少线程切换，实现极致的性能。

本项目目前使用 Meta 的 [libunifex](https://github.com/facebookexperimental/libunifex)作为 `std::execution` 的实现，并可轻松切换到 NVIDIA 的 [stdexec](https://github.com/NVIDIA/stdexec) 或其他兼容实现。

其设计目标是为现代 C++ 应用程序提供一个高效、灵活且可组合的异步 RDMA 通信后端。

![Architecture](doc/ucx_context.png)

## 核心特性

*   **基于 `std::execution`**：采用最新的 C++ 异步模型，提供富有表现力且可组合的 API。
*   **高性能**：所有操作都在专用的 `ucx_context` 线程中执行，避免了多线程同步开销，最大化 UCX 的性能。
*   **Active Message 支持**：内置高效的 Active Message (`ucx_am_context`) 实现，支持零拷贝和回调函数处理。
*   **连接管理**：自动化的连接建立、缓存和管理 (`ucx_connection_manager`)。
*   **内存管理**：集成了 UCX 内存注册/反注册 (`ucx_memory_resource`)，简化了 RDMA 操作。
*   **CUDA 支持**：无缝支持 CUDA 设备内存，可实现 GPU-Direct RDMA (GDR)。
*   **可扩展性**：模块化设计，可以轻松扩展以支持新的协议或硬件。
*   **通用 RPC 框架**：一个灵活的 RPC 模块 (`rpc`)，用于类型安全的跨进程函数调用，并支持服务发现。

## 核心概念

*   `ucx_context`: 核心组件，封装了 `ucp_worker_h`，并驱动所有异步操作。它拥有一个独立的线程，负责轮询 UCX 事件和执行任务。
*   `ucx_am_context`: Active Message 上下文，提供了发送和接收 Active Message 的接口。
*   `RpcDispatcher`: 用于注册和调用 RPC 函数的核心类。它管理函数签名并处理数据序列化。
*   `ucx_connection`: 封装了 `ucp_ep_h`，代表一个到远端的连接。
*   `ucx_connection_manager`: 负责管理和复用 `ucx_connection`，处理连接建立和关闭。
*   `ucx_memory_resource`: 一个 C++ PMR 风格的内存资源，用于分配可被 UCX 直接用于 RDMA 操作的注册内存。

## 依赖

*   **Bazel**: >= 7.0.0
*   **C++ Compiler**: 支持 C++17 标准
*   **OpenUCX**: v1.18.1 或更高版本
*   **libunifex**: (默认) 作为 `std::execution` 的实现
*   **(可选) liburing**: `libunifex` 的依赖项
*   **Googletest**: 用于单元测试
*   **(可选) CUDA Toolkit**: 用于构建 GPU 支持
*   **(可选) 各种通信介质**: 请修改[OpenUCX BUILD文件](third_party/openucx/BUILD.bazel)自行选择，推荐安装[Nvidia HPC SDK](https://developer.nvidia.com/hpc-sdk)

## 构建与测试

项目使用 Bazel 进行构建。

1.  **构建项目**:
    ```bash
    bazel build //ucx_context:ucx_am_context
    ```

2.  **运行测试 (CPU)**:
    ```bash
    bazel test //ucx_context:ucx_am_context_test
    ```

3.  **运行测试 (CUDA)**:
    需要本地安装 CUDA Toolkit，Bazel 会自动检测并启用 CUDA 支持。
    ```bash
    bazel test //ucx_context:ucx_am_context_test --@rules_cuda//cuda:enable=True
    ```

## 使用示例

以下是一个简化的示例，展示如何使用 `ucx_am_context` 发送和接收一个 Active Message。该示例基于 `TEST_F(UcxAmTest, SmallMessageTransfer)` 测试用例的逻辑。
详细请参考[测试代码](ucx_context/ucx_am_context_test.cpp)

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

// 使用声明以提高代码清晰度
using eux::ucxx::accept_endpoint;
using eux::ucxx::active_message_bundle;
using eux::ucxx::connect_endpoint;
using eux::ucxx::connection_recv;
using eux::ucxx::connection_send;
using eux::ucxx::DefaultUcxMemoryResourceManager;
using eux::ucxx::ucx_am_context;
using eux::ucxx::UcxMemoryResourceManager;
using unifex::task;

// 辅助函数：创建套接字地址
static std::unique_ptr<sockaddr> create_socket_address(
  uint16_t port, bool is_server) {
  sockaddr_in* addr = new sockaddr_in{
    .sin_family = AF_INET,
    .sin_port = htons(port),
    .sin_addr = {.s_addr = htonl(is_server ? INADDR_ANY : INADDR_LOOPBACK)}};
  return std::unique_ptr<sockaddr>(reinterpret_cast<sockaddr*>(addr));
}

int main() {
  // 1. 设置服务器和客户端上下文，并在单独的线程中运行它们
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

  // 允许上下文初始化
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  auto server_scheduler = server_context->get_scheduler();
  auto client_scheduler = client_context->get_scheduler();
  unsigned int seed = static_cast<unsigned int>(time(nullptr));
  uint16_t port =
    static_cast<uint16_t>(1024 + (rand_r(&seed) % (65535 - 1024 + 1)));

  // 2. 准备传输数据
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
    recv_data{};  // 空描述符，将由接收方填充
  std::atomic<bool> message_received = false;
  std::atomic<bool> send_success = false;

  // 3. 将服务器和客户端逻辑定义为unifex任务
  auto server_recv_logic =
    [&](std::vector<std::pair<std::uint64_t, std::error_code>>&&
          conn_id_status_vector) -> task<void> {
    // 此调用将在消息到达时填充'recv_data'结构
    active_message_bundle bundle =
      co_await connection_recv(server_scheduler, recv_data);
    if (bundle.connection().is_established()) {
      message_received.store(true);
      server_stop_source
        .request_stop();  // 接收一条消息后停止服务器
      co_await unifex::stop_if_requested();
    }
  };

  auto server_logic = [&]() -> task<void> {
    // 从传入连接流中获取第一个
    unifex::v2::async_scope scope;
    co_await unifex::for_each(
      unifex::take_until(
        accept_endpoint(
          server_scheduler,
          create_socket_address(port, true),
          sizeof(sockaddr_in)),
        unifex::single(
          unifex::stop_on_request(server_stop_source.get_token()))),
      [&](std::vector<std::pair<std::uint64_t, std::error_code>>&&
            conn_id_status_vector) {
        // 在非协程函数中只能使用spawn_detached
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

  // 4. 并发运行任务并等待它们完成
  unifex::sync_wait(unifex::when_all(server_logic(), client_logic()));

  // 5. 验证结果
  assert(message_received.load());
  assert(send_success.load());
  assert(recv_data.header.size == message_size);
  assert(memcmp(recv_data.header.data, test_data.data(), message_size) == 0);
  assert(recv_data.buffer.size == message_size);
  assert(memcmp(recv_data.buffer.data, test_data.data(), message_size) == 0);
  std::cout << "成功传输 " << message_size << " 字节。" << std::endl;

  // 6. 关闭
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
    linkstatic = False,  # 对于OpenUCX特定库链接很重要！
    deps = [
        "@execution-ucx//ucx_context:ucx_am_context",
        "@unifex",
    ],
)
```

## API 参考

本节详细介绍 `ucx_am_context` 提供的用于网络通信的主要 API，这些 API 基于 `std::execution` 模型。这些函数返回可以在协程或与 `unifex::sync_wait` 一起使用的 sender。

#### 连接管理

*   **`connect_endpoint(scheduler, dst_saddr, addrlen)`**
    *   **描述**: 建立到由套接字地址指定的远程对等点的连接。
    *   **参数**:
        *   `scheduler`: `ucx_am_context` 调度器。
        *   `dst_saddr`: 目标地址的 `std::unique_ptr<sockaddr>`。
        *   `addrlen`: 套接字地址的长度。
        *   可以提供一个可选的 `src_saddr`。
    *   **返回值**: 成功时返回一个生成连接 ID (`std::uint64_t`) 的 sender。

*   **`connect_endpoint(scheduler, address_buffer)`**
    *   **描述**: 使用远程工作进程的 UCX 地址建立连接。
    *   **参数**:
        *   `scheduler`: `ucx_am_context` 调度器。
        *   `address_buffer`: 包含远程 UCX 工作进程地址的 `std::vector<std::byte>`。
    *   **返回值**: 成功时返回一个生成连接 ID (`std::uint64_t`) 的 sender。

*   **`accept_endpoint(scheduler, socket, addrlen)`**
    *   **描述**: 在给定的套接字地址上侦听并接受传入的连接。
    *   **参数**:
        *   `scheduler`: `ucx_am_context` 调度器。
        *   `socket`: 用于侦听地址的 `std::unique_ptr<sockaddr>`。
        *   `addrlen`: 套接字地址的长度。
    *   **返回值**: 成功时返回一个生成 `std::vector<std::pair<std::uint64_t, std::error_code>>` 的 sender，其中每个 pair 代表一个新接受的连接的 ID 和状态。

*   **`handle_error_connection(scheduler, handler)`**
    *   **描述**: 注册一个用于处理连接错误的处理器。该处理器决定是否尝试重新连接。
    *   **参数**:
        *   `scheduler`: `ucx_am_context` 调度器。
        *   `handler`: 一个可调用的 `std::function<bool(std::uint64_t, T)>`，其中 T 可以是 `ucs_status_t` 或 `std::error_code`。它接收连接 ID 和状态。返回 `true` 以重新连接，`false` 以关闭。
    *   **返回值**: 一个不带值完成的 sender。

#### 数据传输

*   **`connection_send(scheduler, conn_id, data)`**
    *   **描述**: 发送一个 Active Message。存在用于不同连接标识符 (`conn_pair_t&`, `std::uint64_t`, `UcxConnection&`) 和数据类型 (`ucx_am_data&`, `UcxAmData&&`, `ucx_am_iovec&`, `UcxAmIovec&&`) 的重载。
    *   **参数**:
        *   `scheduler`: `ucx_am_context` 调度器。
        *   `conn_id`: 连接的标识符。
        *   `data`: 要发送的数据负载。右值版本 (`&&`) 会获得所有权。
    *   **返回值**: 成功时返回一个不带值完成的 sender。

*   **`connection_recv(scheduler, data)`**
    *   **描述**: 接收一个 Active Message。
    *   **参数**:
        *   `scheduler`: `ucx_am_context` 调度器。
        *   `data`: 一个 `ucx_am_data&`，用于填充接收到的消息。一个重载版本接受 `ucx_memory_type` 以在内部自分配缓冲区。
    *   **返回值**: 成功时返回一个生成 `active_message_bundle` 的 sender。

*   **`connection_recv_header(scheduler)`**
    *   **描述**: 只接收传入消息的头部。这对于两阶段接收（Rendezvous协议）很有用，其中数据负载在单独的步骤中接收。
    *   **参数**:
        *   `scheduler`: `ucx_am_context` 调度器。
    *   **返回值**: 返回一个生成 `std::variant<std::pair<size_t, UcxHeader>, active_message_bundle>` 的 sender。对于 Rendezvous 消息，它包含一个键 (`size_t`) 和 `UcxHeader`。对于 Eager 消息，它包含完整的 `active_message_bundle`。

*   **`connection_recv_buffer(scheduler, am_desc_key, buffer)`**
    *   **描述**: 接收由键标识的 Rendezvous 消息的数据负载。
    *   **参数**:
        *   `scheduler`: `ucx_am_context` 调度器。
        *   `am_desc_key`: 从 `connection_recv_header` 获取的键。
        *   `buffer`: 用于接收数据的 `UcxBuffer&&` 或 `UcxBufferVec&&`。一个重载版本接受 `ucx_memory_type` 以在内部自分配缓冲区。
    *   **返回值**: 成功时返回一个生成 `active_message_buffer_bundle`（对于 `UcxBuffer`）或 `active_message_iovec_buffer_bundle`（对于 `UcxBufferVec`）的 sender。

## RPC 框架 (`rpc`)

在底层的 Active Message API 之上，`execution-ucx` 提供了一个高级的、类型安全的 RPC 框架。这使得开发者可以在服务端注册 C++ 函数，并像调用本地函数一样从客户端调用它们，框架会自动处理参数和返回值的序列化。此外，它还包含一个服务发现机制。

### 使用示例

此示例演示了一个端到端的 RPC 工作流程：服务端注册一个函数，客户端发现该函数，然后调用它并验证结果。

#### rpc_example.cpp

```cpp
#include <cassert>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "rpc_core/rpc_dispatcher.hpp"

// 使用声明以提高代码清晰度
using eux::rpc::function_id_t;
using eux::rpc::ParamMeta;
using eux::rpc::ParamType;
using eux::rpc::PrimitiveValue;
using eux::rpc::RpcDispatcher;
using eux::rpc::RpcFunctionSignature;
using eux::rpc::RpcRequestHeader;
using eux::rpc::session_id_t;
namespace data = cista::offset;

// 一个将通过 RPC 暴露的简单函数
int add(int a, int b) { return a + b; }

int main() {
  // 1. 一个用于服务发现的模拟中央注册表。
  // 在实际应用中，这可能是一个键值存储系统，如 etcd 或 Redis。
  std::map<std::string, cista::byte_buf> registry;

  // 2. 设置服务端 B (服务提供者)。
  // 它创建一个分发器，注册一个函数，并发布其签名。
  RpcDispatcher dispatcher_B("server_B_instance");
  dispatcher_B.register_function_raw(function_id_t{100}, &add,
                                 data::string{"add_func"});

  // 将签名发布到中央注册表。
  registry["server_B_instance"] = dispatcher_B.get_all_signatures();
  std::cout << "服务端 B 注册了 'add_func' 并发布了其签名。"
            << std::endl;

  // 3. 客户端 A (调用者) 发现并调用该函数。
  // --- 服务发现阶段 ---
  auto serialized_sigs_it = registry.find("server_B_instance");
  assert(serialized_sigs_it != registry.end());

  auto deserialized_sigs =
      cista::deserialize<data::vector<RpcFunctionSignature>>(
          serialized_sigs_it->second);
  assert(deserialized_sigs != nullptr);

  std::optional<function_id_t> target_function_id;
  for (const auto& sig : *deserialized_sigs) {
    if (sig.function_name == data::string("add_func")) {
      target_function_id = sig.id;
      break;
    }
  }
  assert(target_function_id.has_value());
  std::cout << "客户端 A 发现了 'add_func'，其 ID 为: "
            << target_function_id.value().v_ << std::endl;

  // --- RPC 调用阶段 ---
  RpcRequestHeader request{};
  request.function_id = target_function_id.value();
  request.session_id = session_id_t{2025};

  ParamMeta p1{};
  p1.type = ParamType::PRIMITIVE_INT32;
  p1.value.emplace<PrimitiveValue>(25);
  request.add_param(std::move(p1));

  ParamMeta p2{};
  p2.type = ParamType::PRIMITIVE_INT32;
  p2.value.emplace<PrimitiveValue>(17);
  request.add_param(std::move(p2));

  auto request_buffer = cista::serialize(request);

  // 模拟将请求发送到服务端 B 的分发器并接收响应。
  // 在真实系统中，这个缓冲区将通过 ucx_am_context 在网络上传输。
  auto response_pair = dispatcher_B.dispatch(std::move(request_buffer));
  auto& response_header = response_pair.header;

  // --- 验证阶段 ---
  assert(response_header.session_id.v_ == 2025);
  assert(response_header.results.size() == 1);
  int32_t result = response_header.get_primitive<int32_t>(0);
  assert(result == 42); // 25 + 17
  assert(response_header.status.value == 0);

  std::cout << "客户端 A 调用 'add_func(25, 17)' 并得到结果: " << result
            << std::endl;
  std::cout << "RPC 调用成功!" << std::endl;

  return 0;
}
```

#### BUILD.bazel
```python
cc_binary(
    name = "rpc_example",
    srcs = ["rpc_example.cpp"],
    deps = [
        "//rpc",
    ],
)
```

### API 参考

本节详细介绍 `RpcDispatcher` 提供的主要 API。

*   **`RpcDispatcher(instance_name)`**
    *   **描述**: 构造一个 `RpcDispatcher`。
    *   **参数**:
        *   `instance_name`: 一个 `data::string`，用于标识此分发器实例，该名称将用于函数签名中以进行服务发现。

*   **`register_function_raw(id, func, name)`**
    *   **描述**: 将一个 C++ 可调用对象（函数、lambda 表达式等）注册为 RPC 函数。
    *   **参数**:
        *   `id`: 一个 `function_id_t`（`uint64_t` 的强类型），用于唯一标识该函数。
        *   `func`: 要注册的可调用对象。
        *   `name`: 一个人类可读的 `data::string` 名称，用于服务发现。

*   **`dispatch(request_buffer, [input_data])`**
    *   **描述**: 从缓冲区反序列化一个请求，调用对应的函数，并返回结果。此重载适用于不接受上下文或通过引用（`&`）接受上下文的函数。
    *   **参数**:
        *   `request_buffer`: 一个包含序列化的 `RpcRequestHeader` 的 `cista::byte_buf&&`。
        *   `input_data`: （可选）一个要传递给函数的上下文对象的左值引用。
    *   **返回值**: 一个 `RpcInvokeResult`，包含响应头和函数返回的任何上下文对象。

*   **`dispatch_move(request_buffer, input_data)`**
    *   **描述**: 功能与 `dispatch` 类似，但会将上下文对象移动（move）到被调用的函数中。这适用于管理资源且所有权应被转移的上下文类型，例如 `UcxBufferVec`。
    *   **参数**:
        *   `request_buffer`: 包含序列化请求的 `cista::byte_buf&&`。
        *   `input_data`: 上下文对象的右值引用（`&&`）。
    *   **返回值**: 一个 `RpcInvokeResult`，包含响应头和函数返回的任何上下文。

*   **`get_all_signatures()`**
    *   **描述**: 检索所有已注册函数的签名并将其序列化。这是服务发现机制的核心。
    *   **返回值**: 一个 `cista::byte_buf`，其中包含一个序列化的 `data::vector<RpcFunctionSignature>`。客户端可以反序列化此缓冲区以了解服务端上可用的函数。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。