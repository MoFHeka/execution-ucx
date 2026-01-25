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

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rpc_core/utils/tensor_meta.hpp"
#include "ucx_context/ucx_device_context.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

#define LOGGING_ENABLED

#ifdef LOGGING_ENABLED
#define LOG(S)             \
  do {                     \
    ::std::puts(S);        \
    ::std::fflush(stdout); \
  } while (false)
#define LOGX(...)               \
  do {                          \
    ::std::printf(__VA_ARGS__); \
    ::std::fflush(stdout);      \
  } while (false)
#else
#define LOG(S) \
  do {         \
  } while (false)
#define LOGX(...) \
  do {            \
  } while (false)
#endif

namespace eux::axon {

using rpc::ParamMeta;
using rpc::ParamType;
using rpc::utils::TensorMeta;

class AxonWorkerIntegrationTest : public ::testing::Test {
 protected:
  std::unique_ptr<ucxx::DefaultUcxMemoryResourceManager> mr_;
};

TEST_F(AxonWorkerIntegrationTest, ClientServerInteraction) {
  int pipe_fd[2];
  ASSERT_EQ(pipe(pipe_fd), 0);

  int control_pipe[2];
  ASSERT_EQ(pipe(control_pipe), 0);

  LOGX("[Parent] Forking server process\n");
  pid_t server_pid = fork();
  ASSERT_GE(server_pid, 0);

  if (server_pid == 0) {
    // Child 1 - Server
    mr_ = std::make_unique<ucxx::DefaultUcxMemoryResourceManager>();
    close(pipe_fd[0]);       // Close read end
    close(control_pipe[1]);  // Close write end

    {
      auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
      AxonWorker worker(
        *mr_, "server_worker", 2, std::chrono::milliseconds(10),
        std::move(device_ctx));

      // Register a simple function
      worker.RegisterFunction<ucxx::UcxBuffer>(
        rpc::function_id_t{1001},
        [](const rpc::RpcRequestHeader& header, ucxx::UcxBuffer&& payload) {
          LOGX("[Server] Function 1001 called\n");
          // Echo payload
          return unifex::just(std::make_pair(
            rpc::RpcResponseHeader{
              header.session_id,
              header.request_id,
              rpc::utils::HybridLogicalClock{},  // hlc
              rpc::utils::workflow_id_t{0},      // workflow_id
              rpc::RpcStatus(std::make_error_code(rpc::RpcErrc::OK)),
              {}  // results
            },
            rpc::ReturnedPayload(
              std::in_place_type<ucxx::UcxBuffer>, std::move(payload))));
        },
        "echo");

      if (!worker.StartServer().has_value()) {
        LOGX("[Server] Failed to start server\n");
        std::exit(1);
      }

      // Send address to parent (who will ignore it) / client (who will read it)
      // Note: We are piping to the other child via the parent's pipe mechanism,
      // but actually the other child has the read end.
      std::vector<std::byte> addr = worker.GetLocalAddress();
      size_t addr_size = addr.size();
      if (write(pipe_fd[1], &addr_size, sizeof(size_t)) != sizeof(size_t)) {
        LOGX("[Server] Failed to write address size\n");
        std::exit(1);
      }
      if (
        write(pipe_fd[1], addr.data(), addr_size)
        != static_cast<ssize_t>(addr_size)) {
        LOGX("[Server] Failed to write address data\n");
        std::exit(1);
      }
      close(pipe_fd[1]);

      // Keep running until signaled
      char buf;
      if (read(control_pipe[0], &buf, 1) < 0) {
        // Ignore read errors, continue to stop server
      }
      worker.Stop();
    }
    std::exit(0);
  }

  LOGX("[Parent] Forking client process\n");
  pid_t client_pid = fork();
  ASSERT_GE(client_pid, 0);

  if (client_pid == 0) {
    // Child 2 - Client
    mr_ = std::make_unique<ucxx::DefaultUcxMemoryResourceManager>();
    close(pipe_fd[1]);  // Close write end

    {
      size_t addr_size;
      if (read(pipe_fd[0], &addr_size, sizeof(size_t)) != sizeof(size_t)) {
        LOGX("[Client] Failed to read address size\n");
        std::exit(1);
      }
      std::vector<std::byte> addr(addr_size);
      if (
        read(pipe_fd[0], addr.data(), addr_size)
        != static_cast<ssize_t>(addr_size)) {
        LOGX("[Client] Failed to read address data\n");
        std::exit(1);
      }
      close(pipe_fd[0]);

      auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
      AxonWorker worker(
        *mr_, "client_worker", 2, std::chrono::milliseconds(10),
        std::move(device_ctx));
      if (!worker.StartClient().has_value()) {
        LOGX("[Client] Failed to start client\n");
        std::exit(1);
      }

      auto conn_result = worker.ConnectEndpoint(addr, "server_worker");
      if (!conn_result.has_value()) {
        LOGX("[Client] Failed to connect to server\n");
        std::exit(1);
      }

      // Prepare payload
      std::string message = "Hello Axon";
      ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, message.size());
      std::memcpy(payload.data(), message.data(), message.size());

      // Invoke RPC
      auto rpc_result =
        unifex::sync_wait(worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
          "server_worker", rpc::session_id_t{1}, rpc::function_id_t{1001},
          rpc::utils::workflow_id_t{0}, std::move(payload)));

      if (!rpc_result.has_value()) {
        LOGX("[Client] RPC failed\n");
        std::exit(1);
      }
      auto& [header, response_payload] = rpc_result.value();
      if (header->status != std::make_error_code(rpc::RpcErrc::OK)) {
        LOGX("[Client] RPC returned error status\n");
        std::exit(1);
      }

      std::string response_str(
        static_cast<char*>(response_payload.get()->data),
        response_payload.size());
      if (response_str != message) {
        LOGX("[Client] Response mismatch\n");
        std::exit(1);
      }

      worker.Stop();
    }
    std::exit(0);
  }

  // Parent process
  close(pipe_fd[0]);
  close(pipe_fd[1]);
  close(control_pipe[0]);

  int status;
  // Wait for client to finish
  waitpid(client_pid, &status, 0);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);

  // Cleanup server
  close(control_pipe[1]);
  waitpid(server_pid, nullptr, 0);
}

TEST_F(AxonWorkerIntegrationTest, DynamicApiAndErrorHandling) {
  int pipe_fd[2];
  ASSERT_EQ(pipe(pipe_fd), 0);

  int control_pipe[2];
  ASSERT_EQ(pipe(control_pipe), 0);

  LOGX("[Parent] Forking server process (dynamic test)\n");
  pid_t server_pid = fork();
  ASSERT_GE(server_pid, 0);

  if (server_pid == 0) {
    // Child 1 - Server
    mr_ = std::make_unique<ucxx::DefaultUcxMemoryResourceManager>();
    close(pipe_fd[0]);       // Close read end
    close(control_pipe[1]);  // Close write end

    {
      auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
      AxonWorker worker(
        *mr_, "server_worker_dyn", 2, std::chrono::milliseconds(10),
        std::move(device_ctx));

      using data_string = cista::offset::string;
      using data_vector_param = cista::offset::vector<rpc::ParamType>;

      // Dynamic echo function: use DynamicAsyncRpcFunction + dynamic InvokeRpc.
      // Use a function object with overloaded operator() for each payload type
      // to match DynamicAsyncRpcFunctionFacade's signature requirements
      struct DynEchoImpl {
        std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr_;

        // Overload for no payload
        unifex::any_sender_of<std::pair<
          cista::offset::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
        operator()(
          const cista::offset::vector<rpc::ParamMeta>& /*params*/) const {
          return unifex::just_error(
            rpc::MakeRpcExceptionPtr(rpc::RpcErrc::INVALID_ARGUMENT));
        }

        // Overload for monostate
        unifex::any_sender_of<std::pair<
          cista::offset::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
        operator()(
          const cista::offset::vector<rpc::ParamMeta>& /*params*/,
          const std::monostate& /*input*/) const {
          return unifex::just_error(
            rpc::MakeRpcExceptionPtr(rpc::RpcErrc::INVALID_ARGUMENT));
        }

        // Overload for UcxBuffer
        unifex::any_sender_of<std::pair<
          cista::offset::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
        operator()(
          const cista::offset::vector<rpc::ParamMeta>& /*params*/,
          const ucxx::UcxBuffer& input) const {
          // Allocate output buffer and copy input payload.
          ucxx::UcxBuffer output(
            mr_.get(), ucx_memory_type::HOST, input.size());
          std::memcpy(output.data(), input.data(), input.size());

          cista::offset::vector<rpc::ParamMeta> results;
          return unifex::just(std::make_pair(
            std::move(results),
            rpc::ReturnedPayload(
              std::in_place_type<ucxx::UcxBuffer>, std::move(output))));
        }

        // Overload for UcxBufferVec
        unifex::any_sender_of<std::pair<
          cista::offset::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
        operator()(
          const cista::offset::vector<rpc::ParamMeta>& /*params*/,
          const ucxx::UcxBufferVec& /*input*/) const {
          return unifex::just_error(
            rpc::MakeRpcExceptionPtr(rpc::RpcErrc::INVALID_ARGUMENT));
        }
      };

      DynEchoImpl dyn_echo_impl{*mr_};

      rpc::DynamicAsyncRpcFunction dyn_echo =
        pro::make_proxy<rpc::DynamicAsyncRpcFunctionFacade>(
          std::move(dyn_echo_impl));

      worker.RegisterFunction<ucxx::UcxBuffer>(
        rpc::function_id_t{2001}, data_string{"dyn_echo"},
        data_vector_param{},           // no params
        data_vector_param{},           // no returns
        rpc::PayloadType::UCX_BUFFER,  // input payload
        rpc::PayloadType::UCX_BUFFER,  // return payload
        std::move(dyn_echo));

      // Failure function: always returns INTERNAL error.
      worker.RegisterFunction<ucxx::UcxBuffer>(
        rpc::function_id_t{2002},
        [](
          const rpc::RpcRequestHeader& /*header*/,
          ucxx::UcxBuffer&& /*payload*/) {
          LOGX("[Server-Dyn] Failure function invoked\n");
          return unifex::just_error(
            rpc::MakeRpcExceptionPtr(rpc::RpcErrc::INTERNAL));
        },
        "always_fail");

      if (!worker.StartServer().has_value()) {
        LOGX("[Server-Dyn] Failed to start server\n");
        std::exit(1);
      }

      // Send address to client
      std::vector<std::byte> addr = worker.GetLocalAddress();
      size_t addr_size = addr.size();
      if (write(pipe_fd[1], &addr_size, sizeof(size_t)) != sizeof(size_t)) {
        LOGX("[Server-Dyn] Failed to write address size\n");
        std::exit(1);
      }
      if (
        write(pipe_fd[1], addr.data(), addr_size)
        != static_cast<ssize_t>(addr_size)) {
        LOGX("[Server-Dyn] Failed to write address data\n");
        std::exit(1);
      }
      close(pipe_fd[1]);

      // Keep running until signaled by parent.
      char buf;
      if (read(control_pipe[0], &buf, 1) < 0) {
        // Expected EOF or error
      }
      worker.Stop();
    }
    std::exit(0);
  }

  LOGX("[Parent] Forking client process (dynamic test)\n");
  pid_t client_pid = fork();
  ASSERT_GE(client_pid, 0);

  if (client_pid == 0) {
    // Child 2 - Client
    mr_ = std::make_unique<ucxx::DefaultUcxMemoryResourceManager>();
    close(pipe_fd[1]);  // Close write end

    {
      size_t addr_size;
      if (read(pipe_fd[0], &addr_size, sizeof(size_t)) != sizeof(size_t)) {
        LOGX("[Client-Dyn] Failed to read address size\n");
        std::exit(1);
      }
      std::vector<std::byte> addr(addr_size);
      if (
        read(pipe_fd[0], addr.data(), addr_size)
        != static_cast<ssize_t>(addr_size)) {
        LOGX("[Client-Dyn] Failed to read address data\n");
        std::exit(1);
      }
      close(pipe_fd[0]);

      auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
      AxonWorker worker(
        *mr_, "client_worker_dyn", 2, std::chrono::milliseconds(10),
        std::move(device_ctx));

      if (!worker.StartClient().has_value()) {
        LOGX("[Client-Dyn] Failed to start client\n");
        std::exit(1);
      }

      auto conn_result = worker.ConnectEndpoint(addr, "server_worker_dyn");
      if (!conn_result.has_value()) {
        LOGX("[Client-Dyn] Failed to connect to server\n");
        std::exit(1);
      }

      // ---------- 1. Dynamic API: RegisterFunction + InvokeRpc ----------
      {
        std::string message = "Hello Axon Dynamic";
        ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, message.size());
        std::memcpy(payload.data(), message.data(), message.size());

        rpc::RpcRequestHeader header;
        header.session_id = rpc::session_id_t{1};
        header.request_id = rpc::request_id_t{0};
        header.function_id = rpc::function_id_t{2001};
        header.workflow_id = rpc::utils::workflow_id_t{0};

        auto dyn_result =
          unifex::sync_wait(worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
            "server_worker_dyn", std::move(header),
            std::optional<ucxx::UcxBuffer>(std::move(payload))));

        ASSERT_TRUE(dyn_result.has_value()) << "Dynamic RPC failed";
        auto& [resp_header, resp_payload_variant] = dyn_result.value();
        ASSERT_EQ(resp_header->status, std::make_error_code(rpc::RpcErrc::OK))
          << "Dynamic RPC returned error status";

        std::string resp_str(
          static_cast<char*>(resp_payload_variant.get()->data),
          resp_payload_variant.size());
        ASSERT_EQ(resp_str, message) << "Dynamic response mismatch";
      }

      // ---------- 2. Server failure handling ----------
      {
        std::string message = "Trigger Server Failure";
        ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, message.size());
        std::memcpy(payload.data(), message.data(), message.size());

        auto sender =
          worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
            "server_worker_dyn", rpc::session_id_t{1}, rpc::function_id_t{2002},
            rpc::utils::workflow_id_t{0}, std::move(payload))
          | unifex::then([](auto&&...) { return false; })
          | unifex::let_error([](auto&& e) {
              if constexpr (std::is_same_v<
                              std::decay_t<decltype(e)>,
                              errors::AxonErrorContext>) {
                return unifex::just(
                  std::error_code(e.status)
                  == std::make_error_code(rpc::RpcErrc::INTERNAL));
              } else {
                return unifex::just(false);
              }
            });

        auto result = unifex::sync_wait(std::move(sender));
        ASSERT_TRUE(result.has_value());
        EXPECT_TRUE(result.value())
          << "RPC should have failed with INTERNAL error";
      }

      // ---------- 3. Client-side failure (invalid worker name) ----------
      {
        std::string message = "Bad Worker";
        ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, message.size());
        std::memcpy(payload.data(), message.data(), message.size());

        rpc::RpcRequestHeader header;
        header.session_id = rpc::session_id_t{1};
        header.request_id = rpc::request_id_t{0};
        header.function_id = rpc::function_id_t{2001};
        header.workflow_id = rpc::utils::workflow_id_t{0};

        auto sender =
          worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
            "nonexistent_worker", std::move(header),
            std::optional<ucxx::UcxBuffer>(std::move(payload)))
          | unifex::then([](auto&&...) { return false; })
          | unifex::let_error([](auto&& e) {
              if constexpr (std::is_same_v<
                              std::decay_t<decltype(e)>,
                              errors::AxonErrorContext>) {
                bool match =
                  std::error_code(e.status)
                  == std::make_error_code(errors::AxonErrc::WorkerNotFound);
                if (!match) {
                  LOGX(
                    "Error status mismatch: expected WorkerNotFound (%d), got "
                    "%d (category: %s, message: %s)\n",
                    static_cast<int>(errors::AxonErrc::WorkerNotFound),
                    std::error_code(e.status).value(),
                    std::error_code(e.status).category().name(),
                    std::error_code(e.status).message().c_str());
                }
                return unifex::just(match);
              } else {
                LOGX("Error type mismatch: expected AxonErrorContext\n");
                return unifex::just(false);
              }
            });

        auto result = unifex::sync_wait(std::move(sender));
        ASSERT_TRUE(result.has_value());
        EXPECT_TRUE(result.value())
          << "RPC should have failed with WorkerNotFound error";
      }

      worker.Stop();
    }

    if (::testing::Test::HasFailure()) {
      std::exit(1);
    }
    std::exit(0);
  }

  // Parent process
  close(pipe_fd[0]);
  close(pipe_fd[1]);
  close(control_pipe[0]);

  int status;
  // Wait for client to finish
  waitpid(client_pid, &status, 0);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);

  // Cleanup server
  close(control_pipe[1]);
  waitpid(server_pid, nullptr, 0);
}

TEST_F(AxonWorkerIntegrationTest, RobustnessAndConcurrency) {
  int pipe_fd[2];
  ASSERT_EQ(pipe(pipe_fd), 0);

  int control_pipe[2];
  ASSERT_EQ(pipe(control_pipe), 0);

  LOGX("[Parent] Forking server process (robustness test)\n");
  pid_t server_pid = fork();
  ASSERT_GE(server_pid, 0);

  if (server_pid == 0) {
    // Child 1 - Server
    mr_ = std::make_unique<ucxx::DefaultUcxMemoryResourceManager>();
    close(pipe_fd[0]);
    close(control_pipe[1]);

    {
      auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
      // Timeout set to 2 seconds for timeout test
      AxonWorker worker(
        *mr_, "server_worker_rob", 4, std::chrono::milliseconds(100),
        std::move(device_ctx));

      // Register Sleep Function
      worker.RegisterFunction<ucxx::UcxBuffer>(
        rpc::function_id_t{3001},
        [](const rpc::RpcRequestHeader& header, ucxx::UcxBuffer&& payload) {
          std::string msg(static_cast<char*>(payload.data()), payload.size());
          int sleep_ms = std::stoi(msg);
          std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
          return unifex::just(std::make_pair(
            rpc::RpcResponseHeader{
              header.session_id,
              header.request_id,
              rpc::utils::HybridLogicalClock{},
              rpc::utils::workflow_id_t{0},
              rpc::RpcStatus(std::error_code{}),
              {}},
            rpc::ReturnedPayload(
              std::in_place_type<ucxx::UcxBuffer>, std::move(payload))));
        },
        "sleep");

      // Register Control Function (to enable rejection)
      worker.RegisterFunction<ucxx::UcxBuffer>(
        rpc::function_id_t{3002},
        [&worker](
          const rpc::RpcRequestHeader& header, ucxx::UcxBuffer&& payload) {
          std::string msg(static_cast<char*>(payload.data()), payload.size());
          if (msg == "reject_on") {
            worker.SetRejectMessages(true);
          }
          return unifex::just(std::make_pair(
            rpc::RpcResponseHeader{
              header.session_id,
              header.request_id,
              rpc::utils::HybridLogicalClock{},
              rpc::utils::workflow_id_t{0},
              rpc::RpcStatus(std::error_code{}),
              {}},
            rpc::ReturnedPayload(
              std::in_place_type<ucxx::UcxBuffer>, std::move(payload))));
        },
        "control");

      // Register Echo for Concurrency/Large payload
      worker.RegisterFunction<ucxx::UcxBuffer>(
        rpc::function_id_t{3003},
        [](const rpc::RpcRequestHeader& header, ucxx::UcxBuffer&& payload) {
          return unifex::just(std::make_pair(
            rpc::RpcResponseHeader{
              header.session_id,
              header.request_id,
              rpc::utils::HybridLogicalClock{},
              rpc::utils::workflow_id_t{0},
              rpc::RpcStatus(std::error_code{}),
              {}},
            rpc::ReturnedPayload(
              std::in_place_type<ucxx::UcxBuffer>, std::move(payload))));
        },
        "echo");

      ASSERT_TRUE(worker.StartServer().has_value())
        << "[Server-Rob] Failed to start server";

      std::vector<std::byte> addr = worker.GetLocalAddress();
      size_t addr_size = addr.size();
      write(pipe_fd[1], &addr_size, sizeof(size_t));
      write(pipe_fd[1], addr.data(), addr_size);
      close(pipe_fd[1]);

      // Keep running until signaled
      char buf;
      if (read(control_pipe[0], &buf, 1) < 0) {
        // Ignore read errors, continue to stop server
      }
      worker.Stop();
    }
    std::exit(0);
  }

  // Parent process
  LOGX("[Parent] Forking client process (robustness test)\n");
  pid_t client_pid = fork();
  ASSERT_GE(client_pid, 0);

  if (client_pid == 0) {
    // Child 2 - Client
    mr_ = std::make_unique<ucxx::DefaultUcxMemoryResourceManager>();
    close(pipe_fd[1]);
    size_t addr_size;
    read(pipe_fd[0], &addr_size, sizeof(size_t));
    std::vector<std::byte> addr(addr_size);
    read(pipe_fd[0], addr.data(), addr_size);
    close(pipe_fd[0]);

    {
      auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
      AxonWorker worker(
        *mr_, "client_worker_rob", 4, std::chrono::milliseconds(10),
        std::move(device_ctx));
      ASSERT_TRUE(worker.StartClient().has_value())
        << "[Client-Rob] Failed to start client";
      auto conn_result = worker.ConnectEndpoint(addr, "server_worker_rob");
      ASSERT_TRUE(conn_result.has_value())
        << "[Client-Rob] Failed to connect to server";

      // 1. Concurrency Test
      {
        LOGX("[Client-Rob] Starting Concurrency Test\n");
        auto mk_req = [&](int i) {
          std::string msg = "req_" + std::to_string(i);
          ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, msg.size());
          std::memcpy(payload.data(), msg.data(), msg.size());
          return worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
            "server_worker_rob", rpc::session_id_t{1}, rpc::function_id_t{3003},
            rpc::utils::workflow_id_t{0}, std::move(payload));
        };

        auto result = unifex::sync_wait(unifex::when_all(
          mk_req(1), mk_req(2), mk_req(3), mk_req(4), mk_req(5)));

        ASSERT_TRUE(result.has_value());
        auto& t = result.value();
        // when_all returns tuple of variants, each variant contains a tuple
        // with a pair Access pattern: tuple -> variant -> tuple -> pair ->
        // first
        auto get_header = [](auto& variant) {
          return std::move(std::get<0>(std::get<0>(variant)).first);
        };
        EXPECT_EQ(get_header(std::get<0>(t))->status, std::error_code{});
        EXPECT_EQ(get_header(std::get<1>(t))->status, std::error_code{});
        EXPECT_EQ(get_header(std::get<2>(t))->status, std::error_code{});
        EXPECT_EQ(get_header(std::get<3>(t))->status, std::error_code{});
        EXPECT_EQ(get_header(std::get<4>(t))->status, std::error_code{});
        LOGX("[Client-Rob] Concurrency Test Passed\n");
      }

      // 2. Large Payload Test (4MB)
      {
        LOGX("[Client-Rob] Starting Large Payload Test\n");
        size_t size = 4 * 1024 * 1024;
        ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, size);
        std::memset(payload.data(), 'A', size);

        auto res =
          unifex::sync_wait(worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
            "server_worker_rob", rpc::session_id_t{1}, rpc::function_id_t{3003},
            rpc::utils::workflow_id_t{0}, std::move(payload)));
        ASSERT_TRUE(res.has_value());
        auto& [header, ret_payload] = res.value();
        LOGX(
          "[Client-Rob] Large Payload Test: status=%d, error=%s\n",
          std::error_code(header->status).value(),
          std::error_code(header->status).message().c_str());
        ASSERT_EQ(ret_payload.size(), size);
        char* data = static_cast<char*>(ret_payload.data());
        ASSERT_EQ(data[0], 'A');
        ASSERT_EQ(data[size - 1], 'A');
        LOGX("[Client-Rob] Large Payload Test Passed\n");
      }

      // 3. Server Timeout Test
      {
        LOGX("[Client-Rob] Starting Server Timeout Test\n");
        // Server timeout is 2s. We sleep 3000ms.
        std::string msg = "3000";
        ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, msg.size());
        std::memcpy(payload.data(), msg.data(), msg.size());

        try {
          auto res = unifex::sync_wait(
            worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
              "server_worker_rob", rpc::session_id_t{1},
              rpc::function_id_t{3001}, rpc::utils::workflow_id_t{0},
              std::move(payload)));
        } catch (const errors::AxonErrorContext& ctx) {
          // If everything is correct, all error type will be transformed to
          // AxonErrorContext
          LOGX(
            "[Client-Rob] Server Timeout Test: AxonErrorContext: status=%d, "
            "msg=%s\n",
            std::error_code(ctx.status).value(), ctx.what.c_str());
          ASSERT_EQ(
            std::error_code(ctx.status),
            std::make_error_code(rpc::RpcErrc::DEADLINE_EXCEEDED));
        } catch (...) {
          LOGX("[Client-Rob] Server Timeout Test: unknown error\n");
          ASSERT_TRUE(false);
        }

        LOGX("[Client-Rob] Server Timeout Test Passed\n");
      }

      // 4. Backpressure Test
      {
        LOGX("[Client-Rob] Starting Backpressure Test\n");
        // Enable reject
        {
          std::string msg = "reject_on";
          ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, msg.size());
          std::memcpy(payload.data(), msg.data(), msg.size());
          unifex::sync_wait(worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
            "server_worker_rob", rpc::session_id_t{1}, rpc::function_id_t{3002},
            rpc::utils::workflow_id_t{0}, std::move(payload)));
        }

        // Try echo, expect StorageBackpressure
        {
          std::string msg = "ping";
          ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, msg.size());
          std::memcpy(payload.data(), msg.data(), msg.size());
          auto sender = worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
                          "server_worker_rob", rpc::session_id_t{1},
                          rpc::function_id_t{3003},
                          rpc::utils::workflow_id_t{0}, std::move(payload))
                        | unifex::then([](auto&&...) { return false; })
                        | unifex::let_error([](auto&& e) {
                            if constexpr (std::is_same_v<
                                            std::decay_t<decltype(e)>,
                                            errors::AxonErrorContext>) {
                              return unifex::just(
                                std::error_code(e.status)
                                == std::make_error_code(
                                  errors::AxonErrc::StorageBackpressure));
                            }
                            return unifex::just(false);
                          });

          auto res = unifex::sync_wait(std::move(sender));
          ASSERT_TRUE(res.has_value());
          ASSERT_TRUE(res.value());
        }
        LOGX("[Client-Rob] Backpressure Test Passed\n");
      }

      worker.Stop();
    }

    // Check for test failures and exit with appropriate status
    if (::testing::Test::HasFailure()) {
      std::exit(1);
    }
    std::exit(0);
  }

  // Parent process
  close(pipe_fd[0]);
  close(pipe_fd[1]);
  close(control_pipe[0]);

  int status;
  // Wait for client to finish
  waitpid(client_pid, &status, 0);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);

  // Cleanup server
  close(control_pipe[1]);
  waitpid(server_pid, nullptr, 0);
}

TEST_F(AxonWorkerIntegrationTest, BackpressureLargeMessage) {
  int pipe_fd[2];
  ASSERT_EQ(pipe(pipe_fd), 0);

  int control_pipe[2];
  ASSERT_EQ(pipe(control_pipe), 0);

  LOGX("[Parent] Forking server process (backpressure large message test)\n");
  pid_t server_pid = fork();
  ASSERT_GE(server_pid, 0);

  if (server_pid == 0) {
    // Child 1 - Server
    mr_ = std::make_unique<ucxx::DefaultUcxMemoryResourceManager>();
    close(pipe_fd[0]);
    close(control_pipe[1]);

    {
      auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
      AxonWorker worker(
        *mr_, "server_worker_bpl", 4, std::chrono::milliseconds(100),
        std::move(device_ctx));

      // Register Control Function (to enable rejection)
      constexpr rpc::function_id_t CONTROL_FUNCTION_ID{4001};
      worker.RegisterFunction<ucxx::UcxBuffer>(
        CONTROL_FUNCTION_ID,
        [&worker](
          const rpc::RpcRequestHeader& header, ucxx::UcxBuffer&& payload) {
          std::string msg(static_cast<char*>(payload.data()), payload.size());
          if (msg == "reject_on") {
            worker.SetRejectMessages(true);
          } else if (msg == "reject_off") {
            worker.SetRejectMessages(false);
          }
          return unifex::just(std::make_pair(
            rpc::RpcResponseHeader{
              header.session_id,
              header.request_id,
              rpc::utils::HybridLogicalClock{},
              rpc::utils::workflow_id_t{0},
              rpc::RpcStatus(std::error_code{}),
              {}},
            rpc::ReturnedPayload(
              std::in_place_type<ucxx::UcxBuffer>, std::move(payload))));
        },
        "control");

      // Allow control function to bypass backpressure
      auto bypass_checker =
        [CONTROL_FUNCTION_ID](rpc::function_id_t function_id) -> bool {
        return function_id == CONTROL_FUNCTION_ID;
      };
      worker.SetBypassRejectionFunction(bypass_checker);

      // Register Echo for Large payload
      worker.RegisterFunction<ucxx::UcxBuffer>(
        rpc::function_id_t{4002},
        [](const rpc::RpcRequestHeader& header, ucxx::UcxBuffer&& payload) {
          return unifex::just(std::make_pair(
            rpc::RpcResponseHeader{
              header.session_id,
              header.request_id,
              rpc::utils::HybridLogicalClock{},
              rpc::utils::workflow_id_t{0},
              rpc::RpcStatus(std::error_code{}),
              {}},
            rpc::ReturnedPayload(
              std::in_place_type<ucxx::UcxBuffer>, std::move(payload))));
        },
        "echo");

      ASSERT_TRUE(worker.StartServer().has_value())
        << "[Server-BPL] Failed to start server";

      std::vector<std::byte> addr = worker.GetLocalAddress();
      size_t addr_size = addr.size();
      write(pipe_fd[1], &addr_size, sizeof(size_t));
      write(pipe_fd[1], addr.data(), addr_size);
      close(pipe_fd[1]);

      // Keep running until signaled
      char buf;
      if (read(control_pipe[0], &buf, 1) < 0) {
        // Ignore read errors, continue to stop server
      }
      worker.Stop();
    }
    std::exit(0);
  }

  // Parent process
  LOGX("[Parent] Forking client process (backpressure large message test)\n");
  pid_t client_pid = fork();
  ASSERT_GE(client_pid, 0);

  if (client_pid == 0) {
    // Child 2 - Client
    mr_ = std::make_unique<ucxx::DefaultUcxMemoryResourceManager>();
    close(pipe_fd[1]);
    size_t addr_size;
    read(pipe_fd[0], &addr_size, sizeof(size_t));
    std::vector<std::byte> addr(addr_size);
    read(pipe_fd[0], addr.data(), addr_size);
    close(pipe_fd[0]);

    {
      auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
      AxonWorker worker(
        *mr_, "client_worker_bpl", 4, std::chrono::milliseconds(10),
        std::move(device_ctx));
      ASSERT_TRUE(worker.StartClient().has_value())
        << "[Client-BPL] Failed to start client";
      auto conn_result = worker.ConnectEndpoint(addr, "server_worker_bpl");
      ASSERT_TRUE(conn_result.has_value())
        << "[Client-BPL] Failed to connect to server";

      // Backpressure Test with Large Message (10MB)
      {
        LOGX("[Client-BPL] Starting Backpressure Large Message Test\n");
        // Enable reject
        {
          std::string msg = "reject_on";
          ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, msg.size());
          std::memcpy(payload.data(), msg.data(), msg.size());
          unifex::sync_wait(worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
            "server_worker_bpl", rpc::session_id_t{1}, rpc::function_id_t{4001},
            rpc::utils::workflow_id_t{0}, std::move(payload)));
        }

        // Try large echo (10MB), expect StorageBackpressure
        {
          size_t large_size = 10 * 1024 * 1024;  // 10MB
          ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, large_size);
          std::memset(payload.data(), 'B', large_size);

          auto sender = worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
                          "server_worker_bpl", rpc::session_id_t{1},
                          rpc::function_id_t{4002},
                          rpc::utils::workflow_id_t{0}, std::move(payload))
                        | unifex::then([](auto&&...) { return false; })
                        | unifex::let_error([](auto&& e) {
                            if constexpr (std::is_same_v<
                                            std::decay_t<decltype(e)>,
                                            errors::AxonErrorContext>) {
                              return unifex::just(
                                std::error_code(e.status)
                                == std::make_error_code(
                                  errors::AxonErrc::StorageBackpressure));
                            }
                            return unifex::just(false);
                          });

          auto res = unifex::sync_wait(std::move(sender));
          ASSERT_TRUE(res.has_value());
          ASSERT_TRUE(res.value())
            << "Large message RPC should have failed with StorageBackpressure "
               "error";
        }

        // Disable reject
        {
          std::string msg = "reject_off";
          ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, msg.size());
          std::memcpy(payload.data(), msg.data(), msg.size());
          unifex::sync_wait(worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
            "server_worker_bpl", rpc::session_id_t{1}, rpc::function_id_t{4001},
            rpc::utils::workflow_id_t{0}, std::move(payload)));
        }

        // Try large echo again, should succeed now
        {
          size_t large_size = 10 * 1024 * 1024;  // 10MB
          ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, large_size);
          std::memset(payload.data(), 'C', large_size);

          auto res = unifex::sync_wait(
            worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
              "server_worker_bpl", rpc::session_id_t{1},
              rpc::function_id_t{4002}, rpc::utils::workflow_id_t{0},
              std::move(payload)));
          ASSERT_TRUE(res.has_value());
          auto& [header, ret_payload] = res.value();
          ASSERT_EQ(header->status, std::error_code{});
          ASSERT_EQ(ret_payload.size(), large_size);
          char* data = static_cast<char*>(ret_payload.data());
          ASSERT_EQ(data[0], 'C');
          ASSERT_EQ(data[large_size - 1], 'C');
        }

        LOGX("[Client-BPL] Backpressure Large Message Test Passed\n");
      }

      worker.Stop();
    }

    if (::testing::Test::HasFailure()) {
      std::exit(1);
    }
    std::exit(0);
  }

  // Parent process
  close(pipe_fd[0]);
  close(pipe_fd[1]);
  close(control_pipe[0]);

  int status;
  // Wait for client to finish
  waitpid(client_pid, &status, 0);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);

  // Cleanup server
  close(control_pipe[1]);
  waitpid(server_pid, nullptr, 0);
}

TEST_F(AxonWorkerIntegrationTest, TensorMetaBufferTransfer) {
  int pipe_fd[2];
  ASSERT_EQ(pipe(pipe_fd), 0);

  int control_pipe[2];
  ASSERT_EQ(pipe(control_pipe), 0);

  LOGX("[Parent] Forking server process (tensor meta test)\n");
  pid_t server_pid = fork();
  ASSERT_GE(server_pid, 0);

  if (server_pid == 0) {
    // Child 1 - Server
    mr_ = std::make_unique<ucxx::DefaultUcxMemoryResourceManager>();
    close(pipe_fd[0]);
    close(control_pipe[1]);

    {
      auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
      AxonWorker worker(
        *mr_, "server_worker_tm", 4, std::chrono::milliseconds(100),
        std::move(device_ctx));

      // Register function that handles UcxBuffer with TensorMeta
      worker.RegisterFunction<ucxx::UcxBuffer>(
        rpc::function_id_t{5001},
        [](
          const TensorMeta& tm, int multiplier, float scale,
          ucxx::UcxBuffer&& payload) {
          // Verify TensorMeta and primitive parameters
          LOGX(
            "[Server-TM] Received TensorMeta: ndim=%d, shape[0]=%ld, "
            "multiplier=%d, scale=%.2f\n",
            tm.ndim, tm.shape.size() > 0 ? tm.shape[0] : 0, multiplier, scale);
          // Echo TensorMeta, primitive parameters and payload back
          return unifex::just(
            std::make_tuple(tm, multiplier, scale, std::move(payload)));
        },
        "echo_buffer");

      // Register function that handles UcxBufferVec with TensorMetaVec
      worker.RegisterFunction<ucxx::UcxBufferVec>(
        rpc::function_id_t{5002},
        [](
          const rpc::TensorMetaVec& tensor_metas, int batch_size,
          float threshold, ucxx::UcxBufferVec&& payload) {
          // Verify TensorMeta and primitive parameters
          // tensor_metas should have 2 elements for this test
          LOGX(
            "[Server-TM] Received TensorMeta for Vec: tm1.ndim=%d, "
            "tm1.shape[0]=%ld, tm2.ndim=%d, tm2.shape[0]=%ld, "
            "batch_size=%d, threshold=%.2f\n",
            tensor_metas[0].ndim,
            tensor_metas[0].shape.size() > 0 ? tensor_metas[0].shape[0] : 0,
            tensor_metas[1].ndim,
            tensor_metas[1].shape.size() > 0 ? tensor_metas[1].shape[0] : 0,
            batch_size, threshold);
          // Echo TensorMeta, primitive parameters and payload back
          return unifex::just(std::make_tuple(
            tensor_metas, batch_size, threshold, std::move(payload)));
        },
        "echo_buffer_vec");

      // Register function that receives UcxBufferVec and returns UcxBuffer
      // (concatenates all buffers into one)
      worker.RegisterFunction<ucxx::UcxBufferVec>(
        rpc::function_id_t{5003},
        [&worker](
          const rpc::TensorMetaVec& tensor_metas, int batch_size,
          float threshold, ucxx::UcxBufferVec&& payload) {
          LOGX(
            "[Server-TM] Received UcxBufferVec to convert to UcxBuffer: "
            "vec_size=%zu, batch_size=%d\n",
            payload.size(), batch_size);

          // Calculate total size
          size_t total_size = 0;
          for (size_t i = 0; i < payload.size(); ++i) {
            total_size += payload[i].size;
          }

          // Create a single UcxBuffer and concatenate all data
          ucxx::UcxBuffer result(
            worker.GetMemoryResourceManager(), ucx_memory_type::HOST,
            total_size);
          size_t offset = 0;
          for (size_t i = 0; i < payload.size(); ++i) {
            std::memcpy(
              static_cast<char*>(result.data()) + offset, payload[i].data,
              payload[i].size);
            offset += payload[i].size;
          }

          // Create combined TensorMeta (use first tensor meta as base,
          // adjust shape).
          TensorMeta combined_tm = tensor_metas[0];
          combined_tm.shape =
            cista::offset::vector<int64_t>{static_cast<int64_t>(
              total_size / (tensor_metas[0].dtype.bits / 8))};

          return unifex::just(std::make_tuple(
            combined_tm, batch_size, threshold, std::move(result)));
        },
        "vec_to_buffer");

      // Register function that receives UcxBuffer and returns UcxBufferVec
      // (splits buffer into multiple buffers)
      worker.RegisterFunction<ucxx::UcxBuffer>(
        rpc::function_id_t{5004},
        [&worker](
          const TensorMeta& tm, float scale, ucxx::UcxBuffer&& payload) {
          // Fixed num_splits = 4
          constexpr int32_t num_splits = 4;
          LOGX(
            "[Server-TM] Received UcxBuffer to convert to UcxBufferVec: "
            "size=%zu, num_splits=%d\n",
            payload.size(), num_splits);

          // Split buffer into num_splits parts
          size_t split_size = payload.size() / num_splits;
          std::vector<size_t> sizes(num_splits, split_size);
          // Add remainder to last split
          sizes[num_splits - 1] += payload.size() % num_splits;

          ucxx::UcxBufferVec result(
            worker.GetMemoryResourceManager(), ucx_memory_type::HOST, sizes);

          // Copy data to splits
          size_t offset = 0;
          for (size_t i = 0; i < static_cast<size_t>(num_splits); ++i) {
            std::memcpy(
              result[i].data, static_cast<char*>(payload.data()) + offset,
              sizes[i]);
            offset += sizes[i];
          }

          // Create TensorMeta for each split (fixed to 4 splits)
          // Use TENSOR_META_VEC instead of multiple separate TensorMeta
          rpc::TensorMetaVec split_tensor_metas;
          for (size_t i = 0; i < static_cast<size_t>(num_splits); ++i) {
            TensorMeta split_tm = tm;
            split_tm.shape = cista::offset::vector<int64_t>{
              static_cast<int64_t>(sizes[i] / (tm.dtype.bits / 8))};
            split_tensor_metas.push_back(split_tm);
          }
          return unifex::just(
            std::make_tuple(split_tensor_metas, scale, std::move(result)));
        },
        "buffer_to_vec");

      ASSERT_TRUE(worker.StartServer().has_value())
        << "[Server-TM] Failed to start server";

      std::vector<std::byte> addr = worker.GetLocalAddress();
      size_t addr_size = addr.size();
      write(pipe_fd[1], &addr_size, sizeof(size_t));
      write(pipe_fd[1], addr.data(), addr_size);
      close(pipe_fd[1]);

      // Keep running until signaled
      char buf;
      if (read(control_pipe[0], &buf, 1) < 0) {
        // Ignore read errors, continue to stop server
      }
      worker.Stop();
    }
    std::exit(0);
  }

  // Parent process
  LOGX("[Parent] Forking client process (tensor meta test)\n");
  pid_t client_pid = fork();
  ASSERT_GE(client_pid, 0);

  if (client_pid == 0) {
    // Child 2 - Client
    try {
      mr_ = std::make_unique<ucxx::DefaultUcxMemoryResourceManager>();
      close(pipe_fd[1]);
      size_t addr_size;
      read(pipe_fd[0], &addr_size, sizeof(size_t));
      std::vector<std::byte> addr(addr_size);
      read(pipe_fd[0], addr.data(), addr_size);
      close(pipe_fd[0]);

      {
        auto device_ctx = std::make_unique<ucxx::UcxAutoDefaultDeviceContext>();
        AxonWorker worker(
          *mr_, "client_worker_tm", 4, std::chrono::milliseconds(10),
          std::move(device_ctx));
        ASSERT_TRUE(worker.StartClient().has_value())
          << "[Client-TM] Failed to start client";
        auto conn_result = worker.ConnectEndpoint(addr, "server_worker_tm");
        ASSERT_TRUE(conn_result.has_value())
          << "[Client-TM] Failed to connect to server";

        // Test 1: UcxBuffer with TensorMeta
        {
          LOGX("[Client-TM] Testing UcxBuffer with TensorMeta\n");

          // Create TensorMeta
          TensorMeta tm{};
          tm.device = {kDLCPU, 0};
          tm.ndim = 1;
          tm.dtype = {kDLFloat, 32, 1};
          tm.shape = cista::offset::vector<int64_t>{1024};
          tm.strides = cista::offset::vector<int64_t>{1};
          tm.byte_offset = 0;

          // Calculate payload size from TensorMeta
          size_t data_size = rpc::utils::CalculateTensorSize(tm);

          // Create payload buffer
          ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, data_size);
          // Fill with test data
          float* data_ptr = static_cast<float*>(payload.data());
          for (size_t i = 0; i < data_size / sizeof(float); ++i) {
            data_ptr[i] = static_cast<float>(i);
          }

          // Create request header with TensorMeta and primitive parameters
          rpc::RpcRequestHeader header{};
          header.session_id = rpc::session_id_t{1};
          header.request_id = rpc::request_id_t{0};
          header.function_id = rpc::function_id_t{5001};
          header.workflow_id = rpc::utils::workflow_id_t{0};
          header.hlc.TickLocal();

          int multiplier = 42;
          float scale = 3.14f;

          cista::offset::vector<ParamMeta> params;
          params.push_back(
            {ParamType::TENSOR_META, tm, cista::offset::string{"tensor"}});
          params.push_back(
            {ParamType::PRIMITIVE_INT32, rpc::PrimitiveValue{multiplier},
             cista::offset::string{"multiplier"}});
          params.push_back(
            {ParamType::PRIMITIVE_FLOAT32, rpc::PrimitiveValue{scale},
             cista::offset::string{"scale"}});
          header.params = std::move(params);

          // Invoke RPC
          auto res = unifex::sync_wait(
            worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBuffer>(
              "server_worker_tm", std::move(header),
              std::optional<ucxx::UcxBuffer>(std::move(payload))));

          ASSERT_TRUE(res.has_value())
            << "UcxBuffer RPC with TensorMeta failed";
          auto& [resp_header, resp_payload] = res.value();
          ASSERT_EQ(resp_header->status, std::error_code{});
          ASSERT_EQ(resp_payload.size(), data_size);

          // Verify response has TensorMeta and primitive parameters
          ASSERT_GE(resp_header->results.size(), 3);
          ASSERT_EQ(resp_header->results[0].type, ParamType::TENSOR_META);
          const auto& resp_tm =
            cista::get<TensorMeta>(resp_header->results[0].value);
          ASSERT_EQ(resp_tm.ndim, 1);
          ASSERT_EQ(resp_tm.shape.size(), 1);
          ASSERT_EQ(resp_tm.shape[0], 1024);

          // Verify int parameter
          ASSERT_EQ(resp_header->results[1].type, ParamType::PRIMITIVE_INT32);
          const auto& resp_multiplier =
            cista::get<rpc::PrimitiveValue>(resp_header->results[1].value);
          ASSERT_EQ(cista::get<int32_t>(resp_multiplier), multiplier);

          // Verify float parameter
          ASSERT_EQ(resp_header->results[2].type, ParamType::PRIMITIVE_FLOAT32);
          const auto& resp_scale =
            cista::get<rpc::PrimitiveValue>(resp_header->results[2].value);
          ASSERT_FLOAT_EQ(cista::get<float>(resp_scale), scale);

          // Verify payload data
          float* resp_data = static_cast<float*>(resp_payload.data());
          for (size_t i = 0; i < data_size / sizeof(float); ++i) {
            ASSERT_EQ(resp_data[i], static_cast<float>(i));
          }

          LOGX("[Client-TM] UcxBuffer with TensorMeta test passed\n");
        }

        // Test 2: UcxBufferVec with TensorMeta
        {
          LOGX("[Client-TM] Testing UcxBufferVec with TensorMeta\n");

          // Create TensorMeta for multiple tensors
          TensorMeta tm1{};
          tm1.device = {kDLCPU, 0};
          tm1.ndim = 1;
          tm1.dtype = {kDLInt, 32, 1};
          tm1.shape = cista::offset::vector<int64_t>{512};
          tm1.strides = cista::offset::vector<int64_t>{1};
          tm1.byte_offset = 0;

          TensorMeta tm2{};
          tm2.device = {kDLCPU, 0};
          tm2.ndim = 1;
          tm2.dtype = {kDLInt, 32, 1};
          tm2.shape = cista::offset::vector<int64_t>{256};
          tm2.strides = cista::offset::vector<int64_t>{1};
          tm2.byte_offset = 0;

          // Calculate payload sizes from TensorMeta
          size_t data_size1 = rpc::utils::CalculateTensorSize(tm1);
          size_t data_size2 = rpc::utils::CalculateTensorSize(tm2);

          // Create payload buffer vec
          ucxx::UcxBufferVec payload(
            *mr_, ucx_memory_type::HOST, {data_size1, data_size2});

          // Fill with test data
          int32_t* data1 = static_cast<int32_t*>(payload[0].data);
          for (size_t i = 0; i < data_size1 / sizeof(int32_t); ++i) {
            data1[i] = static_cast<int32_t>(i);
          }

          int32_t* data2 = static_cast<int32_t*>(payload[1].data);
          for (size_t i = 0; i < data_size2 / sizeof(int32_t); ++i) {
            data2[i] = static_cast<int32_t>(i + 1000);
          }

          // Create request header with TensorMeta and primitive parameters
          rpc::RpcRequestHeader header{};
          header.session_id = rpc::session_id_t{1};
          header.request_id = rpc::request_id_t{1};
          header.function_id = rpc::function_id_t{5002};
          header.workflow_id = rpc::utils::workflow_id_t{0};
          header.hlc.TickLocal();

          int batch_size = 128;
          float threshold = 0.95f;

          cista::offset::vector<ParamMeta> params;
          // Use TENSOR_META_VEC instead of two separate TENSOR_META parameters
          rpc::TensorMetaVec tensor_metas;
          tensor_metas.push_back(tm1);
          tensor_metas.push_back(tm2);
          params.push_back(
            {ParamType::TENSOR_META_VEC, tensor_metas,
             cista::offset::string{"tensors"}});
          params.push_back(
            {ParamType::PRIMITIVE_INT32, rpc::PrimitiveValue{batch_size},
             cista::offset::string{"batch_size"}});
          params.push_back(
            {ParamType::PRIMITIVE_FLOAT32, rpc::PrimitiveValue{threshold},
             cista::offset::string{"threshold"}});
          header.params = std::move(params);

          // Invoke RPC
          auto res = unifex::sync_wait(
            worker.InvokeRpc<ucxx::UcxBufferVec, ucxx::UcxBufferVec>(
              "server_worker_tm", std::move(header),
              std::optional<ucxx::UcxBufferVec>(std::move(payload))));

          ASSERT_TRUE(res.has_value())
            << "UcxBufferVec RPC with TensorMeta failed";
          auto& [resp_header, resp_payload] = res.value();
          ASSERT_EQ(resp_header->status, std::error_code{});
          ASSERT_EQ(resp_payload.size(), 2);
          ASSERT_EQ(resp_payload[0].size, data_size1);
          ASSERT_EQ(resp_payload[1].size, data_size2);

          // Verify response has TensorMetaVec and primitive parameters
          ASSERT_GE(resp_header->results.size(), 3);
          ASSERT_EQ(resp_header->results[0].type, ParamType::TENSOR_META_VEC);

          const auto& resp_tensor_metas =
            cista::get<rpc::TensorMetaVec>(resp_header->results[0].value);
          ASSERT_EQ(resp_tensor_metas.size(), 2);
          ASSERT_EQ(resp_tensor_metas[0].ndim, 1);
          ASSERT_EQ(resp_tensor_metas[0].shape[0], 512);
          ASSERT_EQ(resp_tensor_metas[1].ndim, 1);
          ASSERT_EQ(resp_tensor_metas[1].shape[0], 256);

          // Verify int parameter
          ASSERT_EQ(resp_header->results[1].type, ParamType::PRIMITIVE_INT32);
          const auto& resp_batch_size =
            cista::get<rpc::PrimitiveValue>(resp_header->results[1].value);
          ASSERT_EQ(cista::get<int32_t>(resp_batch_size), batch_size);

          // Verify float parameter
          ASSERT_EQ(resp_header->results[2].type, ParamType::PRIMITIVE_FLOAT32);
          const auto& resp_threshold =
            cista::get<rpc::PrimitiveValue>(resp_header->results[2].value);
          ASSERT_FLOAT_EQ(cista::get<float>(resp_threshold), threshold);

          // Verify payload data
          int32_t* resp_data1 = static_cast<int32_t*>(resp_payload[0].data);
          for (size_t i = 0; i < data_size1 / sizeof(int32_t); ++i) {
            ASSERT_EQ(resp_data1[i], static_cast<int32_t>(i));
          }

          int32_t* resp_data2 = static_cast<int32_t*>(resp_payload[1].data);
          for (size_t i = 0; i < data_size2 / sizeof(int32_t); ++i) {
            ASSERT_EQ(resp_data2[i], static_cast<int32_t>(i + 1000));
          }

          LOGX("[Client-TM] UcxBufferVec with TensorMeta test passed\n");
        }

        // Test 3: UcxBufferVec with large message
        {
          LOGX("[Client-TM] Testing UcxBufferVec with large message\n");

          // Create large TensorMeta for multiple tensors
          TensorMeta tm1{};
          tm1.device = {kDLCPU, 0};
          tm1.ndim = 1;
          tm1.dtype = {kDLFloat, 32, 1};
          // Large size: 5MB per tensor
          size_t large_size1 = 5 * 1024 * 1024;
          tm1.shape = cista::offset::vector<int64_t>{
            static_cast<int64_t>(large_size1 / sizeof(float))};
          tm1.strides = cista::offset::vector<int64_t>{1};
          tm1.byte_offset = 0;

          TensorMeta tm2{};
          tm2.device = {kDLCPU, 0};
          tm2.ndim = 1;
          tm2.dtype = {kDLFloat, 32, 1};
          // Large size: 3MB per tensor
          size_t large_size2 = 3 * 1024 * 1024;
          tm2.shape = cista::offset::vector<int64_t>{
            static_cast<int64_t>(large_size2 / sizeof(float))};
          tm2.strides = cista::offset::vector<int64_t>{1};
          tm2.byte_offset = 0;

          // Calculate payload sizes from TensorMeta
          size_t data_size1 = rpc::utils::CalculateTensorSize(tm1);
          size_t data_size2 = rpc::utils::CalculateTensorSize(tm2);

          // Create payload buffer vec
          ucxx::UcxBufferVec payload(
            *mr_, ucx_memory_type::HOST, {data_size1, data_size2});

          // Fill with test data
          float* data1 = static_cast<float*>(payload[0].data);
          for (size_t i = 0; i < data_size1 / sizeof(float); ++i) {
            data1[i] = static_cast<float>(i % 1000);
          }

          float* data2 = static_cast<float*>(payload[1].data);
          for (size_t i = 0; i < data_size2 / sizeof(float); ++i) {
            data2[i] = static_cast<float>((i + 5000) % 1000);
          }

          // Create request header with TensorMeta and primitive parameters
          rpc::RpcRequestHeader header{};
          header.session_id = rpc::session_id_t{1};
          header.request_id = rpc::request_id_t{2};
          header.function_id = rpc::function_id_t{5002};
          header.workflow_id = rpc::utils::workflow_id_t{0};
          header.hlc.TickLocal();

          int batch_size = 256;
          float threshold = 0.99f;

          cista::offset::vector<ParamMeta> params;
          // Use TENSOR_META_VEC instead of two separate TENSOR_META parameters
          rpc::TensorMetaVec tensor_metas;
          tensor_metas.push_back(tm1);
          tensor_metas.push_back(tm2);
          params.push_back(
            {ParamType::TENSOR_META_VEC, tensor_metas,
             cista::offset::string{"tensors"}});
          params.push_back(
            {ParamType::PRIMITIVE_INT32, rpc::PrimitiveValue{batch_size},
             cista::offset::string{"batch_size"}});
          params.push_back(
            {ParamType::PRIMITIVE_FLOAT32, rpc::PrimitiveValue{threshold},
             cista::offset::string{"threshold"}});
          header.params = std::move(params);

          // Invoke RPC
          auto res = unifex::sync_wait(
            worker.InvokeRpc<ucxx::UcxBufferVec, ucxx::UcxBufferVec>(
              "server_worker_tm", std::move(header),
              std::optional<ucxx::UcxBufferVec>(std::move(payload))));

          ASSERT_TRUE(res.has_value())
            << "UcxBufferVec large message RPC failed";
          auto& [resp_header, resp_payload] = res.value();
          ASSERT_EQ(resp_header->status, std::error_code{});
          ASSERT_EQ(resp_payload.size(), 2);
          ASSERT_EQ(resp_payload[0].size, data_size1);
          ASSERT_EQ(resp_payload[1].size, data_size2);

          // Verify payload data
          float* resp_data1 = static_cast<float*>(resp_payload[0].data);
          for (size_t i = 0; i < data_size1 / sizeof(float); ++i) {
            ASSERT_EQ(resp_data1[i], static_cast<float>(i % 1000));
          }

          float* resp_data2 = static_cast<float*>(resp_payload[1].data);
          for (size_t i = 0; i < data_size2 / sizeof(float); ++i) {
            ASSERT_EQ(resp_data2[i], static_cast<float>((i + 5000) % 1000));
          }

          LOGX(
            "[Client-TM] UcxBufferVec large message test passed (total size: "
            "%zu bytes)\n",
            data_size1 + data_size2);
        }

        // Test 4: Send UcxBufferVec, return UcxBuffer
        {
          LOGX("[Client-TM] Testing UcxBufferVec -> UcxBuffer conversion\n");

          // Create TensorMeta for multiple tensors
          TensorMeta tm1{};
          tm1.device = {kDLCPU, 0};
          tm1.ndim = 1;
          tm1.dtype = {kDLInt, 32, 1};
          tm1.shape = cista::offset::vector<int64_t>{1024};
          tm1.strides = cista::offset::vector<int64_t>{1};
          tm1.byte_offset = 0;

          TensorMeta tm2{};
          tm2.device = {kDLCPU, 0};
          tm2.ndim = 1;
          tm2.dtype = {kDLInt, 32, 1};
          tm2.shape = cista::offset::vector<int64_t>{512};
          tm2.strides = cista::offset::vector<int64_t>{1};
          tm2.byte_offset = 0;

          // Calculate payload sizes from TensorMeta
          size_t data_size1 = rpc::utils::CalculateTensorSize(tm1);
          size_t data_size2 = rpc::utils::CalculateTensorSize(tm2);

          // Create payload buffer vec
          ucxx::UcxBufferVec payload(
            *mr_, ucx_memory_type::HOST, {data_size1, data_size2});

          // Fill with test data
          int32_t* data1 = static_cast<int32_t*>(payload[0].data);
          for (size_t i = 0; i < data_size1 / sizeof(int32_t); ++i) {
            data1[i] = static_cast<int32_t>(i);
          }

          int32_t* data2 = static_cast<int32_t*>(payload[1].data);
          for (size_t i = 0; i < data_size2 / sizeof(int32_t); ++i) {
            data2[i] = static_cast<int32_t>(i + 2000);
          }

          // Create request header
          rpc::RpcRequestHeader header{};
          header.session_id = rpc::session_id_t{1};
          header.request_id = rpc::request_id_t{3};
          header.function_id = rpc::function_id_t{5003};
          header.workflow_id = rpc::utils::workflow_id_t{0};
          header.hlc.TickLocal();

          int batch_size = 64;
          float threshold = 0.85f;

          cista::offset::vector<ParamMeta> params;
          // Use TENSOR_META_VEC instead of two separate TENSOR_META parameters
          rpc::TensorMetaVec tensor_metas;
          tensor_metas.push_back(tm1);
          tensor_metas.push_back(tm2);
          params.push_back(
            {ParamType::TENSOR_META_VEC, tensor_metas,
             cista::offset::string{"tensors"}});
          params.push_back(
            {ParamType::PRIMITIVE_INT32, rpc::PrimitiveValue{batch_size},
             cista::offset::string{"batch_size"}});
          params.push_back(
            {ParamType::PRIMITIVE_FLOAT32, rpc::PrimitiveValue{threshold},
             cista::offset::string{"threshold"}});
          header.params = std::move(params);

          // Invoke RPC: UcxBufferVec -> UcxBuffer
          auto res = unifex::sync_wait(
            worker.InvokeRpc<ucxx::UcxBufferVec, ucxx::UcxBuffer>(
              "server_worker_tm", std::move(header),
              std::optional<ucxx::UcxBufferVec>(std::move(payload))));

          ASSERT_TRUE(res.has_value())
            << "UcxBufferVec -> UcxBuffer RPC failed";
          auto& [resp_header, resp_payload] = res.value();
          ASSERT_EQ(resp_header->status, std::error_code{});
          ASSERT_EQ(resp_payload.size(), data_size1 + data_size2);

          // Verify response has TensorMeta and primitive parameters
          ASSERT_GE(resp_header->results.size(), 3);
          ASSERT_EQ(resp_header->results[0].type, ParamType::TENSOR_META);

          // Verify int parameter
          ASSERT_EQ(resp_header->results[1].type, ParamType::PRIMITIVE_INT32);
          const auto& resp_batch_size =
            cista::get<rpc::PrimitiveValue>(resp_header->results[1].value);
          ASSERT_EQ(cista::get<int32_t>(resp_batch_size), batch_size);

          // Verify float parameter
          ASSERT_EQ(resp_header->results[2].type, ParamType::PRIMITIVE_FLOAT32);
          const auto& resp_threshold =
            cista::get<rpc::PrimitiveValue>(resp_header->results[2].value);
          ASSERT_FLOAT_EQ(cista::get<float>(resp_threshold), threshold);

          // Verify payload data (concatenated)
          int32_t* resp_data = static_cast<int32_t*>(resp_payload.data());
          for (size_t i = 0; i < data_size1 / sizeof(int32_t); ++i) {
            ASSERT_EQ(resp_data[i], static_cast<int32_t>(i));
          }
          for (size_t i = 0; i < data_size2 / sizeof(int32_t); ++i) {
            ASSERT_EQ(
              resp_data[data_size1 / sizeof(int32_t) + i],
              static_cast<int32_t>(i + 2000));
          }

          LOGX("[Client-TM] UcxBufferVec -> UcxBuffer test passed\n");
        }

        // Test 5: Send UcxBuffer, return UcxBufferVec
        {
          LOGX("[Client-TM] Testing UcxBuffer -> UcxBufferVec conversion\n");

          // Create TensorMeta
          TensorMeta tm{};
          tm.device = {kDLCPU, 0};
          tm.ndim = 1;
          tm.dtype = {kDLFloat, 32, 1};
          tm.shape = cista::offset::vector<int64_t>{2048};
          tm.strides = cista::offset::vector<int64_t>{1};
          tm.byte_offset = 0;

          // Calculate payload size from TensorMeta
          size_t data_size = rpc::utils::CalculateTensorSize(tm);

          // Create payload buffer
          ucxx::UcxBuffer payload(*mr_, ucx_memory_type::HOST, data_size);
          // Fill with test data
          float* data_ptr = static_cast<float*>(payload.data());
          for (size_t i = 0; i < data_size / sizeof(float); ++i) {
            data_ptr[i] = static_cast<float>(i);
          }

          // Create request header
          rpc::RpcRequestHeader header{};
          header.session_id = rpc::session_id_t{1};
          header.request_id = rpc::request_id_t{4};
          header.function_id = rpc::function_id_t{5004};
          header.workflow_id = rpc::utils::workflow_id_t{0};
          header.hlc.TickLocal();

          constexpr int32_t num_splits = 4;
          float scale = 2.5f;

          cista::offset::vector<ParamMeta> params;
          params.push_back(
            {ParamType::TENSOR_META, tm, cista::offset::string{"tensor"}});
          params.push_back(
            {ParamType::PRIMITIVE_FLOAT32, rpc::PrimitiveValue{scale},
             cista::offset::string{"scale"}});
          header.params = std::move(params);

          // Invoke RPC: UcxBuffer -> UcxBufferVec
          auto res = unifex::sync_wait(
            worker.InvokeRpc<ucxx::UcxBuffer, ucxx::UcxBufferVec>(
              "server_worker_tm", std::move(header),
              std::optional<ucxx::UcxBuffer>(std::move(payload))));

          ASSERT_TRUE(res.has_value())
            << "UcxBuffer -> UcxBufferVec RPC failed: no response";
          auto& [resp_header, resp_payload] = res.value();
          if (std::error_code(resp_header->status)) {
            std::string error_msg = resp_header->status.GetErrorMessage();
            if (
              !resp_header->results.empty()
              && resp_header->results[0].type == rpc::ParamType::STRING) {
              const auto& error_str =
                cista::get<data::string>(resp_header->results[0].value);
              if (!error_str.empty()) {
                error_msg = std::string(error_str.data(), error_str.size());
              }
            }
            LOGX(
              "[Client-TM] RPC failed with status: %s, error: %s\n",
              resp_header->status.GetErrorMessage().c_str(), error_msg.c_str());
            ASSERT_EQ(resp_header->status, std::error_code{})
              << "UcxBuffer -> UcxBufferVec RPC failed: " << error_msg;
          }
          ASSERT_EQ(resp_payload.size(), static_cast<size_t>(num_splits));

          // Verify total size matches
          size_t total_resp_size = 0;
          for (size_t i = 0; i < resp_payload.size(); ++i) {
            total_resp_size += resp_payload[i].size;
          }
          ASSERT_EQ(total_resp_size, data_size);

          // Verify response has TensorMetaVec and primitive parameters
          // Should have TENSOR_META_VEC + scale = 2
          ASSERT_GE(resp_header->results.size(), 2);
          // Verify TensorMetaVec
          ASSERT_EQ(resp_header->results[0].type, ParamType::TENSOR_META_VEC);
          const auto& resp_tensor_metas =
            cista::get<rpc::TensorMetaVec>(resp_header->results[0].value);
          ASSERT_EQ(resp_tensor_metas.size(), static_cast<size_t>(num_splits));

          // Verify float parameter (should be at index 1)
          ASSERT_EQ(resp_header->results[1].type, ParamType::PRIMITIVE_FLOAT32);
          const auto& resp_scale =
            cista::get<rpc::PrimitiveValue>(resp_header->results[1].value);
          ASSERT_FLOAT_EQ(cista::get<float>(resp_scale), scale);

          // Verify TensorMetaVec contents
          for (size_t i = 0; i < resp_tensor_metas.size(); ++i) {
            ASSERT_EQ(resp_tensor_metas[i].ndim, 1);
            size_t expected_elements = resp_payload[i].size / sizeof(float);
            ASSERT_EQ(
              resp_tensor_metas[i].shape[0],
              static_cast<int64_t>(expected_elements));
          }

          // Verify payload data (split and concatenated back should match
          // original)
          size_t offset = 0;
          for (size_t split_idx = 0; split_idx < resp_payload.size();
               ++split_idx) {
            float* split_data =
              static_cast<float*>(resp_payload[split_idx].data);
            size_t split_elements =
              resp_payload[split_idx].size / sizeof(float);
            for (size_t i = 0; i < split_elements; ++i) {
              ASSERT_EQ(split_data[i], static_cast<float>(offset + i));
            }
            offset += split_elements;
          }

          LOGX("[Client-TM] UcxBuffer -> UcxBufferVec test passed\n");
        }

        worker.Stop();
      }

      if (::testing::Test::HasFailure()) {
        std::exit(1);
      }
      std::exit(0);
    } catch (const errors::AxonErrorException& e) {
      LOGX(
        "[Client-TM] Caught AxonErrorException: %s (status: %s)\n",
        e.context().what.c_str(), e.context().status.GetErrorMessage().c_str());
      std::exit(1);
    } catch (const rpc::RpcException& e) {
      LOGX(
        "[Client-TM] Caught RpcException: %s (code: %s)\n", e.what(),
        e.code().message().c_str());
      std::exit(1);
    } catch (const std::exception& e) {
      LOGX("[Client-TM] Caught std::exception: %s\n", e.what());
      std::exit(1);
    } catch (...) {
      LOGX("[Client-TM] Caught unknown exception\n");
      std::exit(1);
    }
  }

  // Parent process
  close(pipe_fd[0]);
  close(pipe_fd[1]);
  close(control_pipe[0]);

  int status;
  // Wait for client to finish
  waitpid(client_pid, &status, 0);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);

  // Cleanup server
  close(control_pipe[1]);
  waitpid(server_pid, nullptr, 0);
}

}  // namespace eux::axon
