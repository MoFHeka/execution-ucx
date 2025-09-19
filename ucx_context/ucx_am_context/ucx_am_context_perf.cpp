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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unifex/defer.hpp>
#include <unifex/finally.hpp>
#include <unifex/just_from.hpp>
#include <unifex/let_value.hpp>
#include <unifex/let_value_with.hpp>
#include <unifex/repeat_effect_until.hpp>
#include <unifex/sequence.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/then.hpp>

#include "ucx_context/ucx_connection.hpp"
#include "ucx_context/ucx_context_def.h"
#include "ucx_context/ucx_memory_resource.hpp"
#include "ucx_context/ucx_status.hpp"

using eux::ucxx::connect_endpoint;
using eux::ucxx::connection_recv;
using eux::ucxx::connection_send;
using eux::ucxx::DefaultUcxMemoryResourceManager;
using eux::ucxx::ucx_am_context;
using eux::ucxx::UcxAmData;
using eux::ucxx::UcxMemoryResourceManager;

using unifex::defer;
using unifex::finally;
using unifex::inplace_stop_source;
using unifex::just_from;
using unifex::let_value;
using unifex::let_value_with;
using unifex::repeat_effect_until;
using unifex::sequence;
using unifex::sync_wait;
using unifex::then;

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
      if (thread_.joinable()) {
        thread_.join();
      }
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
      *memoryResource_, name_, timeout_, /*connectionHandleError=*/false));
    thread_ = std::thread([this] { context_->run(stopSource_.get_token()); });
    // Wait for context to initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
};

void print_statistics(
  const std::vector<double>& latencies, size_t msg_size, float factor = 0.5) {
  if (latencies.empty()) {
    std::cout << "No measurements collected." << std::endl;
    return;
  }

  std::vector<double> sorted_latencies = latencies;
  std::sort(sorted_latencies.begin(), sorted_latencies.end());

  double sum =
    std::accumulate(sorted_latencies.begin(), sorted_latencies.end(), 0.0);
  double mean = sum / sorted_latencies.size();
  double min = sorted_latencies.front();
  double max = sorted_latencies.back();

  double p50 =
    sorted_latencies[static_cast<size_t>(sorted_latencies.size() * 0.50)];
  double p99 =
    sorted_latencies[static_cast<size_t>(sorted_latencies.size() * 0.99)];
  double p999 =
    sorted_latencies[static_cast<size_t>(sorted_latencies.size() * 0.999)];

  std::cout << "===== One-Way Latency Test Results (msg size: " << msg_size
            << " bytes) =====" << std::endl;
  std::cout << "Iterations: " << latencies.size() << std::endl;
  std::cout << "Min (us): " << min * factor << std::endl;
  std::cout << "Max (us): " << max * factor << std::endl;
  std::cout << "Mean (us): " << mean * factor << std::endl;
  std::cout << "p50 (us): " << p50 * factor << std::endl;
  std::cout << "p99 (us): " << p99 * factor << std::endl;
  std::cout << "p999 (us): " << p999 * factor << std::endl;
  std::cout << "======================================================"
            << std::endl;
}

void echo_server_task(
  const ucx_am_context::scheduler& scheduler,
  std::vector<std::byte>& client_ucp_address, int warmup_iters, int iters) {
  auto conn_id =
    sync_wait(connect_endpoint(scheduler, client_ucp_address)).value();
  // Warm-up
  for (int i = 0; i < warmup_iters; ++i) {
    connection_recv(scheduler, ucx_memory_type::HOST)
      | let_value([&](auto&& bundle) {
          return connection_send(
            scheduler, conn_id, std::move(bundle.move_data()));
        })
      | sync_wait();
  }

  // Measurement
  defer([&]() {
    return connection_recv(scheduler, ucx_memory_type::HOST)
           | let_value([&](auto&& bundle) {
               return connection_send(
                 scheduler, conn_id, std::move(bundle.move_data()));
             });
  }) | repeat_effect_until([iters]() mutable {
    if (iters == 1) return true;  // 1 is the last iteration then end
    --iters;
    return false;
  }) | sync_wait();

  // // Another code style for Measurement
  // for (int i = 0; i < iters; ++i) {
  //   connection_recv(scheduler, ucx_memory_type::HOST)
  //     | let_value([&](auto&& bundle) {
  //         return connection_send(
  //           scheduler, conn_id, std::move(bundle.move_data()));
  //       })
  //     | sync_wait();
  // }
}

template <typename Sender>
auto measure_time(Sender&& sender, std::vector<double>& latencies) {
  return let_value_with(
    [] { return std::chrono::high_resolution_clock::now(); },
    [sender = std::move(sender),
     &latencies](const std::chrono::high_resolution_clock::time_point& start) {
      return unifex::finally(
        std::move(sender), unifex::just_from([&]() noexcept {
          const auto end = std::chrono::high_resolution_clock::now();
          const auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
          latencies.push_back(duration.count());
        }));
    });
}

std::vector<double> echo_client_task(
  const ucx_am_context::scheduler& scheduler,
  std::vector<std::byte>& server_ucp_address, inplace_stop_source& stop_source,
  size_t header_size, size_t msg_size, int warmup_iters, int iters) {
  auto conn_id =
    sync_wait(connect_endpoint(scheduler, server_ucp_address)).value();

  std::vector<char> send_header(header_size, 'h');
  std::vector<char> send_buffer(msg_size, 'a');
  ucx_am_data send_data{};
  send_data.buffer.data = send_buffer.data();
  send_data.buffer.size = send_buffer.size();
  send_data.buffer_type = ucx_memory_type::HOST;
  send_data.header.data = send_header.data();
  send_data.header.size = send_header.size();

  // Warm-up
  for (int i = 0; i < warmup_iters; ++i) {
    sequence(
      connection_send(scheduler, conn_id, send_data),
      connection_recv(scheduler, ucx_memory_type::HOST))
      | sync_wait();
  }

  std::vector<double> latencies;
  latencies.reserve(iters);

  // Measurement
  for (int i = 0; i < iters; ++i) {
    measure_time(
      defer([&]() {
        return sequence(
          connection_send(scheduler, conn_id, send_data),
          connection_recv(scheduler, ucx_memory_type::HOST));
      }),
      latencies)
      | sync_wait();
  }

  // // Another code style for Measurement
  // // sync_wait() making context switch slows down the performance
  // for (int i = 0; i < iters; ++i) {
  //   auto start = std::chrono::high_resolution_clock::now();
  //   sequence(
  //     connection_send(scheduler, conn_id, send_data),
  //     connection_recv(scheduler, ucx_memory_type::HOST))
  //     | sync_wait();
  //   auto end = std::chrono::high_resolution_clock::now();

  //   auto duration =
  //     std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  //   latencies.push_back(duration.count());
  // }

  stop_source.request_stop();
  return latencies;
}

int main(int argc, char** argv) {
  // Config
  size_t header_size = 8;
  size_t msg_size = 64;
  int warmup_iters = 1;
  int iters = 100;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--header_size" || arg == "-H") && i + 1 < argc) {
      header_size = static_cast<size_t>(std::stoul(argv[++i]));
    } else if ((arg == "--msg_size" || arg == "-M") && i + 1 < argc) {
      msg_size = static_cast<size_t>(std::stoul(argv[++i]));
    } else if ((arg == "--warmup_iters" || arg == "-w") && i + 1 < argc) {
      warmup_iters = std::stoi(argv[++i]);
    } else if ((arg == "--iters" || arg == "-i") && i + 1 < argc) {
      iters = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--header_size N|-H N] [--msg_size N|-M N] "
                   "[--warmup_iters N|-w N] [--iters N|-i N]\n";
      std::exit(0);
    }
  }

  UcxContextHostRunner server_runner("server");
  UcxContextHostRunner client_runner("client");

  std::vector<std::byte> server_addr, client_addr;
  if (server_runner.get_context().get_ucp_address(server_addr).value() != 0) {
    std::cerr << "Failed to get UCP address for server" << std::endl;
    return 1;
  }
  if (client_runner.get_context().get_ucp_address(client_addr).value() != 0) {
    std::cerr << "Failed to get UCP address for client" << std::endl;
    return 1;
  }

  inplace_stop_source stop_source;
  std::vector<double> latencies;

  std::thread server_thread([&]() {
    echo_server_task(
      server_runner.get_context().get_scheduler(), client_addr, warmup_iters,
      iters);
  });

  std::thread client_thread([&]() {
    latencies = echo_client_task(
      client_runner.get_context().get_scheduler(), server_addr, stop_source,
      header_size, msg_size, warmup_iters, iters);
  });

  client_thread.join();
  server_thread.join();

  print_statistics(
    latencies, msg_size, 0.5);  // factor = 0.5 to convert to half route time

  return 0;
}
