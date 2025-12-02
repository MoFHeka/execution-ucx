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

#pragma once

#ifndef AXON_CORE_METRICS_METRICS_OBSERVER_HPP_
#define AXON_CORE_METRICS_METRICS_OBSERVER_HPP_

#include <proxy/proxy.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rpc_core/rpc_types.hpp"

namespace eux {
namespace axon {
namespace metrics {

// General RPC metrics context
struct RpcMetricsContext {
  uint64_t conn_id{0};
  uint64_t session_id{0};
  uint32_t request_id{0};
  uint32_t function_id{0};
  std::string function_name;
  rpc::utils::HybridLogicalClock hlc{};
  uint32_t workflow_id{};
};

// Worker-level metrics
struct WorkerMetrics {
  size_t pending_rpcs_count{0};
  size_t client_pipe_failed_count{0};
  size_t server_pipe_invoked_count{0};
  size_t server_pipe_failed_count{0};
  std::vector<std::pair<uint64_t, time_t>> server_ucx_rejected_messages_info{};
};

PRO_DEF_MEM_DISPATCH(MemOnDispatchStart, OnDispatchStart);
PRO_DEF_MEM_DISPATCH(MemOnDispatchComplete, OnDispatchComplete);
PRO_DEF_MEM_DISPATCH(MemOnWorkerStats, OnWorkerStats);

struct MetricsObserverFacade
  : pro::facade_builder  //
    ::add_convention<
      MemOnDispatchStart,
      void(const RpcMetricsContext&, std::chrono::steady_clock::time_point)>  //
    ::add_convention<
      MemOnDispatchComplete,
      void(const RpcMetricsContext&, std::chrono::steady_clock::time_point)>  //
    ::add_convention<MemOnWorkerStats, void(WorkerMetrics)>                   //
    ::build {};

using MetricsObserver = pro::proxy<MetricsObserverFacade>;

}  // namespace metrics
}  // namespace axon
}  // namespace eux

#endif  // AXON_CORE_METRICS_METRICS_OBSERVER_HPP_
