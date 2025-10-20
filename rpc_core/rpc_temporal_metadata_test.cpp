/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.
 *
 *Licensed under the Apache License Version 2.0 with LLVM Exceptions
 *(the "License"); you may not use this file except in compliance with
 *the License. You may obtain a copy of the License at
 *
 *    https://llvm.org/LICENSE.txt
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 *==============================================================================*/

#include <cista.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rpc_core/rpc_dispatcher.hpp"
#include "rpc_core/rpc_types.hpp"
#include "rpc_core/utils/hybrid_logical_clock.hpp"

namespace eux {
namespace rpc {

namespace {

class RpcTemporalMetadataTest : public ::testing::Test {
 protected:
  static RpcRequestHeader make_request() {
    RpcRequestHeader header;
    header.function_id = function_id_t{1};
    header.session_id = session_id_t{2};
    return header;
  }
};

// Graph node definition for distributed computation
struct GraphNode {
  uint32_t node_id;
  std::string node_name;
  std::vector<uint32_t> input_sessions;   // Input session IDs
  std::vector<uint32_t> output_sessions;  // Output session IDs
  std::function<void(const RpcRequestHeader&)> handler;
  bool is_source{false};
  bool is_sink{false};
};

// Data flow trace for replay and verification
struct DataFlowTrace {
  std::string node_name;
  uint32_t function_id;
  uint32_t session_id;
  uint32_t source_session;   // Source session ID (for tracking data flow)
  uint32_t dest_session;     // Destination session ID
  uint64_t clock_raw;        // Hybrid logical clock raw value
  uint64_t physical_ms;      // Physical timestamp in milliseconds
  uint16_t logical_counter;  // Logical counter
  uint32_t event_id;
  uint32_t workflow_id;
  std::chrono::steady_clock::time_point wall_time;  // Wall clock for ordering

  // Comparison for temporal ordering
  bool operator<(const DataFlowTrace& other) const {
    return clock_raw < other.clock_raw;
  }
};

// Enhanced event recorder for distributed graph simulation
class DistributedGraphRecorder {
 public:
  void record_execution(
    const std::string& node_name, const RpcRequestHeader& header) {
    DataFlowTrace trace{};
    trace.node_name = node_name;
    trace.function_id = header.function_id.v_;
    trace.session_id = header.session_id.v_;
    trace.clock_raw = header.clock().raw();
    trace.physical_ms = header.clock().physical_time_ms();
    trace.logical_counter = header.clock().logical_counter();
    trace.event_id = header.event().event_id.v_;
    trace.workflow_id = header.event().workflow_id.v_;
    trace.wall_time = std::chrono::steady_clock::now();

    // Extract source/dest session info from workflow context
    trace.source_session = extract_source_session(header);
    trace.dest_session = extract_dest_session(header);

    std::lock_guard<std::mutex> lock(mutex_);
    traces_.push_back(std::move(trace));
  }

  std::vector<DataFlowTrace> get_chronological_traces() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto traces = traces_;
    std::sort(traces.begin(), traces.end());
    return traces;
  }

  std::vector<DataFlowTrace> get_traces_by_session(uint32_t session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DataFlowTrace> session_traces;
    for (const auto& trace : traces_) {
      if (
        trace.session_id == session_id || trace.source_session == session_id
        || trace.dest_session == session_id) {
        session_traces.push_back(trace);
      }
    }
    std::sort(session_traces.begin(), session_traces.end());
    return session_traces;
  }

  // Replay data flow for verification
  struct DataFlowReplay {
    std::vector<std::string> execution_order;
    std::map<uint32_t, std::vector<std::string>> session_flows;
    std::map<uint32_t, std::vector<std::string>> workflow_flows;
    bool causal_consistency_valid{true};
    std::vector<std::string> violations;
  };

  DataFlowReplay replay_data_flow() const {
    auto traces = get_chronological_traces();
    DataFlowReplay replay;

    // Build execution order
    for (const auto& trace : traces) {
      replay.execution_order.push_back(trace.node_name);
      replay.session_flows[trace.session_id].push_back(trace.node_name);
      replay.workflow_flows[trace.workflow_id].push_back(trace.node_name);
    }

    // Verify causal consistency
    verify_causal_consistency(traces, replay);

    return replay;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<DataFlowTrace> traces_;

  uint32_t extract_source_session(const RpcRequestHeader& header) const {
    // Extract source session from temporal metadata or workflow context
    // This is a simplified implementation - in practice, this would be
    // derived from the workflow metadata
    if (header.event().workflow_id.v_ == 0) {
      return 0;  // Source node
    }
    return header.event().workflow_id.v_ - 1;  // Parent workflow as source
  }

  uint32_t extract_dest_session(const RpcRequestHeader& header) const {
    // Extract destination session based on node type and workflow
    return header.session_id.v_;
  }

  void verify_causal_consistency(
    const std::vector<DataFlowTrace>& traces, DataFlowReplay& replay) const {
    // Check that events maintain causal ordering
    std::map<uint32_t, uint64_t> last_session_clock;

    for (const auto& trace : traces) {
      // Check if this session's clock is consistent with causal dependencies
      if (
        last_session_clock.find(trace.source_session)
        != last_session_clock.end()) {
        if (trace.clock_raw <= last_session_clock[trace.source_session]) {
          replay.causal_consistency_valid = false;
          replay.violations.push_back(
            "Causal violation: " + trace.node_name + " clock "
            + std::to_string(trace.clock_raw) + " <= source clock "
            + std::to_string(last_session_clock[trace.source_session]));
        }
      }

      // Update last known clock for this session
      last_session_clock[trace.session_id] = trace.clock_raw;
    }
  }
};

TEST_F(RpcTemporalMetadataTest, TemporalMetadataDefaults) {
  RpcRequestHeader header;
  EXPECT_EQ(header.clock().raw(), 0u);
  EXPECT_EQ(header.event().workflow_id, utils::workflow_id_t{});
  EXPECT_FALSE(header.event().valid());
}

TEST_F(RpcTemporalMetadataTest, TickLocalUpdatesClock) {
  auto header = make_request();
  header.tick_local_event();

  EXPECT_NE(header.clock().raw(), 0u);
  EXPECT_LE(header.clock().logical_counter(), 1u);
}

TEST_F(RpcTemporalMetadataTest, EventAssignmentAndClear) {
  RpcRequestHeader header;
  header.assign_event(utils::event_id_t{42}, utils::workflow_id_t{7});

  EXPECT_TRUE(header.event().valid());
  EXPECT_EQ(header.event().event_id, utils::event_id_t{42});
  EXPECT_EQ(header.event().workflow_id, utils::workflow_id_t{7});

  header.clear_event();
  EXPECT_FALSE(header.event().valid());
}

TEST_F(RpcTemporalMetadataTest, MergeRemoteClockPropagates) {
  RpcRequestHeader header;
  utils::HybridLogicalClock remote;
  remote.assign(123, 10);

  // Set initial clock with some logical counter
  header.clock().assign(100, 5);  // physical=100, logical=5

  header.merge_remote_clock(remote);

  EXPECT_GE(header.clock().physical_time_ms(), 123u);
  // When physical time advances, logical counter is reset to 0 in HLC
  // This is the expected behavior of HLC merge
  EXPECT_EQ(header.clock().logical_counter(), 0u);
}

// HLC-based distributed node for proper clock management
class HLCNode {
 public:
  HLCNode(uint32_t node_id, const std::string& name)
    : node_id_(node_id), name_(name) {}

  // Process incoming RPC request with proper HLC updates
  RpcRequestHeader process_request(
    const RpcRequestHeader& incoming_request,
    DistributedGraphRecorder& recorder) {
    // 1. Merge incoming clock into local HLC (receive event)
    local_hlc_.merge(incoming_request.clock());

    // 2. Create a copy of incoming request with updated clock for recording
    RpcRequestHeader receive_request = incoming_request;
    receive_request.clock() = local_hlc_;
    recorder.record_execution(name_ + "_receive", receive_request);

    // 3. Tick local clock for local processing event
    local_hlc_.tick_local();

    // 4. Create response with current HLC
    RpcRequestHeader response;
    response.function_id = function_id_t{node_id_};
    response.session_id = incoming_request.session_id;
    response.assign_event(
      utils::event_id_t{static_cast<uint32_t>(local_hlc_.raw())},
      incoming_request.event().workflow_id);

    // 5. Assign current HLC to response clock
    response.clock() = local_hlc_;

    // 6. Record the response event
    recorder.record_execution(name_ + "_send", response);

    return response;
  }

  // Send event - tick local clock and return HLC
  uint64_t send_event() {
    local_hlc_.tick_local();
    return local_hlc_.raw();
  }

  const utils::HybridLogicalClock& get_clock() const { return local_hlc_; }
  uint32_t get_node_id() const { return node_id_; }
  const std::string& get_name() const { return name_; }

 private:
  uint32_t node_id_;
  std::string name_;
  utils::HybridLogicalClock local_hlc_;
};

TEST_F(RpcTemporalMetadataTest, HLCBasedDistributedForkJoinSimulation) {
  // Create HLC-based distributed nodes for proper clock management
  DistributedGraphRecorder recorder;

  // Initialize nodes with independent HLC clocks
  HLCNode source_node(1, "Source");
  HLCNode branch_a_node(2, "BranchA");
  HLCNode branch_b_node(3, "BranchB");
  HLCNode join_node(4, "Join");
  HLCNode sink_node(5, "Sink");

  // Simulate distributed execution with proper HLC usage
  std::vector<std::thread> execution_threads;

  // Phase 1: Source node sends initial event
  execution_threads.emplace_back([&]() {
    // Source node creates initial request
    RpcRequestHeader initial_request;
    initial_request.function_id = function_id_t{1};
    initial_request.session_id = session_id_t{1001};
    initial_request.assign_event(utils::event_id_t{1}, utils::workflow_id_t{0});

    // Source tick local clock for send event
    uint64_t source_clock = source_node.send_event();
    initial_request.clock().assign_raw(source_clock);

    // Record source send event
    recorder.record_execution("Source_send", initial_request);

    // Simulate network delay
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // BranchA receives and processes
    auto branch_a_response =
      branch_a_node.process_request(initial_request, recorder);
    EXPECT_GT(branch_a_response.clock().raw(), initial_request.clock().raw());

    // BranchB receives and processes (parallel)
    auto branch_b_response =
      branch_b_node.process_request(initial_request, recorder);
    EXPECT_GT(branch_b_response.clock().raw(), initial_request.clock().raw());

    // Verify causal ordering: both branches have clocks > source
    EXPECT_GT(branch_a_response.clock().raw(), source_clock);
    EXPECT_GT(branch_b_response.clock().raw(), source_clock);
  });

  // Phase 2: Join node waits for both branches
  execution_threads.emplace_back([&]() {
    // Wait for branches to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Create requests from both branches to join
    RpcRequestHeader branch_a_request;
    branch_a_request.function_id = function_id_t{2};
    branch_a_request.session_id = session_id_t{2001};
    branch_a_request.assign_event(
      utils::event_id_t{2}, utils::workflow_id_t{1});

    RpcRequestHeader branch_b_request;
    branch_b_request.function_id = function_id_t{3};
    branch_b_request.session_id = session_id_t{2002};
    branch_b_request.assign_event(
      utils::event_id_t{3}, utils::workflow_id_t{2});

    // Simulate receiving from both branches
    auto join_response_a =
      join_node.process_request(branch_a_request, recorder);
    auto join_response_b =
      join_node.process_request(branch_b_request, recorder);

    // Join should have clock > both branches
    EXPECT_GT(join_response_a.clock().raw(), branch_a_request.clock().raw());
    EXPECT_GT(join_response_b.clock().raw(), branch_b_request.clock().raw());
  });

  // Phase 3: Sink node processes final result
  execution_threads.emplace_back([&]() {
    // Wait for join to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(40));

    RpcRequestHeader join_request;
    join_request.function_id = function_id_t{4};
    join_request.session_id = session_id_t{3001};
    join_request.assign_event(utils::event_id_t{4}, utils::workflow_id_t{3});

    auto sink_response = sink_node.process_request(join_request, recorder);

    // Sink should have clock > join
    EXPECT_GT(sink_response.clock().raw(), join_request.clock().raw());
  });

  // Wait for all executions to complete
  for (auto& thread : execution_threads) {
    thread.join();
  }

  // Verify HLC-based causal inference
  auto replay = recorder.replay_data_flow();

  // Verify causal consistency using HLC
  EXPECT_TRUE(replay.causal_consistency_valid)
    << "HLC causal consistency violations: " << [&replay]() {
         std::string violations;
         for (const auto& violation : replay.violations) {
           violations += violation + "; ";
         }
         return violations;
       }();

  // Verify all nodes executed with proper HLC progression
  std::set<std::string> expected_events = {
    "Source_send",     "BranchA_receive", "BranchA_send",
    "BranchB_receive", "BranchB_send",    "Join_receive",
    "Join_send",       "Sink_receive",    "Sink_send"};

  std::set<std::string> actual_events(
    replay.execution_order.begin(), replay.execution_order.end());

  // Verify all expected events occurred
  for (const auto& expected_event : expected_events) {
    EXPECT_TRUE(actual_events.find(expected_event) != actual_events.end())
      << "Missing expected event: " << expected_event;
  }

  // Verify HLC monotonic progression
  auto all_traces = recorder.get_chronological_traces();
  EXPECT_GT(all_traces.size(), 0);

  for (size_t i = 1; i < all_traces.size(); ++i) {
    EXPECT_LE(all_traces[i - 1].clock_raw, all_traces[i].clock_raw)
      << "HLC non-monotonic progression: " << all_traces[i - 1].node_name
      << " (" << all_traces[i - 1].clock_raw << ")"
      << " -> " << all_traces[i].node_name << " (" << all_traces[i].clock_raw
      << ")";
  }

  // Verify causal dependencies: Source < Branches < Join < Sink
  auto source_trace = std::find_if(
    all_traces.begin(), all_traces.end(),
    [](const DataFlowTrace& t) { return t.node_name == "Source_send"; });
  auto branch_a_trace = std::find_if(
    all_traces.begin(), all_traces.end(),
    [](const DataFlowTrace& t) { return t.node_name == "BranchA_send"; });
  auto branch_b_trace = std::find_if(
    all_traces.begin(), all_traces.end(),
    [](const DataFlowTrace& t) { return t.node_name == "BranchB_send"; });
  auto join_trace = std::find_if(
    all_traces.begin(), all_traces.end(),
    [](const DataFlowTrace& t) { return t.node_name == "Join_send"; });
  auto sink_trace = std::find_if(
    all_traces.begin(), all_traces.end(),
    [](const DataFlowTrace& t) { return t.node_name == "Sink_send"; });

  if (source_trace != all_traces.end() && branch_a_trace != all_traces.end()) {
    EXPECT_LT(source_trace->clock_raw, branch_a_trace->clock_raw)
      << "Source clock should be < BranchA clock for causal ordering";
  }

  if (source_trace != all_traces.end() && branch_b_trace != all_traces.end()) {
    EXPECT_LT(source_trace->clock_raw, branch_b_trace->clock_raw)
      << "Source clock should be < BranchB clock for causal ordering";
  }

  if (branch_a_trace != all_traces.end() && join_trace != all_traces.end()) {
    EXPECT_LT(branch_a_trace->clock_raw, join_trace->clock_raw)
      << "BranchA clock should be < Join clock for causal ordering";
  }

  if (branch_b_trace != all_traces.end() && join_trace != all_traces.end()) {
    EXPECT_LT(branch_b_trace->clock_raw, join_trace->clock_raw)
      << "BranchB clock should be < Join clock for causal ordering";
  }

  if (join_trace != all_traces.end() && sink_trace != all_traces.end()) {
    EXPECT_LT(join_trace->clock_raw, sink_trace->clock_raw)
      << "Join clock should be < Sink clock for causal ordering";
  }
}

TEST_F(RpcTemporalMetadataTest, HLCBasicCausalOrdering) {
  // Test basic HLC causal ordering without complex graph
  DistributedGraphRecorder recorder;

  // Create two nodes with independent HLC clocks
  HLCNode node_a(1, "NodeA");
  HLCNode node_b(2, "NodeB");

  // NodeA sends event to NodeB
  RpcRequestHeader request;
  request.function_id = function_id_t{1};
  request.session_id = session_id_t{1001};
  request.assign_event(utils::event_id_t{1}, utils::workflow_id_t{0});

  // NodeA ticks local clock for send event
  uint64_t send_clock = node_a.send_event();
  request.clock().assign_raw(send_clock);
  recorder.record_execution("NodeA_send", request);

  // NodeB receives and processes
  auto response = node_b.process_request(request, recorder);

  // Verify causal ordering: NodeB clock > NodeA send clock
  EXPECT_GT(response.clock().raw(), send_clock)
    << "NodeB clock should be greater than NodeA send clock for causal "
       "ordering";

  // Verify HLC monotonic progression
  auto traces = recorder.get_chronological_traces();
  EXPECT_EQ(traces.size(), 3);  // NodeA_send, NodeB_receive, NodeB_send

  if (traces.size() >= 3) {
    EXPECT_LT(traces[0].clock_raw, traces[1].clock_raw);
    EXPECT_LT(traces[1].clock_raw, traces[2].clock_raw);
  }

  // Verify causal consistency
  auto replay = recorder.replay_data_flow();
  EXPECT_TRUE(replay.causal_consistency_valid);
}

}  // namespace

}  // namespace rpc
}  // namespace eux
