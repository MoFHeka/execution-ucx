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

#include "ucx_context/ucx_connection_manager.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <typeinfo>
#include <utility>

namespace stdexe_ucx_runtime {

////////////////////////////////////////////////////////////
// Connection manager

template <typename MapType>
std::optional<std::pair<size_t, typename MapType::iterator>>
ConnectionManager::find_connection_in_map(
  const std::uint64_t conn_id, MapType& map) const {
  auto it = map.find(conn_id);
  if (it != map.end()) {
    return std::make_pair(it->second, it);
  }
  return std::nullopt;
}

void ConnectionManager::log_connection_operation(
  const std::string& operation, const UcxConnection& conn) const {
  UCX_CTX_TRACE << operation << " connection " << conn.get_log_prefix()
                << std::endl;
}

template <typename TargetContainer, typename MapType>
void ConnectionManager::remove_connection_to_container(
  const std::uint64_t conn_id, TargetContainer& target_container,
  MapType& map) {
  auto result = find_connection_in_map(conn_id, map);
  if (result) {
    auto [slot, it] = *result;
    log_connection_operation(
      std::string("Moving from ") + std::string(typeid(TargetContainer).name())
        + std::string(" to ") + std::string(typeid(MapType).name()),
      *connections_[slot]);
    target_container.push_back(std::move(connections_[slot]));
    map.erase(it);
    freeSlots_.push(slot);
  }
}

uint64_t ConnectionManager::add_connection(
  const std::uint64_t conn_id, std::unique_ptr<UcxConnection> conn) {
  // Get a free slot from the queue
  size_t slot;
  if (!freeSlots_.pop(slot)) {
    // If no free slots, add to the end of connections_
    slot = connections_.size();
    connections_.push_back(std::move(conn));
  } else {
    // Use the free slot
    connections_[slot] = std::move(conn);
  }

  // Add to the map
  active_map_[conn_id] = slot;
  return conn_id;
}

void ConnectionManager::remove_connection(const std::uint64_t conn_id) {
  auto result = find_connection_in_map(conn_id, active_map_);
  if (result) {
    auto [slot, it] = *result;
    log_connection_operation("Removing", *connections_[slot]);
    // Remove from the map
    active_map_.erase(it);
    // Clear the connection
    connections_[slot].reset();
    // Add the slot back to the free queue
    freeSlots_.push(slot);
  }
}

void ConnectionManager::remove_connection_to_inactive_map(
  const std::uint64_t conn_id) {
  auto result = find_connection_in_map(conn_id, active_map_);
  if (result) {
    auto [slot, it] = *result;
    log_connection_operation(
      "Moving from active to inactive", *connections_[slot]);
    active_map_.erase(it);
    inactive_map_[conn_id] = slot;
  }
}

void ConnectionManager::move_connection_from_inactive_to_failed_queue(
  const std::uint64_t conn_id) {
  remove_connection_to_container(conn_id, failedConns_, inactive_map_);
}

void ConnectionManager::move_connection_from_inactive_to_disconnecting_queue(
  const std::uint64_t conn_id) {
  remove_connection_to_container(conn_id, disconnectingConns_, inactive_map_);
}

void ConnectionManager::remove_connection_to_failed_queue(
  const std::uint64_t conn_id) {
  remove_connection_to_container(conn_id, failedConns_, active_map_);
}

void ConnectionManager::remove_connection_to_disconnecting_queue(
  const std::uint64_t conn_id) {
  remove_connection_to_container(conn_id, disconnectingConns_, active_map_);
}

bool ConnectionManager::is_connection_valid(const std::uint64_t conn_id) const {
  return active_map_.find(conn_id) != active_map_.end();
}

bool ConnectionManager::is_in_disconnecting(const std::uint64_t conn_id) const {
  return std::find_if(
           disconnectingConns_.begin(), disconnectingConns_.end(),
           [conn_id](const auto& conn) { return conn->id() == conn_id; })
         != disconnectingConns_.end();
}

conn_opt_t ConnectionManager::get_connection(
  const std::uint64_t conn_id) const {
  auto it = active_map_.find(conn_id);
  if (it == active_map_.end()) {
    return std::nullopt;
  }
  return std::ref(*connections_[it->second]);
}

conn_opt_t ConnectionManager::get_connection_by_slot(const size_t slot) const {
  if (slot >= connections_.size()) {
    return std::nullopt;
  }
  return std::ref(*connections_[slot]);
}

ConnectionManager::conn_deque_t& ConnectionManager::get_failed_connections() {
  return failedConns_;
}

const ConnectionManager::conn_deque_t&
ConnectionManager::get_failed_connections() const {
  return failedConns_;
}

const std::map<std::uint64_t, size_t>&
ConnectionManager::get_active_connection_map() const {
  return active_map_;
}

const ConnectionManager::conn_list_t&
ConnectionManager::get_disconnecting_connections() const {
  return disconnectingConns_;
}

void ConnectionManager::remove_active_connections_to_disconnecting_queue() {
  auto it = active_map_.begin();
  while (it != active_map_.end()) {
    log_connection_operation(
      "Moving from active to disconnecting", *connections_[it->second]);
    disconnectingConns_.push_back(std::move(connections_[it->second]));
    it = active_map_.erase(it);
  }
}

void ConnectionManager::remove_failed_connections() {
  auto it = failedConns_.begin();
  while (it != failedConns_.end()) {
    std::unique_ptr<UcxConnection>& conn = *it;
    if (conn->disconnect_progress()) {
      it = failedConns_.erase(it);
    } else {
      ++it;
    }
  }
}

void ConnectionManager::add_connection_to_disconnecting_queue(
  std::unique_ptr<UcxConnection> conn) {
  disconnectingConns_.push_back(std::move(conn));
}

void ConnectionManager::remove_disconnected_connections() {
  auto it = disconnectingConns_.begin();
  while (it != disconnectingConns_.end()) {
    std::unique_ptr<UcxConnection>& conn = *it;
    if (conn->disconnect_progress()) {
      it = disconnectingConns_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace stdexe_ucx_runtime
