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

#pragma once

#ifndef UCX_CONNECTION_MANAGER_HPP_
#define UCX_CONNECTION_MANAGER_HPP_

#include <deque>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ucx_context/lock_free_queue.hpp"
#include "ucx_context/ucx_connection.hpp"

namespace eux {
namespace ucxx {

// UCX connection type
using conn_pair_t =
  std::pair<std::uint64_t, std::reference_wrapper<UcxConnection>>;
using conn_opt_t = std::optional<std::reference_wrapper<UcxConnection>>;

// shared_ptr + weak_ptr has additional performance overhead
// so we use unique_ptr + index to manage the connections
class ConnectionManager {
 public:
  // Attention:
  // The id is the connection id, not the index of the connection in the
  // connections_ vector
  using conn_vector_t = std::vector<std::unique_ptr<UcxConnection>>;
  using conn_deque_t = std::deque<std::unique_ptr<UcxConnection>>;
  using conn_list_t = std::list<std::unique_ptr<UcxConnection>>;

  std::uint64_t add_connection(
    const std::uint64_t conn_id, std::unique_ptr<UcxConnection> conn);

  void remove_connection(const std::uint64_t conn_id);

  void remove_connection_to_inactive_map(const std::uint64_t conn_id);

  void move_connection_from_inactive_to_failed_queue(
    const std::uint64_t conn_id);

  void move_connection_from_inactive_to_disconnecting_queue(
    const std::uint64_t conn_id);

  void remove_connection_to_failed_queue(const std::uint64_t conn_id);

  void remove_connection_to_disconnecting_queue(const std::uint64_t conn_id);

  void remove_active_connections_to_disconnecting_queue();

  void remove_failed_connections();

  void remove_disconnected_connections();

  bool is_connection_valid(const std::uint64_t conn_id) const;

  bool is_in_disconnecting(const std::uint64_t conn_id) const;

  conn_opt_t get_connection(const std::uint64_t conn_id) const;

  conn_opt_t get_connection_by_slot(const size_t slot) const;

  const std::map<std::uint64_t, size_t>& get_active_connection_map() const;

  const conn_deque_t& get_failed_connections() const;

  conn_deque_t& get_failed_connections();

  const conn_list_t& get_disconnecting_connections() const;

  void add_connection_to_disconnecting_queue(
    std::unique_ptr<UcxConnection> conn);

 private:
  // Alive connections
  conn_vector_t connections_;
  // UCX failed connection queue
  conn_deque_t failedConns_;
  // UCX disconnecting connection list
  conn_list_t disconnectingConns_;
  std::map<std::uint64_t, size_t> active_map_;    // conn_id -> slot
  std::map<std::uint64_t, size_t> inactive_map_;  // conn_id -> slot
  // Lock-free queue for free slots
  LockFreeQueue<size_t> freeSlots_;

  template <typename MapType>
  std::optional<std::pair<size_t, typename MapType::iterator>>
  find_connection_in_map(std::uint64_t conn_id, MapType& map) const;

  void log_connection_operation(
    const std::string& operation, const UcxConnection& conn) const;

  template <typename TargetContainer, typename MapType>
  void remove_connection_to_container(
    std::uint64_t conn_id, TargetContainer& target_container, MapType& map);
};

}  // namespace ucxx
}  // namespace eux

#endif  // UCX_CONNECTION_MANAGER_HPP_
