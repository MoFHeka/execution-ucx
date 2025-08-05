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

#ifndef UCX_CONNECTION_HPP_
#define UCX_CONNECTION_HPP_

#include <ucp/api/ucp.h>
#include <ucs/datastruct/list.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "ucx_context/ucx_context_def.h"
#include "ucx_context/ucx_context_logger.hpp"

namespace stdexe_ucx_runtime {

/*
TODO(He Jia): For now only support Active Message.
Please add Tag Message support in the future.
*/

constexpr const std::uint32_t MAX_LOG_PREFIX_SIZE = 64;

constexpr const uint64_t CLIENT_ID_UNDEFINED = 0;

constexpr const std::uint32_t DEFAULT_AM_MSG_ID = 0;

// description of arrived AM message
// The desc is the data pointer from callback function not the receiving buffer
struct UcxAmDesc {
  UcxAmDesc(
    void* header, size_t header_length, void* desc, size_t data_length,
    const uint64_t conn_id, const ucp_am_recv_attr_t recv_attr)
    : header(header),
      header_length(header_length),
      desc(desc),
      data_length(data_length),
      conn_id(conn_id),
      recv_attr(recv_attr) {}
  void* header;
  size_t header_length;
  void* desc;
  size_t data_length;
  const uint64_t conn_id;
  const ucp_am_recv_attr_t recv_attr;
};

// UCX request structure.
enum class UcxRequestType { Send, Recv, Last, Unknown = Last };

// forward declaration
class UcxConnection;

// UCX callback for send/receive completion
class UcxCallback {
 public:
  virtual ~UcxCallback();
  virtual void operator()(ucs_status_t status) = 0;
  virtual void handle_connection_error(
    ucs_status_t status, UcxConnection& conn) {
    throw std::runtime_error(
      "handle_connection_error with UcxConnection not implemented");
  }
  virtual void handle_connection_error(
    ucs_status_t status, std::uint64_t conn_id) {
    throw std::runtime_error(
      "handle_connection_error with conn_id not implemented");
  }
  virtual void mark_inactive(std::uint64_t conn_id) {
    throw std::runtime_error("mark_inactive not implemented");
  }
  virtual void mark_disconnecting_from_inactive(std::uint64_t conn_id) {
    throw std::runtime_error(
      "mark_disconnecting_from_inactive not implemented");
  }
  virtual void mark_failed_from_inactive(std::uint64_t conn_id) {
    throw std::runtime_error("mark_failed_from_inactive not implemented");
  }
  uint64_t get_client_id() const { return client_id_; }

 protected:
  uint64_t client_id_ = CLIENT_ID_UNDEFINED;
};

// Empty callback singleton
class EmptyCallback : public UcxCallback {
 public:
  virtual void operator()(ucs_status_t status);
  void handle_connection_error(
    ucs_status_t status, UcxConnection& conn) override {}
  void handle_connection_error(
    ucs_status_t status, std::uint64_t conn_id) override {}
  void mark_inactive(std::uint64_t conn_id) override {}
  void mark_disconnecting_from_inactive(std::uint64_t conn_id) override {}
  void mark_failed_from_inactive(std::uint64_t conn_id) override {}

  static EmptyCallback* get();

  static std::unique_ptr<EmptyCallback> get_unique();
};

// CqeEntryCallback for rigister operation in completion queue
class CqeEntryCallback : public UcxCallback {
 public:
  explicit CqeEntryCallback(
    std::uintptr_t user_data, std::function<ucx_am_cqe&()> get_entry_fn)
    : user_data_(user_data), get_entry_fn_(get_entry_fn) {}

  virtual void operator()(ucs_status_t status);

 private:
  std::uintptr_t user_data_;
  std::function<ucx_am_cqe&()> get_entry_fn_;
};

bool is_rdma_transport_available(ucp_ep_h ep);

bool is_zcopy_available(ucp_context_h context, size_t msg_len);

struct UcxRequest {
  std::unique_ptr<UcxCallback> callback = std::unique_ptr<UcxCallback>(nullptr);
  std::optional<std::reference_wrapper<UcxConnection>> conn = std::nullopt;
  ucs_status_t status = UCS_ERR_LAST;
  std::uintptr_t conn_id = std::uintptr_t(nullptr);
  UcxRequestType type = UcxRequestType::Unknown;
  ucs_list_link_t pos;
  std::string_view what;
};

// Global connection ID generator
class ConnectionIdGenerator {
 public:
  static std::uint64_t generate() {
    static std::atomic<std::uint64_t> next_id_{1};
    return next_id_.fetch_add(1, std::memory_order_relaxed);
  }

 private:
  ConnectionIdGenerator() = delete;
  ~ConnectionIdGenerator() = delete;
};

// Default connection ID generation function
static std::uint64_t default_connection_id_generator() {
  return ConnectionIdGenerator::generate();
}

class UcxConnection : public std::enable_shared_from_this<UcxConnection> {
 public:
  /**
   * @brief Constructs a new UcxConnection object
   *
   * @param worker The UCX worker handle
   * @param handle_err_cb Callback for handling connection errors
   * @param get_conn_id Function to generate connection ID (optional)
   */
  explicit UcxConnection(
    ucp_worker_h worker,
    std::unique_ptr<UcxCallback>
      handle_err_cb,
    std::function<std::uint64_t()> get_conn_id =
      default_connection_id_generator);

  /**
   * @brief Destructor for UcxConnection
   *
   * Ensures proper cleanup of UCX resources
   */
  ~UcxConnection();

  /**
   * @brief Establishes a connection to a remote endpoint
   *
   * @param src_saddr Source socket address
   * @param dst_saddr Destination socket address
   * @param addrlen Length of the socket address structure
   * @param callback Callback to be called when connection is established
   */
  void connect(
    const struct sockaddr* src_saddr, const struct sockaddr* dst_saddr,
    socklen_t addrlen, std::unique_ptr<UcxCallback> callback);

  /**
   * @brief Accepts an incoming connection request
   *
   * @param conn_req The connection request handle
   * @param callback Callback to be called when connection is accepted
   */
  void accept(
    ucp_conn_request_h conn_req, std::unique_ptr<UcxCallback> callback);

  /**
   * @brief Disconnects the connection
   *
   * The connection will be destroyed automatically after callback is called.
   *
   * @param callback Callback to be called when disconnection is complete
   */
  void disconnect(
    std::unique_ptr<UcxCallback> callback = EmptyCallback::get_unique());

  /**
   * @brief Directly disconnects the connection without waiting for completion
   */
  void disconnect_direct();

  /**
   * @brief Progresses the disconnection process
   *
   * @param callback Callback to be called when disconnection is complete
   * @return true if disconnection is complete, false otherwise
   */
  bool disconnect_progress(
    std::unique_ptr<UcxCallback> callback = EmptyCallback::get_unique());

  /**
   * @brief Sends active message data
   *
   * @param header Data header buffer
   * @param header_length Length of header
   * @param buffer Data buffer to send
   * @param length Length of data
   * @param memh Memory handle (optional)
   * @param callback Callback to be called when send is complete
   * @return Tuple containing status and request pointer
   */
  std::tuple<ucs_status_t, UcxRequest*> send_am_data(
    const void* header, size_t header_length, const void* buffer, size_t length,
    ucp_mem_h memh,
    std::unique_ptr<UcxCallback> callback = EmptyCallback::get_unique());

  /**
   * @brief Receives active message data
   *
   * @param buffer Buffer to receive data into
   * @param length Length of buffer
   * @param memh Memory handle (optional)
   * @param data_desc Active message descriptor
   * @param callback Callback to be called when receive is complete
   * @return Tuple containing status and request pointer
   */
  std::tuple<ucs_status_t, UcxRequest*> recv_am_data(
    void* buffer, size_t length, ucp_mem_h memh, const UcxAmDesc&& data_desc,
    std::unique_ptr<UcxCallback> callback = EmptyCallback::get_unique());

  /**
   * @brief Cancels a specific request
   *
   * @param request The request to cancel
   */
  void cancel_request(UcxRequest* request);

  /**
   * @brief Cancels all send requests
   */
  void cancel_send();

  /**
   * @brief Cancels all receive requests
   */
  void cancel_recv();

  /**
   * @brief Cancels all pending requests
   */
  void cancel_all();

  /**
   * @brief Gets the connection ID
   *
   * @return The connection ID
   */
  uint64_t id() const { return conn_id_; }

  /**
   * @brief Gets the current UCX status
   *
   * @return The current UCX status
   */
  ucs_status_t ucx_status() const { return ucx_status_; }

  /**
   * @brief Gets the log prefix for this connection
   *
   * @return The log prefix string
   */
  const char* get_log_prefix() const { return log_prefix_; }

  /**
   * @brief Checks if zcopy should be used
   *
   * @param msg_len Message length
   * @return true if zcopy should be used, false otherwise
   */
  bool should_use_zcopy(size_t msg_len) const {
    return msg_len >= zcopy_thresh_ || is_rdma_transport_available_;
  }

  /**
   * @brief Checks if the connection is established
   *
   * @return true if connection is established, false otherwise
   */
  bool is_established() const { return establish_cb_ == nullptr; }

  /**
   * @brief Gets the peer name
   *
   * @return The peer name string
   */
  const std::string& get_peer_name() const { return remote_address_; }

  /**
   * @brief Checks if the connection is in disconnecting state
   *
   * @return true if connection is disconnecting, false otherwise
   */
  bool is_disconnecting() const { return disconnect_cb_ != nullptr; }

  /**
   * @brief Handles connection errors
   *
   * @param status The error status
   */
  void handle_connection_error(ucs_status_t status);

  /**
   * @brief Gets the number of UcxConnection instances
   *
   * @return The number of instances
   */
  static size_t get_num_instances() { return num_instances_; }

  /**
   * @brief Sets the log prefix based on socket address
   *
   * @param saddr Socket address
   * @param addrlen Length of socket address
   */
  void set_log_prefix(const struct sockaddr* saddr, socklen_t addrlen);

  /**
   * @brief Gets the logger instance
   *
   * @return Reference to the logger
   */
  UcxLogger& get_logger() {
    return *UcxLoggerManager::get_instance().get_logger();
  }

  /**
   * @brief Sets the logger instance
   *
   * @param logger Pointer to the logger
   */
  static void set_logger(UcxLogger* logger) {
    UcxLoggerManager::get_instance().set_logger(logger);
  }

  /**
   * @brief Checks if an active message is using rendezvous protocol
   *
   * @param desc Active message descriptor
   * @return true if using rendezvous, false otherwise
   */
  static inline bool ucx_am_is_rndv(const UcxAmDesc& desc) {
    return desc.recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV;
  }

  /**
   * @brief Checks if an active message is using rendezvous protocol
   *
   * @param recv_attr Receive attributes
   * @return true if using rendezvous, false otherwise
   */
  static inline bool ucx_am_is_rndv(const ucp_am_recv_attr_t& recv_attr) {
    return recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV;
  }

  /**
   * @brief Initializes a request
   *
   * @param request Pointer to the request
   */
  static void request_init(void* request);

  /**
   * @brief Resets a request to its initial state
   *
   * @param r Pointer to the request
   */
  static void request_reset(UcxRequest* r);

  /**
   * @brief Releases a request
   *
   * @param r Pointer to the request
   */
  static void request_release(void* r);

  /**
   * @brief Converts socket address to string
   *
   * @param saddr Socket address
   * @param addrlen Length of socket address
   * @return String representation of the address
   */
  static const std::string sockaddr_str(
    const struct sockaddr* saddr, size_t addrlen);

 private:
  static void common_request_callback(void* request, ucs_status_t status);

  static void am_data_recv_callback(
    void* request, ucs_status_t status, size_t length, void* user_data);

  static void error_callback(void* arg, ucp_ep_h ep, ucs_status_t status);

  void print_addresses();

  void connect_common(
    ucp_ep_params_t& ep_params, std::unique_ptr<UcxCallback> callback);

  void connect_am(std::unique_ptr<UcxCallback> callback);

  void established(ucs_status_t status);

  void request_started(UcxRequest* r);

  void request_completed(UcxRequest* r);

  void ep_close(enum ucp_ep_close_mode mode);

  std::tuple<ucs_status_t, UcxRequest*> process_request(
    std::string_view what, ucs_status_ptr_t ptr_status,
    std::unique_ptr<UcxCallback> callback,
    UcxRequestType type = UcxRequestType::Unknown);

  static void invoke_callback(
    std::unique_ptr<UcxCallback>& callback, ucs_status_t status);

  static std::uint32_t num_instances_;

  ucp_worker_h worker_ = nullptr;
  std::unique_ptr<UcxCallback> handle_err_cb_ = nullptr;
  std::unique_ptr<UcxCallback> establish_cb_ = nullptr;
  std::unique_ptr<UcxCallback> disconnect_cb_ = nullptr;
  std::uintptr_t conn_id_ = reinterpret_cast<uintptr_t>(
    nullptr);  // use ucp_ep_h pointer address as conn_id
  std::uintptr_t remote_conn_id_;
  char log_prefix_[MAX_LOG_PREFIX_SIZE];
  ucp_ep_h ep_ = nullptr;
  std::string remote_address_;
  void* close_request_;  // it's a ucs_status_ptr_t or UcxRequest*
  ucs_list_link_t all_requests_;
  ucs_status_t ucx_status_;
  // Fast judge if zcopy is available
  size_t zcopy_thresh_ = 0;
  bool is_rdma_transport_available_ = false;
};

}  // namespace stdexe_ucx_runtime

#endif  // UCX_CONNECTION_HPP_
