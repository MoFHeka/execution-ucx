/*Copyright (c) Facebook, Inc. and its affiliates.
Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.

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

#ifndef UCX_CONTEXT_UCX_AM_CONTEXT_UCX_AM_CONTEXT_HPP_
#define UCX_CONTEXT_UCX_AM_CONTEXT_UCX_AM_CONTEXT_HPP_

#include <netinet/in.h>
#include <ucp/api/ucp.h>
#include <ucs/datastruct/list.h>

#include <cassert>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// stdexe include
#include <unifex/defer.hpp>
#include <unifex/detail/atomic_intrusive_queue.hpp>
#include <unifex/detail/intrusive_heap.hpp>
#include <unifex/detail/intrusive_queue.hpp>
#include <unifex/get_stop_token.hpp>
#include <unifex/inplace_stop_token.hpp>
#include <unifex/just_done.hpp>
#include <unifex/just_error.hpp>
#include <unifex/let_value_with.hpp>
#include <unifex/linux/monotonic_clock.hpp>
#include <unifex/manual_lifetime.hpp>
#include <unifex/receiver_concepts.hpp>
#include <unifex/sender_concepts.hpp>
#include <unifex/socket_concepts.hpp>
#include <unifex/span.hpp>
#include <unifex/stop_token_concepts.hpp>

#include "ucx_context/lock_free_queue.hpp"
#include "ucx_context/ucx_connection.hpp"
#include "ucx_context/ucx_connection_manager.hpp"
#include "ucx_context/ucx_context_concept.hpp"
#include "ucx_context/ucx_context_data.hpp"
#include "ucx_context/ucx_context_def.h"
#include "ucx_context/ucx_context_logger.hpp"
#include "ucx_context/ucx_device_context.hpp"
#include "ucx_context/ucx_memory_resource.hpp"
#include "ucx_context/ucx_status.hpp"

// TODO(He Jia): Discuss whether it should be unified into Google C++ style
// Although according to the project's code style guidelines, class and class
// member function declarations should use PascalCase, and class member
// variables should use snake_case. Considering that ucx_am_context is based on
// libunifex, and libunifex uses a naming style similar to the C++ standard
// template library's snake_case, this project maintains consistency with the
// standard template style only within ucx_am_context.

namespace stdexe_ucx_runtime {

using unifex::atomic_intrusive_queue;
using unifex::blocking_kind;
using unifex::get_stop_token;
using unifex::inplace_stop_token;
using unifex::intrusive_heap;
using unifex::intrusive_queue;
using unifex::is_stop_never_possible_v;
using unifex::manual_lifetime;
using unifex::remove_cvref_t;
using unifex::scope_guard;
using unifex::stop_token_type_t;
using unifex::linuxos::monotonic_clock;

static constexpr const std::uint32_t kCompletionQueueEntryCount = 256;

/**
 * @class active_message_bundle
 * @brief A class that bundles UCX active message data with connection
 * information.
 */
class active_message_bundle {
 public:
  /**
   * @brief Construct a new active message bundle object.
   * @param data The UCX active message data.
   * @param conn The UCX connection associated with the message.
   */
  active_message_bundle(UcxAmData&& data, const UcxConnection& conn)
    : data_(std::move(data)), conn_(conn) {}

  active_message_bundle(const active_message_bundle&) = delete;
  active_message_bundle& operator=(const active_message_bundle&) = delete;
  active_message_bundle(active_message_bundle&&) = default;
  active_message_bundle& operator=(active_message_bundle&&) = default;

  /**
   * @brief Get a constant reference to the data bundle.
   * @return const UcxAmData& A constant reference to the UCX AM data.
   */
  const UcxAmData& data() const { return data_; }
  /**
   * @brief Get a rvalue reference of the data bundle.
   * @return UcxAmData&& A rvalue reference of the UCX AM data.
   */
  UcxAmData&& move_data() { return std::move(data_); }

  /**
   * @brief Get a constant reference to the data bundle.
   * @return const UcxAmData& A constant reference to the UCX AM data.
   */
  const ucx_am_data& get_raw_data() const { return *data_.get(); }

  /**
   * @brief Get a constant reference to the connection info.
   * @return const UcxConnection& A constant reference to the UCX connection.
   */
  const UcxConnection& connection() const { return conn_; }

 private:
  UcxAmData data_;             // Reference to the data bundle
  const UcxConnection& conn_;  // Connection info
};

class active_message_iovec_bundle {
 public:
  /**
   * @brief Construct a new active message bundle object.
   * @param data The UCX active message data.
   * @param conn The UCX connection associated with the message.
   */
  active_message_iovec_bundle(UcxAmIovec&& iovec, const UcxConnection& conn)
    : iovec_(std::move(iovec)), conn_(conn) {}

  active_message_iovec_bundle(const active_message_iovec_bundle&) = delete;
  active_message_iovec_bundle& operator=(const active_message_iovec_bundle&) =
    delete;
  active_message_iovec_bundle(active_message_iovec_bundle&&) = default;
  active_message_iovec_bundle& operator=(active_message_iovec_bundle&&) =
    default;

  /**
   * @brief Get a constant reference to the data bundle.
   * @return const UcxAmIovec& A constant reference to the UCX AM data.
   */
  const UcxAmIovec& data() const { return iovec_; }
  /**
   * @brief Get a rvalue reference of the data bundle.
   * @return UcxAmIovec&& A rvalue reference of the UCX AM data.
   */
  UcxAmIovec&& move_data() { return std::move(iovec_); }

  /**
   * @brief Get a constant reference to the connection info.
   * @return const UcxConnection& A constant reference to the UCX connection.
   */
  const UcxConnection& connection() const { return conn_; }

 private:
  UcxAmIovec iovec_;           // Reference to the data bundle
  const UcxConnection& conn_;  // Connection info
};

template <
  typename BufferType,
  std::enable_if_t<is_receive_buffer_parameter_v<BufferType>, int> = 0>
class active_message_buffer_bundle_t {
 public:
  active_message_buffer_bundle_t(
    BufferType&& buffer_vec, const UcxConnection& conn)
    : buffer_vec_(std::move(buffer_vec)), conn_(conn) {}

  active_message_buffer_bundle_t(const active_message_buffer_bundle_t&) =
    delete;
  active_message_buffer_bundle_t& operator=(
    const active_message_buffer_bundle_t&) = delete;
  active_message_buffer_bundle_t(active_message_buffer_bundle_t&&) = default;
  active_message_buffer_bundle_t& operator=(active_message_buffer_bundle_t&&) =
    default;

  /**
   * @brief Get a constant reference to the data bundle.
   * @return const BufferType& A constant reference to the UCX AM data.
   * @note For now, BufferType is UcxBuffer or std::vector<UcxBuffer>
   */
  const BufferType& data() const { return buffer_vec_; }
  /**
   * @brief Get a rvalue reference of the data bundle.
   * @return BufferType&& A rvalue reference of the UCX AM data.
   * @note For now, BufferType is UcxBuffer or std::vector<UcxBuffer>
   */
  BufferType&& move_data() { return std::move(buffer_vec_); }

  /**
   * @brief Get a constant reference to the connection info.
   * @return const UcxConnection& A constant reference to the UCX connection.
   */
  const UcxConnection& connection() const { return conn_; }

 private:
  BufferType buffer_vec_;      // Reference to the data bundle
  const UcxConnection& conn_;  // Connection info
};
using active_message_buffer_bundle = active_message_buffer_bundle_t<UcxBuffer>;
using active_message_iovec_buffer_bundle =
  active_message_buffer_bundle_t<UcxBufferVec>;

class ucx_am_desc_map {
 public:
  ucx_am_desc_map() { map_.reserve(kCompletionQueueEntryCount); }

  size_t push(UcxAmDesc&& desc) {
    size_t key = next_key_.fetch_add(1, std::memory_order_relaxed);
    map_.emplace(key, std::move(desc));
    return key;
  }

  std::optional<UcxAmDesc> pop(size_t key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return std::nullopt;
    }
    UcxAmDesc desc = std::move(it->second);
    map_.erase(it);
    return std::move(desc);
  }

 private:
  std::unordered_map<size_t, UcxAmDesc> map_;
  std::atomic<size_t> next_key_{0};
};

/**
 * @class ucx_am_context
 * @brief Manages UCX resources and asynchronous operations.
 *
 * This class encapsulates the UCX context, worker, and event loop, providing a
 * high-level interface for sending and receiving active messages, managing
 * connections, and scheduling work.
 */
class ucx_am_context {
 private:
  // To ensure the consistency of code style
  using ucx_am_desc = UcxAmDesc;
  using ucx_callback = UcxCallback;
  using ucx_connection = UcxConnection;
  using ucx_logger = UcxLogger;
  using ucx_logger_manager = UcxLoggerManager;
  using ucx_memory_resource = UcxMemoryResourceManager;
  using ucx_request = UcxRequest;

 public:
  class scheduler;
  class schedule_sender;
  class schedule_at_sender;
  template <typename Duration>
  class schedule_after_sender;
  template <typename Payload>
  class send_sender_t;
  using send_sender = send_sender_t<ucx_am_data>;
  using send_sender_move = send_sender_t<UcxAmData>;
  using send_iovec_sender = send_sender_t<ucx_am_iovec>;
  using send_iovec_sender_move = send_sender_t<UcxAmIovec>;
  class recv_sender;
  class recv_header_sender;
  template <
    typename BufferType,
    std::enable_if_t<is_receive_buffer_parameter_v<BufferType>, int> = 0>
  class recv_buffer_sender_t;
  using recv_buffer_sender = recv_buffer_sender_t<UcxBuffer>;
  using recv_iovec_buffer_sender = recv_buffer_sender_t<UcxBufferVec>;
  class connect_sender;
  class accept_sender;
  class accept_connection;
  class dispatch_connection_error_sender;
  friend class UcxConnection;
  friend class active_message_bundle;
  friend class active_message_iovec_bundle;
  friend class active_message_buffer_bundle_t<UcxBuffer>;
  friend class active_message_buffer_bundle_t<UcxBufferVec>;
  friend class ConnectionManager;
  friend class memory_resource;

  // Time point and duration
  using time_point = monotonic_clock::time_point;
  using time_duration = monotonic_clock::duration;

  /**
   * @brief Constructs a ucx_am_context and initializes a new UCP context.
   *
   * @param memoryResource The memory resource manager to use for allocations.
   * @param ucxContextName A name for the UCX context, used for debugging.
   * @param connectionTimeout The duration to wait before a connection attempt
   * times out.
   * @param deviceContext An optional device context for hardware-specific
   * operations.
   * @param clientId A unique identifier for this client.
   * @param connectionHandleError If true, connection will call
   * ucx_handle_err_callback when error occurs. But it may make UCX TLS only use
   * TCP.
   */
  ucx_am_context(
    ucx_memory_resource& memoryResource,
    const std::string_view ucxContextName = "default",
    const time_duration connectionTimeout = std::chrono::seconds(30),
    const bool connectionHandleError = false,
    const uint64_t clientId = CLIENT_ID_UNDEFINED,
    const std::unique_ptr<UcxAutoDeviceContext> deviceContext = nullptr);
  /**
   * @brief Constructs a ucx_am_context with an existing UCP context.
   *
   * @param memoryResource The memory resource manager to use for allocations.
   * @param ucpContext An existing UCP context handle.
   * @param connectionTimeout The duration to wait before a connection attempt
   * times out.
   * @param deviceContext An optional device context for hardware-specific
   * operations.
   * @param clientId A unique identifier for this client.
   * @param connectionHandleError If true, connection will call
   * ucx_handle_err_callback when error occurs. But it may make UCX TLS only use
   * TCP.
   */
  ucx_am_context(
    ucx_memory_resource& memoryResource,
    const ucp_context_h ucpContext,
    const time_duration connectionTimeout = std::chrono::seconds(30),
    const bool connectionHandleError = false,
    const uint64_t clientId = CLIENT_ID_UNDEFINED,
    const std::unique_ptr<UcxAutoDeviceContext> deviceContext = nullptr);
  /**
   * @brief Destroys the ucx_am_context, cleaning up UCX resources.
   */
  ~ucx_am_context();

  /**
   * @brief Initializes a UCP context.
   * @param name A name for the context.
   * @param ucpContext[out] The created UCP context handle.
   * @param mtWorkersShared If true, use shared context for multiple UCX
   * workers.
   * @param printConfig If true, print the UCP configuration.
   * @return std::error_code The status of the operation.
   */
  static std::error_code init_ucp_context(
    std::string_view name, ucp_context_h* ucpContext, bool mtWorkersShared,
    bool printConfig);
  /**
   * @brief Destroys a UCP context.
   * @param ucpContext The UCP context handle to destroy.
   */
  static void destroy_ucp_context(ucp_context_h ucpContext);
  /**
   * @brief Checks if the UCP context was provided externally.
   * @return bool True if the context is external, false otherwise.
   */
  bool is_ucp_context_external() const noexcept;
  /**
   * @brief Initializes a UCP worker for this context.
   * @return std::error_code The status of the operation.
   */
  std::error_code init_ucp_worker();
  /**
   * @brief Gets the address of the UCP worker.
   * @param address_buffer[out] The address of the UCP worker.
   * @return std::error_code The status of the operation.
   */
  std::error_code get_ucp_address(std::vector<std::byte>& address_buffer);

  /**
   * @brief Initializes the UCX AM context, including worker and AM handlers.
   * @param name The name for the UCP context if it needs to be created.
   * @param forceInit If true, force re-initialization even if already
   * initialized.
   * @return std::error_code The status of the operation.
   */
  std::error_code init(std::string_view name, bool forceInit = false);

  /**
   * @brief Runs the event loop for this context.
   *
   * This function blocks and processes I/O events and scheduled tasks until the
   * provided stop token is triggered.
   *
   * @tparam StopToken The type of the stop token.
   * @param stopToken A token that can be used to stop the event loop.
   */
  template <typename StopToken>
  void run(StopToken stopToken);

  /**
   * @brief Get a scheduler for this context.
   *
   * The scheduler can be used to schedule work on the context's event loop
   * from other threads.
   *
   * @return scheduler An object that can be used to schedule work.
   */
  scheduler get_scheduler() noexcept;

  /**
   * @brief Registers a memory region with the UCX worker.
   *
   * This function registers a memory region so that it can be used for UCX
   * communication. The memory handle (memh) will be set on success.
   *
   * @param ptr Pointer to the memory region to register.
   * @param size Size of the memory region in bytes.
   * @param memh Reference to a ucp_mem_h handle that will be set to the
   * registered memory handle.
   * @return std::error_code The status of the registration operation.
   */
  std::error_code register_ucx_memory(void* ptr, size_t size, ucp_mem_h& memh);

  /**
   * @brief Unregisters a previously registered UCX memory region.
   *
   * This function releases the UCX memory handle and unregisters the memory
   * region.
   *
   * @param memh Reference to the ucp_mem_h handle to unregister.
   */
  void unregister_ucx_memory(ucp_mem_h& memh);

  /**
   * @brief Checks if handle error in connection is enabled.
   *
   * @return true if handle error in connection is enabled, false otherwise.
   */
  bool is_connection_handle_error() const noexcept {
    return connectionHandleError_;
  }

 protected:
  class ucx_accept_callback;
  class ucx_connect_callback;
  class ucx_disconnect_callback;
  class ucx_handle_err_callback;

  enum class wait_status_t {
    WAIT_STATUS_OK,
    WAIT_STATUS_FAILED,
    WAIT_STATUS_TIMED_OUT
  };

  std::uint64_t get_client_id() noexcept;

  std::atomic<std::uint64_t> conn_id_it_{0};

 private:
  ////////
  // Context relative struct

  struct operation_base {
    operation_base() noexcept {}
    std::add_pointer_t<operation_base> next_ = nullptr;
    std::add_pointer_t<void(operation_base* ptr) noexcept> execute_ = nullptr;
  };

  struct completion_base : operation_base {
    int result_;
  };

  struct stop_operation : operation_base {
    stop_operation() noexcept {
      this->execute_ = [](operation_base* op) noexcept {
        static_cast<stop_operation*>(op)->shouldStop_ = true;
      };
    }
    bool shouldStop_ = false;
  };

  struct schedule_at_operation : operation_base {
    explicit schedule_at_operation(
      ucx_am_context& context,
      const time_point& dueTime,
      bool canBeCancelled) noexcept
      : context_(context), dueTime_(dueTime), canBeCancelled_(canBeCancelled) {}

    schedule_at_operation* timerNext_;
    schedule_at_operation* timerPrev_;
    ucx_am_context& context_;
    time_point dueTime_;
    bool canBeCancelled_;
    std::optional<std::reference_wrapper<ucx_am_cqe>> cq_entry_ref_;

    static constexpr std::uint32_t timer_elapsed_flag = 1;
    static constexpr std::uint32_t cancel_pending_flag = 2;
    std::atomic<std::uint32_t> state_ = 0;
  };

  using operation_queue =
    intrusive_queue<operation_base, &operation_base::next_>;

  using timer_heap = intrusive_heap<
    schedule_at_operation, &schedule_at_operation::timerNext_,
    &schedule_at_operation::timerPrev_, time_point,
    &schedule_at_operation::dueTime_>;

  ////////
  // UCX relative struct

  // UCX arrived endpoint connection request
  struct ep_conn_request {
    ucp_conn_request_h conn_request;
    time_point arrival_time;
  };

  ////////
  // UCX relative function

  // Init UCX context
  std::error_code init_with_internal_ucp_context(bool forceInit = false);
  void set_setimer_action_event();

  // Accept a new connection
  ucs_status_t listen(const std::unique_ptr<sockaddr>& socket, size_t addrlen);

  ucs_status_t is_listener_valid() const noexcept;

  static void connect_request_callback(ucp_conn_request_h conn_req, void* arg);

  void dispatch_connection_accepted(std::unique_ptr<ucx_connection> conn);

  void handle_connection_error(std::uint64_t conn_id);

  // Create a new connection
  std::uint64_t create_new_connection(
    const struct sockaddr* src_saddr,
    const struct sockaddr* dst_saddr,
    socklen_t addrlen);

  std::uint64_t create_new_connection(const ucp_address_t* ucp_address);

  // Helper method to handle common connection creation logic
  inline std::uint64_t handle_new_connection(
    std::unique_ptr<ucx_connection> new_conn);

  std::uint64_t recreate_connection_from_failed(conn_pair_t conn);

  // AM receive
  static ucs_status_t am_recv_callback(
    void* arg, const void* header, size_t header_length, void* data,
    size_t data_length, const ucp_am_recv_param_t* param);

  // Logger function
  ucx_logger& get_logger() {
    return *ucx_logger_manager::get_instance().get_logger();
  }

  static void set_logger(ucx_logger* logger) {
    ucx_logger_manager::get_instance().set_logger(logger);
  }

  // Initialize the AM
  void set_am_handler(ucp_am_recv_callback_t cb, void* arg);

  // UCX Context Loop function
  unsigned progress_worker_event();

  std::tuple<ucs_status_t, std::uint64_t> progress_conn_request(
    const ep_conn_request& epConnReq);

  std::vector<std::pair<std::uint64_t, std::error_code>>
  progress_pending_conn_requests();

  void progress_failed_connections();

  void progress_disconnected_connections();

  void wait_disconnected_connections();

  void destroy_connections();

  wait_status_t wait_completion(
    ucs_status_ptr_t statusPtr, std::string_view title,
    time_duration timeout = std::chrono::seconds(300));

  // UCX Clean up function
  void destroy_worker();

  void destroy_listener();

  ////////
  // Context relative function

  bool is_running_on_io_thread() const noexcept;
  void run_impl(const bool& shouldStop);

  void schedule_impl(operation_base* op);
  void schedule_local(operation_base* op) noexcept;
  void schedule_local(operation_queue ops) noexcept;
  void schedule_remote(operation_base* op) noexcept;

  // Schedule some operation to be run when there is next available I/O slots.
  void schedule_pending_io(operation_base* op) noexcept;
  void reschedule_pending_io(operation_base* op) noexcept;

  // Insert the timer operation into the queue of timers.
  // Must be called from the I/O thread.
  void schedule_at_impl(schedule_at_operation* op) noexcept;

  // Execute all ready-to-run items on the local queue.
  // Will not run other items that were enqueued during the execution of the
  // items that were already enqueued.
  // This bounds the amount of work to a finite amount.
  void execute_pending_local() noexcept;

  // Check if any completion queue items are available and if so add them
  // to the local queue.
  void acquire_completion_queue_items() noexcept;

  // Check if any completion queue items have been enqueued and move them
  // to the local queue.
  void acquire_remote_queued_items() noexcept;

  // Sets an atomic variable to ready state for the remote queue to modify as
  // a way of registering for asynchronous notification of someone enqueueing
  //
  // Returns true if successful. If so then it is no longer permitted
  // to call 'acquire_remote_queued_items()' until after the completion
  // for this operation is received.
  //
  // Returns false and sets the atomic variable to pendding state if either no
  // more operations can be submitted at this time (submission queue full or too
  // many pending completions) or if some other thread concurrently enqueued
  // work to the remote queue.
  bool try_register_remote_queue_notification() noexcept;

  // Signal the remote queue atomic variable.
  //
  // This should only be called after trying to enqueue() work
  // to the remoteQueue and being told that the I/O thread is
  // inactive.
  void signal_remote_queue();

  void remove_timer(schedule_at_operation* op) noexcept;
  void update_timers() noexcept;
  bool try_submit_timer_io(const time_point& dueTime) noexcept;
  bool try_submit_timer_io_cancel() noexcept;
  /*
  // BACKUP(He Jia): use pthread signal to trigger timer timeout
  static void timer_timeout_callback(union sigval sv) noexcept;
  */
  static void timer_timeout_callback(int signo, siginfo_t* info, void* _);

  // Try to submit a task to the UCX worker
  //
  // If there is no limit condition then populateSqe
  template <typename PopulateFn>
  bool try_submit_io(PopulateFn populateSqe) noexcept;

  // Get available completion queue entries and update tail
  ucx_am_cqe& get_and_update_cq_entry() noexcept;

  // Total number of operations submitted that have not yet
  // completed.
  std::int32_t pending_operation_count() const noexcept {
    return cqPendingCount_ + sqUnflushedCount_;
  }

  // Query whether there is space in the completion ring buffer
  // for an additional entry and the submission limit is met.
  bool can_submit_io() const noexcept {
    return sqUnflushedCount_ < sqEntryCount_
           && pending_operation_count() < cqEntryCount_;
  }

  // Decrement the submission queue unflushed count and increment
  // the completion queue pending count
  ucx_am_cqe& get_completion_queue_entry() {
    --this->sqUnflushedCount_;
    ++this->cqPendingCount_;
    return this->get_and_update_cq_entry();
  }

  std::uintptr_t timer_user_data() const {
    return reinterpret_cast<std::uintptr_t>(&timers_);
  }

  std::uintptr_t remove_timer_user_data() const {
    return reinterpret_cast<std::uintptr_t>(&currentDueTime_);
  }

  // time relative function
  bool is_timeout_elapsed(
    time_point timestamp, time_duration timeout) const noexcept;
  time_duration get_conn_timeout() const noexcept;

  ////////
  // UCX related data.

  // Memory resource
  ucx_memory_resource& mr_;

  // UCX context init parameters
  const std::string ucxAmContextName_ = "default";

  const time_duration connTimeout_ = std::chrono::seconds(300);

  const bool connectionHandleError_ = false;

  const uint64_t clientId_ = CLIENT_ID_UNDEFINED;

  const std::unique_ptr<UcxAutoDeviceContext> deviceContext_ = nullptr;

  // UCX context handle.
  ucp_context_h ucpContext_ = nullptr;
  ucp_worker_h ucpWorker_ = nullptr;
  ucp_listener_h ucpListener_ = nullptr;
  bool ucpContextInitialized_ = false;
  const bool isUcpContextExternal_ = false;

  // UCX arrived active message pending queue
  std::deque<ucx_am_desc> amDescQueue_;
  // UCX arrived active message pending map, used to delay receiving the buffer
  // until the recv_buffer_sender operation is scheduled
  ucx_am_desc_map amDescMap_;
  // If amDescQueue_ is empty but recv_sender has submited, push this
  // operation into pendingRecvIoQueue_
  std::deque<operation_base*> pendingRecvIoQueue_;

  // UCX arrived endpoint connection request pending queue
  std::deque<ep_conn_request> epConnReqQueue_;
  // If epConnReqQueue_ is empty but accept_sender has submited, push this
  // operation into pendingAcptIoQueue_
  std::deque<operation_base*> pendingAcptIoQueue_;

  // UCX connection struct
  ConnectionManager conn_manager_;

  ////////
  // Data that does not change once initialised.

  // Submission queue state
  // SQE is a concept but not implemented as a queue in this context
  const std::int32_t sqEntryCount_ = kCompletionQueueEntryCount;

  // Completion queue state
  const std::int32_t cqEntryCount_ = kCompletionQueueEntryCount;
  const std::uint32_t cqMask_ = static_cast<std::uint32_t>(cqEntryCount_) - 1;
  std::array<ucx_am_cqe, kCompletionQueueEntryCount> cqEntries_;
  std::atomic<unsigned> cqHead_{0};
  std::atomic<unsigned> cqTail_{0};

  ///////////////////
  // Data that is modified by I/O thread

  // Local queue for operations that are ready to execute.
  operation_queue localQueue_;

  // Operations that are waiting for more space in the I/O queues.
  operation_queue pendingIoQueue_;

  // Set of operations waiting to be executed at a specific time.
  timer_heap timers_;

  // The time that the current timer operation submitted to the kernel
  // is due to elapse.
  std::optional<time_point> currentDueTime_;

  // Number of context unflushed submission.
  std::int32_t sqUnflushedCount_ = 0;

  // Number of submitted operations that have not yet received a completion.
  // We should ensure this number is never greater than cqEntryCount_ so that
  // we don't end up with an overflowed completion queue.
  std::int32_t cqPendingCount_ = 0;

  std::atomic<bool> remoteQueueEventEntry_;
  bool remoteQueueReadSubmitted_ = false;
  bool timersAreDirty_ = false;

  // Timer related data
  std::uint32_t activeTimerCount_ = 0;
  timer_t timerId_;
  struct sigevent timerSigevent_;
  struct sigaction timerSigaction_;
  struct itimerspec timerItimerspec_;
  schedule_at_operation* lastTimerOp_ = nullptr;

  //////////////////
  // Data that is modified by remote threads

  // Queue of operations enqueued by remote threads.
  atomic_intrusive_queue<operation_base, &operation_base::next_> remoteQueue_;
};

template <typename StopToken>
void ucx_am_context::run(StopToken stopToken) {
  stop_operation stopOp;
  auto onStopRequested = [&] { this->schedule_impl(&stopOp); };
  typename StopToken::template callback_type<decltype(onStopRequested)>
    stopCallback{std::move(stopToken), std::move(onStopRequested)};
  run_impl(stopOp.shouldStop_);
}

template <typename PopulateFn>
bool ucx_am_context::try_submit_io(PopulateFn populateSqe) noexcept {
  UNIFEX_ASSERT(is_running_on_io_thread());

  if (pending_operation_count() < cqEntryCount_) {
    // Haven't reached limit of completion-queue yet.
    if (sqUnflushedCount_ < sqEntryCount_) {
      static_assert(noexcept(populateSqe()));

      if constexpr (std::is_void_v<decltype(populateSqe())>) {
        populateSqe();
      } else {
        if (!populateSqe()) {
          return false;
        }
      }

      ++sqUnflushedCount_;
      return true;
    }
  }

  return false;
}

class ucx_am_context::schedule_sender {
  template <typename Receiver>
  class operation : private operation_base {
   public:
    void start() noexcept {
      UNIFEX_TRY { context_.schedule_impl(this); }
      UNIFEX_CATCH(...) {
        unifex::set_error(
          static_cast<Receiver&&>(receiver_), std::current_exception());
      }
    }

   private:
    friend schedule_sender;

    template <typename Receiver2>
    explicit operation(ucx_am_context& context, Receiver2&& r)
      : context_(context), receiver_(static_cast<Receiver2&&>(r)) {
      this->execute_ = &execute_impl;
    }

    static void execute_impl(operation_base* p) noexcept {
      operation& op = *static_cast<operation*>(p);
      if constexpr (!is_stop_never_possible_v<stop_token_type_t<Receiver>>) {
        if (get_stop_token(op.receiver_).stop_requested()) {
          unifex::set_done(static_cast<Receiver&&>(op.receiver_));
          return;
        }
      }

      if constexpr (noexcept(unifex::set_value(
                      static_cast<Receiver&&>(op.receiver_)))) {
        unifex::set_value(static_cast<Receiver&&>(op.receiver_));
      } else {
        UNIFEX_TRY { unifex::set_value(static_cast<Receiver&&>(op.receiver_)); }
        UNIFEX_CATCH(...) {
          unifex::set_error(
            static_cast<Receiver&&>(op.receiver_), std::current_exception());
        }
      }
    }

    ucx_am_context& context_;
    Receiver receiver_;
  };

 public:
  template <
    template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<>>;

  template <template <typename...> class Variant>
  using error_types = Variant<std::exception_ptr>;

  static constexpr bool sends_done = true;

  template <typename Receiver>
  operation<std::remove_reference_t<Receiver>> connect(Receiver&& r) {
    return operation<std::remove_reference_t<Receiver>>{
      context_, static_cast<Receiver&&>(r)};
  }

 private:
  friend ucx_am_context::scheduler;

  explicit schedule_sender(ucx_am_context& context) noexcept
    : context_(context) {}

  ucx_am_context& context_;
};

class ucx_am_context::recv_sender {
  template <typename Receiver>
  class operation : private completion_base {
    friend ucx_am_context;

   public:
    template <typename Receiver2>
    explicit operation(const recv_sender& sender, Receiver2&& r)
      : context_(sender.context_),
        ucpWorker_(sender.context_.ucpWorker_),
        data_own_(
          sender.data_own_.has_value()
            ? std::move(const_cast<recv_sender&>(sender).data_own_)
            : std::nullopt),
        data_(data_own_.has_value() ? *data_own_.value().get() : sender.data_),
        mr_(sender.mr_),
        receiver_(static_cast<Receiver2&&>(r)) {
      UNIFEX_ASSERT(data_.header.data == nullptr &&
                    data_.header.size == 0 &&
                    "The passed-in ucx_am_data object's header "
                    "must be nullptr and size 0 (no allocation performed)");
      UNIFEX_ASSERT(
          (data_.buffer.data ? data_.buffer.size > 0
                             : data_.buffer.size == 0) &&
          "The data buffer must be nullptr initialized when data length is 0 "
          "passed to recv_sender constructor");
    }

    ~operation() {}

    void start() noexcept {
      if (!context_.is_running_on_io_thread()) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_remote(this);
      } else {
        start_io();
      }
    }

   private:
    static void on_schedule_complete(operation_base* op) noexcept {
      static_cast<operation*>(op)->start_io();
    }

    void start_io() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      if (!stopCallbackConstructed_) {
        stopCallback_.construct(
          get_stop_token(receiver_), cancel_callback{*this});
        stopCallbackConstructed_ = true;
      }
      auto populateSqe = [this]() noexcept {
        // Handle cancellation
        if (cancel_flag_.load(std::memory_order_consume)) {
          this->result_ = UCS_ERR_CANCELED;
          return true;
        }
        // Get operation pointer for completion
        auto op_ =
          reinterpret_cast<std::uintptr_t>(static_cast<completion_base*>(this));

        // Helper function to set completion entry and update counters
        auto set_completion_entry = [this, op_](ucs_status_t status) {
          auto& entry = this->context_.get_completion_queue_entry();
          entry.user_data = op_;
          entry.res = status;
          this->execute_ = &operation::on_schedule_complete;
          this->result_ = status;
        };

        // No pending messages - schedule for later processing
        if (context_.amDescQueue_.empty()) {
          onPending_.store(true, std::memory_order_relaxed);
          this->execute_ = &operation::on_schedule_complete;
          this->result_ = UCS_ERR_NO_MESSAGE;
          --this->context_.sqUnflushedCount_;
          this->context_.pendingRecvIoQueue_.push_front(this);
          return true;
        }

        // Process pending AM message
        onPending_.store(false, std::memory_order_relaxed);
        const ucx_am_desc amDesc = std::move(context_.amDescQueue_.front());
        context_.amDescQueue_.pop_front();

        // Get connection ID from reply_ep
        auto conn_opt = context_.conn_manager_.get_connection(amDesc.conn_id);

        UNIFEX_TRY {
          if (!conn_opt.has_value()) {
            // Connection not found
            set_completion_entry(UCS_ERR_UNREACHABLE);
            return true;
          }

          // Setup connection and data
          auto conn = conn_opt.value();
          this->conn_ = conn;

          // Header has been copied in the message callback
          // TODO(He Jia): Make it more efficient and safe
          data_.header.data = amDesc.header;
          data_.header.size = amDesc.header_length;

          bool allocatedInnerData =
            !data_.buffer.data || data_.buffer.size < amDesc.data_length;
          // TODO(He Jia): Be careful, the data_ will be reallocated when its
          // length is not enough. Not sure if it's a good idea.
          if (allocatedInnerData && data_.buffer.data) {
            mr_.deallocate(
              data_.buffer_type, data_.buffer.data, data_.buffer.size);
          }
          data_.buffer.size = amDesc.data_length;

          this->execute_ = &operation::on_read_complete;

          if (UcxConnection::ucx_am_is_rndv(amDesc)) {
            // Handle rendezvous protocol
            // Setup memory and callback
            if (allocatedInnerData) {
              data_.buffer.data =
                mr_.allocate(data_.buffer_type, amDesc.data_length);
            }
            auto am_recv_cb = std::make_unique<CqeEntryCallback>(
              op_, &(this->context_), [](void* context_ptr) -> ucx_am_cqe& {
                return reinterpret_cast<ucx_am_context*>(context_ptr)
                  ->get_completion_queue_entry();
              });
            auto impl_memh_ = reinterpret_cast<ucp_mem_h>(data_.mem_h);
            std::tie(this->result_, request_) = conn.get().recv_am_data(
              data_.buffer.data, amDesc.data_length, impl_memh_,
              std::move(amDesc), mem_type_map(data_.buffer_type),
              std::move(am_recv_cb));
          } else {
            // Handle eager protocol
            if (amDesc.recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA) {
              bool use_ucp_buffer_mem = false;
              if (data_own_.has_value()) {
                use_ucp_buffer_mem =
                  data_own_.value().set_ucp_release_fn([this](void* data) {
                    if (this->ucpWorker_ != nullptr) {
                      ucp_am_data_release(this->ucpWorker_, data);
                    } else {
                      UCX_CTX_ERROR
                        << "ucpWorker_ is nullptr when release data " << data
                        << std::endl;
                    }
                  });
              }
              if (use_ucp_buffer_mem) {
                data_own_.value().get()->buffer.data = amDesc.desc;
                data_own_.value().get()->buffer.size = amDesc.data_length;
                data_own_.value().get()->buffer_type = ucx_memory_type::HOST;
              } else {
                if (allocatedInnerData) {
                  data_.buffer.data =
                    mr_.allocate(data_.buffer_type, amDesc.data_length);
                }
                mr_.memcpy(
                  data_.buffer_type, data_.buffer.data, ucx_memory_type::HOST,
                  amDesc.desc, amDesc.data_length);
                if (this->ucpWorker_ != nullptr) {
                  ucp_am_data_release(this->ucpWorker_, amDesc.desc);
                } else {
                  UCX_CTX_ERROR << "ucpWorker_ is nullptr when release data "
                                << amDesc.desc << std::endl;
                }
              }
            } else {
              // Has been handled in the message callback
              // Eager message is always in host memory
              data_.buffer.data = amDesc.desc;
              data_.buffer_type = ucx_memory_type::HOST;
            }
            this->result_ = conn.get().ucx_status();
            // Schedule the completion to the pending IO queue front
            --this->context_.sqUnflushedCount_;
            // TODO(He Jia): This is may cause segment fault issue.
            this->on_read_complete(this);
          }
        }
        UNIFEX_CATCH(...) { set_completion_entry(UCS_ERR_NO_MESSAGE); }

        return true;
      };

      if (!context_.try_submit_io(populateSqe)) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_pending_io(this);
      }
    }

    void request_stop() noexcept {
      if (!onPending_.load(std::memory_order_relaxed)) {
        return;
      }
      if (char expected = 1; !refCount_.compare_exchange_strong(
            expected, 2, std::memory_order_relaxed)) {
        // If refCount_ is 0 means the operation is already completed
        // If onPending_ is true means the operation is pending to be
        // rescheduled, no side effect to be cancelled
        UNIFEX_ASSERT(expected == 0);
        return;
      }

      cancel_flag_.store(true, std::memory_order_release);

      if (context_.is_running_on_io_thread()) {
        request_stop_local();
      } else {
        request_stop_remote();
      }
    }

    void request_stop_local() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      auto populateSqe = [this]() noexcept {
        auto cop = reinterpret_cast<std::uintptr_t>(
          static_cast<completion_base*>(&cop_));
        if (__builtin_expect(conn_.has_value(), 1)) {
          auto& conn_ref = conn_.value().get();
          if (request_) {
            conn_ref.cancel_request(request_);
          }
        }
        auto& entry = this->context_.get_completion_queue_entry();
        entry.user_data = cop;
        entry.res = UCS_ERR_CANCELED;
        this->result_ = UCS_ERR_CANCELED;
        cop_.execute_ = &cancel_operation::on_stop_complete;

        // Ensure the operation is removed from the queue
        std::atomic_thread_fence(std::memory_order_acquire);
        try {
          context_.pendingRecvIoQueue_.erase(
            std::remove_if(
              context_.pendingRecvIoQueue_.begin(),
              context_.pendingRecvIoQueue_.end(),
              [this](operation_base* op) { return op == this; }),
            context_.pendingRecvIoQueue_.end());
        } catch (const std::exception& e) {
          UCX_CTX_ERROR << "std::erase_if context_.pendingRecvIoQueue_ failed: "
                        << e.what() << std::endl;
        }
        std::atomic_thread_fence(std::memory_order_release);
      };

      if (!context_.try_submit_io(populateSqe)) {
        cop_.execute_ = &cancel_operation::on_schedule_stop_complete;
        context_.schedule_pending_io(&cop_);
      }
    }

    void request_stop_remote() noexcept {
      cop_.execute_ = &cancel_operation::on_schedule_stop_complete;
      context_.schedule_remote(&cop_);
    }

    static void on_read_complete(operation_base* op) noexcept {
      auto& self = *static_cast<operation*>(op);
      if (
        self.refCount_.fetch_sub(1, std::memory_order_acq_rel) != 1
        && !self.onPending_.load(std::memory_order_relaxed)) {
        // stop callback is running, must complete the op
        // If onPending_ is true, means the operation is not be rescheduled,
        // so on_read_complete is legal to be called
        return;
      }

      self.stopCallback_.destruct();

      // If data is owned, move it; otherwise, copy from the external buffer.
      UcxAmData am_data_wrapper =
        self.data_own_.has_value() ? std::move(self.data_own_.value())
                                   : UcxAmData(
                                     self.mr_, self.data_, /*own_header=*/false,
                                     /*own_buffer=*/false);

      if (get_stop_token(self.receiver_).stop_requested()) {
        unifex::set_done(std::move(self.receiver_));
      } else if (self.result_ >= 0) {
        active_message_bundle bundle{
          std::move(am_data_wrapper), self.conn_.value()};
        if constexpr (noexcept(unifex::set_value(
                        std::move(self.receiver_), std::move(bundle)))) {
          unifex::set_value(std::move(self.receiver_), std::move(bundle));
        } else {
          UNIFEX_TRY {
            unifex::set_value(std::move(self.receiver_), std::move(bundle));
          }
          UNIFEX_CATCH(...) {
            unifex::set_error(
              std::move(self.receiver_), std::current_exception());
          }
        }
      } else if (self.result_ == UCS_ERR_CANCELED) {
        UNIFEX_TRY { am_data_wrapper.~UcxAmData(); }
        UNIFEX_CATCH(...) {
          UCX_CTX_ERROR
            << "Failed to deallocate data in recv_sender on_read_complete";
        }
        unifex::set_done(std::move(self.receiver_));
      } else {
        UNIFEX_TRY { am_data_wrapper.~UcxAmData(); }
        UNIFEX_CATCH(...) {
          UCX_CTX_ERROR
            << "Failed to deallocate data in recv_sender on_read_complete";
        }
        unifex::set_error(
          std::move(self.receiver_),
          stdexe_ucx_runtime::make_error_code(
            static_cast<ucs_status_t>(self.result_)));
      }
    }

    struct cancel_operation final : completion_base {
      operation& op_;

      explicit cancel_operation(operation& op) noexcept : op_(op) {}
      // intrusive list breaks if the same operation is submitted twice
      // break the cycle: `on_stop_complete` delegates to the parent operation
      static void on_stop_complete(operation_base* op) noexcept {
        operation::on_read_complete(&(static_cast<cancel_operation*>(op)->op_));
      }

      static void on_schedule_stop_complete(operation_base* op) noexcept {
        static_cast<cancel_operation*>(op)->op_.request_stop_local();
      }
    };

    struct cancel_callback final {
      operation& op_;

      void operator()() noexcept { op_.request_stop(); }
    };

    ucx_am_context& context_;
    ucp_worker_h ucpWorker_;
    conn_opt_t conn_;
    std::optional<UcxAmData> data_own_;
    ucx_am_data& data_;
    ucx_request* request_ = nullptr;
    ucx_memory_resource& mr_;
    std::atomic_bool cancel_flag_{false};
    std::atomic_bool onPending_{false};
    Receiver receiver_;
    manual_lifetime<typename stop_token_type_t<
      Receiver>::template callback_type<cancel_callback>>
      stopCallback_;
    bool stopCallbackConstructed_ = false;
    std::atomic_char refCount_{1};
    cancel_operation cop_{*this};
  };

 public:
  // Produces active_message_bundle
  template <
    template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<active_message_bundle>>;

  // Note: Only case it might complete with exception_ptr is if the
  // receiver's set_value() exits with an exception.
  template <template <typename...> class Variant>
  using error_types = Variant<std::error_code, std::exception_ptr>;

  static constexpr bool sends_done = true;

  explicit recv_sender(ucx_am_context& context, ucx_am_data& data) noexcept
    : context_(context),
      data_own_(std::nullopt),
      data_(data),
      mr_(context.mr_) {
    UNIFEX_ASSERT(data_.header.data == nullptr &&
                  data_.header.size == 0 &&
                  "The passed-in ucx_am_data object's header "
                  "must be nullptr and size 0 (no allocation performed)");
  }

  explicit recv_sender(
    ucx_am_context& context, ucx_memory_type data_type) noexcept
    : context_(context),
      data_own_(
        std::in_place, context.mr_, 0, 0, data_type, /*own_header=*/true,
        /*own_buffer=*/true),
      data_(*data_own_.value().get()),
      mr_(context.mr_) {}

  template <typename Receiver>
  operation<remove_cvref_t<Receiver>> connect(Receiver&& r) && {
    return operation<remove_cvref_t<Receiver>>{
      *this, static_cast<Receiver&&>(r)};
  }

 private:
  friend scheduler;

  ucx_am_context& context_;
  std::optional<UcxAmData> data_own_;
  ucx_am_data& data_;
  ucx_memory_resource& mr_;  // Use pmr for receiving buffer allocation
};

template <
  typename BufferType,
  std::enable_if_t<is_receive_buffer_parameter_v<BufferType>, int> = 0>
class ucx_am_context::recv_buffer_sender_t {
  template <typename Receiver>
  class operation : private completion_base {
    friend ucx_am_context;

   public:
    template <typename Receiver2>
    explicit operation(const recv_buffer_sender_t& sender, Receiver2&& r)
      : context_(sender.context_),
        am_desc_key_(sender.am_desc_key_),
        buffer_own_(
          std::move(const_cast<recv_buffer_sender_t&>(sender).buffer_own_)),
        mr_(sender.mr_),
        receiver_(static_cast<Receiver2&&>(r)) {}

    ~operation() {}

    void start() noexcept {
      if (!context_.is_running_on_io_thread()) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_remote(this);
      } else {
        start_io();
      }
    }

   private:
    static void on_schedule_complete(operation_base* op) noexcept {
      static_cast<operation*>(op)->start_io();
    }

    void start_io() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      if (!stopCallbackConstructed_) {
        stopCallback_.construct(
          get_stop_token(receiver_), cancel_callback{*this});
        stopCallbackConstructed_ = true;
      }
      auto populateSqe = [this]() noexcept {
        // Handle cancellation
        if (cancel_flag_.load(std::memory_order_consume)) {
          this->result_ = UCS_ERR_CANCELED;
          return true;
        }
        // Get operation pointer for completion
        auto op_ =
          reinterpret_cast<std::uintptr_t>(static_cast<completion_base*>(this));

        // Helper function to set completion entry and update counters
        auto set_completion_entry = [this, op_](ucs_status_t status) {
          auto& entry = this->context_.get_completion_queue_entry();
          entry.user_data = op_;
          entry.res = status;
          this->execute_ = &operation::on_schedule_complete;
          this->result_ = status;
        };

        auto amDescOpt = context_.amDescMap_.pop(this->am_desc_key_);
        if (!amDescOpt.has_value()) {
          set_completion_entry(UCS_ERR_NO_MESSAGE);
          return true;
        }
        auto& amDesc = amDescOpt.value();

        // Get connection ID from reply_ep
        auto conn_opt = context_.conn_manager_.get_connection(amDesc.conn_id);

        UNIFEX_TRY {
          if (!conn_opt.has_value()) {
            // Connection not found
            set_completion_entry(UCS_ERR_UNREACHABLE);
            return true;
          }

          // Setup connection and data
          auto conn = conn_opt.value();
          this->conn_ = conn;

          this->execute_ = &operation::on_read_complete;

          if (UcxConnection::ucx_am_is_rndv(amDesc)) {
            auto am_recv_cb = std::make_unique<CqeEntryCallback>(
              op_, &(this->context_), [](void* context_ptr) -> ucx_am_cqe& {
                return reinterpret_cast<ucx_am_context*>(context_ptr)
                  ->get_completion_queue_entry();
              });

            constexpr bool is_vec =
              std::is_same_v<std::decay_t<BufferType>, UcxBufferVec>;
            if constexpr (std::is_same_v<BufferType, UcxBuffer>) {
              auto buffer = buffer_own_.get();
              bool allocatedInnerData =
                !buffer->data || buffer->size < amDesc.data_length;
              // TODO(He Jia): Be careful, the data_ will be reallocated when
              // its length is not enough. Not sure if it's a good idea.
              if (allocatedInnerData && buffer->data) {
                mr_.deallocate(buffer_own_.type(), buffer->data, buffer->size);
                UCX_CTX_INFO << "recv_buffer_sender: deallocated inner data"
                             << std::endl;
              }
              buffer->size = amDesc.data_length;

              // Handle rendezvous protocol
              // Setup memory and callback
              if (allocatedInnerData) {
                buffer->data =
                  mr_.allocate(buffer_own_.type(), amDesc.data_length);
                UCX_CTX_INFO << "recv_buffer_sender: allocated inner data"
                             << std::endl;
              }

              auto impl_memh_ =
                reinterpret_cast<ucp_mem_h>(buffer_own_.mem_h());
              std::tie(this->result_, request_) = conn.get().recv_am_data(
                buffer->data, amDesc.data_length, impl_memh_, std::move(amDesc),
                mem_type_map(buffer_own_.type()), std::move(am_recv_cb));
            } else if constexpr (is_vec) {
              auto& buffer_vec = buffer_own_;
              auto impl_memh_ = reinterpret_cast<ucp_mem_h>(buffer_vec.mem_h());
              auto buffer_vec_data =
                reinterpret_cast<ucp_dt_iov_t*>(buffer_vec.data());
              std::tie(this->result_, request_) = conn.get().recv_am_iov_data(
                buffer_vec_data, buffer_vec.size(), impl_memh_,
                std::move(amDesc), mem_type_map(buffer_vec.type()),
                std::move(am_recv_cb));
            } else {
              UNIFEX_ASSERT(false && "recv_buffer_sender not implemented");
              UCX_CTX_ERROR << "recv_buffer_sender not implemented"
                            << std::endl;
              set_completion_entry(UCS_ERR_NOT_IMPLEMENTED);
            }
          } else {
            // recv_buffer_sender should not handle eager protocol
            UNIFEX_ASSERT(
              false && "recv_buffer_sender should not handle eager protocol");
            UCX_CTX_ERROR
              << "recv_buffer_sender should not handle eager protocol"
              << std::endl;
            set_completion_entry(UCS_ERR_NOT_IMPLEMENTED);
          }
        }
        UNIFEX_CATCH(...) { set_completion_entry(UCS_ERR_NO_MESSAGE); }

        return true;
      };

      if (!context_.try_submit_io(populateSqe)) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_pending_io(this);
      }
    }

    void request_stop() noexcept {
      if (char expected = 1; !refCount_.compare_exchange_strong(
            expected, 2, std::memory_order_relaxed)) {
        // If refCount_ is 0 means the operation is already completed
        UNIFEX_ASSERT(expected == 0);
        return;
      }

      cancel_flag_.store(true, std::memory_order_release);

      if (context_.is_running_on_io_thread()) {
        request_stop_local();
      } else {
        request_stop_remote();
      }
    }

    void request_stop_local() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      auto populateSqe = [this]() noexcept {
        auto cop = reinterpret_cast<std::uintptr_t>(
          static_cast<completion_base*>(&cop_));
        if (__builtin_expect(conn_.has_value(), 1)) {
          auto& conn_ref = conn_.value().get();
          if (request_) {
            conn_ref.cancel_request(request_);
          }
        }
        auto& entry = this->context_.get_completion_queue_entry();
        entry.user_data = cop;
        entry.res = UCS_ERR_CANCELED;
        this->result_ = UCS_ERR_CANCELED;
        cop_.execute_ = &cancel_operation::on_stop_complete;

        // Ensure the operation is removed from the map
        context_.amDescMap_.pop(this->am_desc_key_);
      };

      if (!context_.try_submit_io(populateSqe)) {
        cop_.execute_ = &cancel_operation::on_schedule_stop_complete;
        context_.schedule_pending_io(&cop_);
      }
    }

    void request_stop_remote() noexcept {
      cop_.execute_ = &cancel_operation::on_schedule_stop_complete;
      context_.schedule_remote(&cop_);
    }

    static void on_read_complete(operation_base* op) noexcept {
      auto& self = *static_cast<operation*>(op);
      if (self.refCount_.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        // stop callback is running, must complete the op
        return;
      }

      self.stopCallback_.destruct();

      if (get_stop_token(self.receiver_).stop_requested()) {
        unifex::set_done(std::move(self.receiver_));
      } else if (self.result_ >= 0) {
        active_message_buffer_bundle_t<BufferType> bundle{
          std::move(self.buffer_own_), self.conn_.value()};
        if constexpr (noexcept(unifex::set_value(
                        std::move(self.receiver_), std::move(bundle)))) {
          unifex::set_value(std::move(self.receiver_), std::move(bundle));
        } else {
          UNIFEX_TRY {
            unifex::set_value(std::move(self.receiver_), std::move(bundle));
          }
          UNIFEX_CATCH(...) {
            unifex::set_error(
              std::move(self.receiver_), std::current_exception());
          }
        }
      } else if (self.result_ == UCS_ERR_CANCELED) {
        unifex::set_done(std::move(self.receiver_));
      } else {
        unifex::set_error(
          std::move(self.receiver_),
          stdexe_ucx_runtime::make_error_code(
            static_cast<ucs_status_t>(self.result_)));
      }
    }

    struct cancel_operation final : completion_base {
      operation& op_;

      explicit cancel_operation(operation& op) noexcept : op_(op) {}
      // intrusive list breaks if the same operation is submitted twice
      // break the cycle: `on_stop_complete` delegates to the parent operation
      static void on_stop_complete(operation_base* op) noexcept {
        operation::on_read_complete(&(static_cast<cancel_operation*>(op)->op_));
      }

      static void on_schedule_stop_complete(operation_base* op) noexcept {
        static_cast<cancel_operation*>(op)->op_.request_stop_local();
      }
    };

    struct cancel_callback final {
      operation& op_;

      void operator()() noexcept { op_.request_stop(); }
    };

    ucx_am_context& context_;
    conn_opt_t conn_;
    size_t am_desc_key_;
    BufferType buffer_own_;
    ucx_request* request_ = nullptr;
    ucx_memory_resource& mr_;
    std::atomic_bool cancel_flag_{false};
    Receiver receiver_;
    manual_lifetime<typename stop_token_type_t<
      Receiver>::template callback_type<cancel_callback>>
      stopCallback_;
    bool stopCallbackConstructed_ = false;
    std::atomic_char refCount_{1};
    cancel_operation cop_{*this};
  };

 public:
  // Produces BufferType
  template <
    template <typename...> class Variant, template <typename...> class Tuple>
  using value_types =
    Variant<Tuple<active_message_buffer_bundle_t<BufferType>>>;

  // Note: Only case it might complete with exception_ptr is if the
  // receiver's set_value() exits with an exception.
  template <template <typename...> class Variant>
  using error_types = Variant<std::error_code, std::exception_ptr>;

  static constexpr bool sends_done = true;

  explicit recv_buffer_sender_t(
    ucx_am_context& context, size_t key, BufferType&& buffer_own) noexcept
    : am_desc_key_(key),
      context_(context),
      buffer_own_(std::move(buffer_own)),
      mr_(context.mr_) {
    if constexpr (std::is_same_v<std::decay_t<BufferType>, UcxBufferVec>) {
      UNIFEX_ASSERT(
        buffer_own_.size() > 0 && "At least one buffer is required");
      // It's a span of buffers, so we need to check each buffer
      for (size_t i = 0; i < buffer_own_.size(); ++i) {
        UNIFEX_ASSERT(buffer_own_[i].data != nullptr &&
                      buffer_own_[i].size != 0 &&
                      "The passed-in buffer must be allocated "
                      "when receiving IO vector data");
      }
    }
  }

  template <typename B = BufferType>
  explicit recv_buffer_sender_t(
    ucx_am_context& context, size_t key, ucx_memory_type memory_type,
    std::enable_if_t<std::is_same_v<B, UcxBuffer>>* = nullptr) noexcept
    : am_desc_key_(key),
      context_(context),
      buffer_own_({context.mr_, memory_type, 0}),
      mr_(context.mr_) {}

  template <typename Receiver>
  operation<remove_cvref_t<Receiver>> connect(Receiver&& r) && {
    return operation<remove_cvref_t<Receiver>>{
      *this, static_cast<Receiver&&>(r)};
  }

 private:
  friend scheduler;

  size_t am_desc_key_;
  ucx_am_context& context_;
  BufferType buffer_own_;
  ucx_memory_resource& mr_;  // Use pmr for receiving buffer allocation
};

class ucx_am_context::recv_header_sender {
  template <typename Receiver>
  class operation : private completion_base {
    friend ucx_am_context;

   public:
    template <typename Receiver2>
    explicit operation(const recv_header_sender& sender, Receiver2&& r)
      : context_(sender.context_),
        ucpWorker_(sender.context_.ucpWorker_),
        mr_(sender.mr_),
        receiver_(static_cast<Receiver2&&>(r)) {}

    ~operation() {}

    void start() noexcept {
      if (!context_.is_running_on_io_thread()) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_remote(this);
      } else {
        start_io();
      }
    }

   private:
    static void on_schedule_complete(operation_base* op) noexcept {
      static_cast<operation*>(op)->start_io();
    }

    void start_io() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      if (!stopCallbackConstructed_) {
        stopCallback_.construct(
          get_stop_token(receiver_), cancel_callback{*this});
        stopCallbackConstructed_ = true;
      }
      auto populateSqe = [this]() noexcept {
        // Handle cancellation
        if (cancel_flag_.load(std::memory_order_consume)) {
          this->result_ = UCS_ERR_CANCELED;
          return true;
        }

        // Get operation pointer for completion
        auto op_ =
          reinterpret_cast<std::uintptr_t>(static_cast<completion_base*>(this));

        // Helper function to set completion entry and update counters
        auto set_completion_entry = [this, op_](ucs_status_t status) {
          auto& entry = this->context_.get_completion_queue_entry();
          entry.user_data = op_;
          entry.res = status;
          this->execute_ = &operation::on_schedule_complete;
          this->result_ = status;
        };

        // No pending messages - schedule for later processing
        if (context_.amDescQueue_.empty()) {
          onPending_.store(true, std::memory_order_relaxed);
          this->execute_ = &operation::on_schedule_complete;
          this->result_ = UCS_ERR_NO_MESSAGE;
          --this->context_.sqUnflushedCount_;
          this->context_.pendingRecvIoQueue_.push_front(this);
          return true;
        }

        // Process pending AM message
        onPending_.store(false, std::memory_order_relaxed);
        const ucx_am_desc amDesc = std::move(context_.amDescQueue_.front());
        context_.amDescQueue_.pop_front();

        // Get connection ID from reply_ep
        auto conn_opt = context_.conn_manager_.get_connection(amDesc.conn_id);

        UNIFEX_TRY {
          if (!conn_opt.has_value()) {
            // Connection not found
            set_completion_entry(UCS_ERR_UNREACHABLE);
            return true;
          }

          // Setup connection and data
          auto conn = conn_opt.value();
          this->conn_ = conn;

          if (UcxConnection::ucx_am_is_rndv(amDesc)) {
            // Handle rendezvous protocol
            header_own_.emplace(UcxHeader(
              context_.mr_, ucx_memory_type::HOST, amDesc.header,
              amDesc.header_length, nullptr, /*own_buffer=*/true));
            am_desc_key_ = context_.amDescMap_.push(
              std::move(const_cast<ucx_am_desc&>(amDesc)));
          } else {
            // Handle eager protocol
            ucx_am_data_t am_data = {
              .header = {.data = amDesc.header, .size = amDesc.header_length},
              .buffer = {.data = amDesc.desc, .size = amDesc.data_length},
              .buffer_type = ucx_memory_type::HOST,
            };
            if (amDesc.recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA) {
              data_own_.emplace(UcxAmData(
                context_.mr_, std::move(am_data), /*own_header=*/true,
                /*own_buffer=*/true, [this](void* data) {
                  if (this->ucpWorker_ != nullptr) {
                    ucp_am_data_release(this->ucpWorker_, data);
                  } else {
                    UCX_CTX_ERROR << "ucpWorker_ is nullptr when release data "
                                  << data << std::endl;
                  }
                }));
            } else {
              // Has been handled in the message callback
              // Eager message is always in host memory
              data_own_.emplace(UcxAmData(
                context_.mr_, std::move(am_data), /*own_header=*/true,
                /*own_buffer=*/true));
            }
          }

          this->execute_ = &operation::on_read_complete;
          this->result_ = conn.get().ucx_status();
          // Schedule the completion to the pending IO queue front
          --this->context_.sqUnflushedCount_;
          // TODO(He Jia): This is may cause segment fault issue.
          this->on_read_complete(this);
        }
        UNIFEX_CATCH(...) { set_completion_entry(UCS_ERR_NO_MESSAGE); }

        return true;
      };

      if (!context_.try_submit_io(populateSqe)) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_pending_io(this);
      }
    }

    void request_stop() noexcept {
      if (!onPending_.load(std::memory_order_relaxed)) {
        return;
      }
      if (char expected = 1; !refCount_.compare_exchange_strong(
            expected, 2, std::memory_order_relaxed)) {
        // If refCount_ is 0 means the operation is already completed
        // If onPending_ is true means the operation is pending to be
        // rescheduled, no side effect to be cancelled
        UNIFEX_ASSERT(expected == 0);
        return;
      }

      cancel_flag_.store(true, std::memory_order_release);

      if (context_.is_running_on_io_thread()) {
        request_stop_local();
      } else {
        request_stop_remote();
      }
    }

    void request_stop_local() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      auto populateSqe = [this]() noexcept {
        auto cop = reinterpret_cast<std::uintptr_t>(
          static_cast<completion_base*>(&cop_));
        if (__builtin_expect(conn_.has_value(), 1)) {
          auto& conn_ref = conn_.value().get();
          if (request_) {
            conn_ref.cancel_request(request_);
          }
        }
        auto& entry = this->context_.get_completion_queue_entry();
        entry.user_data = cop;
        entry.res = UCS_ERR_CANCELED;
        this->result_ = UCS_ERR_CANCELED;
        cop_.execute_ = &cancel_operation::on_stop_complete;

        // Ensure the operation is removed from the queue
        std::atomic_thread_fence(std::memory_order_acquire);
        try {
          context_.pendingRecvIoQueue_.erase(
            std::remove_if(
              context_.pendingRecvIoQueue_.begin(),
              context_.pendingRecvIoQueue_.end(),
              [this](operation_base* op) { return op == this; }),
            context_.pendingRecvIoQueue_.end());
        } catch (const std::exception& e) {
          UCX_CTX_ERROR << "std::erase_if context_.pendingRecvIoQueue_ failed: "
                        << e.what() << std::endl;
        }
        std::atomic_thread_fence(std::memory_order_release);
      };

      if (!context_.try_submit_io(populateSqe)) {
        cop_.execute_ = &cancel_operation::on_schedule_stop_complete;
        context_.schedule_pending_io(&cop_);
      }
    }

    void request_stop_remote() noexcept {
      cop_.execute_ = &cancel_operation::on_schedule_stop_complete;
      context_.schedule_remote(&cop_);
    }

    static void on_read_complete(operation_base* op) noexcept {
      auto& self = *static_cast<operation*>(op);
      if (
        self.refCount_.fetch_sub(1, std::memory_order_acq_rel) != 1
        && !self.onPending_.load(std::memory_order_relaxed)) {
        // stop callback is running, must complete the op
        return;
      }

      self.stopCallback_.destruct();

      if (get_stop_token(self.receiver_).stop_requested()) {
        unifex::set_done(std::move(self.receiver_));
      } else if (self.result_ >= 0) {
        auto return_fn = [&](auto&&... rr) {
          if constexpr (noexcept(unifex::set_value(
                          std::move(self.receiver_), std::move(rr)...))) {
            unifex::set_value(std::move(self.receiver_), std::move(rr)...);
          } else {
            UNIFEX_TRY {
              unifex::set_value(std::move(self.receiver_), std::move(rr)...);
            }
            UNIFEX_CATCH(...) {
              unifex::set_error(
                std::move(self.receiver_), std::current_exception());
            }
          }
        };
        using value_variant =
          std::variant<std::pair<size_t, UcxHeader>, active_message_bundle>;
        if (self.header_own_.has_value()) {
          return_fn(value_variant(
            std::in_place_index<0>, self.am_desc_key_,
            std::move(self.header_own_.value())));
        } else if (self.data_own_.has_value()) {
          return_fn(value_variant(
            std::in_place_index<1>, std::move(self.data_own_.value()),
            self.conn_.value()));
        } else {
          // TODO(He Jia): UCS_ERR_LAST is not a valid error code
          unifex::set_error(
            std::move(self.receiver_),
            stdexe_ucx_runtime::make_error_code(UCS_ERR_LAST));
        }
      } else if (self.result_ == UCS_ERR_CANCELED) {
        UNIFEX_TRY {
          if (self.header_own_.has_value()) {
            self.header_own_.value().~UcxHeader();
          } else if (self.data_own_.has_value()) {
            self.data_own_.value().~UcxAmData();
          }
        }
        UNIFEX_CATCH(...) {
          UCX_CTX_ERROR << "Failed to deallocate data in recv_header_sender "
                           "on_read_complete";
        }
        unifex::set_done(std::move(self.receiver_));
      } else {
        UNIFEX_TRY {
          if (self.header_own_.has_value()) {
            self.header_own_.value().~UcxHeader();
          } else if (self.data_own_.has_value()) {
            self.data_own_.value().~UcxAmData();
          }
        }
        UNIFEX_CATCH(...) {
          UCX_CTX_ERROR << "Failed to deallocate data in recv_header_sender "
                           "on_read_complete";
        }
        unifex::set_error(
          std::move(self.receiver_),
          stdexe_ucx_runtime::make_error_code(
            static_cast<ucs_status_t>(self.result_)));
      }
    }

    struct cancel_operation final : completion_base {
      operation& op_;

      explicit cancel_operation(operation& op) noexcept : op_(op) {}
      // intrusive list breaks if the same operation is submitted twice
      // break the cycle: `on_stop_complete` delegates to the parent operation
      static void on_stop_complete(operation_base* op) noexcept {
        operation::on_read_complete(&(static_cast<cancel_operation*>(op)->op_));
      }

      static void on_schedule_stop_complete(operation_base* op) noexcept {
        static_cast<cancel_operation*>(op)->op_.request_stop_local();
      }
    };

    struct cancel_callback final {
      operation& op_;

      void operator()() noexcept { op_.request_stop(); }
    };

    ucx_am_context& context_;
    ucp_worker_h ucpWorker_;
    conn_opt_t conn_;
    ucx_request* request_ = nullptr;
    size_t am_desc_key_;
    std::optional<UcxHeader> header_own_;
    std::optional<UcxAmData> data_own_;
    ucx_memory_resource& mr_;
    std::atomic_bool cancel_flag_{false};
    std::atomic_bool onPending_{false};
    Receiver receiver_;
    manual_lifetime<typename stop_token_type_t<
      Receiver>::template callback_type<cancel_callback>>
      stopCallback_;
    bool stopCallbackConstructed_ = false;
    std::atomic_char refCount_{1};
    cancel_operation cop_{*this};
  };

 public:
  template <
    template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<
    Tuple<std::variant<std::pair<size_t, UcxHeader>, active_message_bundle>>>;

  // Note: Only case it might complete with exception_ptr is if the
  // receiver's set_value() exits with an exception.
  template <template <typename...> class Variant>
  using error_types = Variant<std::error_code, std::exception_ptr>;

  static constexpr bool sends_done = true;

  explicit recv_header_sender(ucx_am_context& context) noexcept
    : context_(context), mr_(context.mr_) {}

  template <typename Receiver>
  operation<remove_cvref_t<Receiver>> connect(Receiver&& r) && {
    return operation<remove_cvref_t<Receiver>>{
      *this, static_cast<Receiver&&>(r)};
  }

 private:
  friend scheduler;

  ucx_am_context& context_;
  ucx_memory_resource& mr_;
};

template <typename Payload>
class ucx_am_context::send_sender_t {
 private:
  // Map wrapper to underlying C struct; identity otherwise
  template <typename T>
  using payload_view_t_ = std::conditional_t<
    std::is_same_v<std::decay_t<T>, UcxAmData>, ucx_am_data,
    std::conditional_t<
      std::is_same_v<std::decay_t<T>, UcxAmIovec>, ucx_am_iovec,
      std::decay_t<T>>>;

  using payload_view_t = payload_view_t_<Payload>;
  // For static_assert in payload_view_t
  struct payload_always_false : std::false_type {};

  // Get reference to underlying C struct (wrapper -> underlying, raw -> self)
  template <typename T>
  static payload_view_t_<T>& to_view(T& v) noexcept {
    if constexpr (std::is_same_v<std::decay_t<T>, UcxAmData>) {
      return *v.get();
    } else if constexpr (std::is_same_v<std::decay_t<T>, UcxAmIovec>) {
      return *v.get();
    } else {
      return v;
    }
  }

  template <typename Receiver>
  class operation : private completion_base {
    friend ucx_am_context;

   public:
    template <typename Receiver2>
    explicit operation(const send_sender_t& sender, Receiver2&& r)
      : context_(sender.context_),
        conn_(sender.conn_),
        data_owned_(std::move(const_cast<send_sender_t&>(sender).data_owned_)),
        data_(
          data_owned_ ? send_sender_t::to_view(*data_owned_) : sender.data_),
        mr_(sender.mr_),
        receiver_(static_cast<Receiver2&&>(r)) {
      UNIFEX_ASSERT(
        data_.header.size > 0 && data_.header.data &&
          "The header must not be nullptr initialized when "
          "passed to send_sender constructor with header_length > 0");
      if constexpr (std::is_same_v<payload_view_t, ucx_am_data>) {
        UNIFEX_ASSERT(
          data_.buffer.size > 0 && data_.buffer.data &&
          "The data buffer must not be nullptr initialized when "
          "passed to send_sender constructor with data size > 0");
      } else if constexpr (std::is_same_v<payload_view_t, ucx_am_iovec>) {
        UNIFEX_ASSERT(
          data_.buffer_count > 0 && data_.buffer_vec &&
          "The buffer vector must not be nullptr initialized when "
          "passed to send_sender constructor with buffer_count > 0");
      } else {
        static_assert(
          payload_always_false::value,
          "Invalid send_sender payload type. Please check "
          "ucx_context/ucx_context_def.h "
          "and ucx_context/ucx_context_data.hpp for "
          "definition of ucx_am_data and ucx_am_iovec and "
          "other available types.");
      }
    }

    ~operation() {}

    void start() noexcept {
      if (!context_.is_running_on_io_thread()) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_remote(this);
      } else {
        start_io();
      }
    }

   private:
    static void on_schedule_complete(operation_base* op) noexcept {
      static_cast<operation*>(op)->start_io();
    }

    void start_io() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      if (!stopCallbackConstructed_) {
        stopCallback_.construct(
          get_stop_token(receiver_), cancel_callback{*this});
        stopCallbackConstructed_ = true;
      }
      auto populateSqe = [this]() noexcept {
        if (cancel_flag_.load(std::memory_order_consume)) {
          this->result_ = UCS_ERR_CANCELED;
          return true;
        }
        auto op_ =
          reinterpret_cast<std::uintptr_t>(static_cast<completion_base*>(this));
        this->execute_ = &operation::on_write_complete;
        auto& conn_ref = conn_.value().get();
        std::unique_ptr<UcxCallback> am_send_cb;
        bool use_rndv = true;
        if constexpr (std::is_same_v<payload_view_t, ucx_am_data>) {
          use_rndv = conn_ref.should_use_zcopy(data_.buffer.size);
        }
        if (use_rndv) {
          am_send_cb = std::move(std::make_unique<CqeEntryCallback>(
            op_, reinterpret_cast<void*>(&(this->context_)),
            [](void* context_ptr) -> ucx_am_cqe& {
              return reinterpret_cast<ucx_am_context*>(context_ptr)
                ->get_completion_queue_entry();
            }));
        } else {
          // TODO(He Jia): pendingIOQueue_ is faster than CQE
          am_send_cb = std::move(std::make_unique<DirectEntryCallback>(
            this, [](ucs_status_t status, void* op) {
              auto this_ = reinterpret_cast<operation*>(op);
              this_->result_ = status;
              // Schedule the completion to the pending IO queue front
              --this_->context_.sqUnflushedCount_;
              this_->context_.reschedule_pending_io(this_);
            }));
        }
        // Prepare buffer
        auto impl_memh_ = reinterpret_cast<ucp_mem_h>(data_.mem_h);
        // Call the send function
        if constexpr (std::is_same_v<payload_view_t, ucx_am_data>) {
          std::tie(this->result_, this->request_) = conn_ref.send_am_data(
            data_.header.data, data_.header.size, data_.buffer.data,
            data_.buffer.size, impl_memh_, mem_type_map(data_.buffer_type),
            std::move(am_send_cb));
        } else if constexpr (std::is_same_v<payload_view_t, ucx_am_iovec>) {
          static_assert(sizeof(ucp_dt_iov_t) == sizeof(data_.buffer_vec[0]));
          std::tie(this->result_, this->request_) = conn_ref.send_am_iov_data(
            data_.header.data, data_.header.size,
            reinterpret_cast<ucp_dt_iov_t*>(data_.buffer_vec),
            data_.buffer_count, impl_memh_, mem_type_map(data_.buffer_type),
            std::move(am_send_cb));
        } else {
          static_assert(
            payload_always_false::value,
            "Invalid send_sender payload type. Please check "
            "ucx_context/ucx_context_def.h "
            "and ucx_context/ucx_context_data.hpp for "
            "definition of ucx_am_data and ucx_am_iovec and "
            "other available types.");
        }

        return true;
      };

      if (!context_.try_submit_io(populateSqe)) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_pending_io(this);
      }
    }

    void request_stop() noexcept {
      if (char expected = 1; !refCount_.compare_exchange_strong(
            expected, 2, std::memory_order_relaxed)) {
        // lost race with on_write_complete
        UNIFEX_ASSERT(expected == 0);
        return;
      }

      cancel_flag_.store(true, std::memory_order_release);

      if (context_.is_running_on_io_thread()) {
        request_stop_local();
      } else {
        request_stop_remote();
      }
    }

    void request_stop_local() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      auto populateSqe = [this]() noexcept {
        auto cop = reinterpret_cast<std::uintptr_t>(
          static_cast<completion_base*>(&cop_));
        // Ensure the connection is valid
        auto& conn_ref = conn_.value().get();
        const auto conn_id = conn_ref.id();
        if (context_.conn_manager_.is_connection_valid(conn_id)) {
          conn_ref.cancel_send();
        }
        // Update the cqe entry
        auto& entry = this->context_.get_completion_queue_entry();
        entry.user_data = cop;
        entry.res = UCS_ERR_CANCELED;
        this->result_ = UCS_ERR_CANCELED;
        cop_.execute_ = &cancel_operation::on_stop_complete;
      };

      if (!context_.try_submit_io(populateSqe)) {
        cop_.execute_ = &cancel_operation::on_schedule_stop_complete;
        context_.schedule_pending_io(&cop_);
      }
    }

    void request_stop_remote() noexcept {
      cop_.execute_ = &cancel_operation::on_schedule_stop_complete;
      context_.schedule_remote(&cop_);
    }

    static void on_write_complete(operation_base* op) noexcept {
      auto& self = *static_cast<operation*>(op);
      if (self.refCount_.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        // stop callback is running, must complete the op
        return;
      }
      self.stopCallback_.destruct();
      if (get_stop_token(self.receiver_).stop_requested()) {
        unifex::set_done(std::move(self.receiver_));
      } else if (self.result_ >= 0) {
        if constexpr (noexcept(unifex::set_value(std::move(self.receiver_)))) {
          unifex::set_value(std::move(self.receiver_));
        } else {
          UNIFEX_TRY { unifex::set_value(std::move(self.receiver_)); }
          UNIFEX_CATCH(...) {
            unifex::set_error(
              std::move(self.receiver_), std::current_exception());
          }
        }
      } else if (self.result_ == UCS_ERR_CANCELED) {
        unifex::set_done(std::move(self.receiver_));
      } else {
        unifex::set_error(
          std::move(self.receiver_),
          stdexe_ucx_runtime::make_error_code(
            static_cast<ucs_status_t>(self.result_)));
      }
    }

    struct cancel_operation final : completion_base {
      operation& op_;

      explicit cancel_operation(operation& op) noexcept : op_(op) {}
      // intrusive list breaks if the same operation is submitted twice
      // break the cycle: `on_stop_complete` delegates to the parent operation
      static void on_stop_complete(operation_base* op) noexcept {
        operation::on_write_complete(
          &(static_cast<cancel_operation*>(op)->op_));
      }

      static void on_schedule_stop_complete(operation_base* op) noexcept {
        static_cast<cancel_operation*>(op)->op_.request_stop_local();
      }
    };

    struct cancel_callback final {
      operation& op_;

      void operator()() noexcept { op_.request_stop(); }
    };

    ucx_am_context& context_;
    conn_opt_t conn_;
    std::optional<Payload> data_owned_;
    payload_view_t& data_;
    ucx_request* request_ = nullptr;
    ucx_memory_resource& mr_;
    std::atomic_bool cancel_flag_{false};
    Receiver receiver_;
    manual_lifetime<typename stop_token_type_t<
      Receiver>::template callback_type<cancel_callback>>
      stopCallback_;
    bool stopCallbackConstructed_ = false;
    std::atomic_char refCount_{1};
    cancel_operation cop_{*this};
  };

 public:
  // Produces number of bytes read.
  template <
    template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<>>;

  // Note: Only case it might complete with exception_ptr is if the
  // receiver's set_value() exits with an exception.
  template <template <typename...> class Variant>
  using error_types = Variant<std::error_code, std::exception_ptr>;

  static constexpr bool sends_done = true;

  template <typename P = Payload>
  inline static constexpr bool is_ucx_data_v =
    std::is_same_v<std::decay_t<P>, ucx_am_data>
    || std::is_same_v<std::decay_t<P>, ucx_am_iovec>;

  template <typename P = Payload>
  explicit send_sender_t(
    ucx_am_context& context, conn_pair_t& conn, P& data,
    std::enable_if_t<is_ucx_data_v<P>>* = nullptr) noexcept
    : context_(context),
      conn_(conn.second),
      data_owned_(std::nullopt),
      data_(to_view(data)),
      mr_(context.mr_) {
    [[maybe_unused]] auto conn_opt =
      context_.conn_manager_.get_connection(conn.first);
    UNIFEX_ASSERT(
      conn_opt.has_value() && "The connection must exist in context");
    UNIFEX_ASSERT(
      &(conn_opt.value().get()) == &(conn.second.get())
      && "The connection must be the same");
  }

  template <typename P = Payload>
  explicit send_sender_t(
    ucx_am_context& context, std::uint64_t conn_id, P& data,
    std::enable_if_t<is_ucx_data_v<P>>* = nullptr) noexcept
    : context_(context),
      data_owned_(std::nullopt),
      data_(to_view(data)),
      mr_(context.mr_) {
    [[maybe_unused]] auto conn_opt =
      context_.conn_manager_.get_connection(conn_id);
    UNIFEX_ASSERT(
      conn_opt.has_value() && "The connection must exist in context");
    conn_ = conn_opt.value();
  }

  template <typename P>
  explicit send_sender_t(
    ucx_am_context& context, UcxConnection& conn, P& data,
    std::enable_if_t<is_ucx_data_v<P>>* = nullptr) noexcept
    : context_(context),
      conn_(conn),
      data_owned_(std::nullopt),
      data_(to_view(data)),
      mr_(context.mr_) {
    [[maybe_unused]] auto conn_opt =
      context_.conn_manager_.get_connection(conn.id());
    UNIFEX_ASSERT(
      conn_opt.has_value() && "The connection must exist in context");
  }

  template <typename P>
  inline static constexpr bool is_ucx_data_wrapper_v =
    std::is_same_v<std::decay_t<P>, UcxAmData>
    || std::is_same_v<std::decay_t<P>, UcxAmIovec>;

  template <typename P = Payload>
  explicit send_sender_t(
    ucx_am_context& context, conn_pair_t& conn, P&& data,
    std::enable_if_t<is_ucx_data_wrapper_v<P>>* = nullptr) noexcept
    : context_(context),
      conn_(conn.second),
      data_owned_(std::forward<P>(data)),
      data_(to_view(*data_owned_)),
      mr_(context.mr_) {
    [[maybe_unused]] auto conn_opt =
      context_.conn_manager_.get_connection(conn.first);
    UNIFEX_ASSERT(
      conn_opt.has_value() && "The connection must exist in context");
    UNIFEX_ASSERT(
      &(conn_opt.value().get()) == &(conn.second.get())
      && "The connection must be the same");
  }

  template <typename P = Payload>
  explicit send_sender_t(
    ucx_am_context& context, std::uint64_t conn_id, P&& data,
    std::enable_if_t<is_ucx_data_wrapper_v<P>>* = nullptr) noexcept
    : context_(context),
      data_owned_(std::forward<P>(data)),
      data_(to_view(*data_owned_)),
      mr_(context.mr_) {
    [[maybe_unused]] auto conn_opt =
      context_.conn_manager_.get_connection(conn_id);
    UNIFEX_ASSERT(
      conn_opt.has_value() && "The connection must exist in context");
    conn_ = conn_opt.value();
  }

  template <typename P = Payload>
  explicit send_sender_t(
    ucx_am_context& context, UcxConnection& conn, P&& data,
    std::enable_if_t<is_ucx_data_wrapper_v<P>>* = nullptr) noexcept
    : context_(context),
      conn_(conn),
      data_owned_(std::forward<P>(data)),
      data_(to_view(*data_owned_)),
      mr_(context.mr_) {
    [[maybe_unused]] auto conn_opt =
      context_.conn_manager_.get_connection(conn.id());
    UNIFEX_ASSERT(
      conn_opt.has_value() && "The connection must exist in context");
  }

  template <typename Receiver>
  operation<remove_cvref_t<Receiver>> connect(Receiver&& r) {
    return operation<remove_cvref_t<Receiver>>{
      *this, static_cast<Receiver&&>(r)};
  }

 private:
  friend scheduler;

  ucx_am_context& context_;
  conn_opt_t conn_;
  std::optional<Payload> data_owned_;
  payload_view_t& data_;
  ucx_memory_resource& mr_;
};

class ucx_am_context::schedule_at_sender {
  template <typename Receiver>
  struct operation : schedule_at_operation {
    static constexpr bool is_stop_ever_possible =
      !is_stop_never_possible_v<stop_token_type_t<Receiver>>;

   public:
    explicit operation(
      ucx_am_context& context, const time_point& dueTime, Receiver&& r)
      : schedule_at_operation(
        context, dueTime, get_stop_token(r).stop_possible()),
        receiver_(static_cast<Receiver&&>(r)) {}

    void start() noexcept {
      if (this->context_.is_running_on_io_thread()) {
        start_local();
      } else {
        start_remote();
      }
    }

   private:
    static void on_schedule_complete(operation_base* op) noexcept {
      static_cast<operation*>(op)->start_local();
    }

    static void complete_with_done(operation_base* op) noexcept {
      // Avoid instantiating set_done() if we're not going to call it.
      if constexpr (is_stop_ever_possible) {
        auto& timerOp = *static_cast<operation*>(op);
        unifex::set_done(std::move(timerOp).receiver_);
      } else {
        // This should never be called if stop is not possible.
        UNIFEX_ASSERT(false);
      }
    }

    // Executed when the timer gets to the front of the ready-to-run queue.
    static void maybe_complete_with_value(operation_base* op) noexcept {
      auto& timerOp = *static_cast<operation*>(op);
      if constexpr (is_stop_ever_possible) {
        timerOp.stopCallback_.destruct();

        if (get_stop_token(timerOp.receiver_).stop_requested()) {
          complete_with_done(op);
          return;
        }
      }

      if constexpr (noexcept(unifex::set_value(std::move(timerOp).receiver_))) {
        unifex::set_value(std::move(timerOp).receiver_);
      } else {
        UNIFEX_TRY { unifex::set_value(std::move(timerOp).receiver_); }
        UNIFEX_CATCH(...) {
          unifex::set_error(
            std::move(timerOp).receiver_, std::current_exception());
        }
      }
    }

    static void remove_timer_from_queue_and_complete_with_done(
      operation_base* op) noexcept {
      // Avoid instantiating set_done() if we're never going to call it.
      if constexpr (is_stop_ever_possible) {
        auto& timerOp = *static_cast<operation*>(op);
        UNIFEX_ASSERT(timerOp.context_.is_running_on_io_thread());

        timerOp.stopCallback_.destruct();

        auto state = timerOp.state_.load(std::memory_order_relaxed);
        if ((state & schedule_at_operation::timer_elapsed_flag) == 0) {
          // Timer not yet removed from the timers_ list. Do that now.
          timerOp.context_.remove_timer(&timerOp);
        }

        unifex::set_done(std::move(timerOp).receiver_);
      } else {
        // Should never be called if stop is not possible.
        UNIFEX_ASSERT(false);
      }
    }

    void start_local() noexcept {
      if constexpr (is_stop_ever_possible) {
        if (get_stop_token(receiver_).stop_requested()) {
          // Stop already requested. Don't bother adding the timer.
          this->execute_ = &operation::complete_with_done;
          this->context_.schedule_local(this);
          return;
        }
      }

      this->execute_ = &operation::maybe_complete_with_value;
      this->context_.schedule_at_impl(this);

      if constexpr (is_stop_ever_possible) {
        stopCallback_.construct(
          get_stop_token(receiver_), cancel_callback{*this});
      }
    }

    void start_remote() noexcept {
      this->execute_ = &operation::on_schedule_complete;
      this->context_.schedule_remote(this);
    }

    void request_stop() noexcept {
      if (context_.is_running_on_io_thread()) {
        request_stop_local();
      } else {
        request_stop_remote();
      }
    }

    void request_stop_local() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());

      stopCallback_.destruct();

      this->execute_ = &operation::complete_with_done;

      auto state = this->state_.load(std::memory_order_relaxed);
      if ((state & schedule_at_operation::timer_elapsed_flag) == 0) {
        // Timer not yet elapsed.
        // Remove timer from list of timers and enqueue cancellation.
        context_.remove_timer(this);
        context_.schedule_local(this);
      } else {
        // Timer already elapsed and added to ready-to-run queue.
      }
    }

    void request_stop_remote() noexcept {
      auto oldState = this->state_.fetch_add(
        schedule_at_operation::cancel_pending_flag, std::memory_order_acq_rel);
      if ((oldState & schedule_at_operation::timer_elapsed_flag) == 0) {
        // Timer had not yet elapsed.
        // We are responsible for scheduling the completion of this timer
        // operation.
        this->execute_ =
          &operation::remove_timer_from_queue_and_complete_with_done;
        this->context_.schedule_remote(this);
      }
    }

    struct cancel_callback {
      operation& op_;

      void operator()() noexcept { op_.request_stop(); }
    };

    Receiver receiver_;
    manual_lifetime<typename stop_token_type_t<
      Receiver>::template callback_type<cancel_callback>>
      stopCallback_;
  };

 public:
  template <
    template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<>>;

  // Note: Only case it might complete with exception_ptr is if the
  // receiver's set_value() exits with an exception.
  template <template <typename...> class Variant>
  using error_types = Variant<std::exception_ptr>;

  static constexpr bool sends_done = true;

  explicit schedule_at_sender(
    ucx_am_context& context, const time_point dueTime) noexcept
    : context_(context), dueTime_(dueTime) {}

  template <typename Receiver>
  operation<remove_cvref_t<Receiver>> connect(Receiver&& r) {
    return operation<remove_cvref_t<Receiver>>{
      context_, dueTime_, static_cast<Receiver&&>(r)};
  }

 private:
  ucx_am_context& context_;
  time_point dueTime_;
};

class ucx_am_context::scheduler {
 public:
  scheduler(const scheduler&) noexcept = default;
  scheduler& operator=(const scheduler&) = default;
  ~scheduler() = default;

  /**
   * @brief Creates a sender that completes on the context's event loop.
   * @return A schedule_sender.
   */
  schedule_sender schedule() const noexcept {
    return schedule_sender{*context_};
  }

  /**
   * @brief Returns the current time from the context's monotonic clock.
   * @return The current time point.
   */
  time_point now() const noexcept { return monotonic_clock::now(); }

  /**
   * @brief Creates a sender that completes at a specific time.
   * @param dueTime The time point for completion.
   * @return A schedule_at_sender.
   */
  schedule_at_sender schedule_at(const time_point dueTime) const noexcept {
    return schedule_at_sender{*context_, dueTime};
  }

  /**
   * @brief Creates a sender that completes after a specified duration.
   * @tparam Rep The representation type of the duration.
   * @tparam Ratio The ratio type of the duration.
   * @param duration The duration to wait before completion.
   * @return A schedule_at_sender.
   */
  template <typename Rep, typename Ratio>
  schedule_at_sender schedule_after(
    const std::chrono::duration<Rep, Ratio> duration) const noexcept {
    return schedule_at_sender{
      *context_, now() + std::chrono::duration_cast<time_duration>(duration)};
  }

 private:
  friend ucx_am_context;

  friend accept_connection tag_invoke(
    tag_t<accept_endpoint>, scheduler scheduler, port_t port);
  friend accept_connection tag_invoke(
    tag_t<accept_endpoint>,
    scheduler scheduler,
    std::unique_ptr<sockaddr>
      socket,
    size_t addrlen);
  friend connect_sender tag_invoke(
    tag_t<connect_endpoint>, scheduler scheduler,
    std::unique_ptr<sockaddr> src_saddr, std::unique_ptr<sockaddr> dst_saddr,
    socklen_t addrlen);
  friend connect_sender tag_invoke(
    tag_t<connect_endpoint>, scheduler scheduler, std::nullptr_t src_saddr,
    std::unique_ptr<sockaddr> dst_saddr, socklen_t addrlen);
  friend connect_sender tag_invoke(
    tag_t<connect_endpoint>, scheduler scheduler,
    std::unique_ptr<sockaddr> src_saddr, std::unique_ptr<sockaddr> dst_saddr,
    size_t addrlen);
  friend connect_sender tag_invoke(
    tag_t<connect_endpoint>, scheduler scheduler, std::nullptr_t src_saddr,
    std::unique_ptr<sockaddr> dst_saddr, size_t addrlen);
  friend connect_sender tag_invoke(
    tag_t<connect_endpoint>, scheduler scheduler,
    std::vector<std::byte>& address_buffer);
  friend connect_sender tag_invoke(
    tag_t<connect_endpoint>, scheduler scheduler,
    std::vector<std::byte>&& address_buffer);
  friend dispatch_connection_error_sender tag_invoke(
    tag_t<handle_error_connection>, scheduler scheduler,
    std::function<bool(std::uint64_t conn_id, ucs_status_t status)> handler);
  friend dispatch_connection_error_sender tag_invoke(
    tag_t<handle_error_connection>, scheduler scheduler,
    std::function<bool(std::uint64_t conn_id, std::error_code status)> handler);
  friend send_sender tag_invoke(
    tag_t<connection_send>, scheduler scheduler, conn_pair_t& conn,
    ucx_am_data& data);
  friend send_sender tag_invoke(
    tag_t<connection_send>, scheduler scheduler, std::uint64_t conn_id,
    ucx_am_data& data);
  friend send_sender tag_invoke(
    tag_t<connection_send>, scheduler scheduler, UcxConnection& conn,
    ucx_am_data& data);
  friend send_sender_move tag_invoke(
    tag_t<connection_send>, scheduler scheduler, conn_pair_t& conn,
    UcxAmData&& data);
  friend send_sender_move tag_invoke(
    tag_t<connection_send>, scheduler scheduler, std::uint64_t conn_id,
    UcxAmData&& data);
  friend send_sender_move tag_invoke(
    tag_t<connection_send>, scheduler scheduler, UcxConnection& conn,
    UcxAmData&& data);
  friend send_iovec_sender tag_invoke(
    tag_t<connection_send>, scheduler scheduler, conn_pair_t& conn,
    ucx_am_iovec& iovec);
  friend send_iovec_sender tag_invoke(
    tag_t<connection_send>, scheduler scheduler, std::uint64_t conn_id,
    ucx_am_iovec& iovec);
  friend send_iovec_sender tag_invoke(
    tag_t<connection_send>, scheduler scheduler, UcxConnection& conn,
    ucx_am_iovec& iovec);
  friend send_iovec_sender_move tag_invoke(
    tag_t<connection_send>, scheduler scheduler, conn_pair_t& conn,
    UcxAmIovec&& iovec);
  friend send_iovec_sender_move tag_invoke(
    tag_t<connection_send>, scheduler scheduler, std::uint64_t conn_id,
    UcxAmIovec&& iovec);
  friend send_iovec_sender_move tag_invoke(
    tag_t<connection_send>, scheduler scheduler, UcxConnection& conn,
    UcxAmIovec&& iovec);
  friend recv_sender tag_invoke(
    tag_t<connection_recv>, scheduler scheduler, ucx_am_data& data);
  friend recv_sender tag_invoke(
    tag_t<connection_recv>, scheduler scheduler, ucx_memory_type data_type);
  friend recv_header_sender tag_invoke(
    tag_t<connection_recv_header>, scheduler scheduler);
  friend recv_buffer_sender tag_invoke(
    tag_t<connection_recv_buffer>, scheduler scheduler, size_t am_desc_key,
    ucx_memory_type memory_type);
  friend recv_buffer_sender tag_invoke(
    tag_t<connection_recv_buffer>, scheduler scheduler, size_t am_desc_key,
    UcxBuffer&& buffer);
  friend recv_iovec_buffer_sender tag_invoke(
    tag_t<connection_recv_buffer>, scheduler scheduler, size_t am_desc_key,
    UcxBufferVec&& buffers);

  friend bool operator==(scheduler a, scheduler b) noexcept {
    return a.context_ == b.context_;
  }
  friend bool operator!=(scheduler a, scheduler b) noexcept {
    return a.context_ != b.context_;
  }

  explicit scheduler(ucx_am_context& context) noexcept : context_(&context) {}

  ucx_am_context* context_;
};

/**
 * @brief Gets a scheduler for the ucx_am_context.
 * @return A scheduler associated with this context.
 */
inline ucx_am_context::scheduler ucx_am_context::get_scheduler() noexcept {
  return scheduler{*this};
}

class ucx_am_context::ucx_accept_callback : public ucx_callback {
 public:
  ucx_accept_callback(ucx_am_context& context, ucx_connection* connection)
    : context_(context), connection_(connection) {
    client_id_ = context.get_client_id();
  }

  virtual void operator()(ucs_status_t status) {
    UNIFEX_ASSERT(connection_->ucx_status() == status);
  }

 private:
  ucx_am_context& context_;
  ucx_connection* connection_;
};

class ucx_am_context::ucx_connect_callback : public ucx_callback {
 public:
  ucx_connect_callback(ucx_am_context& context, ucx_connection* connection)
    : context_(context), connection_(connection) {
    client_id_ = context.get_client_id();
  }

  virtual void operator()(ucs_status_t status) {
    UNIFEX_ASSERT(connection_->ucx_status() == status);
  }

 private:
  ucx_am_context& context_;
  ucx_connection* connection_;
};

class ucx_am_context::ucx_disconnect_callback : public ucx_callback {
 public:
  explicit ucx_disconnect_callback(ucx_am_context& context) noexcept
    : context_(context) {}

  virtual void operator()(ucs_status_t status);

  void mark_inactive(std::uint64_t conn_id) override;
  void mark_disconnecting_from_inactive(std::uint64_t conn_id) override;
  void mark_failed_from_inactive(std::uint64_t conn_id) override;

 private:
  ucx_am_context& context_;
};

class ucx_am_context::ucx_handle_err_callback : public ucx_callback {
 public:
  explicit ucx_handle_err_callback(ucx_am_context& context) noexcept
    : context_(context) {}

  virtual void operator()(ucs_status_t status);

  void handle_connection_error(
    ucs_status_t status, std::uint64_t conn_id) override;

 private:
  ucx_am_context& context_;
};

/**
 * @class dispatch_connection_error_sender
 * @brief The connection_error_handler_ is a callback function that processes
 * connection errors. It takes a connection ID and a UCX status code as
 * parameters and returns a boolean value.
 *
 * This handler allows custom error handling logic for different connection
 * scenarios:
 * - When a connection error occurs, this handler is invoked with the affected
 * connection ID and the specific error status from UCX
 * - The return value determines whether the endpoint should be reconnected:
 *   - Return true: The system will attempt to reestablish the connection
 *   - Return false: The connection will be permanently closed
 *
 * This provides flexibility to implement different recovery strategies based on
 * the specific error type or connection importance.
 */
class ucx_am_context::dispatch_connection_error_sender {
 public:
  template <typename Receiver>
  class operation : private completion_base {
    friend ucx_am_context;

   public:
    template <typename Receiver2>
    explicit operation(
      const dispatch_connection_error_sender& sender,
      Receiver2&&
        r) noexcept(std::is_nothrow_constructible_v<Receiver, Receiver2>)
      : context_(sender.context_),
        connection_error_handler_(sender.connection_error_handler_),
        receiver_(static_cast<Receiver2&&>(r)) {}

    operation(operation&&) = delete;

    void start() noexcept {
      if (!context_.is_running_on_io_thread()) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_remote(this);
      } else {
        start_io();
      }
    }

   private:
    static void on_schedule_complete(operation_base* op) noexcept {
      static_cast<operation*>(op)->start_io();
    }

    void start_io() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      if (!stopCallbackConstructed_) {
        stopCallback_.construct(
          get_stop_token(receiver_), cancel_callback{*this});
        stopCallbackConstructed_ = true;
      }
      auto populateSqe = [this]() noexcept {
        if (cancel_flag_.load(std::memory_order_consume)) {
          this->result_ = UCS_ERR_CANCELED;
          return true;
        }
        // Get failed connections
        auto& failed_conns = context_.conn_manager_.get_failed_connections();
        if (
          failed_conns.empty() && retryTimes_ <= HANDLE_IF_ERROR_RETRY_TIMES) {
          // No failed connections, add this operation to pending queue
          this->execute_ = &operation::on_schedule_complete;
          this->retryTimes_++;
          return false;
        } else {
          // Process all failed connections
          ucs_status_t res = UCS_OK;
          while (!failed_conns.empty()) {
            auto old_conn = std::move(failed_conns.front());
            failed_conns.pop_front();
            auto old_conn_id = old_conn->id();
            auto should_reconnect =
              connection_error_handler_(old_conn_id, old_conn->ucx_status());
            // If should reconnect, reconnect the connection
            if (should_reconnect) {
              auto new_conn_id = context_.recreate_connection_from_failed(
                conn_pair_t{old_conn_id, std::ref(*old_conn)});
              auto conn_opt =
                context_.conn_manager_.get_connection(new_conn_id);
              assert(conn_opt.has_value());
              res = conn_opt.value().get().ucx_status();
            }
            // Add the connection to the disconnecting queue and disconnect it
            // asynchronously when context is stopped.
            // TODO(He Jia): Refactor this ugly code, should not use raw
            // pointer and better do more things inside the UcxConnection class
            // but not here.
            auto conn_ptr = old_conn.get();
            context_.conn_manager_.add_connection_to_disconnecting_queue(
              std::move(old_conn));
            conn_ptr->disconnect_direct();
          }

          auto& entry = this->context_.get_completion_queue_entry();
          entry.user_data = reinterpret_cast<std::uintptr_t>(
            static_cast<completion_base*>(this));
          entry.res = res;
          this->execute_ = &operation::on_dispatch;
          this->result_ = entry.res;
        }

        return true;
      };

      if (!context_.try_submit_io(populateSqe)) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_pending_io(this);
      }
    }

    void request_stop() noexcept {
      if (char expected = 1; !refCount_.compare_exchange_strong(
            expected, 2, std::memory_order_relaxed)) {
        UNIFEX_ASSERT(expected == 0);
        return;
      }

      cancel_flag_.store(true, std::memory_order_release);

      if (context_.is_running_on_io_thread()) {
        request_stop_local();
      } else {
        request_stop_remote();
      }
    }

    void request_stop_local() noexcept {
      this->execute_ = &operation::on_dispatch;
      context_.schedule_local(this);
    }

    void request_stop_remote() noexcept {
      this->execute_ = &operation::on_schedule_stop_complete;
      context_.schedule_remote(this);
    }

    static void on_schedule_stop_complete(operation_base* op) noexcept {
      static_cast<operation*>(op)->request_stop_local();
    }

    static void on_dispatch(operation_base* op) noexcept {
      auto& self = *static_cast<operation*>(op);
      if (self.refCount_.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        // Waiting for stop function to complete
        return;
      }
      self.stopCallback_.destruct();
      if (get_stop_token(self.receiver_).stop_requested()) {
        unifex::set_done(std::move(self.receiver_));
      } else if (self.result_ >= 0) {
        if constexpr (noexcept(unifex::set_value(std::move(self.receiver_)))) {
          unifex::set_value(std::move(self.receiver_));
        } else {
          UNIFEX_TRY { unifex::set_value(std::move(self.receiver_)); }
          UNIFEX_CATCH(...) {
            unifex::set_error(
              std::move(self.receiver_), std::current_exception());
          }
        }
      } else if (self.result_ == UCS_ERR_CANCELED) {
        unifex::set_done(std::move(self.receiver_));
      } else {
        unifex::set_error(
          std::move(self.receiver_),
          stdexe_ucx_runtime::make_error_code(
            static_cast<ucs_status_t>(self.result_)));
      }
    }

    struct cancel_operation final : completion_base {
      operation& op_;

      explicit cancel_operation(operation& op) noexcept : op_(op) {}

      static void on_stop_complete(operation_base* op) noexcept {
        operation::on_dispatch(&(static_cast<cancel_operation*>(op)->op_));
      }

      static void on_schedule_stop_complete(operation_base* op) noexcept {
        static_cast<cancel_operation*>(op)->op_.request_stop_local();
      }
    };

    struct cancel_callback final {
      operation& op_;

      void operator()() noexcept { op_.request_stop(); }
    };

    ucx_am_context& context_;
    std::function<bool(std::uint64_t conn_id, ucs_status_t status)>
      connection_error_handler_;
    static constexpr const int HANDLE_IF_ERROR_RETRY_TIMES = 1;
    int retryTimes_{0};
    std::atomic_bool cancel_flag_{false};
    Receiver receiver_;
    manual_lifetime<typename stop_token_type_t<
      Receiver>::template callback_type<cancel_callback>>
      stopCallback_;
    bool stopCallbackConstructed_ = false;
    std::atomic_char refCount_{1};
    cancel_operation cop_{*this};
  };

 public:
  template <
    template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<>>;

  template <template <typename...> class Variant>
  using error_types = Variant<std::error_code, std::exception_ptr>;

  static constexpr bool sends_done = true;
  static constexpr blocking_kind blocking = blocking_kind::never;
  static constexpr bool is_always_scheduler_affine = false;

  /**
   * @brief Construct a new dispatch connection error sender object
   *
   * @param context The ucx_am_context to use.
   * @param connection_error_handler A function to call when a connection error
   * occurs. It should return true to attempt reconnection, false otherwise.
   */
  explicit dispatch_connection_error_sender(
    ucx_am_context& context,
    std::function<bool(std::uint64_t, ucs_status_t status)>
      connection_error_handler =
        []([[maybe_unused]] std::uint64_t, [[maybe_unused]] ucs_status_t status)
      -> bool { return false; })
    : context_(context),
      connection_error_handler_(std::move(connection_error_handler)) {}

  explicit dispatch_connection_error_sender(
    ucx_am_context& context,
    std::function<bool(std::uint64_t, std::error_code status)>
      connection_error_handler =
        [](
          [[maybe_unused]] std::uint64_t,
          [[maybe_unused]] std::error_code status) -> bool { return false; })
    : context_(context) {
    connection_error_handler_ = [handler = std::move(connection_error_handler)](
                                  std::uint64_t conn_id,
                                  ucs_status_t status) -> bool {
      return handler(conn_id, stdexe_ucx_runtime::make_error_code(status));
    };
  }

  /**
   * @brief Connects a receiver to the sender.
   * @tparam Receiver The type of the receiver.
   * @param r The receiver to connect.
   * @return An operation state object.
   */
  template <typename Receiver>
  operation<remove_cvref_t<Receiver>> connect(Receiver&& r) && {
    return operation<remove_cvref_t<Receiver>>{
      *this, static_cast<Receiver&&>(r)};
  }

 private:
  friend scheduler;

  ucx_am_context& context_;
  std::function<bool(std::uint64_t, ucs_status_t status)>
    connection_error_handler_;
};

class ucx_am_context::connect_sender {
  template <typename Receiver>
  class operation : private completion_base {
    friend ucx_am_context;

   public:
    template <typename Receiver2>
    explicit operation(const connect_sender& sender, Receiver2&& r) noexcept(
      std::is_nothrow_constructible_v<Receiver, Receiver2>)
      : context_(sender.context_),
        src_saddr_(std::move(const_cast<connect_sender&>(sender).src_saddr_)),
        dst_saddr_(std::move(const_cast<connect_sender&>(sender).dst_saddr_)),
        addrlen_(sender.addrlen_),
        ucp_address_buffer_(std::move(sender.ucp_address_buffer_)),
        receiver_(static_cast<Receiver2&&>(r)) {}

    operation(operation&&) = delete;

    void start() noexcept {
      if (!context_.is_running_on_io_thread()) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_remote(this);
      } else {
        start_io();
      }
    }

   private:
    static void on_schedule_complete(operation_base* op) noexcept {
      static_cast<operation*>(op)->start_io();
    }

    void start_io() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      if (!stopCallbackConstructed_) {
        stopCallback_.construct(
          get_stop_token(receiver_), cancel_callback{*this});
        stopCallbackConstructed_ = true;
      }
      auto populateSqe = [this]() noexcept {
        // Check if operation has been canceled
        if (cancel_flag_.load(std::memory_order_consume)) {
          this->result_ = UCS_ERR_CANCELED;
          return true;
        }

        auto& entry = this->context_.get_completion_queue_entry();
        entry.user_data =
          reinterpret_cast<std::uintptr_t>(static_cast<completion_base*>(this));
        this->execute_ = &operation::on_connect;

        // Create a new connection using the provided addresses
        if (ucp_address_buffer_.size() > 0) {
          conn_id_ = context_.create_new_connection(
            reinterpret_cast<const ucp_address_t*>(ucp_address_buffer_.data()));
        } else {
          conn_id_ = context_.create_new_connection(
            src_saddr_.get(), dst_saddr_.get(), addrlen_);
        }
        auto conn_opt = context_.conn_manager_.get_connection(conn_id_);

        if (!conn_opt.has_value()) {
          // Connection creation failed
          entry.res = UCS_ERR_UNREACHABLE;
          this->result_ = UCS_ERR_UNREACHABLE;
        } else {
          // Store the connection reference
          conn_ = conn_opt.value();
          entry.res = conn_.value().get().ucx_status();
          this->result_ = entry.res;
        }

        return true;
      };

      if (!context_.try_submit_io(populateSqe)) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_pending_io(this);
      }
    }

    void request_stop() noexcept {
      if (char expected = 1; !refCount_.compare_exchange_strong(
            expected, 2, std::memory_order_relaxed)) {
        // lost race with on_connect
        UNIFEX_ASSERT(expected == 0);
        return;
      }

      cancel_flag_.store(true, std::memory_order_release);

      if (context_.is_running_on_io_thread()) {
        request_stop_local();
      } else {
        request_stop_remote();
      }
    }

    void request_stop_local() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      auto populateSqe = [this]() noexcept {
        auto cop = reinterpret_cast<std::uintptr_t>(
          static_cast<completion_base*>(&cop_));
        cop_.execute_ = &cancel_operation::on_stop_complete;

        // Update the cqe entry
        auto& entry = this->context_.get_completion_queue_entry();
        entry.user_data = cop;
        entry.res = UCS_ERR_CANCELED;
        this->result_ = UCS_ERR_CANCELED;

        // If we have a valid connection, disconnect it
        if (
          conn_id_ != 0
          && context_.conn_manager_.is_connection_valid(conn_id_)) {
          if (conn_.has_value()) {
            // Create a callback for disconnection
            auto disconnect_cb =
              std::make_unique<ucx_disconnect_callback>(context_);
            conn_.value().get().disconnect(std::move(disconnect_cb));
          }
        }
      };

      if (!context_.try_submit_io(populateSqe)) {
        cop_.execute_ = &cancel_operation::on_schedule_stop_complete;
        context_.schedule_pending_io(&cop_);
      }
    }

    void request_stop_remote() noexcept {
      cop_.execute_ = &cancel_operation::on_schedule_stop_complete;
      context_.schedule_remote(&cop_);
    }

    static void on_connect(operation_base* op) noexcept {
      auto& self = *static_cast<operation*>(op);
      if (self.refCount_.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        // Waiting for stop function to complete
        return;
      }
      self.stopCallback_.destruct();
      if (get_stop_token(self.receiver_).stop_requested()) {
        unifex::set_done(std::move(self.receiver_));
      } else if (self.result_ >= 0) {
        if constexpr (noexcept(unifex::set_value(
                        std::move(self.receiver_),
                        static_cast<std::uint64_t>(self.conn_id_)))) {
          unifex::set_value(
            std::move(self.receiver_),
            static_cast<std::uint64_t>(self.conn_id_));
        } else {
          UNIFEX_TRY {
            unifex::set_value(
              std::move(self.receiver_),
              static_cast<std::uint64_t>(self.conn_id_));
          }
          UNIFEX_CATCH(...) {
            unifex::set_error(
              std::move(self.receiver_), std::current_exception());
          }
        }
      } else if (self.result_ == UCS_ERR_CANCELED) {
        unifex::set_done(std::move(self.receiver_));
      } else {
        unifex::set_error(
          std::move(self.receiver_),
          stdexe_ucx_runtime::make_error_code(
            static_cast<ucs_status_t>(self.result_)));
      }
    }

    struct cancel_operation final : completion_base {
      operation& op_;

      explicit cancel_operation(operation& op) noexcept : op_(op) {}
      // intrusive list breaks if the same operation is submitted twice
      // break the cycle: `on_stop_complete` delegates to the parent operation
      static void on_stop_complete(operation_base* op) noexcept {
        operation::on_connect(&(static_cast<cancel_operation*>(op)->op_));
      }

      static void on_schedule_stop_complete(operation_base* op) noexcept {
        static_cast<cancel_operation*>(op)->op_.request_stop_local();
      }
    };

    struct cancel_callback final {
      operation& op_;

      void operator()() noexcept { op_.request_stop(); }
    };

    ucx_am_context& context_;
    std::unique_ptr<sockaddr> src_saddr_;
    std::unique_ptr<sockaddr> dst_saddr_;
    socklen_t addrlen_;
    std::vector<std::byte> ucp_address_buffer_;
    std::uint64_t conn_id_ = 0;
    std::optional<std::reference_wrapper<UcxConnection>> conn_;
    std::atomic_bool cancel_flag_{false};
    Receiver receiver_;
    manual_lifetime<typename stop_token_type_t<
      Receiver>::template callback_type<cancel_callback>>
      stopCallback_;
    bool stopCallbackConstructed_ = false;
    std::atomic_char refCount_{1};
    cancel_operation cop_{*this};
  };

 public:
  // Produces the connection ID of the established connection
  template <
    template <typename...> class Variant, template <typename...> class Tuple>
  using value_types = Variant<Tuple<std::uint64_t>>;

  // Note: Only case it might complete with exception_ptr is if the
  // receiver's set_value() exits with an exception.
  template <template <typename...> class Variant>
  using error_types = Variant<std::error_code, std::exception_ptr>;

  static constexpr bool sends_done = true;
  static constexpr blocking_kind blocking = blocking_kind::never;
  // always completes on the ucx_am context
  static constexpr bool is_always_scheduler_affine = false;

  /**
   * @brief Construct a new connect sender object
   *
   * @param context The ucx_am_context to use.
   * @param src_saddr The source socket address. Can be null.
   * @param dst_saddr The destination socket address.
   * @param addrlen The length of the socket address structure.
   */
  connect_sender(
    ucx_am_context& context,
    std::unique_ptr<sockaddr>
      src_saddr,
    std::unique_ptr<sockaddr>
      dst_saddr,
    socklen_t addrlen) noexcept
    : context_(context),
      src_saddr_(std::move(src_saddr)),
      dst_saddr_(std::move(dst_saddr)),
      addrlen_(addrlen),
      ucp_address_buffer_() {}

  /**
   * @brief Construct a new connect sender object with a UCX address.
   *
   * @param context The ucx_am_context to use.
   * @param ucp_address_buffer The UCP address from ucp_worker_get_address.
   */
  connect_sender(
    ucx_am_context& context,
    std::vector<std::byte>&& ucp_address_buffer) noexcept
    : context_(context), ucp_address_buffer_(std::move(ucp_address_buffer)) {}

  /**
   * @brief Move constructor for connect_sender.
   */
  connect_sender(connect_sender&& other) noexcept
    : context_(other.context_),
      src_saddr_(std::move(other.src_saddr_)),
      dst_saddr_(std::move(other.dst_saddr_)),
      addrlen_(other.addrlen_),
      ucp_address_buffer_(std::move(other.ucp_address_buffer_)) {}

  /**
   * @brief Connects a receiver to the sender.
   * @tparam Receiver The type of the receiver.
   * @param r The receiver to connect.
   * @return An operation state object.
   */
  template <typename Receiver>
  operation<remove_cvref_t<Receiver>> connect(Receiver&& r) && {
    return operation<remove_cvref_t<Receiver>>{
      *this, static_cast<Receiver&&>(r)};
  }

 private:
  friend scheduler;

  ucx_am_context& context_;
  std::unique_ptr<sockaddr> src_saddr_;
  std::unique_ptr<sockaddr> dst_saddr_;
  socklen_t addrlen_;
  std::vector<std::byte> ucp_address_buffer_;
};

class ucx_am_context::accept_sender {
  template <typename Receiver>
  class operation : private completion_base {
    friend ucx_am_context;

   public:
    template <typename Receiver2>
    explicit operation(const accept_sender& sender, Receiver2&& r) noexcept(
      std::is_nothrow_constructible_v<Receiver, Receiver2>)
      : context_(sender.context_), receiver_(static_cast<Receiver2&&>(r)) {}

    operation(operation&&) = delete;

    void start() noexcept {
      if (!context_.is_running_on_io_thread()) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_remote(this);
      } else {
        start_io();
      }
    }

   private:
    static void on_schedule_complete(operation_base* op) noexcept {
      static_cast<operation*>(op)->start_io();
    }

    void start_io() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      if (!stopCallbackConstructed_) {
        stopCallback_.construct(
          get_stop_token(receiver_), cancel_callback{*this});
        stopCallbackConstructed_ = true;
      }
      auto populateSqe = [this]() noexcept {
        // Check if operation has been canceled
        if (cancel_flag_.load(std::memory_order_consume)) {
          this->result_ = UCS_ERR_CANCELED;
          return true;
        }
        // Check pending connection requests being created by callback
        if (context_.epConnReqQueue_.empty()) {
          onPending_.store(true, std::memory_order_relaxed);
          // No pending connection request, add this operation to pending queue
          this->execute_ = &operation::on_schedule_complete;
          this->result_ = UCS_ERR_NO_MESSAGE;
          --this->context_.sqUnflushedCount_;
          this->context_.pendingAcptIoQueue_.push_front(this);
          return true;
        } else {
          onPending_.store(false, std::memory_order_relaxed);
          connIdStatusVector_ = context_.progress_pending_conn_requests();
          // Update the cqe entry
          auto& entry = this->context_.get_completion_queue_entry();
          entry.user_data = reinterpret_cast<std::uintptr_t>(
            static_cast<completion_base*>(this));
          entry.res = UCS_OK;
          this->execute_ = &operation::on_accept;
          this->result_ = UCS_OK;
        }

        return true;
      };

      if (!context_.try_submit_io(populateSqe)) {
        this->execute_ = &operation::on_schedule_complete;
        context_.schedule_pending_io(this);
      }
    }

    void request_stop() noexcept {
      if (!onPending_.load(std::memory_order_relaxed)) {
        return;
      }
      if (char expected = 1; !refCount_.compare_exchange_strong(
            expected, 2, std::memory_order_relaxed)) {
        // If refCount_ is 0 means the operation is already completed
        // If onPending_ is true means the operation is pending to be
        // rescheduled, no side effect to be cancelled
        UNIFEX_ASSERT(expected == 0);  // lost race with on_accept
        return;
      }

      cancel_flag_.store(true, std::memory_order_release);

      if (context_.is_running_on_io_thread()) {
        request_stop_local();
      } else {
        request_stop_remote();
      }
    }

    void request_stop_local() noexcept {
      UNIFEX_ASSERT(context_.is_running_on_io_thread());
      auto populateSqe = [this]() noexcept {
        auto cop = reinterpret_cast<std::uintptr_t>(
          static_cast<completion_base*>(&cop_));
        cop_.execute_ = &cancel_operation::on_stop_complete;

        // Update the cqe entry
        auto& entry = this->context_.get_completion_queue_entry();
        entry.user_data = cop;
        entry.res = UCS_ERR_CANCELED;
        this->result_ = UCS_ERR_CANCELED;

        // Ensure the operation is removed from the queue
        std::atomic_thread_fence(std::memory_order_acquire);
        try {
          context_.pendingAcptIoQueue_.erase(
            std::remove_if(
              context_.pendingAcptIoQueue_.begin(),
              context_.pendingAcptIoQueue_.end(),
              [this](operation_base* op) { return op == this; }),
            context_.pendingAcptIoQueue_.end());
        } catch (const std::exception& e) {
          UCX_CTX_ERROR << "std::erase_if context_.pendingAcptIoQueue_ failed: "
                        << e.what() << std::endl;
        }
        std::atomic_thread_fence(std::memory_order_release);
      };

      if (!context_.try_submit_io(populateSqe)) {
        cop_.execute_ = &cancel_operation::on_schedule_stop_complete;
        context_.schedule_pending_io(&cop_);
      }
    }

    void request_stop_remote() noexcept {
      cop_.execute_ = &cancel_operation::on_schedule_stop_complete;
      context_.schedule_remote(&cop_);
    }

    static void on_accept(operation_base* op) noexcept {
      auto& self = *static_cast<operation*>(op);
      if (
        self.refCount_.fetch_sub(1, std::memory_order_acq_rel) != 1
        && !self.onPending_.load(std::memory_order_acquire)) {
        // stop callback is running, must complete the op
        return;
      }
      self.stopCallback_.destruct();
      if (get_stop_token(self.receiver_).stop_requested()) {
        unifex::set_done(std::move(self.receiver_));
      } else if (self.result_ >= 0) {
        std::vector<std::pair<std::uint64_t, std::error_code>>
          conn_id_status_copy = self.connIdStatusVector_;
        if constexpr (noexcept(unifex::set_value(
                        std::move(self.receiver_),
                        std::move(conn_id_status_copy)))) {
          unifex::set_value(
            std::move(self.receiver_), std::move(conn_id_status_copy));
        } else {
          UNIFEX_TRY {
            unifex::set_value(
              std::move(self.receiver_), std::move(conn_id_status_copy));
          }
          UNIFEX_CATCH(...) {
            unifex::set_error(
              std::move(self.receiver_), std::current_exception());
          }
        }
      } else if (self.result_ == UCS_ERR_CANCELED) {
        unifex::set_done(std::move(self.receiver_));
      } else {
        unifex::set_error(
          std::move(self.receiver_),
          stdexe_ucx_runtime::make_error_code(
            static_cast<ucs_status_t>(self.result_)));
      }
    }

    struct cancel_operation final : completion_base {
      operation& op_;

      explicit cancel_operation(operation& op) noexcept : op_(op) {}
      // intrusive list breaks if the same operation is submitted twice
      // break the cycle: `on_stop_complete` delegates to the parent operation
      static void on_stop_complete(operation_base* op) noexcept {
        operation::on_accept(&(static_cast<cancel_operation*>(op)->op_));
      }

      static void on_schedule_stop_complete(operation_base* op) noexcept {
        static_cast<cancel_operation*>(op)->op_.request_stop_local();
      }
    };

    struct cancel_callback final {
      operation& op_;

      void operator()() noexcept { op_.request_stop(); }
    };

    ucx_am_context& context_;
    std::vector<std::pair<std::uint64_t, std::error_code>> connIdStatusVector_;
    std::atomic_bool cancel_flag_{false};
    Receiver receiver_;
    manual_lifetime<typename stop_token_type_t<
      Receiver>::template callback_type<cancel_callback>>
      stopCallback_;
    bool stopCallbackConstructed_ = false;
    std::atomic_char refCount_{1};
    std::atomic_bool onPending_{true};
    cancel_operation cop_{*this};
  };

 public:
  // Produces an open read-write file corresponding to the accepted connection.
  template <
    template <typename...> class Variant, template <typename...> class Tuple>
  using value_types =
    Variant<Tuple<std::vector<std::pair<std::uint64_t, std::error_code>>>>;

  // Note: Only case it might complete with exception_ptr is if the
  // receiver's set_value() exits with an exception.
  template <template <typename...> class Variant>
  using error_types = Variant<std::error_code, std::exception_ptr>;

  static constexpr bool sends_done = true;
  static constexpr blocking_kind blocking = blocking_kind::never;
  // always completes on the ucx_am context
  static constexpr bool is_always_scheduler_affine = false;

  /**
   * @brief Construct a new accept sender object
   * @param context The ucx_am_context to use.
   */
  explicit accept_sender(ucx_am_context& context) noexcept
    : context_(context) {}

  /**
   * @brief Connects a receiver to the sender.
   * @tparam Receiver The type of the receiver.
   * @param r The receiver to connect.
   * @return An operation state object.
   */
  template <typename Receiver>
  operation<remove_cvref_t<Receiver>> connect(Receiver&& r) && {
    return operation<remove_cvref_t<Receiver>>{
      *this, static_cast<Receiver&&>(r)};
  }

 private:
  friend scheduler;

  ucx_am_context& context_;
};

class ucx_am_context::accept_connection {
 public:
  /**
   * @brief Construct a new accept connection object
   *
   * @param context The ucx_am_context to use.
   * @param socket The socket address to listen on.
   * @param addrlen The length of the socket address structure.
   */
  accept_connection(
    ucx_am_context& context,
    std::unique_ptr<sockaddr>
      socket,
    size_t addrlen) noexcept
    : context_(context), socket_(std::move(socket)), addrlen_(addrlen) {}

  /**
   * @brief Construct a new accept connection object
   *
   * @param context The ucx_am_context to use.
   * @param port The port number to listen on.
   */
  accept_connection(ucx_am_context& context, port_t port) noexcept
    : context_(context) {
    sockaddr_in* addr = new sockaddr_in{
      .sin_family = AF_INET,
      .sin_port = htons(port),
      .sin_addr = {.s_addr = INADDR_ANY}};
    socket_ = std::unique_ptr<sockaddr>(reinterpret_cast<sockaddr*>(addr));
    addrlen_ = sizeof(sockaddr_in);
  }

  /**
   * @brief Move constructor for accept_connection.
   */
  accept_connection(accept_connection&& other) noexcept
    : context_(other.context_),
      socket_(std::move(other.socket_)),
      addrlen_(other.addrlen_) {}

 private:
  class conditional_listener_sender {
   public:
    conditional_listener_sender(
      ucx_am_context& context, ucs_status_t listener_status)
      : context_(context), listener_status_(listener_status) {}

    template <
      template <typename...> class Variant, template <typename...> class Tuple>
    using value_types =
      Variant<Tuple<std::vector<std::pair<std::uint64_t, ucs_status_t>>>>;

    template <template <typename...> class Variant>
    using error_types = Variant<std::error_code, std::exception_ptr>;

    static constexpr bool sends_done = true;

   private:
    friend class ucx_am_context::accept_connection;
    friend class ucx_am_context::accept_sender;

    template <typename Receiver>
    class operation {
     public:
      operation(
        ucx_am_context& context, ucs_status_t listener_status,
        Receiver&& receiver)
        : context_(context),
          listener_status_(listener_status),
          receiver_(std::move(receiver)) {
        auto stop_token = get_stop_token(receiver_);
        if constexpr (std::is_same_v<
                        decltype(stop_token), inplace_stop_token>) {
          receiver_stop_token_ = stop_token;
        } else {
          receiver_stop_token_ = inplace_stop_token{};
        }
      }

      ~operation() {
        if (!isAccepOpConstructed_) {
          receiver_.~Receiver();
        }
      }

      void start() noexcept {
        if (listener_status_ != UCS_OK) {
          unifex::set_error(
            std::move(receiver_),
            stdexe_ucx_runtime::make_error_code(listener_status_));
        } else {
          start_accept();
        }
      }

     private:
      class receiver_wrapper {
       private:
        operation<Receiver>* op_;
        inplace_stop_token stopToken_;

       public:
        receiver_wrapper(operation<Receiver>* op, inplace_stop_token stopToken)
          : op_(op), stopToken_(stopToken) {}

        receiver_wrapper(receiver_wrapper&& other) noexcept
          : op_(std::exchange(other.op_, nullptr)),
            stopToken_(std::move(other.stopToken_)) {}

        template <typename... Values>
        void set_value(Values&&... values) && noexcept {
          UNIFEX_ASSERT(op_ != nullptr);
          unifex::deactivate_union_member(op_->acceptOp_);
          UNIFEX_TRY {
            // Take a copy of the values before destroying the next operation
            // state in case the values are references to objects stored in
            // the operation object.
            [&](Values... values) {
              unifex::set_value(
                std::move(op_->receiver_), static_cast<Values&&>(values)...);
            }(static_cast<Values&&>(values)...);
          }
          UNIFEX_CATCH(...) {
            unifex::set_error(
              std::move(op_->receiver_), std::current_exception());
          }
        }

        void set_value(std::vector<std::pair<std::uint64_t, ucs_status_t>>&&
                         conn_id_status_vector) && noexcept {
          UNIFEX_ASSERT(op_ != nullptr);
          unifex::deactivate_union_member(op_->acceptOp_);
          unifex::set_value(
            std::move(op_->receiver_), std::move(conn_id_status_vector));
        }

        void set_done() && noexcept {
          UNIFEX_ASSERT(op_ != nullptr);
          unifex::deactivate_union_member(op_->acceptOp_);
          unifex::set_done(std::move(op_->receiver_));
        }

        template <typename Error>
        void set_error(Error&& error) && noexcept {
          UNIFEX_ASSERT(op_ != nullptr);
          unifex::deactivate_union_member(op_->acceptOp_);
          // Type-erase any errors that come through.
          unifex::set_error(
            std::move(op_->receiver_),
            make_exception_ptr(static_cast<Error&&>(error)));
        }

        void set_error(std::exception_ptr ex) && noexcept {
          UNIFEX_ASSERT(op_ != nullptr);
          unifex::deactivate_union_member(op_->acceptOp_);
          unifex::set_error(std::move(op_->receiver_), std::move(ex));
        }

        void set_error(std::error_code ec) && noexcept {
          UNIFEX_ASSERT(op_ != nullptr);
          unifex::deactivate_union_member(op_->acceptOp_);
          unifex::set_error(std::move(op_->receiver_), ec);
        }

        friend const inplace_stop_token& tag_invoke(
          tag_t<get_stop_token>, const receiver_wrapper& r) noexcept {
          return r.stopToken_;
        }

       private:
        UNIFEX_TEMPLATE(typename CPO)  //
        (requires unifex::is_receiver_query_cpo_v<CPO> UNIFEX_AND
           std::is_invocable_v<
             CPO,
             const Receiver&>)  //
          friend auto tag_invoke(CPO cpo, const receiver_wrapper& r) noexcept(
            std::is_nothrow_invocable_v<CPO, const Receiver&>)
            -> std::invoke_result_t<CPO, const Receiver&> {
          return std::move(cpo)(r.get_receiver());
        }

#if UNIFEX_ENABLE_CONTINUATION_VISITATIONS
        template <typename Func>
        friend void tag_invoke(
          tag_t<visit_continuations>, const receiver_wrapper& r, Func&& func) {
          visit_continuations(r.get_receiver(), static_cast<Func&&>(func));
        }
#endif

        const Receiver& get_receiver() const noexcept {
          UNIFEX_ASSERT(op_ != nullptr);
          return op_->receiver_;
        }
      };

      ucx_am_context& context_;
      ucs_status_t listener_status_;
      UNIFEX_NO_UNIQUE_ADDRESS Receiver receiver_;
      bool isAccepOpConstructed_ = false;

      using accept_op_t =
        unifex::connect_result_t<accept_sender&&, receiver_wrapper>;
      union {
        manual_lifetime<accept_op_t> acceptOp_;
      };

      inplace_stop_token receiver_stop_token_;

      void start_accept() {
        UNIFEX_TRY {
          auto accept_sender_ = accept_sender{context_};
          unifex::activate_union_member_with(acceptOp_, [&]() {
            return unifex::connect(
              std::move(accept_sender_),
              receiver_wrapper{this, receiver_stop_token_});
          });
          isAccepOpConstructed_ = true;
          unifex::start(acceptOp_.get());
        }
        UNIFEX_CATCH(...) {
          unifex::set_error(
            static_cast<Receiver&&>(receiver_), std::current_exception());
        }
      }
    };

   public:
    template <typename Receiver>
    operation<remove_cvref_t<Receiver>> connect(Receiver&& r) && {
      return operation<remove_cvref_t<Receiver>>{
        context_, listener_status_, std::move(r)};
    }

   private:
    ucx_am_context& context_;
    ucs_status_t listener_status_;
  };

 public:
  auto next() noexcept {
    return unifex::let_value_with(
      [this]() noexcept {
        // TODO(libunifex): move to operation.start_io()
        // TODO(He Jia): I'm not sure if not to check the listener validation
        // every times.
        auto status = UCS_OK;
        if (__builtin_expect(!context_.ucpListener_, 0)) {
          status = context_.listen(socket_, addrlen_);
        }
        return status;
      },
      [this](ucs_status_t status) noexcept {
        return conditional_listener_sender{context_, status};
      });
  }

  auto cleanup() noexcept {
    return unifex::defer([this]() noexcept {
      // Check pending connection requests being created by callback
      while (!context_.epConnReqQueue_.empty()) {
        const auto epConnReq = std::move(context_.epConnReqQueue_.front());
        UCX_CTX_TRACE << "reject connection request " << epConnReq.conn_request;
        auto status =
          ucp_listener_reject(context_.ucpListener_, epConnReq.conn_request);
        // (status < 0) == (status != UCS_OK || status != UCS_INPROGRESS)
        if (status < 0) {
          UCX_CTX_ERROR << "reject connection request "
                        << epConnReq.conn_request
                        << " failed: " << ucs_status_string(status)
                        << std::endl;
        }
        context_.epConnReqQueue_.pop_front();
      }
      context_.destroy_listener();
      return unifex::just_done();
    });
  }

 private:
  friend scheduler;

  ucx_am_context& context_;
  std::unique_ptr<sockaddr> socket_;
  size_t addrlen_;
};

}  // namespace stdexe_ucx_runtime

#endif  // UCX_CONTEXT_UCX_AM_CONTEXT_UCX_AM_CONTEXT_HPP_
