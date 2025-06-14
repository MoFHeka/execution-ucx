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

#ifndef UCX_CONTEXT_CONCEPT_HPP_
#define UCX_CONTEXT_CONCEPT_HPP_

#include <netinet/in.h>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include <unifex/detail/prologue.hpp>
#include <unifex/tag_invoke.hpp>

#include "ucx_context/ucx_memory_resource.hpp"

namespace stdexe_ucx_runtime {

using ::unifex::is_nothrow_tag_invocable_v;
using ::unifex::tag_invoke_result_t;
using ::unifex::tag_t;

namespace _endpoint_cpo {
using port_t = std::uint16_t;

/**
 * @brief CPO for accepting incoming connections on a listening socket.
 *
 * Creates a sender that, when started, begins listening on the specified
 * local endpoint and produces a stream of incoming connections.
 */
inline constexpr struct accept_endpoint_cpo final {
  template <typename Scheduler, typename SocketDescriptor>
  constexpr auto operator()(Scheduler&& sched, SocketDescriptor desc) const
    noexcept(is_nothrow_tag_invocable_v<
             accept_endpoint_cpo, Scheduler, SocketDescriptor>)
      -> tag_invoke_result_t<accept_endpoint_cpo, Scheduler, SocketDescriptor> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), desc);
  }

  /**
   * @brief Creates a sender for accepting connections.
   * @param sched The scheduler to use for the operation.
   * @param desc The socket descriptor (e.g., port number or sockaddr).
   * @param addrlen The length of the socket address.
   * @return A sender that, when connected, yields a connection object.
   *
   * This overload is used when the socket descriptor is a pointer (like
   * `sockaddr*`).
   */
  template <typename Scheduler, typename SocketDescriptor, typename SocklenT>
  constexpr auto operator()(
    Scheduler&& sched, SocketDescriptor desc, SocklenT addrlen) const
    noexcept(is_nothrow_tag_invocable_v<
             accept_endpoint_cpo, Scheduler, SocketDescriptor, SocklenT>)
      -> tag_invoke_result_t<
        accept_endpoint_cpo, Scheduler, SocketDescriptor, SocklenT> {
    if constexpr (!std::is_copy_constructible_v<SocketDescriptor>) {
      return tag_invoke(
        *this, static_cast<Scheduler&&>(sched), std::move(desc), addrlen);
    } else {
      return tag_invoke(*this, static_cast<Scheduler&&>(sched), desc, addrlen);
    }
  }
} accept_endpoint{};

/**
 * @brief CPO for establishing an outbound connection to a remote endpoint.
 *
 * Creates a sender that, when started, attempts to connect to the specified
 * remote endpoint.
 */
inline constexpr struct connect_endpoint_cpo final {
  /**
   * @brief Creates a sender for establishing a connection.
   * @param sched The scheduler to use.
   * @param src_saddr The source socket address (can be null).
   * @param dst_saddr The destination socket address.
   * @param addrlen The length of the socket address structure.
   * @return A sender that yields a connection object upon successful
   * connection.
   */
  template <
    typename Scheduler, typename SrcSaddr, typename DstSaddr, typename SocklenT>
  constexpr auto operator()(
    Scheduler&& sched, SrcSaddr src_saddr, DstSaddr dst_saddr,
    SocklenT addrlen) const
    noexcept(is_nothrow_tag_invocable_v<
             connect_endpoint_cpo, Scheduler, SrcSaddr, DstSaddr, SocklenT>)
      -> tag_invoke_result_t<
        connect_endpoint_cpo, Scheduler, SrcSaddr, DstSaddr, SocklenT> {
    if constexpr (
      std::is_copy_constructible_v<DstSaddr>
      && std::is_copy_constructible_v<SrcSaddr>) {
      return tag_invoke(
        *this, static_cast<Scheduler&&>(sched), src_saddr, dst_saddr, addrlen);
    } else {
      return tag_invoke(
        *this, static_cast<Scheduler&&>(sched), std::move(src_saddr),
        std::move(dst_saddr), addrlen);
    }
  }
} connect_endpoint{};
}  // namespace _endpoint_cpo

namespace _io_cpo {
using conn_pair_t = std::pair<
  uint64_t, std::reference_wrapper<stdexe_ucx_runtime::UcxConnection>>;

/**
 * @brief CPO for sending data over a connection.
 *
 * Creates a sender that, when started, sends the provided data buffer over the
 * specified connection.
 */
inline constexpr struct send_cpo final {
  /**
   * @brief Creates a sender to send data over a given connection.
   * @param sched The scheduler for the operation.
   * @param conn A pair containing the connection ID and reference.
   * @param data The data to be sent.
   * @return A sender that completes when the send operation is finished.
   */
  template <typename Scheduler>
  constexpr auto operator()(
    Scheduler&& sched, conn_pair_t& conn, ucx_am_data& data) const
    noexcept(is_nothrow_tag_invocable_v<
             send_cpo, Scheduler, conn_pair_t&, ucx_am_data&>)
      -> tag_invoke_result_t<send_cpo, Scheduler, conn_pair_t&, ucx_am_data&> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), conn, data);
  }

  /**
   * @brief Creates a sender to send data using a connection ID.
   * @param sched The scheduler for the operation.
   * @param conn_id The ID of the connection.
   * @param data The data to be sent.
   * @return A sender that completes when the send operation is finished.
   */
  template <typename Scheduler>
  constexpr auto operator()(
    Scheduler&& sched, std::uintptr_t conn_id, ucx_am_data& data) const
    noexcept(is_nothrow_tag_invocable_v<
             send_cpo, Scheduler, std::uintptr_t, ucx_am_data&>)
      -> tag_invoke_result_t<
        send_cpo, Scheduler, std::uintptr_t, ucx_am_data&> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), conn_id, data);
  }

  /**
   * @brief Creates a sender to send data over a given connection object.
   * @param sched The scheduler for the operation.
   * @param conn A reference to the UcxConnection object.
   * @param data The data to be sent.
   * @return A sender that completes when the send operation is finished.
   */
  template <typename Scheduler>
  constexpr auto operator()(
    Scheduler&& sched, UcxConnection& conn, ucx_am_data& data) const
    noexcept(is_nothrow_tag_invocable_v<
             send_cpo, Scheduler, UcxConnection&, ucx_am_data&>)
      -> tag_invoke_result_t<
        send_cpo, Scheduler, UcxConnection&, ucx_am_data&> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), conn, data);
  }
} connection_send{};

/**
 * @brief CPO for receiving data from a connection.
 *
 * Creates a sender that, when started, receives data from a connection and
 * places it into the provided buffer.
 */
inline constexpr struct recv_cpo final {
  /**
   * @brief Creates a sender to receive data into a provided buffer.
   * @param sched The scheduler for the operation.
   * @param data A reference to a `ucx_am_data` struct to store the received
   * data. The `data` member of this struct should be null, as the receiver
   * will allocate the buffer.
   * @return A sender that completes with the received data.
   */
  template <typename Scheduler>
  constexpr auto operator()(Scheduler&& sched, ucx_am_data& data) const
    noexcept(is_nothrow_tag_invocable_v<recv_cpo, Scheduler, ucx_am_data&>)
      -> tag_invoke_result_t<recv_cpo, Scheduler, ucx_am_data&> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), data);
  }

  /**
   * @brief Creates a sender that allocates a buffer and receives data into it.
   * @param sched The scheduler for the operation.
   * @param data_type The memory type for the buffer to be allocated for the
   * received data.
   * @return A sender that completes with the received data.
   */
  template <typename Scheduler>
  constexpr auto operator()(Scheduler&& sched, ucx_memory_type data_type) const
    noexcept(is_nothrow_tag_invocable_v<recv_cpo, Scheduler, ucx_memory_type>)
      -> tag_invoke_result_t<recv_cpo, Scheduler, ucx_memory_type> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), data_type);
  }
} connection_recv{};

/**
 * @brief CPO for handling connection errors.
 *
 * Creates a sender that monitors for connection errors and invokes a handler
 * when one occurs.
 */
inline constexpr struct handle_error_cpo final {
  /**
   * @brief Creates a sender for handling connection errors.
   * @param sched The scheduler for the operation.
   * @param handler A function to be called when an error occurs. The handler
   * receives the connection ID and status, and returns true to attempt
   * reconnection.
   * @return A sender that completes when the error handling is finished.
   */
  template <typename Scheduler>
  constexpr auto operator()(
    Scheduler&& sched,
    std::function<bool(std::uint64_t conn_id, ucs_status_t status)> handler =
      [](
        [[maybe_unused]] std::uint64_t conn_id,
        [[maybe_unused]] ucs_status_t status) -> bool { return false; }) const
    noexcept(is_nothrow_tag_invocable_v<
             handle_error_cpo, Scheduler,
             std::function<bool(std::uint64_t conn_id, ucs_status_t status)>>)
      -> tag_invoke_result_t<
        handle_error_cpo, Scheduler,
        std::function<bool(std::uint64_t conn_id, ucs_status_t status)>> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), handler);
  }
} handle_error_connection{};

}  // namespace _io_cpo

using _endpoint_cpo::accept_endpoint;
using _endpoint_cpo::connect_endpoint;
using _endpoint_cpo::port_t;
using _io_cpo::connection_recv;
using _io_cpo::connection_send;
using _io_cpo::handle_error_connection;

}  // namespace stdexe_ucx_runtime

#include <unifex/detail/epilogue.hpp>

#endif  // UCX_CONTEXT_CONCEPT_HPP_
