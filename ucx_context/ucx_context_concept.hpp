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

using unifex::is_nothrow_tag_invocable_v;
using unifex::remove_cvref_t;
using unifex::tag_invoke_result_t;
using unifex::tag_t;

using port_t = std::uint16_t;

/**
 * @brief Type trait to check if a type is a valid socket descriptor for
 * accepting connections.
 *
 * A type T satisfies this trait if it is either a
 * `std::unique_ptr<sockaddr>` or a `port_t`.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_socket_descriptor_v =
  std::is_same_v<std::decay_t<T>, std::unique_ptr<sockaddr>>
  || std::is_same_v<std::decay_t<T>, port_t>;

/**
 * @brief CPO for accepting incoming connections on a listening socket.
 *
 * Creates a sender that, when started, begins listening on the specified
 * local endpoint and produces a stream of incoming connections.
 */
inline constexpr struct accept_endpoint_cpo final {
  /**
   * @brief Creates a sender for accepting connections.
   * @tparam Scheduler The type of the scheduler.
   * @tparam SocketDescriptor The type of the socket descriptor, constrained by
   * the `is_socket_descriptor_v` trait.
   * @param sched The scheduler to use for the operation.
   * @param desc The socket descriptor (either a port number or a
   * `std::unique_ptr<sockaddr>`).
   * @param addrlen The length of the socket address, used when `desc` is a
   * `sockaddr`. Defaults to 0.
   * @return A sender that, when connected, yields a connection object.
   */
  template <
    typename Scheduler,
    typename SocketDescriptor,
    std::enable_if_t<is_socket_descriptor_v<SocketDescriptor>, int> = 0>
  constexpr auto operator()(
    Scheduler&& sched, SocketDescriptor&& desc, size_t addrlen = 0) const
    noexcept(is_nothrow_tag_invocable_v<
             accept_endpoint_cpo,
             Scheduler,
             SocketDescriptor,
             size_t>)
      -> tag_invoke_result_t<
        accept_endpoint_cpo,
        Scheduler,
        SocketDescriptor,
        size_t> {
    return tag_invoke(
      *this,
      static_cast<Scheduler&&>(sched),
      static_cast<SocketDescriptor&&>(desc),
      addrlen);
  }
} accept_endpoint{};

/**
 * @brief Type trait to constrain the type of a source socket address.
 *
 * A type T satisfies this trait if it is either a
 * `std::unique_ptr<sockaddr>` or `std::nullptr_t`.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_source_socket_address_v =
  std::is_same_v<std::decay_t<T>, std::unique_ptr<sockaddr>>
  || std::is_same_v<std::decay_t<T>, std::nullptr_t>;

/**
 * @brief Type trait to constrain the type for an address length.
 *
 * A type T satisfies this trait if it is either a `size_t` or a
 * `socklen_t`.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_length_type_v =
  std::is_same_v<std::decay_t<T>, size_t>
  || std::is_same_v<std::decay_t<T>, socklen_t>;

/**
 * @brief CPO for establishing an outbound connection to a remote endpoint.
 *
 * Creates a sender that, when started, attempts to connect to the specified
 * remote endpoint.
 */
inline constexpr struct connect_endpoint_cpo final {
  /**
   * @brief Creates a sender for establishing a connection.
   * @tparam Scheduler The type of the scheduler.
   * @tparam SrcSaddr The type of the source socket address, constrained by
   * `is_source_socket_address_v`.
   * @tparam SocklenT The type of the address length, constrained by
   * `is_length_type_v`.
   * @param sched The scheduler to use for the operation.
   * @param src_saddr The source socket address (can be a
   * `std::unique_ptr<sockaddr>` or `nullptr`).
   * @param dst_saddr The destination socket address.
   * @param addrlen The length of the socket address structure.
   * @return A sender that yields a connection object upon successful
   * connection.
   */
  template <
    typename Scheduler,
    typename SrcSaddr,
    typename SocklenT,
    std::enable_if_t<
      is_source_socket_address_v<SrcSaddr> && is_length_type_v<SocklenT>,
      int> = 0>
  constexpr auto operator()(
    Scheduler&& sched,
    SrcSaddr src_saddr,
    std::unique_ptr<sockaddr>
      dst_saddr,
    SocklenT addrlen) const
    noexcept(is_nothrow_tag_invocable_v<
             connect_endpoint_cpo,
             Scheduler,
             SrcSaddr,
             std::unique_ptr<sockaddr>,
             SocklenT>)
      -> tag_invoke_result_t<
        connect_endpoint_cpo,
        Scheduler,
        SrcSaddr,
        std::unique_ptr<sockaddr>,
        SocklenT> {
    return tag_invoke(
      *this,
      static_cast<Scheduler&&>(sched),
      std::move(src_saddr),
      std::move(dst_saddr),
      addrlen);
  }
} connect_endpoint{};

class UcxConnection;

using conn_pair_t = std::pair<uint64_t, std::reference_wrapper<UcxConnection>>;

/**
 * @brief Type trait to check if a type is a valid connection identifier for
 * I/O operations.
 *
 * A type T satisfies this trait if it is `conn_pair_t` or
 * `stdexe_ucx_runtime::UcxConnection`.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_connection_type_v =
  std::is_same_v<remove_cvref_t<T>, conn_pair_t>
  || std::is_same_v<remove_cvref_t<T>, UcxConnection>;

/**
 * @brief Type trait to check if a type is a valid data payload for send
 * operations.
 *
 * A type T satisfies this trait if it is either `ucx_am_data` or
 * `ucx_iov_data`.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_data_type_v =
  std::is_same_v<remove_cvref_t<T>, ucx_am_data>
  || std::is_same_v<remove_cvref_t<T>, ucx_iov_data>;

/**
 * @brief CPO for sending data over a connection.
 *
 * Creates a sender that, when started, sends the provided data buffer over the
 * specified connection.
 */
inline constexpr struct send_cpo final {
  /**
   * @brief Creates a sender to send data over a given connection.
   * @tparam Scheduler The type of the scheduler.
   * @tparam Conn The type of the connection identifier, constrained by trait.
   * @tparam Data The type of the data payload, constrained by
   * `is_data_type_v`.
   * @param sched The scheduler for the operation.
   * @param conn A connection identifier (`conn_pair_t&`, `UcxConnection&`, or
   * `std::uintptr_t`).
   * @param data The data to be sent (`ucx_am_data&` or `ucx_iov_data&`).
   * @return A sender that completes when the send operation is finished.
   */
  template <typename Scheduler, typename Conn, typename Data>
  constexpr auto operator()(Scheduler&& sched, Conn&& conn, Data& data) const
    noexcept(is_nothrow_tag_invocable_v<send_cpo, Scheduler, Conn, Data&>)
      -> std::enable_if_t<
        (is_connection_type_v<Conn>
         || std::is_same_v<
           remove_cvref_t<Conn>,
           std::uintptr_t>)&&(is_data_type_v<Data>),
        tag_invoke_result_t<send_cpo, Scheduler, Conn, Data&>> {
    return tag_invoke(
      *this, static_cast<Scheduler&&>(sched), static_cast<Conn&&>(conn), data);
  }
} connection_send{};

/**
 * @brief Type trait to constrain the parameter type for receive operations.
 *
 * A type T satisfies this trait if it is either `ucx_am_data&` or
 * `ucx_memory_type`.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_receive_parameter_v =
  std::is_same_v<std::decay_t<T>, ucx_am_data>
  || std::is_same_v<std::decay_t<T>, ucx_memory_type>;

/**
 * @brief CPO for receiving data from a connection.
 *
 * Creates a sender that, when started, receives data from a connection and
 * places it into the provided buffer.
 */
inline constexpr struct recv_cpo final {
  /**
   * @brief Creates a sender to receive data.
   * @tparam Scheduler The type of the scheduler.
   * @tparam RecvParam The type of the receive parameter, constrained by
   * `is_receive_parameter_v`.
   * @param sched The scheduler for the operation.
   * @param param The receive parameter, which can be a reference to a
   * `ucx_am_data` struct (for receiving into a user-provided buffer
   * descriptor) or a `ucx_memory_type` (for allocating a new buffer).
   * @return A sender that completes with the received data.
   */
  template <
    typename Scheduler,
    typename RecvParam,
    std::enable_if_t<is_receive_parameter_v<RecvParam>, int> = 0>
  constexpr auto operator()(Scheduler&& sched, RecvParam&& param) const
    noexcept(is_nothrow_tag_invocable_v<recv_cpo, Scheduler, RecvParam>)
      -> tag_invoke_result_t<recv_cpo, Scheduler, RecvParam> {
    return tag_invoke(
      *this, static_cast<Scheduler&&>(sched), static_cast<RecvParam&&>(param));
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
   * @tparam Scheduler The type of the scheduler.
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
             handle_error_cpo,
             Scheduler,
             std::function<bool(std::uint64_t conn_id, ucs_status_t status)>>)
      -> tag_invoke_result_t<
        handle_error_cpo,
        Scheduler,
        std::function<bool(std::uint64_t conn_id, ucs_status_t status)>> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), handler);
  }
} handle_error_connection{};

}  // namespace stdexe_ucx_runtime

#include <unifex/detail/epilogue.hpp>

#endif  // UCX_CONTEXT_CONCEPT_HPP_
