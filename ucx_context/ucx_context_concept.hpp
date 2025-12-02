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
#include <ucp/api/ucp.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include <unifex/detail/prologue.hpp>
#include <unifex/tag_invoke.hpp>

#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace ucxx {

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
    typename Scheduler, typename SocketDescriptor,
    std::enable_if_t<is_socket_descriptor_v<SocketDescriptor>, int> = 0>
  constexpr auto operator()(
    Scheduler&& sched, SocketDescriptor&& desc, size_t addrlen = 0) const
    noexcept(is_nothrow_tag_invocable_v<
             accept_endpoint_cpo, Scheduler, SocketDescriptor, size_t>)
      -> tag_invoke_result_t<
        accept_endpoint_cpo, Scheduler, SocketDescriptor, size_t> {
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
 * @brief Type trait to determine if a type is a valid send parameter for
 * connect_endpoint.
 *
 * This trait evaluates to true if the type T (after removing cv and reference
 * qualifiers) is one of the following:
 *   - std::string_view
 *   - std::string
 *   - void* (will be casted to ucp_address_t*)
 *
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_send_parameter_v =
  std::is_same_v<std::decay_t<T>, std::string_view>
  || std::is_same_v<std::decay_t<T>, std::string>
  || std::is_same_v<std::decay_t<T>, void*>
  || std::is_same_v<std::decay_t<T>, std::vector<std::byte>>;

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
    typename Scheduler, typename SrcSaddr, typename SocklenT,
    std::enable_if_t<
      is_source_socket_address_v<SrcSaddr> && is_length_type_v<SocklenT>, int> =
      0>
  auto operator()(
    Scheduler&& sched,
    SrcSaddr src_saddr,
    std::unique_ptr<sockaddr>
      dst_saddr,
    SocklenT addrlen) const
    noexcept(is_nothrow_tag_invocable_v<
             connect_endpoint_cpo, Scheduler, SrcSaddr,
             std::unique_ptr<sockaddr>, SocklenT>)
      -> tag_invoke_result_t<
        connect_endpoint_cpo, Scheduler, SrcSaddr, std::unique_ptr<sockaddr>,
        SocklenT> {
    return tag_invoke(
      *this,
      static_cast<Scheduler&&>(sched),
      std::move(src_saddr),
      std::move(dst_saddr),
      addrlen);
  }

  /**
   * @brief Overload of connect_endpoint_cpo for send parameters.
   *
   * This overload enables connect_endpoint to be invoked with a scheduler and a
   * send parameter (such as std::string_view, std::string, or void*). The
   * function forwards the call to tag_invoke, enabling customization point
   * behavior.
   *
   * @tparam Scheduler The type of the scheduler.
   * @tparam SendParam The type of the send parameter, constrained by
   * is_send_parameter_v.
   * @param sched The scheduler to use for the operation.
   * @param send_param The send parameter to be used in the connection
   * operation.
   * @return The result of tag_invoke for connect_endpoint_cpo with the given
   * arguments.
   */
  template <
    typename Scheduler, typename SendParam,
    std::enable_if_t<is_send_parameter_v<SendParam>, int> = 0>
  constexpr auto operator()(Scheduler&& sched, SendParam&& send_param) const
    noexcept(
      is_nothrow_tag_invocable_v<connect_endpoint_cpo, Scheduler, SendParam>)
      -> tag_invoke_result_t<connect_endpoint_cpo, Scheduler, SendParam> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), send_param);
  }
} connect_endpoint{};

class UcxConnection;

using conn_pair_t = std::pair<uint64_t, std::reference_wrapper<UcxConnection>>;

/**
 * @brief Type trait to check if a type is a valid connection identifier for
 * I/O operations.
 *
 * A type T satisfies this trait if it is `conn_pair_t` or
 * `eux::ucxx::UcxConnection`.
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
 * `ucx_am_iovec`.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_data_type_v =
  // raw payload types
  std::is_same_v<remove_cvref_t<T>, ucx_am_data>
  || std::is_same_v<remove_cvref_t<T>, ucx_am_iovec>
  // wrapper payload types
  || std::is_same_v<remove_cvref_t<T>, UcxAmData>
  || std::is_same_v<remove_cvref_t<T>, UcxAmIovec>;

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
   * `std::uint64_t`).
   * @param data The data to be sent (`ucx_am_data&` or `ucx_am_iovec_t&`).
   * @return A sender that completes when the send operation is finished.
   */
  template <typename Scheduler, typename Conn, typename Data>
  constexpr auto operator()(Scheduler&& sched, Conn&& conn, Data& data) const
    noexcept(is_nothrow_tag_invocable_v<send_cpo, Scheduler, Conn, Data&>)
      -> std::enable_if_t<
        (is_connection_type_v<Conn>
         || std::is_same_v<
           remove_cvref_t<Conn>, std::uint64_t>)&&(is_data_type_v<Data>),
        tag_invoke_result_t<send_cpo, Scheduler, Conn, Data&>> {
    return tag_invoke(
      *this, static_cast<Scheduler&&>(sched), static_cast<Conn&&>(conn), data);
  }

  template <typename Scheduler, typename Conn, typename Data>
  constexpr auto operator()(Scheduler&& sched, Conn&& conn, Data&& data) const
    noexcept(is_nothrow_tag_invocable_v<send_cpo, Scheduler, Conn, Data&&>)
      -> std::enable_if_t<
        (is_connection_type_v<Conn>
         || std::is_same_v<
           remove_cvref_t<Conn>, std::uint64_t>)&&(is_data_type_v<Data>),
        tag_invoke_result_t<send_cpo, Scheduler, Conn, Data&&>> {
    return tag_invoke(
      *this,
      static_cast<Scheduler&&>(sched),
      static_cast<Conn&&>(conn),
      static_cast<Data&&>(data));
  }
} connection_send{};

/**
 * @brief Type trait to constrain the parameter type for receive operations.
 *
 * A type T satisfies this trait if it is either `ucx_am_data&` or
 * `ucx_memory_type` or `ucx_am_iovec`.
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
    typename Scheduler, typename RecvParam,
    std::enable_if_t<is_receive_parameter_v<RecvParam>, int> = 0>
  constexpr auto operator()(Scheduler&& sched, RecvParam&& param) const
    noexcept(is_nothrow_tag_invocable_v<recv_cpo, Scheduler, RecvParam>)
      -> tag_invoke_result_t<recv_cpo, Scheduler, RecvParam> {
    return tag_invoke(
      *this, static_cast<Scheduler&&>(sched), static_cast<RecvParam&&>(param));
  }
} connection_recv{};

/**
 * @brief Type trait to constrain the parameter type for receive header
 * operations.
 *
 * A type T satisfies this trait if it is `ucx_buffer`.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_receive_header_parameter_v =
  std::is_same_v<std::decay_t<T>, ucx_buffer>;

/**
 * @brief CPO for receiving header data from a connection.
 *
 * Creates a sender that, when started, receives header data from a connection.
 */
inline constexpr struct recv_header_cpo final {
  /**
   * @brief Creates a sender to receive header data.
   * @tparam Scheduler The type of the scheduler.
   * @param sched The scheduler for the operation.
   * @return A sender that completes with the received header data.
   */
  template <typename Scheduler>
  constexpr auto operator()(Scheduler&& sched) const
    noexcept(is_nothrow_tag_invocable_v<recv_header_cpo, Scheduler>)
      -> tag_invoke_result_t<recv_header_cpo, Scheduler> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched));
  }

  /**
   * @brief Creates a sender to receive header data into a user-provided buffer.
   * @tparam Scheduler The type of the scheduler.
   * @tparam Header The type of the header buffer, constrained by
   * `is_receive_header_parameter_v`.
   * @param sched The scheduler for the operation.
   * @param header The header buffer to receive data into.
   * @return A sender that completes with the received header data.
   */
  template <
    typename Scheduler, typename Header,
    std::enable_if_t<is_receive_header_parameter_v<Header>, int> = 0>
  constexpr auto operator()(Scheduler&& sched, Header&& header) const
    noexcept(is_nothrow_tag_invocable_v<recv_header_cpo, Scheduler, Header>)
      -> tag_invoke_result_t<recv_header_cpo, Scheduler, Header> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), header);
  }
} connection_recv_header{};

/**
 * @brief Type trait to constrain the parameter type for receive buffer flag
 * operations.
 *
 * A type T satisfies this trait if it is `ucx_memory_type`.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_receive_buffer_flag_parameter_v =
  std::is_same_v<std::decay_t<T>, ucx_memory_type>;

/**
 * @brief Type trait to check if a type is an indexed container.
 *
 * A type T is considered an indexed container if it has a size() method
 * returning a value convertible to std::size_t, and supports operator[].
 * @tparam T The type to check.
 */
template <typename T, typename = std::void_t<>>
struct is_indexed_buffer_container : std::false_type {};

template <typename T>
struct is_indexed_buffer_container<
  T, std::void_t<
       // Check 1: Existence of t.size() expression, convertible to std::size_t
       std::enable_if_t<std::is_convertible_v<
         decltype(std::declval<T&>().size()), std::size_t>>,
       // Check 2: Existence of t[0] index access expression
       decltype(std::declval<T&>()[0]),
       // Check 3: The value_type is UcxBuffer
       std::enable_if_t<std::is_same_v<
         std::decay_t<decltype(std::declval<T&>()[0])>, UcxBuffer>>>>
  : std::true_type {};

/**
 * @brief Type trait to constrain the parameter type for receive buffer
 * operations.
 *
 * A type T satisfies this trait if it is `UcxBuffer` or an indexed Buffer
 * container.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_receive_buffer_parameter_v =
  std::is_same_v<std::decay_t<T>, UcxBuffer>
  || std::is_same_v<std::decay_t<T>, UcxBufferVec>;

/**
 * @brief CPO for receiving buffer data from a connection.
 *
 * Creates a sender that, when started, receives buffer data from a connection.
 */
inline constexpr struct recv_buffer_cpo final {
  /**
   * @brief Creates a sender to receive buffer data into a user-provided buffer.
   * @tparam Scheduler The type of the scheduler.
   * @tparam Key The type of the key, constrained by `std::size_t`.
   * @tparam Data The type of the buffer, constrained by
   * `is_receive_buffer_parameter_v`.
   * @param sched The scheduler for the operation.
   * @param key The key of the receive buffer.
   * @param data The buffer to receive data into.
   * @return A sender that completes with the received buffer data.
   */
  template <
    typename Scheduler, typename Data,
    std::enable_if_t<is_receive_buffer_parameter_v<Data>, int> = 0>
  constexpr auto operator()(Scheduler&& sched, size_t key, Data&& data) const
    noexcept(
      is_nothrow_tag_invocable_v<recv_buffer_cpo, Scheduler, size_t, Data>)
      -> tag_invoke_result_t<recv_buffer_cpo, Scheduler, size_t, Data> {
    return tag_invoke(
      *this, static_cast<Scheduler&&>(sched), key, static_cast<Data&&>(data));
  }

  /**
   * @brief Creates a sender to receive buffer data with a key.
   * @tparam Scheduler The type of the scheduler.
   * @tparam Flag The type of the memory type flag, constrained by
   * `is_receive_buffer_flag_parameter_v`.
   * @param sched The scheduler for the operation.
   * @param key The key of the receive buffer.
   * @param flag The memory type flag.
   * @return A sender that completes with the received buffer data.
   */
  template <
    typename Scheduler, typename Flag,
    std::enable_if_t<is_receive_buffer_flag_parameter_v<Flag>, int> = 0>
  constexpr auto operator()(Scheduler&& sched, size_t key, Flag&& flag) const
    noexcept(
      is_nothrow_tag_invocable_v<recv_buffer_cpo, Scheduler, size_t, Flag>)
      -> tag_invoke_result_t<recv_buffer_cpo, Scheduler, size_t, Flag> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), key, flag);
  }
} connection_recv_buffer{};

/**
 * @brief Type trait to check if a type is a valid status type.
 *
 * A type T satisfies this trait if it is `ucs_status_t` or `std::error_code`.
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_status_type_v =
  std::is_same_v<std::decay_t<T>, ucs_status_t>
  || std::is_same_v<std::decay_t<T>, std::error_code>;

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
   * @tparam Handler The type of the handler, must be callable with
   * (std::uint64_t, Status) -> bool.
   * @tparam Status The type of the status, constrained by `is_status_type_v`.
   * @param sched The scheduler for the operation.
   * @param handler A function to be called when an error occurs. The handler
   * receives the connection ID and status, and returns true to attempt
   * reconnection.
   * @return A sender that completes when the error handling is finished.
   */
  template <
    typename Scheduler, typename Handler, typename Status = ucs_status_t,
    std::enable_if_t<
      is_status_type_v<Status>
        && std::is_invocable_r_v<bool, Handler, std::uint64_t, Status>,
      int> = 0>
  constexpr auto operator()(Scheduler&& sched, Handler&& handler) const
    noexcept(is_nothrow_tag_invocable_v<
             handle_error_cpo, Scheduler,
             std::function<bool(std::uint64_t conn_id, Status)>>)
      -> tag_invoke_result_t<
        handle_error_cpo, Scheduler,
        std::function<bool(std::uint64_t conn_id, Status)>> {
    return tag_invoke(
      *this, static_cast<Scheduler&&>(sched),
      std::function<bool(std::uint64_t, Status)>(
        static_cast<Handler&&>(handler)));
  }

  template <typename Scheduler>
  constexpr auto operator()(
    Scheduler&& sched, bool (*handler)(std::uint64_t, ucs_status_t)) const
    noexcept(
      is_nothrow_tag_invocable_v<
        handle_error_cpo, Scheduler, bool (*)(std::uint64_t, ucs_status_t)>)
      -> tag_invoke_result_t<
        handle_error_cpo, Scheduler, bool (*)(std::uint64_t, ucs_status_t)> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), handler);
  }

  template <typename Scheduler>
  constexpr auto operator()(
    Scheduler&& sched, bool (*handler)(std::uint64_t, std::error_code)) const
    noexcept(
      is_nothrow_tag_invocable_v<
        handle_error_cpo, Scheduler, bool (*)(std::uint64_t, std::error_code)>)
      -> tag_invoke_result_t<
        handle_error_cpo, Scheduler, bool (*)(std::uint64_t, std::error_code)> {
    return tag_invoke(*this, static_cast<Scheduler&&>(sched), handler);
  }
} handle_error_connection{};

}  // namespace ucxx
}  // namespace eux

#include <unifex/detail/epilogue.hpp>

#endif  // UCX_CONTEXT_CONCEPT_HPP_
