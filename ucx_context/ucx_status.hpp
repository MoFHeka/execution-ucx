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

#ifndef UCX_STATUS_HPP_
#define UCX_STATUS_HPP_

#include <ucs/type/status.h>

#include <string>
#include <system_error>

namespace stdexe_ucx_runtime {

/**
 * @brief Error category for UCX status codes
 *
 * This class integrates UCX status codes with the std::error_code system,
 * allowing UCX errors to be used with standard C++ error handling mechanisms.
 */
class UcxErrorCategory : public std::error_category {
 public:
  /**
   * @brief Get the name of the error category
   *
   * @return const char* The name of the error category
   */
  const char* name() const noexcept override { return "UCX"; }

  /**
   * @brief Get the error message for a given error code
   *
   * @param ev The error code value
   * @return std::string The error message
   */
  std::string message(int ev) const override {
    // Convert the error code to a ucs_status_t
    ucs_status_t status = static_cast<ucs_status_t>(ev);

    // Use UCX's built-in status string function if available
    const char* status_str = ucs_status_string(status);
    if (status_str != nullptr) {
      return status_str;
    }

    // Fallback for unknown status codes
    return "Unknown UCX error: " + std::to_string(ev);
  }

  /**
   * @brief Get the default error condition for a given error code
   *
   * @param ev The error code value
   * @return std::error_condition The default error condition
   */
  std::error_condition default_error_condition(int ev) const noexcept override {
    // Map UCX error codes to standard error conditions
    ucs_status_t status = static_cast<ucs_status_t>(ev);

    if (status == UCS_OK) {
      return std::error_condition(0, std::generic_category());
    }

    if (status == UCS_INPROGRESS) {
      return std::error_condition(0, std::generic_category());
    }

    // Map common error codes to standard error conditions
    switch (status) {
      case UCS_ERR_NO_MEMORY:
        return std::error_condition(
          static_cast<int>(std::errc::not_enough_memory),
          std::generic_category());
      case UCS_ERR_INVALID_PARAM:
        return std::error_condition(
          static_cast<int>(std::errc::invalid_argument),
          std::generic_category());
      case UCS_ERR_UNREACHABLE:
        return std::error_condition(
          static_cast<int>(std::errc::network_unreachable),
          std::generic_category());
      case UCS_ERR_TIMED_OUT:
        return std::error_condition(
          static_cast<int>(std::errc::timed_out), std::generic_category());
      case UCS_ERR_BUSY:
        return std::error_condition(
          static_cast<int>(std::errc::resource_unavailable_try_again),
          std::generic_category());
      case UCS_ERR_CANCELED:
        return std::error_condition(
          static_cast<int>(std::errc::operation_canceled),
          std::generic_category());
      case UCS_ERR_ALREADY_EXISTS:
        return std::error_condition(
          static_cast<int>(std::errc::file_exists), std::generic_category());
      case UCS_ERR_OUT_OF_RANGE:
        return std::error_condition(
          static_cast<int>(std::errc::result_out_of_range),
          std::generic_category());
      case UCS_ERR_UNSUPPORTED:
        return std::error_condition(
          static_cast<int>(std::errc::operation_not_supported),
          std::generic_category());
      case UCS_ERR_NOT_CONNECTED:
        return std::error_condition(
          static_cast<int>(std::errc::not_connected), std::generic_category());
      case UCS_ERR_CONNECTION_RESET:
        return std::error_condition(
          static_cast<int>(std::errc::connection_reset),
          std::generic_category());
      default:
        // For other error codes, return a generic error condition
        return std::error_condition(
          static_cast<int>(std::errc::io_error), std::generic_category());
    }
  }
};

/**
 * @brief Get the singleton instance of the UCX error category
 *
 * @return const UcxErrorCategory& The UCX error category instance
 */
inline const UcxErrorCategory& ucx_error_category() noexcept {
  static const UcxErrorCategory category;
  return category;
}

/**
 * @brief Create an std::error_code from a UCX status code
 *
 * @param status The UCX status code
 * @return std::error_code The corresponding std::error_code
 */
inline std::error_code make_error_code(ucs_status_t status) noexcept {
  return std::error_code(static_cast<int>(status), ucx_error_category());
}

/**
 * @brief Create an std::error_condition from a UCX status code
 *
 * @param status The UCX status code
 * @return std::error_condition The corresponding std::error_condition
 */
inline std::error_condition make_error_condition(ucs_status_t status) noexcept {
  return std::error_condition(static_cast<int>(status), ucx_error_category());
}

}  // namespace stdexe_ucx_runtime

// Register UCX status codes with std::error_code system
namespace std {
template <>
struct is_error_code_enum<ucs_status_t> : public true_type {};
}  // namespace std

#endif  // UCX_STATUS_HPP_
