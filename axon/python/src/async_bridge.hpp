/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#ifndef AXON_PYTHON_ASYNC_BRIDGE_HPP_
#define AXON_PYTHON_ASYNC_BRIDGE_HPP_

#include <nanobind/nanobind.h>

#include <utility>

#include <unifex/spawn_detached.hpp>
#include <unifex/then.hpp>
#include <unifex/upon_error.hpp>
#include <unifex/v2/async_scope.hpp>

#include "axon/python/src/python_helpers.hpp"
#include "axon/python/src/python_wake_manager.hpp"

namespace eux {
namespace axon {
namespace python {

namespace {
// Debug log helper function for async_bridge

}  // namespace

namespace nb = nanobind;

/**
 * @brief Bridge unifex sender to Python asyncio.Future.
 *
 * This class creates a Python Future and connects a unifex sender to it.
 * When the sender completes, the Future is resolved with the result or
 * exception.
 */
template <typename Sender>
class SenderToFuture {
 public:
  /**
   * @brief Create a Python Future and connect the sender to it.
   *
   * @param sender The unifex sender to connect
   * @param scope The async scope to spawn the operation in
   * @return Python Future object that will be resolved when sender completes
   */
  static nb::object CreateFuture(
    Sender&& sender, unifex::v2::async_scope& scope) {
    // Import asyncio
    nb::module_ asyncio = GetAsyncioModule();
    // Create a new Future
    nb::object future = asyncio.attr("Future")();

    // Create a receiver that will set the Future result/exception
    // Note: Using a lambda-based approach instead of local struct to avoid
    // template member function in local class issue
    auto receiver = [future](auto&&... values) mutable {
      auto& wake_manager = GetPythonWakeManager();
      wake_manager.Enqueue(pro::make_proxy<TaskFacade>(
        [future = std::move(future),
         values = std::make_tuple(
           std::forward<decltype(values)>(values)...)]() mutable {
          // Set result on the Future
          if constexpr (std::tuple_size_v<decltype(values)> == 0) {
            future.attr("set_result")(nb::none());
          } else if constexpr (std::tuple_size_v<decltype(values)> == 1) {
            // Single value - wrap it appropriately
            auto value = std::get<0>(std::move(values));
            future.attr("set_result")(nb::cast(std::move(value)));
          } else {
            // Multiple values - return as tuple
            future.attr("set_result")(nb::cast(std::move(values)));
          }
        }));
    };

    auto error_handler = [future](std::exception_ptr ep) mutable noexcept {
      auto& wake_manager = GetPythonWakeManager();
      wake_manager.Enqueue(
        pro::make_proxy<TaskFacade>([future = std::move(future), ep]() mutable {
          try {
            std::rethrow_exception(ep);
          } catch (const nb::python_error& e) {
            // Python exception - set it directly
            // python_error doesn't have error() method, use PyErr_Occurred to
            // get current exception
            PyObject *exc_type, *exc_value, *exc_traceback;
            PyErr_Fetch(&exc_type, &exc_value, &exc_traceback);
            if (exc_value) {
              future.attr("set_exception")(nb::steal<nb::object>(exc_value));
            } else if (exc_type) {
              future.attr("set_exception")(nb::steal<nb::object>(exc_type));
            } else {
              // Fallback: create RuntimeError
              nb::module_ builtins = GetBuiltinsModule();
              nb::object runtime_error =
                builtins.attr("RuntimeError")(e.what());
              future.attr("set_exception")(runtime_error);
            }
            PyErr_Restore(exc_type, exc_value, exc_traceback);
          } catch (const std::exception& e) {
            // C++ exception - convert to Python RuntimeError
            nb::module_ builtins = GetBuiltinsModule();
            nb::object runtime_error = builtins.attr("RuntimeError")(e.what());
            future.attr("set_exception")(runtime_error);
          } catch (...) {
            // Unknown exception
            nb::module_ builtins = GetBuiltinsModule();
            nb::object runtime_error =
              builtins.attr("RuntimeError")("Unknown error from unifex sender");
            future.attr("set_exception")(runtime_error);
          }
        }));
    };

    auto done_handler = [future]() mutable noexcept {
      auto& wake_manager = GetPythonWakeManager();
      wake_manager.Enqueue(
        pro::make_proxy<TaskFacade>([future = std::move(future)]() mutable {
          // Cancelled - set CancelledError
          nb::module_ asyncio = GetAsyncioModule();
          nb::object cancelled_error = asyncio.attr("CancelledError")();
          future.attr("set_exception")(cancelled_error);
        }));
    };

    // Connect sender to receiver and spawn
    unifex::spawn_detached(
      std::forward<Sender>(sender) | unifex::then(std::move(receiver))
        | unifex::upon_error(
          [error_handler = std::move(error_handler)](auto&& error) mutable {
            error_handler(
              std::is_same_v<std::decay_t<decltype(error)>, std::exception_ptr>
                ? std::forward<decltype(error)>(error)
                : std::make_exception_ptr(
                  std::forward<decltype(error)>(error)));
          }),
      scope);

    return future;
  }
};

}  // namespace python
}  // namespace axon
}  // namespace eux

#endif  // AXON_PYTHON_ASYNC_BRIDGE_HPP_
