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

#include "axon/python/python_wake_manager.hpp"

#include <nanobind/nanobind.h>

#include <sys/eventfd.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "axon/python/python_module.hpp"

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;

PythonWakeManager::PythonWakeManager() {
  event_fd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (event_fd_ == -1) {
    throw std::runtime_error("Failed to create eventfd for Python wakeup");
  }
}

PythonWakeManager::~PythonWakeManager() {
  if (event_fd_ >= 0) {
    ::close(event_fd_);
  }
}

void PythonWakeManager::NotifyPython() {
  uint64_t one = 1;
  ssize_t written = ::write(event_fd_, &one, sizeof(one));
  if (written < 0 && errno != EAGAIN) {
    // TODO(He Jia): Handle error.
  }
}

void PythonWakeManager::Enqueue(Task&& task) {
  auto* node = new TaskNode{.task = std::move(task)};
  bool need_wakeup = queue_.enqueue(node);
  if (need_wakeup) {
    NotifyPython();
  } else {
  }
}

std::size_t PythonWakeManager::ProcessQueue() {
  // Ensure we are active if a caller invokes while inactive with an empty
  // queue.
  (void)queue_.try_mark_active();

  std::size_t processed = 0;
  for (;;) {
    auto batch = queue_.dequeue_all();
    if (batch.empty()) {
      // Attempt to mark inactive. If it fails, new items arrived; loop again.
      if (queue_.try_mark_inactive()) {
        break;
      }
      continue;
    }

    nb::gil_scoped_acquire acquire;
    while (!batch.empty()) {
      auto* node = batch.pop_front();
      if (!node) continue;
      if (node->task) {
        try {
          (*node->task)();
        } catch (const std::exception& e) {
          std::cerr << "PythonWakeManager task threw exception: " << e.what()
                    << std::endl;
        } catch (...) {
          std::cerr << "PythonWakeManager task threw unknown exception"
                    << std::endl;
        }
      }
      delete node;
      ++processed;
    }
  }
  return processed;
}

std::size_t PythonWakeManager::ClearQueue() {
  // Ensure we are active if a caller invokes while inactive with an empty
  // queue.
  (void)queue_.try_mark_active();

  std::size_t discarded = 0;
  for (;;) {
    auto batch = queue_.dequeue_all();
    if (batch.empty()) {
      // Attempt to mark inactive. If it fails, new items arrived; loop again.
      if (queue_.try_mark_inactive()) {
        break;
      }
      continue;
    }

    // No GIL acquire needed as we are just deleting nodes
    while (!batch.empty()) {
      auto* node = batch.pop_front();
      // If Python is still initialized, it's safe to delete the node
      // (which may trigger Py_DECREF).
      // If NOT initialized, we must leak to avoid crash.
      if (Py_IsInitialized()) {
        delete node;
      }
      ++discarded;
    }
  }
  return discarded;
}

bool PythonWakeManager::RegisterAsyncioReader() {
  // Increment reference count
  int old_count = reader_ref_count_.fetch_add(1, std::memory_order_acq_rel);

  // If already registered, just increment count and return
  if (old_count > 0) {
    return true;
  }

  // First registration, actually register with asyncio
  nb::gil_scoped_acquire acquire;
  try {
    nb::module_ asyncio = GetAsyncioModule();

    // Try to get running event loop first (for async contexts)
    nb::object loop;
    try {
      loop = asyncio.attr("get_running_loop")();
    } catch (const nb::python_error&) {
      // No running loop, use event loop policy to get or create loop
      // This avoids DeprecationWarning in Python 3.10+
      nb::object policy = asyncio.attr("get_event_loop_policy")();
      try {
        loop = policy.attr("get_event_loop")();
        // Check if loop is None or closed
        if (loop.is_none() || nb::cast<bool>(loop.attr("is_closed")())) {
          // Create a new event loop
          loop = asyncio.attr("new_event_loop")();
          policy.attr("set_event_loop")(loop);
        }
      } catch (const nb::python_error&) {
        // If get_event_loop() fails, create a new event loop
        loop = asyncio.attr("new_event_loop")();
        policy.attr("set_event_loop")(loop);
      }
    }

    // Register the eventfd with asyncio's add_reader
    // Save the callback object to ensure remove_reader can properly remove it
    reader_callback_ = nb::cpp_function([this]() {
      [[maybe_unused]] std::size_t processed = this->ProcessQueue();
    });
    // Store the callback object reference to ensure it's not garbage collected
    // and can be properly removed later
    loop.attr("add_reader")(event_fd_, reader_callback_);

    return true;
  } catch (const nb::python_error& e) {
    // Registration failed, decrement count
    reader_ref_count_.fetch_sub(1, std::memory_order_acq_rel);
    return false;
  }
}

bool PythonWakeManager::UnregisterAsyncioReader() {
  // Decrement reference count
  int old_count = reader_ref_count_.fetch_sub(1, std::memory_order_acq_rel);

  // If count is still positive, just decrement and return
  if (old_count > 1) {
    // Ensure we process any pending tasks to avoid "stuck active" state
    // caused by race conditions during stop sequence.
    // We use ProcessQueue (not ClearQueue) because other components are still
    // active, so tasks must be executed, not discarded.
    // This fixes the HANG in concurrent tests.
    ProcessQueue();
    return true;
  }

  // If count was already 0 or negative, something is wrong, but be safe
  if (old_count <= 0) {
    reader_ref_count_.store(0, std::memory_order_release);
    return true;
  }

  // Last unregistration, actually unregister from asyncio
  nb::gil_scoped_acquire acquire;

  try {
    nb::module_ asyncio = GetAsyncioModule();
    nb::object loop = asyncio.attr("get_running_loop")();
    loop.attr("remove_reader")(event_fd_);
  } catch (...) {
    return false;
  }

  // Always process queue to ensure we don't leave it in "stuck active" state
  // If tasks were added just before we unregistered, they might have marked
  // the queue active, but the event loop might not have seen the event yet.
  // Or if we drained it above, we processed them. But if read failed or was
  // empty, we still need to check if the queue thinks it's active but has
  // pending work that won't be picked up because we're removing the reader.
  // ClearQueue() handles the logic of marking inactive if empty, and discards
  // tasks to avoid unsafe execution during shutdown.
  ClearQueue();

  return true;
}

PythonWakeManager& GetPythonWakeManager() {
  // static singleton destructed by OS
  static PythonWakeManager* manager = new PythonWakeManager();
  return *manager;
}

}  // namespace python
}  // namespace axon
}  // namespace eux
