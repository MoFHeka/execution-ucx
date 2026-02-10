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

#ifndef AXON_PYTHON_PYTHON_WAKE_MANAGER_HPP_
#define AXON_PYTHON_PYTHON_WAKE_MANAGER_HPP_

#include <nanobind/nanobind.h>

#include <proxy/proxy.h>

#include <atomic>

#include <unifex/detail/atomic_intrusive_queue.hpp>
#include <unifex/detail/intrusive_queue.hpp>
#include <unifex/just.hpp>
#include <unifex/on.hpp>
#include <unifex/spawn_detached.hpp>
#include <unifex/then.hpp>
#include <unifex/timed_single_thread_context.hpp>

namespace eux {
namespace axon {
class AxonRuntime;
}  // namespace axon
}  // namespace eux

namespace eux {
namespace axon {
namespace python {

/**
 * @brief Facade for task callable interface.
 */
struct TaskFacade : pro::facade_builder  //
                    ::add_convention<
                      pro::operator_dispatch<"()">,
                      void()>  //
                    ::build {};

/**
 * @brief Manage wakeups from C++ async completion to Python.
 *
 * The manager owns an eventfd and a lock-free queue of completion tasks.
 * Producers push tasks without holding the GIL. When the queue transitions
 * from inactive to active, the manager writes to the eventfd once (edge
 * triggered). Python can wait on the eventfd (e.g., with io_uring) and call
 * ProcessQueue to execute all pending tasks with the GIL held.
 */
class PythonWakeManager {
 public:
  using Scheduler = decltype(std::declval<unifex::timed_single_thread_context>()
                               .get_scheduler());
  using Task = pro::proxy<TaskFacade>;

  explicit PythonWakeManager();
  ~PythonWakeManager();

  PythonWakeManager(const PythonWakeManager&) = delete;
  PythonWakeManager& operator=(const PythonWakeManager&) = delete;

  /**
   * @brief Enqueue a completion task. Thread-safe.
   *
   * Uses edge-triggered strategy: if the consumer was inactive, writes to
   * eventfd once to wake Python.
   */
  void Enqueue(Task&& task);

  /**
   * @brief Drain and execute all queued tasks with the GIL held.
   * @return Number of tasks executed.
   */
  std::size_t ProcessQueue();

  /**
   * @brief Drain the queue and discard all tasks without executing them.
   * This is used during shutdown to ensure the queue is empty and marked
   * inactive, preventing "stuck active" state while avoiding unsafe execution
   * of Python code during interpreter shutdown or GC.
   * @return Number of tasks discarded.
   */
  std::size_t ClearQueue();

  /**
   * @brief Automatically register asyncio reader for eventfd.
   * This sets up automatic queue processing when eventfd becomes readable.
   * @return True if registration succeeded, false otherwise.
   */
  bool RegisterAsyncioReader();

  /**
   * @brief Unregister asyncio reader for eventfd.
   * This removes the reader from the event loop to allow proper cleanup.
   * @return True if unregistration succeeded, false otherwise.
   */
  bool UnregisterAsyncioReader();

 private:
  /**
   * @brief Return eventfd descriptor for Python-side polling.
   * @note This is for internal use only.
   */
  int GetEventFd() const noexcept { return event_fd_; }

 private:
  struct TaskNode {
    TaskNode* next = nullptr;
    Task task;
  };

  using TaskQueue = unifex::atomic_intrusive_queue<TaskNode, &TaskNode::next>;

  void NotifyPython();

  TaskQueue queue_{false};  // start inactive so first enqueue triggers wake.
  int event_fd_;
  std::atomic<int> reader_ref_count_{0};  // Reference count for asyncio reader
  nanobind::object
    reader_callback_;  // Save the callback object to ensure remove_reader works
  nanobind::object stored_loop_;  // Track the loop we are registered with to
                                  // handle loop changes
};

/**
 * @brief Get (or create) the global wake manager.
 *
 * A single global manager instance is shared. It does not depend on
 * AxonRuntime.
 */
PythonWakeManager& GetPythonWakeManager();

}  // namespace python
}  // namespace axon
}  // namespace eux

#endif  // AXON_PYTHON_PYTHON_WAKE_MANAGER_HPP_
