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

#ifndef UCX_CONTEXT_LOCK_FREE_QUEUE_HPP_
#define UCX_CONTEXT_LOCK_FREE_QUEUE_HPP_

#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace eux {

/**
 * @brief A high-performance lock-free queue implementation
 *
 * This queue is designed for high-concurrency scenarios where multiple threads
 * need to access the queue simultaneously without blocking each other.
 *
 * @tparam T The type of elements stored in the queue
 */
template <typename T>
class LockFreeQueue {
 private:
  /**
   * @brief Node structure for the lock-free queue
   */
  struct Node {
    T data;
    std::atomic<Node*> next;

    explicit Node(const T& value) : data(value), next(nullptr) {}
  };

  // Head and tail pointers for the queue
  std::atomic<Node*> head_;
  std::atomic<Node*> tail_;

 public:
  /**
   * @brief Construct a new Lock Free Queue object
   *
   * Initializes the queue with a dummy node to simplify push/pop operations
   */
  LockFreeQueue() {
    Node* dummy = new Node(T{});  // Dummy node to simplify push/pop logic
    head_.store(dummy);
    tail_.store(dummy);
  }

  /**
   * @brief Destroy the Lock Free Queue object
   *
   * Cleans up all nodes in the queue
   */
  ~LockFreeQueue() {
    while (Node* old_head = head_.load()) {
      head_.store(old_head->next);
      delete old_head;
    }
  }

  /**
   * @brief Push a value to the queue
   *
   * @param value The value to push
   */
  void push(const T& value) {
    Node* new_node = new Node(value);
    Node* old_tail;

    while (true) {
      old_tail = tail_.load();
      Node* tail_next = old_tail->next;

      if (old_tail == tail_.load()) {  // Check if tail hasn't changed
        if (tail_next == nullptr) {    // Tail is truly the last node
          if (old_tail->next.compare_exchange_weak(tail_next, new_node)) {
            break;  // Successfully added the new node
          }
        } else {  // Tail is lagging; advance it
          tail_.compare_exchange_weak(old_tail, tail_next);
        }
      }
    }
    // Move tail to the new node
    tail_.compare_exchange_weak(old_tail, new_node);
  }

  /**
   * @brief Pop a value from the queue
   *
   * @param result Reference to store the popped value
   * @return true if a value was successfully popped
   * @return false if the queue was empty
   */
  bool pop(T& result) {
    Node* old_head;

    while (true) {
      old_head = head_.load();
      Node* old_tail = tail_.load();
      Node* head_next = old_head->next;

      if (old_head == head_.load()) {  // Consistency check
        if (old_head == old_tail) {    // Queue might be empty
          if (head_next == nullptr) {  // Queue is empty
            return false;
          }
          tail_.compare_exchange_weak(old_tail, head_next);  // Advance tail
        } else {  // Queue is not empty
          if (head_.compare_exchange_weak(old_head, head_next)) {
            result = head_next->data;
            delete old_head;  // Free the old dummy head node
            return true;
          }
        }
      }
    }
  }

  /**
   * @brief Check if the queue is empty
   *
   * @return true if the queue is empty
   * @return false if the queue contains elements
   */
  bool empty() const {
    Node* old_head = head_.load();
    Node* old_tail = tail_.load();
    Node* head_next = old_head->next;

    return (old_head == old_tail && head_next == nullptr);
  }

  /**
   * @brief Get the current size of the queue
   *
   * Note: This is an approximate size as the queue can change while counting
   *
   * @return size_t The approximate number of elements in the queue
   */
  size_t size() const {
    size_t count = 0;
    Node* current = head_.load()->next;

    while (current != nullptr) {
      count++;
      current = current->next;
    }

    return count;
  }
};

}  // namespace eux

#endif  // UCX_CONTEXT_LOCK_FREE_QUEUE_HPP_
