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

#ifndef AXON_UTILS_RING_BUFFER_HPP_
#define AXON_UTILS_RING_BUFFER_HPP_

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace eux {
namespace axon {
namespace utils {

/**
 * @brief High-performance ring buffer using request_id as direct index
 *
 * This ring buffer uses request_id directly as an index (with modulo),
 * and uses session_id for validation. This eliminates the need for
 * hashing and provides O(1) lookup performance.
 *
 * @tparam SessionIdT The session ID type (for validation)
 * @tparam Value The value type to store
 * @tparam Size The size of the ring buffer (must be power of 2)
 */
template <typename SessionIdT, typename Value, size_t Size>
class RingBuffer {
  static_assert((Size & (Size - 1)) == 0, "Size must be a power of 2");

 public:
  using session_id_type = SessionIdT;
  using value_type = Value;
  using size_type = size_t;
  using request_id_type = uint32_t;

 private:
  // Slot state: empty or occupied
  enum class SlotState : uint8_t { Empty = 0, Occupied = 1 };

  struct Slot {
    std::atomic<SlotState> state{SlotState::Empty};
    SessionIdT session_id{};
    Value value{};

    Slot() = default;
    // Non-copyable, non-movable for lock-free safety
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    Slot(Slot&&) = delete;
    Slot& operator=(Slot&&) = delete;

    ~Slot() {
      SlotState s = state.load(std::memory_order_acquire);
      if (s == SlotState::Occupied) {
        session_id.~SessionIdT();
        value.~Value();
      }
    }
  };

  alignas(64) Slot slots_[Size];
  alignas(64) std::atomic<size_type> size_{0};

  // Fast modulo operation for power-of-two size
  static constexpr size_type mask_ = Size - 1;

  // Compute slot index from request_id
  static constexpr size_type request_id_to_index(request_id_type request_id) {
    return static_cast<size_type>(request_id) & mask_;
  }

 public:
  /**
   * @brief Construct a new RingBuffer object
   */
  RingBuffer() {
    // Initialize all slots as empty (already initialized by default)
  }

  /**
   * @brief Destroy the RingBuffer object
   */
  ~RingBuffer() { clear(); }

  // Non-copyable
  RingBuffer(const RingBuffer&) = delete;
  RingBuffer& operator=(const RingBuffer&) = delete;

  // Non-movable for lock-free safety
  RingBuffer(RingBuffer&&) = delete;
  RingBuffer& operator=(RingBuffer&&) = delete;

  /**
   * @brief Insert or update an entry
   *
   * @param request_id The request ID (used as index). Because the cista strong
   * type does not support the "&=" operation on general C++ types, normal
   * unit32_t are used for request_id here.
   * @param session_id The session ID (used for validation)
   * @param value The value to store
   * @return true if inserted/updated successfully, false if slot is occupied
   * by different session_id or buffer is full
   */
  template <typename S, typename V>
  bool emplace(request_id_type request_id, S&& session_id, V&& value) {
    size_type idx = request_id_to_index(request_id);
    auto& slot = slots_[idx];

    // Try to acquire the slot (lock-free)
    SlotState expected = SlotState::Empty;
    if (!slot.state.compare_exchange_weak(
          expected, SlotState::Occupied, std::memory_order_acq_rel)) {
      // Slot is occupied, check if session_id matches
      if (expected == SlotState::Occupied) {
        // Need to check session_id, but we need to be careful about
        // concurrent access. For safety, we'll use acquire semantics.
        std::atomic_thread_fence(std::memory_order_acquire);
        if (slot.session_id == session_id) {
          // Same session_id, update the value (this doesn't change the count)
          slot.value = std::forward<V>(value);
          return true;
        } else {
          // Different session_id, this is a collision - reject it
          // Note: This could happen even if buffer is not full due to
          // hash collision (different request_id mapping to same slot)
          return false;
        }
      }
      // Retry if compare_exchange_weak failed due to spurious failure
      // (This should be rare, but handle it by returning false)
      return false;
    }

    // Successfully acquired empty slot
    // Check if buffer is full before inserting (best-effort check)
    // Note: This check is not atomic with the slot acquisition, but provides
    // early detection. The actual limit is enforced by slot availability.
    size_type current_size = size_.load(std::memory_order_relaxed);
    if (current_size >= Size) {
      // Buffer appears full, release the slot we just acquired
      slot.state.store(SlotState::Empty, std::memory_order_release);
      return false;
    }

    // Now construct the objects
    new (&slot.session_id) SessionIdT(std::forward<S>(session_id));
    new (&slot.value) Value(std::forward<V>(value));
    size_.fetch_add(1, std::memory_order_relaxed);

    return true;
  }

  /**
   * @brief Find an entry by request_id and validate with session_id
   *
   * @param request_id The request ID (used as index)
   * @param session_id The session ID (for validation)
   * @return size_type Index of the slot if found and validated, or Size if not
   * found
   */
  size_type find(
    request_id_type request_id, const SessionIdT& session_id) const {
    size_type idx = request_id_to_index(request_id);
    const auto& slot = slots_[idx];

    SlotState state = slot.state.load(std::memory_order_acquire);
    if (state == SlotState::Occupied) {
      // Validate session_id matches (read after acquire barrier)
      if (slot.session_id == session_id) {
        return idx;
      }
    }

    return Size;
  }

  /**
   * @brief Erase an entry by request_id
   *
   * @param request_id The request ID
   * @return true if erased successfully
   */
  bool erase(request_id_type request_id) {
    size_type idx = request_id_to_index(request_id);
    auto& slot = slots_[idx];

    SlotState expected = SlotState::Occupied;
    if (!slot.state.compare_exchange_strong(
          expected, SlotState::Empty, std::memory_order_acq_rel)) {
      return false;
    }

    // Successfully acquired the slot, now destruct the objects
    slot.session_id.~SessionIdT();
    slot.value.~Value();
    size_.fetch_sub(1, std::memory_order_relaxed);

    return true;
  }

  /**
   * @brief Erase an entry by request_id and validate with session_id
   *
   * @param request_id The request ID
   * @param session_id The session ID (for validation)
   * @return true if erased successfully
   */
  bool erase(request_id_type request_id, const SessionIdT& session_id) {
    size_type idx = request_id_to_index(request_id);
    auto& slot = slots_[idx];

    // First check if occupied and session_id matches
    SlotState state = slot.state.load(std::memory_order_acquire);
    if (state != SlotState::Occupied) {
      return false;
    }

    // Validate session_id matches (read after acquire barrier)
    if (slot.session_id != session_id) {
      return false;
    }

    // Try to acquire and clear the slot
    SlotState expected = SlotState::Occupied;
    if (!slot.state.compare_exchange_strong(
          expected, SlotState::Empty, std::memory_order_acq_rel)) {
      return false;
    }

    // Successfully acquired the slot, now destruct the objects
    slot.session_id.~SessionIdT();
    slot.value.~Value();
    size_.fetch_sub(1, std::memory_order_relaxed);

    return true;
  }

  /**
   * @brief Get the current size
   *
   * @return size_type Number of occupied slots
   */
  size_type size() const { return size_.load(std::memory_order_relaxed); }

  /**
   * @brief Check if the buffer is empty
   *
   * @return true if empty
   */
  bool empty() const { return size() == 0; }

  /**
   * @brief Check if the buffer is full
   *
   * @return true if full (all slots occupied)
   */
  bool full() const { return size() >= Size; }

  /**
   * @brief Get the maximum capacity
   *
   * @return size_type Maximum number of slots
   */
  static constexpr size_type capacity() { return Size; }

  /**
   * @brief Clear all entries
   */
  void clear() {
    for (size_type i = 0; i < Size; ++i) {
      auto& slot = slots_[i];
      SlotState state = slot.state.load(std::memory_order_acquire);
      if (state == SlotState::Occupied) {
        slot.session_id.~SessionIdT();
        slot.value.~Value();
      }
      slot.state.store(SlotState::Empty, std::memory_order_release);
    }
    size_.store(0, std::memory_order_relaxed);
  }

  /**
   * @brief Get value at index
   *
   * @param idx The index
   * @return Value& Reference to the value
   */
  Value& at(size_type idx) {
    assert(idx < Size);
    SlotState state = slots_[idx].state.load(std::memory_order_acquire);
    assert(state == SlotState::Occupied);
    return slots_[idx].value;
  }

  /**
   * @brief Get value at index (const version)
   *
   * @param idx The index
   * @return const Value& Reference to the value
   */
  const Value& at(size_type idx) const {
    assert(idx < Size);
    SlotState state = slots_[idx].state.load(std::memory_order_acquire);
    assert(state == SlotState::Occupied);
    return slots_[idx].value;
  }

  /**
   * @brief Get session_id at index
   *
   * @param idx The index
   * @return const SessionIdT& Reference to the session_id
   */
  const SessionIdT& session_id_at(size_type idx) const {
    assert(idx < Size);
    SlotState state = slots_[idx].state.load(std::memory_order_acquire);
    assert(state == SlotState::Occupied);
    return slots_[idx].session_id;
  }

  /**
   * @brief Iterator-like interface for iteration
   */
  class Iterator {
   private:
    const RingBuffer* buffer_;
    size_type idx_;

    void advance_to_next_occupied() {
      while (idx_ < Size) {
        SlotState state =
          buffer_->slots_[idx_].state.load(std::memory_order_acquire);
        if (state == SlotState::Occupied) {
          break;
        }
        ++idx_;
      }
    }

   public:
    Iterator(const RingBuffer* buffer, size_type idx)
      : buffer_(buffer), idx_(idx) {
      advance_to_next_occupied();
    }

    Iterator& operator++() {
      ++idx_;
      advance_to_next_occupied();
      return *this;
    }

    bool operator==(const Iterator& other) const {
      return buffer_ == other.buffer_ && idx_ == other.idx_;
    }

    bool operator!=(const Iterator& other) const { return !(*this == other); }

    struct Entry {
      request_id_type request_id;
      const SessionIdT& session_id;
      const Value& value;
    };

    Entry operator*() const {
      // Reconstruct request_id from index (this is approximate)
      // For iteration, we need to track the actual request_id
      // For now, return the index as request_id (this is a limitation)
      // Note: We've already verified the slot is occupied in
      // advance_to_next_occupied
      return Entry{
        static_cast<request_id_type>(idx_), buffer_->slots_[idx_].session_id,
        buffer_->slots_[idx_].value};
    }

    size_type index() const { return idx_; }
  };

  /**
   * @brief Get iterator to first occupied slot
   */
  Iterator begin() const { return Iterator(this, 0); }

  /**
   * @brief Get iterator to end
   */
  Iterator end() const { return Iterator(this, Size); }
};

}  // namespace utils
}  // namespace axon
}  // namespace eux

#endif  // AXON_UTILS_RING_BUFFER_HPP_
