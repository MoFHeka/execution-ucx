/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.
 *
 *Licensed under the Apache License Version 2.0 with LLVM Exceptions
 *(the "License"); you may not use this file except in compliance with
 *the License. You may obtain a copy of the License at
 *
 *    https://llvm.org/LICENSE.txt
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 *==============================================================================*/

#pragma once

#ifndef AXON_CORE_UTILS_SLOT_MAP_HPP_
#define AXON_CORE_UTILS_SLOT_MAP_HPP_

#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace eux {
namespace axon {
namespace utils {

// Key used to access items in the SlotMap.
// Maintains a generation counter to detect ABA problems (stale keys).
struct SlotKey {
  // Default initialized to max value, ensuring SlotKey{} generates invalid Key.
  uint32_t index = std::numeric_limits<uint32_t>::max();
  uint32_t generation = 0;

  // C++20/23 default comparison (spaceship operator)
  friend bool operator==(const SlotKey&, const SlotKey&) = default;
};

template <typename T>
class SlotMap {
 public:
  using value_type = T;
  using size_type = uint32_t;

  static constexpr size_type InvalidIndex =
    std::numeric_limits<size_type>::max();

 private:
  struct Slot {
    // Even generation = Slot is occupied (Active).
    // Odd generation  = Slot is part of free list (Free).
    uint32_t generation = 0;

    union {
      T value;
      size_type next_free;
    };

    // Standard C++ union management.
    // Default constructor sets generation to 1 (Free) to ensure exception
    // safety during emplace (if construction throws, we shouldn't destroy
    // uninitialized value).
    Slot() : generation(1), next_free(InvalidIndex) {}

    Slot(Slot&& other) noexcept : generation(other.generation) {
      if (generation % 2 == 0) {
        // Active: move value
        std::construct_at(&value, std::move(other.value));
      } else {
        // Free: copy next_free
        next_free = other.next_free;
      }
    }

    // No copy constructor/assignment to enforce move-only if T is move-only,
    // and to avoid accidental copies in vector resize if T is heavy.
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;

    Slot& operator=(Slot&& other) noexcept {
      if (this != &other) {
        // Destroy current if active
        if (generation % 2 == 0) {
          std::destroy_at(&value);
        }

        generation = other.generation;
        if (generation % 2 == 0) {
          std::construct_at(&value, std::move(other.value));
        } else {
          next_free = other.next_free;
        }
      }
      return *this;
    }

    ~Slot() {
      if (generation % 2 == 0) {
        std::destroy_at(&value);
      }
    }
  };

  std::vector<Slot> slots_;
  size_type free_head_ = InvalidIndex;
  size_type count_ = 0;

 public:
  SlotMap() = default;

  // Disable copy to prevent accidental deep copies of heavy resources.
  // Move is allowed.
  SlotMap(const SlotMap&) = delete;
  SlotMap& operator=(const SlotMap&) = delete;

  SlotMap(SlotMap&& other) noexcept
    : slots_(std::move(other.slots_)),
      free_head_(other.free_head_),
      count_(other.count_) {
    other.free_head_ = InvalidIndex;
    other.count_ = 0;
  }

  SlotMap& operator=(SlotMap&& other) noexcept {
    if (this != &other) {
      clear();
      slots_ = std::move(other.slots_);
      free_head_ = other.free_head_;
      count_ = other.count_;
      other.free_head_ = InvalidIndex;
      other.count_ = 0;
    }
    return *this;
  }

  ~SlotMap() { clear(); }

  // Emplace a new element into the map.
  // Returns a stable Key used to access the element later.
  template <typename... Args>
  [[nodiscard]] SlotKey emplace(Args&&... args) {
    size_type index;

    if (free_head_ != InvalidIndex) {
      // Pop from free list
      index = free_head_;
      Slot& slot = slots_[index];
      free_head_ = slot.next_free;

      // Construct value first for exception safety.
      // If it throws, generation is still Odd (Free), so ~Slot won't destroy
      // uninitialized value.
      std::construct_at(&slot.value, std::forward<Args>(args)...);

      // Transition generation from Odd (Free) to Even (Active)
      // New Generation = Old Odd + 1
      slot.generation++;
    } else {
      // Append new slot
      index = static_cast<size_type>(slots_.size());
      Slot& slot = slots_.emplace_back();

      // Slot starts as Free (generation 1) to be exception-safe.
      // Construct value first.
      std::construct_at(&slot.value, std::forward<Args>(args)...);

      // Then mark as Active (generation 0).
      slot.generation = 0;
    }

    count_++;
    return {index, slots_[index].generation};
  }

  // Access an element by Key (Non-const version).
  // Returns pointer if valid, nullptr if Key is stale or invalid.
  [[nodiscard]] T* get(SlotKey key) {
    // 1. Bounds check
    if (key.index >= slots_.size()) {
      return nullptr;
    }

    // 2. Generation check (covers IsFree check implicitely)
    auto& slot = slots_[key.index];
    if (slot.generation != key.generation) {
      return nullptr;
    }

    return &slot.value;
  }

  // Access an element by Key (Const version).
  // Returns pointer if valid, nullptr if Key is stale or invalid.
  [[nodiscard]] const T* get(SlotKey key) const {
    // 1. Bounds check
    if (key.index >= slots_.size()) {
      return nullptr;
    }

    // 2. Generation check (covers IsFree check implicitely)
    auto& slot = slots_[key.index];
    if (slot.generation != key.generation) {
      return nullptr;
    }

    return &slot.value;
  }

  // Access by operator[] (Unsafe Non-const)
  // Undefined Behavior if key is invalid.
  [[nodiscard]] T& operator[](SlotKey key) {
    assert(
      key.index < slots_.size()
      && slots_[key.index].generation == key.generation);
    return slots_[key.index].value;
  }

  // Access by operator[] (Unsafe Const)
  // Undefined Behavior if key is invalid.
  [[nodiscard]] const T& operator[](SlotKey key) const {
    assert(
      key.index < slots_.size()
      && slots_[key.index].generation == key.generation);
    return slots_[key.index].value;
  }

  // Removes an element. Returns true if successful, false if key was invalid.
  bool erase(SlotKey key) {
    if (key.index >= slots_.size()) return false;

    Slot& slot = slots_[key.index];
    if (slot.generation != key.generation) return false;

    // 1. Destroy value manually.
    // We must do this because we are transitioning to Free state,
    // where ~Slot() will no longer destroy the value.
    std::destroy_at(&slot.value);

    // 2. Update generation: Even (Active) -> Odd (Free)
    slot.generation++;

    // 3. Push to free list
    slot.next_free = free_head_;
    free_head_ = key.index;

    count_--;
    return true;
  }

  void clear() {
    if (count_ == 0) return;

    // We must iterate and destroy only active slots (Even generation)
    // before clearing vector, because vector clear will call ~Slot.
    // We destroy manually and update generation so ~Slot does nothing.
    for (auto& slot : slots_) {
      if (slot.generation % 2 == 0) {
        std::destroy_at(&slot.value);
        // Increment to mark as effectively free
        slot.generation++;
      }
    }

    slots_.clear();
    free_head_ = InvalidIndex;
    count_ = 0;
  }

  [[nodiscard]] size_type size() const { return count_; }
  [[nodiscard]] size_type capacity() const { return slots_.capacity(); }
  [[nodiscard]] bool empty() const { return count_ == 0; }

  // Reserve memory to prevent reallocations
  void reserve(size_type n) { slots_.reserve(n); }
};

template <typename T>
class ThreadSafeSlotMap {
 private:
  SlotMap<T> impl_;
  // Mutable allows locking even in const member functions
  mutable std::shared_mutex mutex_;

 public:
  SlotKey emplace(auto&&... args) {
    std::unique_lock lock(mutex_);
    return impl_.emplace(std::forward<decltype(args)>(args)...);
  }

  bool erase(SlotKey key) {
    std::unique_lock lock(mutex_);
    return impl_.erase(key);
  }

  void clear() {
    std::unique_lock lock(mutex_);
    impl_.clear();
  }

  // Return: pointer to T if found, nullptr if not found.
  auto access(SlotKey key) const -> const T* {
    std::shared_lock lock(mutex_);
    return impl_.get(key);
  }

  auto access(SlotKey key) -> T* {
    std::shared_lock lock(mutex_);
    return impl_.get(key);
  }

  auto access_lockless(SlotKey key) -> T* { return impl_.get(key); }

  auto access_lockless(SlotKey key) const -> const T* { return impl_.get(key); }

  size_t size() const {
    std::shared_lock lock(mutex_);
    return impl_.size();
  }
};

}  // namespace utils
}  // namespace axon
}  // namespace eux

#endif  // AXON_CORE_UTILS_SLOT_MAP_HPP_
