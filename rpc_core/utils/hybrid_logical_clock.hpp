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

#ifndef RPC_CORE_UTILS_HYBRID_LOGICAL_CLOCK_HPP_
#define RPC_CORE_UTILS_HYBRID_LOGICAL_CLOCK_HPP_

#include <cista.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <tuple>

namespace eux {
namespace rpc {
namespace utils {

namespace data = cista::offset;

/**
 * @brief Hybrid logical clock with millisecond physical component and 16-bit
 * logical counter.
 *
 * Layout: [63:16] physical milliseconds, [15:0] logical counter. A fully
 * inlineable API is provided to maintain the clock without branching hot paths.
 * Methods leverage branchless arithmetic to minimize latency in RPC middleware
 * interceptors where this clock lives on the critical path.
 */
class HybridLogicalClock final {
 public:
  static constexpr uint64_t kLogicalBits = 16;
  static constexpr uint64_t kLogicalMask = (1ULL << kLogicalBits) - 1ULL;
  static constexpr uint64_t kPhysicalMask = ~kLogicalMask;

  constexpr HybridLogicalClock() noexcept = default;
  explicit constexpr HybridLogicalClock(uint64_t raw_timestamp) noexcept
    : raw_(raw_timestamp) {}

  constexpr HybridLogicalClock(
    uint64_t physical_time_ms, uint16_t logical_counter) noexcept
    : raw_(Pack(physical_time_ms, logical_counter)) {}

  static HybridLogicalClock Now() noexcept {
    return HybridLogicalClock{CurrentPhysicalTimeMs(), 0};
  }

  [[nodiscard]] constexpr uint64_t raw() const noexcept { return raw_; }
  [[nodiscard]] constexpr uint64_t physical_time_ms() const noexcept {
    return raw_ >> kLogicalBits;
  }
  [[nodiscard]] constexpr uint16_t logical_counter() const noexcept {
    return static_cast<uint16_t>(raw_ & kLogicalMask);
  }

  void TickLocal() noexcept { TickLocal(CurrentPhysicalTimeMs()); }

  void TickLocal(uint64_t observed_physical_ms) noexcept {
    // A local event is equivalent to receiving a message from our past self.
    // The logic is unified by deferring to the merge function.
    Merge(raw_, observed_physical_ms);
  }

  void Merge(HybridLogicalClock remote) noexcept { Merge(remote.raw_); }

  void Merge(uint64_t remote_raw) noexcept {
    Merge(remote_raw, CurrentPhysicalTimeMs());
  }

  // Merge remote HLC (remote_raw) and observed physical time into the local
  // clock.
  void Merge(uint64_t remote_raw, uint64_t observed_physical_ms) noexcept {
    const uint64_t local_physical = raw_ >> kLogicalBits;
    const uint64_t remote_physical = remote_raw >> kLogicalBits;

// Fast path: The most common case is that the wall clock is ahead.
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_expect(
          observed_physical_ms > local_physical
            && observed_physical_ms > remote_physical,
          1)) {
#else
    if (
      observed_physical_ms > local_physical
      && observed_physical_ms > remote_physical) {
#endif
      // Physical time has advanced past both clocks. Reset logical time.
      raw_ = observed_physical_ms << kLogicalBits;
      return;
    }

    // Slower path: Timestamps are close or equal, logical counters matter.
    const uint64_t max_physical =
      std::max({observed_physical_ms, local_physical, remote_physical});

    uint64_t next_logical = 0;
    if (max_physical == local_physical && max_physical == remote_physical) {
      const uint64_t local_logical = raw_ & kLogicalMask;
      const uint64_t remote_logical = remote_raw & kLogicalMask;
      next_logical = BumpLogical(std::max(local_logical, remote_logical));
    } else if (max_physical == local_physical) {
      const uint64_t local_logical = raw_ & kLogicalMask;
      next_logical = BumpLogical(local_logical);
    } else if (max_physical == remote_physical) {
      const uint64_t remote_logical = remote_raw & kLogicalMask;
      next_logical = BumpLogical(remote_logical);
    }
    // else: max_physical is from observed_physical_ms, next_logical remains 0.

    raw_ = (max_physical << kLogicalBits) | next_logical;
  }

  void Assign(uint64_t physical_ms, uint16_t logical) noexcept {
    raw_ = Pack(physical_ms, logical);
  }

  void AssignRaw(uint64_t raw_timestamp) noexcept { raw_ = raw_timestamp; }

  void BumpLogicalCounter() noexcept {
    raw_ = (raw_ & kPhysicalMask)
           | BumpLogical(static_cast<uint64_t>(raw_ & kLogicalMask));
  }

  [[nodiscard]] std::string ToString() const {
    return std::to_string(physical_time_ms()) + "."
           + std::to_string(logical_counter());
  }

  [[nodiscard]] constexpr bool operator==(
    const HybridLogicalClock& other) const noexcept {
    return raw_ == other.raw_;
  }
  [[nodiscard]] constexpr bool operator!=(
    const HybridLogicalClock& other) const noexcept {
    return raw_ != other.raw_;
  }
  [[nodiscard]] constexpr bool operator<(
    const HybridLogicalClock& other) const noexcept {
    return raw_ < other.raw_;
  }
  [[nodiscard]] constexpr bool operator<=(
    const HybridLogicalClock& other) const noexcept {
    return raw_ <= other.raw_;
  }
  [[nodiscard]] constexpr bool operator>(
    const HybridLogicalClock& other) const noexcept {
    return raw_ > other.raw_;
  }
  [[nodiscard]] constexpr bool operator>=(
    const HybridLogicalClock& other) const noexcept {
    return raw_ >= other.raw_;
  }

  [[nodiscard]] constexpr auto cista_members() const noexcept {
    return std::tie(raw_);
  }

 private:
  static constexpr uint64_t Pack(
    uint64_t physical_ms, uint16_t logical) noexcept {
    return (physical_ms << kLogicalBits) | static_cast<uint64_t>(logical);
  }

  static constexpr uint64_t BumpLogical(uint64_t logical) noexcept {
    const uint64_t incremented = (logical + 1ULL) & kLogicalMask;
    return incremented;
  }

  static uint64_t CurrentPhysicalTimeMs() noexcept {
    using clock = std::chrono::system_clock;
    using duration = std::chrono::milliseconds;
    return static_cast<uint64_t>(
      std::chrono::duration_cast<duration>(clock::now().time_since_epoch())
        .count());
  }

  uint64_t raw_{0};
};

using workflow_id_t = cista::strong<uint32_t, struct workflow_id_tag>;

}  // namespace utils
}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_UTILS_HYBRID_LOGICAL_CLOCK_HPP_
