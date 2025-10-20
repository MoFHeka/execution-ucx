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
    : raw_(pack(physical_time_ms, logical_counter)) {}

  static HybridLogicalClock now() noexcept {
    return HybridLogicalClock{current_physical_time_ms(), 0};
  }

  [[nodiscard]] constexpr uint64_t raw() const noexcept { return raw_; }
  [[nodiscard]] constexpr uint64_t physical_time_ms() const noexcept {
    return raw_ >> kLogicalBits;
  }
  [[nodiscard]] constexpr uint16_t logical_counter() const noexcept {
    return static_cast<uint16_t>(raw_ & kLogicalMask);
  }

  void tick_local() noexcept { tick_local(current_physical_time_ms()); }

  void tick_local(uint64_t observed_physical_ms) noexcept {
    const uint64_t current_raw = raw_;
    const uint64_t current_physical = current_raw >> kLogicalBits;
    const uint64_t advance_mask =
      static_cast<uint64_t>(observed_physical_ms > current_physical);

    const uint64_t physical_part = (advance_mask * observed_physical_ms
                                    + (1ULL - advance_mask) * current_physical)
                                   << kLogicalBits;

    const uint64_t logical_base =
      advance_mask * 0U
      + (1ULL - advance_mask)
          * static_cast<uint64_t>(current_raw & kLogicalMask);
    raw_ = physical_part | bump_logical(logical_base);
  }

  void merge(HybridLogicalClock remote) noexcept { merge(remote.raw_); }

  void merge(uint64_t remote_raw) noexcept {
    merge(remote_raw, current_physical_time_ms());
  }

  // Merge remote HLC (remote_raw) and observed physical time into the local
  // clock.
  void merge(uint64_t remote_raw, uint64_t observed_physical_ms) noexcept {
    // Extract fields from current and remote HLC
    const uint64_t local_raw = raw_;
    const uint64_t local_physical = local_raw >> kLogicalBits;
    const uint64_t local_logical = local_raw & kLogicalMask;

    const uint64_t remote_physical = remote_raw >> kLogicalBits;
    const uint64_t remote_logical = remote_raw & kLogicalMask;

    // Fast path: if our wall clock time is max, prefer to early-out with
    // logical=0
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
      raw_ = (observed_physical_ms << kLogicalBits);  // logical is 0
      return;
    }

    // Otherwise, need to determine which clock(s) are max, and merge logical
    // counters Compute the maximum physical timestamp (most often, this is
    // observed_physical_ms)
    const uint64_t max_physical =
      std::max({observed_physical_ms, local_physical, remote_physical});
    const bool local_is_max = local_physical == max_physical;
    const bool remote_is_max = remote_physical == max_physical;

    // If both are max, take max(logical); else use the logical from the max
    // one.
    const uint64_t candidate_logical =
      (local_is_max && remote_is_max)
        ? std::max(local_logical, remote_logical)
        : (local_is_max ? local_logical : remote_logical);

    const uint64_t merged_logical = bump_logical(candidate_logical);
    raw_ = (max_physical << kLogicalBits) | merged_logical;
  }

  void assign(uint64_t physical_ms, uint16_t logical) noexcept {
    raw_ = pack(physical_ms, logical);
  }

  void assign_raw(uint64_t raw_timestamp) noexcept { raw_ = raw_timestamp; }

  void bump_logical_counter() noexcept {
    raw_ = (raw_ & kPhysicalMask)
           | bump_logical(static_cast<uint64_t>(raw_ & kLogicalMask));
  }

  [[nodiscard]] std::string to_string() const {
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
  static constexpr uint64_t pack(
    uint64_t physical_ms, uint16_t logical) noexcept {
    return (physical_ms << kLogicalBits) | static_cast<uint64_t>(logical);
  }

  static constexpr uint64_t bump_logical(uint64_t logical) noexcept {
    const uint64_t incremented = (logical + 1ULL) & kLogicalMask;
    return incremented;
  }

  static uint64_t current_physical_time_ms() noexcept {
    using clock = std::chrono::system_clock;
    using duration = std::chrono::milliseconds;
    return static_cast<uint64_t>(
      std::chrono::duration_cast<duration>(clock::now().time_since_epoch())
        .count());
  }

  uint64_t raw_{0};
};

using event_id_t = cista::strong<uint32_t, struct event_id_tag>;
using workflow_id_t = cista::strong<uint32_t, struct workflow_id_tag>;

/**
 * @brief Metadata for tracing directed cyclic workflows in DAG-like schedulers.
 */
struct EventMetadata {
  event_id_t event_id{};
  workflow_id_t workflow_id{};

  EventMetadata() = default;

  EventMetadata(event_id_t eid, workflow_id_t wid)
    : event_id{eid}, workflow_id{wid} {}

  EventMetadata(unsigned int eid, unsigned int wid)
    : event_id{eid}, workflow_id{wid} {}

  [[nodiscard]] constexpr bool valid() const noexcept {
    return event_id != event_id_t{};
  }

  constexpr void clear() noexcept {
    event_id = event_id_t{};
    workflow_id = workflow_id_t{};
  }

  constexpr void assign(
    event_id_t new_id, workflow_id_t new_workflow_id = {}) noexcept {
    event_id = new_id;
    workflow_id = new_workflow_id;
  }

  [[nodiscard]] constexpr auto cista_members() const noexcept {
    return std::tie(event_id, workflow_id);
  }
};

/**
 * @brief Aggregated temporal metadata propagated through RPC middleware.
 */
struct RpcTemporalMetadata {
  HybridLogicalClock clock{};
  EventMetadata event{};

  [[nodiscard]] constexpr auto cista_members() const noexcept {
    return std::tie(clock, event);
  }
};

}  // namespace utils
}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_UTILS_HYBRID_LOGICAL_CLOCK_HPP_
