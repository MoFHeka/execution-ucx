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

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "rpc_core/utils/hybrid_logical_clock.hpp"

namespace eux {
namespace rpc {
namespace utils {

namespace {

class HybridLogicalClockTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kBasePhysical = 10'000ULL;

  HybridLogicalClock make_clock(uint64_t physical_ms, uint16_t logical) {
    HybridLogicalClock clock;
    clock.assign(physical_ms, logical);
    return clock;
  }
};

TEST_F(HybridLogicalClockTest, DefaultConstructionStartsAtZero) {
  HybridLogicalClock clock;
  EXPECT_EQ(clock.raw(), 0u);
  EXPECT_EQ(clock.physical_time_ms(), 0u);
  EXPECT_EQ(clock.logical_counter(), 0u);
}

TEST_F(HybridLogicalClockTest, TickLocalAdvancesWhenPhysicalIncreases) {
  HybridLogicalClock clock = make_clock(kBasePhysical, /*logical=*/42);
  const uint64_t next_physical = kBasePhysical + 5;

  clock.tick_local(next_physical);

  EXPECT_EQ(clock.physical_time_ms(), next_physical);
  EXPECT_EQ(clock.logical_counter(), 1u);
}

TEST_F(HybridLogicalClockTest, TickLocalIncrementsLogicalWhenPhysicalStalled) {
  HybridLogicalClock clock = make_clock(kBasePhysical, /*logical=*/1);

  clock.tick_local(kBasePhysical);
  EXPECT_EQ(clock.physical_time_ms(), kBasePhysical);
  EXPECT_EQ(clock.logical_counter(), 2u);

  clock.tick_local(kBasePhysical - 1);
  EXPECT_EQ(clock.physical_time_ms(), kBasePhysical);
  EXPECT_EQ(clock.logical_counter(), 3u);
}

TEST_F(HybridLogicalClockTest, MergeChoosesMaximumPhysicalComponent) {
  HybridLogicalClock local = make_clock(kBasePhysical + 3, 5);
  HybridLogicalClock remote = make_clock(kBasePhysical + 5, 9);

  local.merge(remote.raw(), kBasePhysical + 4);

  EXPECT_EQ(local.physical_time_ms(), kBasePhysical + 5);
  EXPECT_EQ(local.logical_counter(), 10u);
}

TEST_F(HybridLogicalClockTest, MergeTiesUseMaxLogicalBeforeIncrement) {
  HybridLogicalClock local = make_clock(kBasePhysical, 10);
  HybridLogicalClock remote = make_clock(kBasePhysical, 42);

  local.merge(remote.raw(), kBasePhysical);

  EXPECT_EQ(local.physical_time_ms(), kBasePhysical);
  EXPECT_EQ(local.logical_counter(), 43u);
}

TEST_F(HybridLogicalClockTest, MergeLowerPhysicalKeepsLocal) {
  HybridLogicalClock local = make_clock(kBasePhysical + 10, 7);
  HybridLogicalClock remote = make_clock(kBasePhysical + 2, 100);

  local.merge(remote.raw(), kBasePhysical + 5);

  EXPECT_EQ(local.physical_time_ms(), kBasePhysical + 10);
  EXPECT_EQ(local.logical_counter(), 8u);
}

TEST_F(HybridLogicalClockTest, BumpLogicalCounterWraps) {
  HybridLogicalClock clock =
    make_clock(kBasePhysical, HybridLogicalClock::kLogicalMask);

  clock.bump_logical_counter();

  EXPECT_EQ(clock.physical_time_ms(), kBasePhysical);
  EXPECT_EQ(clock.logical_counter(), 0u);
}

TEST_F(HybridLogicalClockTest, ToStringContainsPhysicalAndLogical) {
  HybridLogicalClock clock = make_clock(123, 4);
  EXPECT_EQ(clock.to_string(), "123.4");
}

TEST_F(HybridLogicalClockTest, NowProducesNonZeroPhysicalComponent) {
  const auto before = HybridLogicalClock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const auto after = HybridLogicalClock::now();

  EXPECT_GT(before.physical_time_ms(), 0u);
  EXPECT_GE(after.physical_time_ms(), before.physical_time_ms());
  EXPECT_EQ(before.logical_counter(), 0u);
  EXPECT_EQ(after.logical_counter(), 0u);
}

TEST_F(HybridLogicalClockTest, MergeWithSystemNowKeepsProgressMonotonic) {
  HybridLogicalClock local = make_clock(kBasePhysical, 0);
  const auto remote = HybridLogicalClock::now();

  local.merge(remote.raw());

  EXPECT_GE(local.physical_time_ms(), remote.physical_time_ms());
  EXPECT_LT(local.logical_counter(), HybridLogicalClock::kLogicalMask);
}

TEST_F(HybridLogicalClockTest, AssignRawOverwritesStateExactly) {
  HybridLogicalClock clock;
  const uint64_t raw =
    (kBasePhysical << HybridLogicalClock::kLogicalBits) | 0x55u;

  clock.assign_raw(raw);

  EXPECT_EQ(clock.raw(), raw);
  EXPECT_EQ(clock.physical_time_ms(), kBasePhysical);
  EXPECT_EQ(clock.logical_counter(), 0x55u);
}

TEST_F(HybridLogicalClockTest, MergeWithLowerObservedPhysicalPrefersObserved) {
  HybridLogicalClock local = make_clock(kBasePhysical + 5, 2);
  HybridLogicalClock remote = make_clock(kBasePhysical + 10, 1);

  local.merge(remote.raw(), kBasePhysical + 20);

  EXPECT_EQ(local.physical_time_ms(), kBasePhysical + 20);
  EXPECT_EQ(local.logical_counter(), 1u);
}

}  // namespace

}  // namespace utils
}  // namespace rpc
}  // namespace eux
