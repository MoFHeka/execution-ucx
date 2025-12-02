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

#include "rpc_core/utils/hybrid_logical_clock.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

namespace eux {
namespace rpc {
namespace utils {

namespace {

class HybridLogicalClockTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kBasePhysical = 10'000ULL;

  HybridLogicalClock make_clock(uint64_t physical_ms, uint16_t logical) {
    HybridLogicalClock clock;
    clock.Assign(physical_ms, logical);
    return clock;
  }
};

TEST_F(HybridLogicalClockTest, DefaultConstructionStartsAtZero) {
  HybridLogicalClock clock;
  EXPECT_EQ(clock.Raw(), 0u);
  EXPECT_EQ(clock.PhysicalTimeMs(), 0u);
  EXPECT_EQ(clock.LogicalCounter(), 0u);
}

TEST_F(HybridLogicalClockTest, TickLocalAdvancesWhenPhysicalIncreases) {
  HybridLogicalClock clock = make_clock(kBasePhysical, /*logical=*/42);
  const uint64_t next_physical = kBasePhysical + 5;

  clock.TickLocal(next_physical);

  EXPECT_EQ(clock.PhysicalTimeMs(), next_physical);
  EXPECT_EQ(clock.LogicalCounter(), 0u);
}

TEST_F(HybridLogicalClockTest, TickLocalIncrementsLogicalWhenPhysicalStalled) {
  HybridLogicalClock clock = make_clock(kBasePhysical, /*logical=*/1);

  clock.TickLocal(kBasePhysical);
  EXPECT_EQ(clock.PhysicalTimeMs(), kBasePhysical);
  EXPECT_EQ(clock.LogicalCounter(), 2u);

  clock.TickLocal(kBasePhysical - 1);
  EXPECT_EQ(clock.PhysicalTimeMs(), kBasePhysical);
  EXPECT_EQ(clock.LogicalCounter(), 3u);
}

TEST_F(HybridLogicalClockTest, MergeChoosesMaximumPhysicalComponent) {
  HybridLogicalClock local = make_clock(kBasePhysical + 3, 5);
  HybridLogicalClock remote = make_clock(kBasePhysical + 5, 9);

  local.Merge(remote.Raw(), kBasePhysical + 4);

  EXPECT_EQ(local.PhysicalTimeMs(), kBasePhysical + 5);
  EXPECT_EQ(local.LogicalCounter(), 10u);
}

TEST_F(HybridLogicalClockTest, MergeTiesUseMaxLogicalBeforeIncrement) {
  HybridLogicalClock local = make_clock(kBasePhysical, 10);
  HybridLogicalClock remote = make_clock(kBasePhysical, 42);

  local.Merge(remote.Raw(), kBasePhysical);

  EXPECT_EQ(local.PhysicalTimeMs(), kBasePhysical);
  EXPECT_EQ(local.LogicalCounter(), 43u);
}

TEST_F(HybridLogicalClockTest, MergeLowerPhysicalKeepsLocal) {
  HybridLogicalClock local = make_clock(kBasePhysical + 10, 7);
  HybridLogicalClock remote = make_clock(kBasePhysical + 2, 100);

  local.Merge(remote.Raw(), kBasePhysical + 5);

  EXPECT_EQ(local.PhysicalTimeMs(), kBasePhysical + 10);
  EXPECT_EQ(local.LogicalCounter(), 8u);
}

TEST_F(HybridLogicalClockTest, BumpLogicalCounterWraps) {
  HybridLogicalClock clock =
    make_clock(kBasePhysical, HybridLogicalClock::kLogicalMask);

  clock.BumpLogicalCounter();

  EXPECT_EQ(clock.PhysicalTimeMs(), kBasePhysical);
  EXPECT_EQ(clock.LogicalCounter(), 0u);
}

TEST_F(HybridLogicalClockTest, ToStringContainsPhysicalAndLogical) {
  HybridLogicalClock clock = make_clock(123, 4);
  EXPECT_EQ(clock.ToString(), "123.4");
}

TEST_F(HybridLogicalClockTest, NowProducesNonZeroPhysicalComponent) {
  const auto before = HybridLogicalClock::Now();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  const auto after = HybridLogicalClock::Now();

  EXPECT_GT(before.PhysicalTimeMs(), 0u);
  EXPECT_GE(after.PhysicalTimeMs(), before.PhysicalTimeMs());
  EXPECT_EQ(before.LogicalCounter(), 0u);
  EXPECT_EQ(after.LogicalCounter(), 0u);
}

TEST_F(HybridLogicalClockTest, MergeWithSystemNowKeepsProgressMonotonic) {
  HybridLogicalClock local = make_clock(kBasePhysical, 0);
  const auto remote = HybridLogicalClock::Now();

  local.Merge(remote.Raw());

  EXPECT_GE(local.PhysicalTimeMs(), remote.PhysicalTimeMs());
  EXPECT_LT(local.LogicalCounter(), HybridLogicalClock::kLogicalMask);
}

TEST_F(HybridLogicalClockTest, AssignRawOverwritesStateExactly) {
  HybridLogicalClock clock;
  const uint64_t raw =
    (kBasePhysical << HybridLogicalClock::kLogicalBits) | 0x55u;

  clock.AssignRaw(raw);

  EXPECT_EQ(clock.Raw(), raw);
  EXPECT_EQ(clock.PhysicalTimeMs(), kBasePhysical);
  EXPECT_EQ(clock.LogicalCounter(), 0x55u);
}

TEST_F(HybridLogicalClockTest, MergeWithLowerObservedPhysicalPrefersObserved) {
  HybridLogicalClock local = make_clock(kBasePhysical + 5, 2);
  HybridLogicalClock remote = make_clock(kBasePhysical + 10, 1);

  local.Merge(remote.Raw(), kBasePhysical + 20);

  EXPECT_EQ(local.PhysicalTimeMs(), kBasePhysical + 20);
  EXPECT_EQ(local.LogicalCounter(), 0u);
}

}  // namespace

}  // namespace utils
}  // namespace rpc
}  // namespace eux
