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

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <ostream>

#include "axon/message_lifecycle_policy.hpp"
#include "axon/storage/axon_storage.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

// Explicitly use namespaces to match headers
using namespace eux::axon;
using namespace eux::axon::storage;
using namespace eux::axon::utils;
namespace ucxx = eux::ucxx;

// allow GTest to print readable enum names (e.g. "Discard") instead of raw
// numbers when assertions fail.
namespace eux {
namespace axon {
std::ostream& operator<<(std::ostream& os, LifecycleStatus status) {
  switch (status) {
    case LifecycleStatus::Discard:
      os << "Discard";
      break;
    case LifecycleStatus::Preserve:
      os << "Preserve";
      break;
    case LifecycleStatus::Error:
      os << "Error";
      break;
    default:
      os << "Unknown";
      break;
  }
  return os;
}
}  // namespace axon
}  // namespace eux

class MessageLifecyclePolicyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mr_ = std::make_shared<ucxx::DefaultUcxMemoryResourceManager>();
  }

  std::shared_ptr<AxonRequest> CreateRequest(uint32_t id) {
    eux::rpc::RpcRequestHeader header{};
    header.request_id = eux::rpc::request_id_t{id};
    header.session_id = eux::rpc::session_id_t{100};  // Dummy session

    // Create empty buffer vec for payload
    ucxx::UcxBufferVec payload(
      *mr_, ucx_memory_type::HOST, std::vector<size_t>{});

    return std::make_shared<AxonRequest>(std::move(header), std::move(payload));
  }

  AxonStorage storage_;
  std::shared_ptr<ucxx::UcxMemoryResourceManager> mr_;
};

// Strict logic policy to verify hive scanning
struct StrictHiveScanningPolicy {
  AxonStorage* storage_ptr;
  uint32_t target_id;

  LifecycleStatus operator()(
    std::shared_ptr<eux::axon::utils::AxonRequest> /*req*/,
    eux::axon::utils::AxonMessageID /*id*/) {
    if (!storage_ptr) return LifecycleStatus::Error;

    for (auto it = storage_ptr->begin(); it != storage_ptr->end(); ++it) {
      if (static_cast<uint32_t>((*it)->header.request_id) == target_id) {
        return LifecycleStatus::Discard;
      }
    }
    return LifecycleStatus::Preserve;
  }
};

TEST_F(MessageLifecyclePolicyTest, ScanHiveTest) {
  // Populate hive with some requests
  storage_.emplace(CreateRequest(1));
  storage_.emplace(CreateRequest(2));
  storage_.emplace(CreateRequest(3));

  // Define policy to scan for ID 2 (should discard)
  StrictHiveScanningPolicy policy_discard{&storage_, 2};
  auto req_new = CreateRequest(99);

  // Verify Logic directly
  EXPECT_EQ(policy_discard(req_new, 0), LifecycleStatus::Discard);

  // Define policy to scan for ID 4 (should preserve)
  StrictHiveScanningPolicy policy_preserve{&storage_, 4};

  // Verify Logic directly
  EXPECT_EQ(policy_preserve(req_new, 0), LifecycleStatus::Preserve);
}

struct StrictLatestElementPolicy {
  AxonStorage* storage_ptr;

  LifecycleStatus operator()(
    std::shared_ptr<eux::axon::utils::AxonRequest> req,
    eux::axon::utils::AxonMessageID /*id*/) {
    if (!storage_ptr) return LifecycleStatus::Error;
    if (storage_ptr->empty()) {
      return LifecycleStatus::Preserve;
    }

    auto& last_req = storage_ptr->back();
    if (
      static_cast<uint32_t>(req->header.request_id)
      > static_cast<uint32_t>(last_req->header.request_id)) {
      return LifecycleStatus::Preserve;
    }
    return LifecycleStatus::Discard;
  }
};

TEST_F(MessageLifecyclePolicyTest, LatestElementTest) {
  // Populate hive
  storage_.emplace(CreateRequest(10));
  storage_.emplace(CreateRequest(20));

  StrictLatestElementPolicy policy{&storage_};

  // Case 1: New ID 21 > 20 -> Preserve
  auto req_ok = CreateRequest(21);
  EXPECT_EQ(policy(req_ok, 0), LifecycleStatus::Preserve);

  // Case 2: New ID 15 < 20 -> Discard
  auto req_bad = CreateRequest(15);
  EXPECT_EQ(policy(req_bad, 0), LifecycleStatus::Discard);
}

// 3. Test calculation using Hive data
// Example: Calculate sum of existing IDs, if new ID + sum > threshold, Discard.
struct CalculationPolicy {
  AxonStorage* storage_ptr;
  uint32_t threshold;

  LifecycleStatus operator()(
    std::shared_ptr<eux::axon::utils::AxonRequest> req,
    eux::axon::utils::AxonMessageID /*id*/) {
    if (!storage_ptr) return LifecycleStatus::Error;
    uint32_t sum = 0;
    for (auto it = storage_ptr->begin(); it != storage_ptr->end(); ++it) {
      sum += static_cast<uint32_t>((*it)->header.request_id);
    }

    if (sum + static_cast<uint32_t>(req->header.request_id) > threshold) {
      return LifecycleStatus::Discard;
    }
    return LifecycleStatus::Preserve;
  }
};

TEST_F(MessageLifecyclePolicyTest, CalculationTest) {
  storage_.emplace(CreateRequest(10));
  storage_.emplace(CreateRequest(10));
  // Sum = 20

  CalculationPolicy policy{&storage_, 50};  // Threshold 50

  // 20 + 20 = 40 <= 50 -> Preserve
  EXPECT_EQ(policy(CreateRequest(20), 0), LifecycleStatus::Preserve);

  // 20 + 31 = 51 > 50 -> Discard
  EXPECT_EQ(policy(CreateRequest(31), 0), LifecycleStatus::Discard);
}
