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

#include "axon/axon_worker.hpp"
#include "ucx_context/ucx_device_context.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace eux::axon;
using namespace eux::axon::utils;

class ProcessStoredRequestsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mr_ = std::make_unique<eux::ucxx::DefaultUcxMemoryResourceManager>();
  }

  void TearDown() override { mr_.reset(); }

  std::unique_ptr<eux::ucxx::DefaultUcxMemoryResourceManager> mr_;
};

TEST_F(ProcessStoredRequestsTest, BasicProcessTest) {
  auto device_ctx = std::make_unique<eux::ucxx::UcxAutoDefaultDeviceContext>();
  // Use a minimal timeout and thread pool
  AxonWorker worker(
    *mr_, "test_worker", 2, std::chrono::milliseconds(100),
    std::move(device_ctx));

  // 1. Manually inject a request into storage
  auto& storage = worker.GetStorage();

  eux::rpc::RpcRequestHeader header{};
  header.request_id = eux::rpc::request_id_t{123};
  header.session_id = eux::rpc::session_id_t{1};
  header.function_id = eux::rpc::function_id_t{99};

  // Empty payload
  eux::ucxx::UcxBufferVec payload(
    *mr_, ucx_memory_type::HOST, std::vector<size_t>{});

  auto req =
    std::make_shared<AxonRequest>(std::move(header), std::move(payload));
  storage.emplace(req);

  ASSERT_FALSE(storage.empty());

  // 2. Define a StorageIteratorVisitor
  bool visitor_called = false;
  uint32_t visited_req_id = 0;

  struct MockStorageVisitor {
    bool* called_ptr;
    uint32_t* req_id_ptr;

    LifecycleStatus operator()(storage::AxonStorage::request_iterator it) {
      *called_ptr = true;
      // Check if iterator is valid
      auto& req_ptr = *it;
      if (!req_ptr) {
        return LifecycleStatus::Discard;
      }
      *req_id_ptr = static_cast<uint32_t>(req_ptr->header.request_id);
      return LifecycleStatus::Discard;
    }
  };

  AxonWorker::StorageIteratorVisitor visitor =
    pro::make_proxy<AxonWorker::StorageIteratorVisitorFacade>(
      MockStorageVisitor{&visitor_called, &visited_req_id});

  // 3. Call ProcessStoredRequests(visitor)
  // Use 1-arg overload to iterate all stored requests
  auto result = worker.ProcessStoredRequests(std::move(visitor));

  EXPECT_TRUE(result.has_value());
  EXPECT_TRUE(visitor_called);
  EXPECT_EQ(visited_req_id, 123);

  // Should be removed because visitor returned Discard
  EXPECT_TRUE(storage.empty());
}
