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

#include "axon/storage/axon_storage.hpp"

#include <cista.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <unifex/inplace_stop_token.hpp>

#include "ucx_context/ucx_memory_resource.hpp"

namespace eux {
namespace axon {
namespace storage {

namespace data = cista::offset;

using eux::rpc::ParamMeta;
using eux::rpc::ParamType;
using eux::rpc::TensorMeta;

// Helper function to compare two requests
void CompareRequests(
  const std::shared_ptr<AxonRequest>& lhs,
  const std::shared_ptr<AxonRequest>& rhs) {
  ASSERT_NE(lhs, nullptr);
  ASSERT_NE(rhs, nullptr);
  EXPECT_EQ(lhs->header.session_id, rhs->header.session_id);
  EXPECT_EQ(lhs->header.function_id, rhs->header.function_id);

  // Verify TensorMeta
  ASSERT_EQ(lhs->header.params.size(), rhs->header.params.size());
  if (!lhs->header.params.empty()) {
    const auto& lhs_tm = cista::get<TensorMeta>(lhs->header.params[0].value);
    const auto& rhs_tm = cista::get<TensorMeta>(rhs->header.params[0].value);
    EXPECT_EQ(lhs_tm.shape, rhs_tm.shape);
    EXPECT_EQ(lhs_tm.dtype.code, rhs_tm.dtype.code);
  }

  // Verify payload
  ASSERT_EQ(lhs->payload.index(), rhs->payload.index());
  switch (lhs->payload.index()) {
    case 0: {  // std::monostate
      break;
    }
    case 1: {  // UcxBuffer
      const auto& lb = std::get<ucxx::UcxBuffer>(lhs->payload);
      const auto& rb = std::get<ucxx::UcxBuffer>(rhs->payload);
      EXPECT_EQ(lb.size(), rb.size());
      EXPECT_EQ(0, memcmp(lb.data(), rb.data(), lb.size()));
      break;
    }
    case 2: {  // UcxBufferVec
      const auto& lv = std::get<ucxx::UcxBufferVec>(lhs->payload);
      const auto& rv = std::get<ucxx::UcxBufferVec>(rhs->payload);
      ASSERT_EQ(lv.size(), rv.size());
      for (size_t j = 0; j < lv.size(); ++j) {
        EXPECT_EQ(lv[j].size, rv[j].size);
        EXPECT_EQ(0, memcmp(lv[j].data, rv[j].data, lv[j].size));
      }
      break;
    }
    default:
      FAIL() << "Unknown payload variant index";
  }
}

class AxonStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a temporary directory for test files
    test_dir_ = std::filesystem::temp_directory_path() / "axon_storage_test";
    std::filesystem::remove_all(test_dir_);  // Ensure the directory is clean
    std::filesystem::create_directory(test_dir_);

    // Initialize memory resource
    mr_ = std::make_shared<ucxx::DefaultUcxMemoryResourceManager>();

    // Start the io_context on a background thread.
    context_thread_ =
      std::thread([this] { context_.run(stop_source_.get_token()); });
  }

  void TearDown() override {
    // Request stop to the context.
    stop_source_.request_stop();

    // Join the context thread.
    if (context_thread_.joinable()) {
      context_thread_.join();
    }
    // Clean up the temporary directory
    std::filesystem::remove_all(test_dir_);
  }

  // Helper to create a sample AxonRequest
  std::shared_ptr<AxonRequest> CreateRequest(
    uint32_t session_id, uint32_t function_id) {
    rpc::RpcRequestHeader header{};
    header.session_id = rpc::session_id_t{session_id};
    header.function_id = rpc::function_id_t{function_id};

    // Add a TensorMeta to the header
    data::vector<ParamMeta> params;
    TensorMeta tm{};
    tm.device = {kDLCPU, 0};
    tm.ndim = 1;
    tm.dtype = {kDLFloat, 32, 1};
    tm.shape = data::vector<int64_t>{128};
    params.push_back({ParamType::TENSOR_META, tm, "tensor_param"});
    header.params = std::move(params);

    // Calculate payload size from TensorMeta
    size_t data_size = 128 * (32 / 8);

    ucxx::UcxBufferVec payload(
      *mr_, ucx_memory_type::HOST, std::vector<size_t>{data_size});
    if (data_size > 0) {
      std::iota(
        static_cast<uint8_t*>(payload[0].data),
        static_cast<uint8_t*>(payload[0].data) + data_size,
        static_cast<uint8_t>(function_id));
    }

    return std::make_shared<AxonRequest>(std::move(header), std::move(payload));
  }

  std::filesystem::path test_dir_;
  std::shared_ptr<ucxx::UcxMemoryResourceManager> mr_;
  io_context context_{};
  unifex::inplace_stop_source stop_source_{};
  std::thread context_thread_{};
};

TEST_F(AxonStorageTest, EmplaceAndAccess) {
  AxonStorage storage;
  ASSERT_TRUE(storage.empty());
  ASSERT_EQ(storage.size(), 0);

  auto req1 = CreateRequest(1, 101);
  auto req2 = CreateRequest(1, 102);

  storage.emplace(CreateRequest(1, 100));
  storage.emplace(req1);
  storage.emplace(std::move(req2));

  ASSERT_FALSE(storage.empty());
  ASSERT_EQ(storage.size(), 3);

  // plf::hive does not guarantee insertion order for front() and back()
  // so we check if all elements are present by iterating.
  std::vector<uint32_t> ids;
  for (const auto& req : storage) {
    ids.push_back(cista::to_idx(req->header.function_id));
  }
  std::sort(ids.begin(), ids.end());

  ASSERT_EQ(ids.size(), 3);
  EXPECT_EQ(ids[0], 100);
  EXPECT_EQ(ids[1], 101);
  EXPECT_EQ(ids[2], 102);
}

TEST_F(AxonStorageTest, Erase) {
  AxonStorage storage;
  storage.emplace(CreateRequest(1, 1));
  auto it_to_erase = storage.emplace(CreateRequest(1, 2));
  storage.emplace(CreateRequest(1, 3));

  ASSERT_EQ(storage.size(), 3);

  storage.erase(it_to_erase);

  ASSERT_EQ(storage.size(), 2);
  for (const auto& req : storage) {
    ASSERT_NE(req->header.function_id, rpc::function_id_t{2});
  }
}

TEST_F(AxonStorageTest, SaveAndLoadSuccessfully) {
  // 1. Create storage and add data
  AxonStorage storage_to_save(std::ref(context_));
  std::vector<std::shared_ptr<AxonRequest>> original_requests;
  original_requests.push_back(CreateRequest(1, 10));
  original_requests.push_back(CreateRequest(1, 20));
  original_requests.push_back(CreateRequest(2, 30));
  original_requests.push_back(CreateRequest(2, 40));
  original_requests.push_back(CreateRequest(3, 50));

  for (const auto& req : original_requests) {
    storage_to_save.emplace(req);
  }

  // 2. Save the data
  std::error_code save_ec =
    storage_to_save.save(*mr_, test_dir_, "test_storage");
  ASSERT_FALSE(save_ec) << "Save failed with: " << save_ec.message();

  // Find the created file
  std::filesystem::path saved_file;
  int file_count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(test_dir_)) {
    saved_file = entry.path();
    file_count++;
  }
  ASSERT_EQ(file_count, 1);

  // 3. Load the data into a new storage object
  AxonStorage storage_to_load(std::ref(context_));
  auto load_result = storage_to_load.load(*mr_, saved_file);
  ASSERT_TRUE(load_result.has_value())
    << "Load failed with: " << load_result.error().message();

  // 4. Verify the data
  ASSERT_EQ(storage_to_load.size(), storage_to_save.size());
  ASSERT_EQ(storage_to_load.size(), original_requests.size());

  // Copy loaded requests to a vector for sorting and comparison
  std::vector<std::shared_ptr<AxonRequest>> loaded_requests;
  for (const auto& req : storage_to_load) {
    loaded_requests.push_back(req);
  }

  // Sort both vectors by function_id to ensure order-independent comparison
  auto sort_fn = [](
                   const std::shared_ptr<AxonRequest>& a,
                   const std::shared_ptr<AxonRequest>& b) {
    return a->header.function_id < b->header.function_id;
  };
  std::sort(original_requests.begin(), original_requests.end(), sort_fn);
  std::sort(loaded_requests.begin(), loaded_requests.end(), sort_fn);

  for (size_t i = 0; i < original_requests.size(); ++i) {
    CompareRequests(loaded_requests[i], original_requests[i]);
  }
}

TEST_F(AxonStorageTest, SaveRequestSuccessfully) {
  AxonStorage storage(std::ref(context_));
  storage.emplace(CreateRequest(1, 10));
  auto it_to_save = storage.emplace(CreateRequest(1, 20));
  storage.emplace(CreateRequest(1, 30));

  std::error_code ec = storage.save_request(*mr_, it_to_save, test_dir_);
  ASSERT_FALSE(ec) << "SaveRequest failed: " << ec.message();

  // Find the created file
  std::filesystem::path saved_file;
  int file_count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(test_dir_)) {
    saved_file = entry.path();
    file_count++;
  }
  ASSERT_EQ(file_count, 1);

  // Load and verify
  AxonStorage loaded_storage(std::ref(context_));
  auto load_result = loaded_storage.load(*mr_, saved_file);
  ASSERT_TRUE(load_result.has_value());
  ASSERT_EQ(loaded_storage.size(), 1);
  CompareRequests(loaded_storage.front(), *it_to_save);
}

TEST_F(AxonStorageTest, SaveEmptyStorage) {
  AxonStorage storage(std::ref(context_));
  std::error_code ec = storage.save(*mr_, test_dir_, "empty");
  ASSERT_FALSE(ec);
  ASSERT_TRUE(std::filesystem::is_empty(test_dir_));
}

TEST_F(AxonStorageTest, LoadNonExistentFile) {
  AxonStorage storage(std::ref(context_));
  auto non_existent_file = test_dir_ / "non_existent.avro";
  auto load_result = storage.load(*mr_, non_existent_file);
  ASSERT_FALSE(load_result.has_value());
  EXPECT_EQ(load_result.error(), std::errc::no_such_file_or_directory);
}

}  // namespace storage
}  // namespace axon
}  // namespace eux
