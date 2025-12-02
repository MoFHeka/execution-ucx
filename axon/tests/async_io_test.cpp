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

#include "axon/storage/async_io.hpp"

#include <cista.h>
#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <unifex/inplace_stop_token.hpp>

#include "axon/utils/axon_message.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

namespace eux {
namespace axon {
namespace storage {

namespace data = cista::offset;

using eux::rpc::ParamMeta;
using eux::rpc::ParamType;
using eux::rpc::TensorMeta;

// Provide an operator== for UcxBufferVec for testing purposes.
bool operator==(const ucxx::UcxBufferVec& lhs, const ucxx::UcxBufferVec& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i].size != rhs[i].size) {
      return false;
    }
    if (std::memcmp(lhs[i].data, rhs[i].data, lhs[i].size) != 0) {
      return false;
    }
  }
  return true;
}

class AsyncUnifexAvroIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a temporary directory for test files
    test_dir_ = std::filesystem::temp_directory_path() / "async_io_test";
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
    header.request_id = rpc::request_id_t{function_id};
    header.hlc.TickLocal();
    header.workflow_id = rpc::workflow_id_t{session_id};

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

TEST_F(AsyncUnifexAvroIOTest, SaveAndLoadSuccessfully) {
  AsyncUnifexAvroIO avro_io(context_);

  // 1. Create some data
  std::vector<std::shared_ptr<AxonRequest>> requests_to_save;
  requests_to_save.push_back(CreateRequest(1, 1));
  requests_to_save.push_back(CreateRequest(1, 2));
  requests_to_save.push_back(CreateRequest(1, 3));

  // 2. Save the data
  std::error_code save_ec =
    avro_io.Save(requests_to_save, *mr_, test_dir_, "test");
  ASSERT_FALSE(save_ec) << "Save failed with: " << save_ec.message();

  // Wait for all async save operations to complete before proceeding
  avro_io.sync();

  // Find the created file
  std::filesystem::path saved_file;
  int file_count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(test_dir_)) {
    saved_file = entry.path();
    file_count++;
  }
  ASSERT_EQ(file_count, 1);

  // 3. Load the data back
  auto load_result = avro_io.Load(*mr_, saved_file);
  ASSERT_TRUE(load_result.has_value())
    << "Load failed with: " << load_result.error().message()
    << " for file: " << saved_file;

  const auto& loaded_requests = load_result.value();

  // 4. Verify the data
  ASSERT_EQ(loaded_requests.size(), requests_to_save.size());

  for (size_t i = 0; i < loaded_requests.size(); ++i) {
    const auto& original = requests_to_save[i];
    const auto& loaded = loaded_requests[i];
    EXPECT_EQ(loaded->header.session_id, original->header.session_id);
    EXPECT_EQ(loaded->header.function_id, original->header.function_id);
    EXPECT_EQ(loaded->header.request_id, original->header.request_id);
    EXPECT_EQ(loaded->header.hlc.Raw(), original->header.hlc.Raw());
    EXPECT_EQ(loaded->header.workflow_id, original->header.workflow_id);

    // Verify TensorMeta
    ASSERT_EQ(loaded->header.params.size(), 1);
    const auto& loaded_tm =
      cista::get<TensorMeta>(loaded->header.params[0].value);
    const auto& original_tm =
      cista::get<TensorMeta>(original->header.params[0].value);
    EXPECT_EQ(loaded_tm.shape, original_tm.shape);
    EXPECT_EQ(loaded_tm.dtype.code, original_tm.dtype.code);

    ASSERT_EQ(loaded->payload.index(), original->payload.index());
    switch (loaded->payload.index()) {
      case 0: {  // std::monostate
        break;
      }
      case 1: {  // UcxBuffer
        const auto& lb = std::get<ucxx::UcxBuffer>(loaded->payload);
        const auto& ob = std::get<ucxx::UcxBuffer>(original->payload);
        EXPECT_EQ(lb.size(), ob.size());
        EXPECT_EQ(0, memcmp(lb.data(), ob.data(), lb.size()));
        break;
      }
      case 2: {  // UcxBufferVec
        const auto& lv = std::get<ucxx::UcxBufferVec>(loaded->payload);
        const auto& ov = std::get<ucxx::UcxBufferVec>(original->payload);
        ASSERT_EQ(lv.size(), ov.size());
        for (size_t j = 0; j < lv.size(); ++j) {
          EXPECT_EQ(lv[j].size, ov[j].size);
          EXPECT_EQ(0, memcmp(lv[j].data, ov[j].data, lv[j].size));
        }
        break;
      }
      default:
        FAIL() << "Unknown payload variant index";
    }
  }
}

TEST_F(AsyncUnifexAvroIOTest, SaveEmptyVector) {
  AsyncUnifexAvroIO avro_io(context_);

  std::vector<std::shared_ptr<AxonRequest>> empty_requests;
  std::error_code ec = avro_io.Save(empty_requests, *mr_, test_dir_, "test");

  ASSERT_FALSE(ec) << "Saving an empty vector should succeed.";

  // Check that no file was created
  ASSERT_TRUE(std::filesystem::is_empty(test_dir_));
}

TEST_F(AsyncUnifexAvroIOTest, LoadNonExistentFile) {
  AsyncUnifexAvroIO avro_io(context_);

  auto non_existent_file = test_dir_ / "non_existent.avro";
  auto load_result = avro_io.Load(*mr_, non_existent_file);

  ASSERT_FALSE(load_result.has_value());
  EXPECT_EQ(load_result.error(), std::errc::no_such_file_or_directory);
}

// Add a test for saving multiple batches to the same directory
TEST_F(AsyncUnifexAvroIOTest, SaveMultipleBatches) {
  AsyncUnifexAvroIO avro_io(context_);

  // Batch 1
  std::vector<std::shared_ptr<AxonRequest>> requests1;
  requests1.push_back(CreateRequest(1, 1));
  std::error_code save_ec1 = avro_io.Save(requests1, *mr_, test_dir_, "test");
  ASSERT_FALSE(save_ec1);

  // sleep to get a different timestamp for the filename
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  // Batch 2
  std::vector<std::shared_ptr<AxonRequest>> requests2;
  requests2.push_back(CreateRequest(1, 2));
  std::error_code save_ec2 = avro_io.Save(requests2, *mr_, test_dir_, "test");
  ASSERT_FALSE(save_ec2);

  int file_count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(test_dir_)) {
    (void)entry;  // Suppress unused variable warning
    file_count++;
  }
  ASSERT_EQ(file_count, 2);
}

}  // namespace storage
}  // namespace axon
}  // namespace eux
