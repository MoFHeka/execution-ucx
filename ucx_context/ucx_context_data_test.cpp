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

#include <cstring>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "ucx_context/ucx_context_data.hpp"
#include "ucx_context/ucx_memory_resource.hpp"

namespace eux {
namespace ucxx {

class UcxBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mr_ = std::make_unique<DefaultUcxMemoryResourceManager>();
  }

  void TearDown() override { mr_.reset(); }

  std::unique_ptr<UcxMemoryResourceManager> mr_;
};

TEST_F(UcxBufferTest, ZeroCopyConversion) {
  const size_t buffer_size = 100;
  UcxBuffer buffer(*mr_, ucx_memory_type::HOST, buffer_size);

  // Fill the buffer with some data
  std::vector<uint8_t> source_data(buffer_size);
  std::iota(source_data.begin(), source_data.end(), 0);
  std::memcpy(buffer.data(), source_data.data(), buffer_size);

  // Store the original pointer before moving the buffer
  void* original_buffer_ptr = buffer.data();

  const std::vector<size_t> sizes = {10, 20, 30, 40};
  auto buffer_vec = std::move(buffer).to_buffer_vec(sizes);

  EXPECT_EQ(buffer_vec.size(), sizes.size());

  size_t offset = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    EXPECT_EQ(buffer_vec[i].size, sizes[i]);
    // Verify that the data pointer is just an offset into the original buffer
    EXPECT_EQ(
      buffer_vec[i].data, static_cast<uint8_t*>(original_buffer_ptr) + offset);
    // Verify the content of the sliced buffer
    EXPECT_EQ(
      0, std::memcmp(
           buffer_vec[i].data,
           static_cast<uint8_t*>(original_buffer_ptr) + offset, sizes[i]));
    offset += sizes[i];
  }

  // The original buffer should be invalidated
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size(), 0);
}

TEST_F(UcxBufferTest, ThrowsOnSizeMismatch) {
  const size_t buffer_size = 100;
  UcxBuffer buffer(*mr_, ucx_memory_type::HOST, buffer_size);
  const std::vector<size_t> sizes = {50, 51};  // Total > 100

  EXPECT_THROW(std::move(buffer).to_buffer_vec(sizes), std::length_error);
}

}  // namespace ucxx
}  // namespace eux
