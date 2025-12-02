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

#include "axon/storage/unifex_io.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <unifex/get_stop_token.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/v2/async_scope.hpp>

namespace eux {
namespace axon {
namespace storage {

class UnifexIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a unique temporary file path for each test
    test_file_path_ =
      std::filesystem::temp_directory_path() / "unifex_io_test.bin";
    // Clean up any old file before the test
    std::filesystem::remove(test_file_path_);
    // Start the io_context on a background thread.
    context_thread_ =
      std::thread([this] { context_.run(stopSource_.get_token()); });
  }

  void TearDown() override {
    // First, wait for all tasks to complete. This ensures that any file
    // operations are finished before we try to delete the file.
    unifex::sync_wait(async_scope_.join());

    // Second, request stop to the context.
    stopSource_.request_stop();

    // Join the context thread.
    if (context_thread_.joinable()) {
      context_thread_.join();
    }

    // Clean up the file after the test
    std::filesystem::remove(test_file_path_);
  }

  // Helper function to write data to a file using standard library
  void write_file_sync(const std::vector<uint8_t>& data) {
    std::ofstream file(test_file_path_, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
  }

  // Helper function to read data from a file using standard library
  std::vector<uint8_t> read_file_sync() {
    std::ifstream file(test_file_path_, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), {}};
  }

  std::filesystem::path test_file_path_;
  unifex::inplace_stop_source stopSource_;
  io_context context_{};
  unifex::v2::async_scope async_scope_{};
  std::thread context_thread_;
};

TEST_F(UnifexIOTest, OutputStream_WriteAndFlush) {
  const size_t buffer_size = 128;
  std::filesystem::path path = test_file_path_;
  UnifexOutputStream out_stream(async_scope_, context_, path, buffer_size);

  std::vector<uint8_t> test_data(100);
  std::iota(test_data.begin(), test_data.end(), 0);

  uint8_t* buffer_ptr;
  size_t len;
  ASSERT_TRUE(out_stream.Next(&buffer_ptr, &len));

  ASSERT_GE(len, test_data.size());
  std::memcpy(buffer_ptr, test_data.data(), test_data.size());
  out_stream.Backup(len - test_data.size());

  EXPECT_EQ(out_stream.ByteCount(), test_data.size());

  out_stream.Flush();

  auto file_content = read_file_sync();
  EXPECT_EQ(file_content, test_data);
}

TEST_F(UnifexIOTest, OutputStream_MultiBufferWrite) {
  const size_t buffer_size = 128;
  std::filesystem::path path = test_file_path_;
  UnifexOutputStream out_stream(async_scope_, context_, path, buffer_size);

  std::vector<uint8_t> test_data(
    buffer_size * 2 + 50);  // More than two buffers
  std::iota(test_data.begin(), test_data.end(), 0);

  size_t bytes_written = 0;
  while (bytes_written < test_data.size()) {
    uint8_t* buffer_ptr;
    size_t len;
    ASSERT_TRUE(out_stream.Next(&buffer_ptr, &len));

    size_t to_write = std::min(len, test_data.size() - bytes_written);
    std::memcpy(buffer_ptr, test_data.data() + bytes_written, to_write);
    out_stream.Backup(len - to_write);
    bytes_written += to_write;
  }

  EXPECT_EQ(out_stream.ByteCount(), test_data.size());

  out_stream.Flush();

  auto file_content = read_file_sync();
  EXPECT_EQ(file_content, test_data);
}

TEST_F(UnifexIOTest, OutputStream_DestructorFlushes) {
  std::vector<uint8_t> test_data(100);
  std::iota(test_data.begin(), test_data.end(), 0);

  {
    const size_t buffer_size = 128;
    std::filesystem::path path = test_file_path_;
    UnifexOutputStream out_stream(async_scope_, context_, path, buffer_size);

    uint8_t* buffer_ptr;
    size_t len;
    ASSERT_TRUE(out_stream.Next(&buffer_ptr, &len));
    ASSERT_GE(len, test_data.size());
    std::memcpy(buffer_ptr, test_data.data(), test_data.size());
    out_stream.Backup(len - test_data.size());
    // out_stream goes out of scope here, destructor should flush.
  }

  auto file_content = read_file_sync();
  EXPECT_EQ(file_content, test_data);
}

TEST_F(UnifexIOTest, InputStream_ReadFullFile) {
  std::vector<uint8_t> test_data(100);
  std::iota(test_data.begin(), test_data.end(), 0);
  write_file_sync(test_data);

  const size_t buffer_size = 128;
  std::filesystem::path path = test_file_path_;
  UnifexInputStream in_stream(async_scope_, context_, path, buffer_size);

  std::vector<uint8_t> read_data;
  const uint8_t* buffer_ptr;
  size_t len;
  while (in_stream.Next(&buffer_ptr, &len) && len > 0) {
    read_data.insert(read_data.end(), buffer_ptr, buffer_ptr + len);
  }

  EXPECT_EQ(in_stream.ByteCount(), test_data.size());
  EXPECT_EQ(read_data, test_data);
}

TEST_F(UnifexIOTest, InputStream_MultiBufferRead) {
  const size_t buffer_size = 128;
  std::vector<uint8_t> test_data(
    buffer_size * 2 + 50);  // More than two buffers
  std::iota(test_data.begin(), test_data.end(), 0);
  write_file_sync(test_data);

  std::filesystem::path path = test_file_path_;
  UnifexInputStream in_stream(async_scope_, context_, path, buffer_size);

  std::vector<uint8_t> read_data;
  const uint8_t* buffer_ptr;
  size_t len;
  while (in_stream.Next(&buffer_ptr, &len) && len > 0) {
    read_data.insert(read_data.end(), buffer_ptr, buffer_ptr + len);
  }

  EXPECT_EQ(in_stream.ByteCount(), test_data.size());
  EXPECT_EQ(read_data, test_data);
}

TEST_F(UnifexIOTest, InputStream_BackupAndSkip) {
  std::vector<uint8_t> test_data(200);
  std::iota(test_data.begin(), test_data.end(), 0);
  write_file_sync(test_data);

  const size_t buffer_size = 128;
  std::filesystem::path path = test_file_path_;
  UnifexInputStream in_stream(async_scope_, context_, path, buffer_size);

  const uint8_t* buffer_ptr;
  size_t len;

  // 1. Read first part (50 bytes)
  ASSERT_TRUE(in_stream.Next(&buffer_ptr, &len));
  ASSERT_GE(len, 50);
  // The next() consumes the whole buffer, so we backup the part we didn't
  // "read"
  in_stream.Backup(len - 50);
  EXPECT_EQ(in_stream.ByteCount(), 50);

  // 2. Backup 20 bytes
  in_stream.Backup(20);
  EXPECT_EQ(in_stream.ByteCount(), 30);

  // 3. Read again, should get the backed-up data
  ASSERT_TRUE(in_stream.Next(&buffer_ptr, &len));
  ASSERT_GE(len, 20);
  std::vector<uint8_t> part2(buffer_ptr, buffer_ptr + 20);
  std::vector<uint8_t> expected_part2(
    test_data.begin() + 30, test_data.begin() + 50);
  EXPECT_EQ(part2, expected_part2);
  in_stream.Backup(len - 20);  // Consume the 20 bytes we just read
  EXPECT_EQ(in_stream.ByteCount(), 50);

  // 4. Skip 60 bytes
  in_stream.Skip(60);
  EXPECT_EQ(in_stream.ByteCount(), 110);

  // 5. Read the rest and verify
  std::vector<uint8_t> final_part;
  while (in_stream.Next(&buffer_ptr, &len) && len > 0) {
    final_part.insert(final_part.end(), buffer_ptr, buffer_ptr + len);
  }
  std::vector<uint8_t> expected_final_part(
    test_data.begin() + 110, test_data.end());
  EXPECT_EQ(final_part, expected_final_part);
  EXPECT_EQ(in_stream.ByteCount(), test_data.size());
}

TEST_F(UnifexIOTest, WriteThenRead) {
  const size_t buffer_size = 256;
  const size_t data_size = buffer_size * 5 + 123;

  std::vector<uint8_t> write_data(data_size);
  for (size_t i = 0; i < data_size; ++i) {
    write_data[i] = i % 256;
  }

  // Write phase
  {
    std::filesystem::path path = test_file_path_;
    UnifexOutputStream out_stream(async_scope_, context_, path, buffer_size);

    size_t bytes_written = 0;
    while (bytes_written < write_data.size()) {
      uint8_t* buffer_ptr;
      size_t len;
      ASSERT_TRUE(out_stream.Next(&buffer_ptr, &len));

      size_t to_write = std::min(len, write_data.size() - bytes_written);
      std::memcpy(buffer_ptr, write_data.data() + bytes_written, to_write);
      out_stream.Backup(len - to_write);
      bytes_written += to_write;
    }
    // flush() is called in destructor
  }

  // Read phase
  {
    std::filesystem::path path = test_file_path_;
    UnifexInputStream in_stream(async_scope_, context_, path, buffer_size);

    std::vector<uint8_t> read_data;
    const uint8_t* buffer_ptr;
    size_t len;
    while (in_stream.Next(&buffer_ptr, &len) && len > 0) {
      read_data.insert(read_data.end(), buffer_ptr, buffer_ptr + len);
    }

    EXPECT_EQ(in_stream.ByteCount(), write_data.size());
    EXPECT_EQ(read_data.size(), write_data.size());
    EXPECT_EQ(read_data, write_data);
  }
}

}  // namespace storage
}  // namespace axon
}  // namespace eux
