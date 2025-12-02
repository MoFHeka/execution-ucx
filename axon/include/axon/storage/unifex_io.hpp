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

#pragma once

#ifndef AXON_CORE_STORAGE_UNIFEX_IO_HPP_
#define AXON_CORE_STORAGE_UNIFEX_IO_HPP_

#include <array>
#include <memory>
#include <span>
#include <vector>

#include <unifex/spawn_future.hpp>
#include <unifex/v2/async_scope.hpp>
#if !UNIFEX_NO_LIBURING
#include <unifex/linux/io_uring_context.hpp>
#else
// #include <unifex/linux/io_epoll_context.hpp>
#error "UNIFEX_NO_LIBURING is true. Compilation is terminated."
#endif

namespace eux {
namespace axon {
namespace storage {

#if !UNIFEX_NO_LIBURING
using io_context = unifex::linuxos::io_uring_context;
#else
// using io_context = unifex::linuxos::io_epoll_context;
#error "UNIFEX_NO_LIBURING is true. Compilation is terminated."
#endif

namespace _ {
using async_scope = unifex::v2::async_scope;
using future = unifex::v2::future<async_scope, int64_t>;
}  // namespace _

/**
 * @brief An output stream that uses libunifex for double-buffered asynchronous
 * writes.
 *
 * This class provides a high-performance, standalone streaming output
 * mechanism. It uses two internal buffers to overlap I/O operations with client
 * data production (e.g., serialization). While one buffer is being filled by
 * the client, the other can be written to the underlying file asynchronously.
 *
 * Data Flow Diagram:
 *
 *   Client Code (e.g., Serializer)
 *        |
 *        | 1. Calls Next() to get a writable buffer segment.
 *        V
 * +----------------+      +----------------+
 * |    Buffer 0    |      |    Buffer 1    |  (inactive, write may be pending)
 * |  (Active)      <------+
 * +----------------+      +----------------+
 *        ^
 *        | 2. Client writes data into the buffer.
 *        | 3. Calls Backup(bytes_written) to commit the write.
 *        |
 *        | 4. When Buffer 0 is full, the next call to Next() triggers:
 *        |    a. Submit Buffer 0 for an asynchronous write to the file.
 *        |    b. Wait for Buffer 1's previous write (if any) to complete.
 *        |    c. Switch the active buffer to Buffer 1.
 *        V
 * +----------------+      +----------------+
 * |    Buffer 0    |----->|      File      |
 * | (Writing...)   |      +----------------+
 * +----------------+      ^
 *                       |
 * +----------------+    |
 * |    Buffer 1    |<---| (Becomes available after its write completes)
 * | (Available)    |
 * +----------------+
 *
 */
class UnifexOutputStream {
 public:
  UnifexOutputStream(
    _::async_scope& async_scope, io_context& context,
    std::filesystem::path file_path, size_t buffer_size = 8192);
  ~UnifexOutputStream();

  /**
   * @brief Obtains the next chunk of the buffer for writing.
   *
   * The caller can write up to `*len` bytes into the buffer pointed to by
   * `*data`.
   *
   * @param data Pointer to a uint8_t* that will be set to the beginning of the
   *             available buffer space.
   * @param len  Pointer to a size_t that will be set to the number of available
   *             bytes in the buffer.
   * @return Always returns true in this implementation.
   */
  bool Next(uint8_t** data, size_t* len);

  /**
   * @brief Commits bytes written to the buffer obtained from Next().
   *
   * This function advances the internal write pointer. The name `backup` is a
   * legacy from the Avro API and might be counter-intuitive.
   *
   * For example, after a call to `Next()` returns a buffer of size `L`, if the
   * client writes N bytes into it, it should call `Backup(L - N)` to signal
   * that `N` bytes have been consumed.
   *
   * @param len The number of unused bytes in the buffer.
   */
  void Backup(size_t len);

  /**
   * @brief Returns the total number of bytes written to the stream so far.
   * This includes bytes that may still be in the buffer and not yet flushed to
   * the file.
   */
  uint64_t ByteCount() const;

  /**
   * @brief Flushes any buffered data to the underlying file.
   *
   * This function submits the current buffer for writing (if it contains data)
   * and blocks until all pending asynchronous write operations for both buffers
   * have completed.
   */
  void Flush();

 private:
  struct Buffer {
    std::vector<std::byte> data_;
    std::optional<_::future> write_future_;
  };

  void submit_buffer(int buffer_idx, size_t size);
  void wait_for_buffer(int buffer_idx);

  _::async_scope& async_scope_;
  io_context& context_;
  const std::filesystem::path file_path_;
  std::optional<io_context::async_write_only_file> file_;
  uint64_t byte_count_{0};
  uint64_t file_offset_{0};
  int active_buffer_idx_{0};
  std::array<Buffer, 2> buffers_;
  std::span<std::byte> current_span_;
};

/**
 * @brief An input stream that uses libunifex for double-buffered asynchronous
 * reads (read-ahead).
 *
 * This class prefetches data from the underlying file into one buffer while the
 * client is consuming data from another. This overlapping of I/O and
 * computation (e.g., deserialization) can significantly improve read
 * throughput.
 *
 * Data Flow Diagram:
 *
 *   Client Code (e.g., Deserializer)
 *        ^
 *        | 1. Calls Next() to get a readable buffer segment.
 *        |
 * +----------------+      +----------------+
 * |    Buffer 0    |      |    Buffer 1    |  (pre-fetching data from file)
 * |   (Active)     +----->|      File      |
 * +----------------+      +----------------+
 *        |
 *        | 2. Client reads and processes data from Buffer 0.
 *        |
 *        | 3. When Buffer 0 is fully consumed, the next call to Next()
 * triggers: |    a. Start a new asynchronous pre-fetch operation for Buffer 0.
 *        |    b. Wait for Buffer 1's pre-fetch operation to complete.
 *        |    c. Switch the active buffer to Buffer 1.
 *        V
 * +----------------+      +----------------+
 * |    Buffer 0    |<-----|      File      |
 * | (Pre-fetching) |      +----------------+
 * +----------------+      ^
 *                       |
 * +----------------+    |
 * |    Buffer 1    |----| (Data becomes ready for client consumption)
 * |    (Ready)     |
 * +----------------+
 */
class UnifexInputStream {
 public:
  UnifexInputStream(
    _::async_scope& async_scope, io_context& context,
    std::filesystem::path file_path, size_t buffer_size = 8192);
  ~UnifexInputStream();

  /**
   * @brief Obtains the next chunk of data from the stream.
   *
   * The caller can read up to `*len` bytes from the buffer pointed to by
   * `*data`.
   *
   * @param data Pointer to a const uint8_t* that will be set to the beginning
   *             of the available data buffer.
   * @param len  Pointer to a size_t that will be set to the number of available
   *             bytes in the buffer.
   * @return Returns true if data is available, false if the end of the stream
   *         is reached.
   */
  bool Next(const uint8_t** data, size_t* len);

  /**
   * @brief "Un-reads" a number of bytes, moving the read cursor backward.
   *
   * The specified number of bytes will be made available again on the next
   * call to Next().
   *
   * @param len The number of bytes to back up. This must be less than or equal
   *            to the number of bytes returned by the last call to Next().
   */
  void Backup(size_t len);

  /**
   * @brief Skips a specified number of bytes in the input stream.
   * @param len The number of bytes to skip.
   */
  void Skip(size_t len);

  /**
   * @brief Returns the total number of bytes read from the stream so far.
   */
  uint64_t ByteCount() const;

 private:
  struct Buffer {
    std::vector<std::byte> data_;
    std::optional<_::future> read_future_;
    size_t data_len_{0};
    size_t read_pos_{0};
  };

  void prefetch_buffer(int buffer_idx);
  void wait_for_buffer_ready(int buffer_idx);

  _::async_scope& async_scope_;
  io_context& context_;
  const std::filesystem::path file_path_;
  std::optional<io_context::async_read_only_file> file_;
  uint64_t file_offset_{0};
  uint64_t byte_count_{0};
  int active_buffer_idx_{0};
  std::array<Buffer, 2> buffers_;
};

}  // namespace storage
}  // namespace axon
}  // namespace eux

#endif  // AXON_CORE_STORAGE_UNIFEX_IO_HPP_
