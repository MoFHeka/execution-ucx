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

#include <algorithm>
#include <iostream>
#include <span>
#include <string>
#include <utility>

#include <unifex/let_error.hpp>
#include <unifex/let_value_with.hpp>
#include <unifex/spawn_future.hpp>
#include <unifex/sync_wait.hpp>

namespace eux {
namespace axon {
namespace storage {

// UnifexOutputStream implementation
UnifexOutputStream::UnifexOutputStream(
  _::async_scope& async_scope, io_context& context,
  std::filesystem::path file_path, size_t buffer_size)
  : async_scope_(async_scope), context_(context), file_path_(file_path) {
  buffers_[0].data_.resize(buffer_size);
  buffers_[1].data_.resize(buffer_size);
  current_span_ = std::span<std::byte>(buffers_[0].data_);
  auto ret = unifex::open_file_write_only(context_.get_scheduler(), file_path);
  file_.emplace(std::move(ret));
}

UnifexOutputStream::~UnifexOutputStream() {
  try {
    Flush();
  } catch (const std::exception& e) {
    // TODO(He Jia): Handle error case
    std::cerr << "Error in ~UnifexOutputStream: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown error in ~UnifexOutputStream" << std::endl;
  }
}

bool UnifexOutputStream::Next(uint8_t** data, size_t* len) {
  if (current_span_.empty()) {
    int next_buffer_idx = 1 - active_buffer_idx_;
    wait_for_buffer(next_buffer_idx);

    size_t write_size = buffers_[active_buffer_idx_].data_.size();
    submit_buffer(active_buffer_idx_, write_size);
    byte_count_ += write_size;

    active_buffer_idx_ = next_buffer_idx;
    current_span_ = std::span<std::byte>(buffers_[active_buffer_idx_].data_);
  }
  *data = reinterpret_cast<uint8_t*>(current_span_.data());
  *len = current_span_.size();
  return true;
}

void UnifexOutputStream::Backup(size_t len) {
  // 'len' is the number of unused bytes remaining in the buffer provided by
  // the last call to Next().
  // We need to advance our span by the number of bytes that were actually
  // written.
  size_t bytes_written = current_span_.size() - len;
  current_span_ = current_span_.subspan(bytes_written);
}

uint64_t UnifexOutputStream::ByteCount() const {
  return byte_count_
         + (buffers_[active_buffer_idx_].data_.size() - current_span_.size());
}

void UnifexOutputStream::Flush() {
  size_t remaining_size =
    buffers_[active_buffer_idx_].data_.size() - current_span_.size();
  if (remaining_size > 0) {
    // To maintain write order, wait for the other buffer to finish before
    // submitting the current one.
    wait_for_buffer(1 - active_buffer_idx_);
    submit_buffer(active_buffer_idx_, remaining_size);
  }

  // Now, wait for all pending writes to complete.
  wait_for_buffer(0);
  wait_for_buffer(1);
  byte_count_ += remaining_size;
  current_span_ = std::span<std::byte>(buffers_[active_buffer_idx_].data_);
}

void UnifexOutputStream::submit_buffer(int buffer_idx, size_t size) {
  if (size == 0) return;
  auto& buffer = buffers_[buffer_idx];
  auto write_task =
    unifex::let_value_with(
      [&buffer, file_offset_ = file_offset_]() {
        return std::tie(buffer, file_offset_);
      },
      [&ifile = *file_, size](auto& state) {
        auto& [buffer, offset] = state;
        return unifex::async_write_some_at(
          ifile, offset,
          unifex::as_bytes(unifex::span{buffer.data_.data(), size}));
      })
    | unifex::let_error([](auto ec) {
        std::exception_ptr p;
        if constexpr (std::is_same_v<
                        std::decay_t<decltype(ec)>, std::error_code>) {
          p = std::make_exception_ptr(std::system_error{ec});
        } else {
          p = ec;
        }
        return unifex::just_error(std::move(p));
      });
  buffer.write_future_.emplace(
    unifex::spawn_future(std::move(write_task), async_scope_));
  file_offset_ += size;
}

void UnifexOutputStream::wait_for_buffer(int buffer_idx) {
  auto& buffer = buffers_[buffer_idx];
  if (buffer.write_future_.has_value()) {
    unifex::sync_wait(std::move(*buffer.write_future_));
    buffer.write_future_.reset();
  }
}

// UnifexInputStream implementation
UnifexInputStream::UnifexInputStream(
  _::async_scope& async_scope, io_context& context,
  std::filesystem::path file_path, size_t buffer_size)
  : async_scope_(async_scope), context_(context), file_path_(file_path) {
  buffers_[0].data_.resize(buffer_size);
  buffers_[1].data_.resize(buffer_size);
  auto ret = unifex::open_file_read_only(context_.get_scheduler(), file_path);
  file_.emplace(std::move(ret));
  prefetch_buffer(0);
  prefetch_buffer(1);
}

UnifexInputStream::~UnifexInputStream() {}

bool UnifexInputStream::Next(const uint8_t** data, size_t* len) {
  wait_for_buffer_ready(active_buffer_idx_);

  auto& active_buffer = buffers_[active_buffer_idx_];
  if (active_buffer.read_pos_ >= active_buffer.data_len_) {
    // EOF
    if (active_buffer.data_len_ < active_buffer.data_.size()) {
      *len = 0;
      return false;
    }

    // Switch to next buffer
    active_buffer_idx_ = 1 - active_buffer_idx_;
    prefetch_buffer(
      1 - active_buffer_idx_);  // Prefetch the one we just finished
    wait_for_buffer_ready(active_buffer_idx_);

    auto& new_active_buffer = buffers_[active_buffer_idx_];
    if (new_active_buffer.read_pos_ >= new_active_buffer.data_len_) {
      *len = 0;
      return false;
    }
  }

  auto& final_active_buffer = buffers_[active_buffer_idx_];
  *data = reinterpret_cast<const uint8_t*>(
    final_active_buffer.data_.data() + final_active_buffer.read_pos_);
  *len = final_active_buffer.data_len_ - final_active_buffer.read_pos_;
  final_active_buffer.read_pos_ += *len;
  byte_count_ += *len;

  return true;
}

void UnifexInputStream::Backup(size_t len) {
  buffers_[active_buffer_idx_].read_pos_ -= len;
  byte_count_ -= len;
}

void UnifexInputStream::Skip(size_t len) {
  // This is a naive implementation. A better one might involve seeking.
  const uint8_t* data;
  size_t chunk_len;
  while (len > 0 && Next(&data, &chunk_len)) {
    size_t to_skip = std::min(len, chunk_len);
    Backup(chunk_len - to_skip);
    len -= to_skip;
  }
}

uint64_t UnifexInputStream::ByteCount() const { return byte_count_; }

void UnifexInputStream::prefetch_buffer(int buffer_idx) {
  auto& buffer = buffers_[buffer_idx];
  auto read_task =
    unifex::let_value_with(
      [&buffer, file_offset_ = file_offset_]() {
        return std::tie(buffer, file_offset_);
      },
      [&ifile = *file_](auto& state) {
        auto& [buffer, offset] = state;
        return unifex::async_read_some_at(
          ifile, offset,
          unifex::as_writable_bytes(
            unifex::span{buffer.data_.data(), buffer.data_.size()}));
      })
    | unifex::let_error([](auto ec) {
        std::exception_ptr p;
        if constexpr (std::is_same_v<
                        std::decay_t<decltype(ec)>, std::error_code>) {
          p = std::make_exception_ptr(std::system_error{ec});
        } else {
          p = ec;
        }
        return unifex::just_error(std::move(p));
      });
  buffer.read_future_.emplace(
    unifex::spawn_future(std::move(read_task), async_scope_));
  file_offset_ += buffer.data_.size();
}

void UnifexInputStream::wait_for_buffer_ready(int buffer_idx) {
  auto& buffer = buffers_[buffer_idx];
  if (buffer.read_future_.has_value()) {
    if (auto result = unifex::sync_wait(std::move(*buffer.read_future_))) {
      buffer.data_len_ = *result;
    } else {
      buffer.data_len_ = 0;  // TODO(He Jia): Handle error case
    }
    buffer.read_pos_ = 0;
    buffer.read_future_.reset();
  }
}

}  // namespace storage
}  // namespace axon
}  // namespace eux
