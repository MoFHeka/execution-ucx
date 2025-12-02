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

#ifndef RPC_CORE_UTILS_CISTA_SERIALIZE_HELPER_HPP_
#define RPC_CORE_UTILS_CISTA_SERIALIZE_HELPER_HPP_

#include <cista.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace eux {
namespace rpc {
namespace utils {

#if WITH_CISTA_VERSION && WITH_CISTA_INTEGRITY
constexpr auto const SerializerMode =  // opt. versioning + check sum
  cista::mode::WITH_VERSION | cista::mode::WITH_INTEGRITY;
#elif WITH_CISTA_INTEGRITY
constexpr auto const SerializerMode = cista::mode::WITH_INTEGRITY;
#elif WITH_CISTA_VERSION
constexpr auto const SerializerMode = cista::mode::WITH_VERSION;
#else
constexpr auto const SerializerMode = cista::mode::NONE;
#endif

// A Cista serialization target that only calculates the required size.
struct CistaSizeCalculator {
  // Called by cista::serialize to write a block of data.
  cista::offset_t write(
    void const* /*ptr*/, std::size_t const num_bytes,
    std::size_t alignment = 0) {
    // 1. Add padding for alignment.
    if (alignment > 1) {
      auto const misalignment = pos_ % alignment;
      if (misalignment != 0) {
        pos_ += (alignment - misalignment);
      }
    }

    // 2. "Write" the data by advancing the position counter.
    auto const start = pos_;
    pos_ += num_bytes;
    total_size_ = std::max(total_size_, static_cast<std::size_t>(pos_));
    return start;
  }

  // Called for back-patching pointer offsets. Does not increase total size,
  // but we track the max write position to be safe.
  template <typename T>
  void write(std::size_t const pos, T const& /*val*/) {
    total_size_ = std::max(total_size_, pos + cista::serialized_size<T>());
  }

  // Dummy checksum to satisfy the target interface.
  std::uint64_t checksum(cista::offset_t const = 0) const { return 0; }

  // Returns the final calculated size.
  std::size_t size() const { return total_size_; }

  cista::offset_t pos_{0};
  std::size_t total_size_{0};
};

// A Cista serialization target that writes to a fixed-size raw buffer.
struct FixedBufferWriter {
  FixedBufferWriter(std::byte* buf, std::size_t size)
    : base_{buf}, size_{size} {}

  cista::offset_t write(
    void const* ptr, std::size_t const num_bytes, std::size_t alignment = 0) {
    // 1. Add padding for alignment.
    if (alignment > 1) {
      auto const misalignment = pos_ % alignment;
      if (misalignment != 0) {
        pos_ += (alignment - misalignment);
      }
    }

    // 2. Write the data.
    auto const start = pos_;
    pos_ += num_bytes;
    cista::verify(
      static_cast<std::size_t>(pos_) <= size_, "fixed_buffer_writer overflow");
    std::memcpy(base_ + start, ptr, num_bytes);
    return start;
  }

  // For back-patching pointer offsets.
  template <typename T>
  void write(std::size_t const pos, T const& val) {
    cista::verify(
      pos + cista::serialized_size<T>() <= size_,
      "fixed_buffer_writer patch overflow");
    std::memcpy(base_ + pos, &val, cista::serialized_size<T>());
  }

  // Dummy checksum.
  std::uint64_t checksum(cista::offset_t const from) const {
    return cista::hash(std::string_view{
      reinterpret_cast<char const*>(base_ + from), size_ - from});
  }

  std::size_t size() const { return size_; }

  // Returns the actual number of bytes written.
  std::size_t written_size() const { return static_cast<std::size_t>(pos_); }

  std::byte* base_{nullptr};
  std::size_t size_{0};
  cista::offset_t pos_{0};
};

template <typename T>
size_t GetSerializedSize(const T& value) {
  CistaSizeCalculator calculator;
  cista::serialize<SerializerMode>(calculator, value);
  return calculator.size();
}

}  // namespace utils
}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_UTILS_CISTA_SERIALIZE_HELPER_HPP_
