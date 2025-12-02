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

#ifndef AXON_CORE_STORAGE_AVRO_UNIFEX_IO_HPP_
#define AXON_CORE_STORAGE_AVRO_UNIFEX_IO_HPP_

#include <filesystem>
#include <memory>

#include <avro/Stream.hh>

// To define the types used in the constructor signatures, without pulling in
// all of unifex_io.hpp
#include <unifex/v2/async_scope.hpp>
#if !UNIFEX_NO_LIBURING
#include <unifex/linux/io_uring_context.hpp>
#else
#error "UNIFEX_NO_LIBURING is true. Compilation is terminated."
#endif

namespace eux {
namespace axon {
namespace storage {

// Recreate aliases needed for the public API
#if !UNIFEX_NO_LIBURING
using io_context = unifex::linuxos::io_uring_context;
#else
#error "UNIFEX_NO_LIBURING is true. Compilation is terminated."
#endif

namespace _ {
using async_scope = unifex::v2::async_scope;
}  // namespace _

// Forward declarations for the Pimpl implementation
class UnifexOutputStream;
class UnifexInputStream;

class UnifexAvroOutputStream : public avro::OutputStream {
 public:
  UnifexAvroOutputStream(
    _::async_scope& async_scope, io_context& context,
    std::filesystem::path file_path, size_t buffer_size = 8192);

  ~UnifexAvroOutputStream() override;

  bool next(uint8_t** data, size_t* len) override;

  void backup(size_t len) override;

  uint64_t byteCount() const override;

  void flush() override;

 private:
  std::unique_ptr<UnifexOutputStream> impl_;
};

class UnifexAvroInputStream : public avro::InputStream {
 public:
  UnifexAvroInputStream(
    _::async_scope& async_scope, io_context& context,
    std::filesystem::path file_path, size_t buffer_size = 8192);

  ~UnifexAvroInputStream() override;

  bool next(const uint8_t** data, size_t* len) override;

  void backup(size_t len) override;

  void skip(size_t len) override;

  uint64_t byteCount() const override;

 private:
  std::unique_ptr<UnifexInputStream> impl_;
};

}  // namespace storage
}  // namespace axon
}  // namespace eux

#endif  // AXON_CORE_STORAGE_AVRO_UNIFEX_IO_HPP_
