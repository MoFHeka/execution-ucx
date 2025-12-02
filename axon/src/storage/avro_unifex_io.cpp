/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUTHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <utility>

#include "axon/storage/avro_unifex_io.hpp"
#include "axon/storage/unifex_io.hpp"

namespace eux {
namespace axon {
namespace storage {

// ~~~~~~~~~~~~~~~~~ UnifexAvroOutputStream Implementation ~~~~~~~~~~~~~~~~~

UnifexAvroOutputStream::UnifexAvroOutputStream(
  _::async_scope& async_scope, io_context& context,
  std::filesystem::path file_path, size_t buffer_size)
  : impl_(std::make_unique<UnifexOutputStream>(
    async_scope, context, std::move(file_path), buffer_size)) {}

UnifexAvroOutputStream::~UnifexAvroOutputStream() {
  if (impl_) {
    impl_->Flush();
  }
}

bool UnifexAvroOutputStream::next(uint8_t** data, size_t* len) {
  return impl_->Next(data, len);
}

void UnifexAvroOutputStream::backup(size_t len) { impl_->Backup(len); }

uint64_t UnifexAvroOutputStream::byteCount() const {
  return impl_->ByteCount();
}

void UnifexAvroOutputStream::flush() { impl_->Flush(); }

// ~~~~~~~~~~~~~~~~~ UnifexAvroInputStream Implementation ~~~~~~~~~~~~~~~~~

UnifexAvroInputStream::UnifexAvroInputStream(
  _::async_scope& async_scope, io_context& context,
  std::filesystem::path file_path, size_t buffer_size)
  : impl_(std::make_unique<UnifexInputStream>(
    async_scope, context, std::move(file_path), buffer_size)) {}

UnifexAvroInputStream::~UnifexAvroInputStream() = default;

bool UnifexAvroInputStream::next(const uint8_t** data, size_t* len) {
  return impl_->Next(data, len);
}

void UnifexAvroInputStream::backup(size_t len) { impl_->Backup(len); }

void UnifexAvroInputStream::skip(size_t len) { impl_->Skip(len); }

uint64_t UnifexAvroInputStream::byteCount() const { return impl_->ByteCount(); }

}  // namespace storage
}  // namespace axon
}  // namespace eux
