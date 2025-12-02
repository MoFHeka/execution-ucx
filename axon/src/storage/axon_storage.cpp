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

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "axon/storage/async_io.hpp"

namespace eux {
namespace axon {
namespace storage {

AxonStorage::AxonStorage(
  std::optional<std::reference_wrapper<io_context>> io_context)
  : io_context_{
    io_context.has_value() ? io_context.value().get()
                           : AsyncUnifexAvroIO::GetDefaultIOContext()} {
  // io_context_ is used to create the AsyncUnifexAvroIO instance.
  // For now only suppport io_uring context from libunifex.
}

auto AxonStorage::save(
  ucxx::UcxMemoryResourceManager& mr, const std::filesystem::path& base_path,
  const std::string& prefix) const -> std::error_code {
  AsyncUnifexAvroIO io_service(io_context_);
  return io_service.Save(arrival_requests_, mr, base_path, prefix);
}

auto AxonStorage::load(
  ucxx::UcxMemoryResourceManager& mr, const std::filesystem::path& file_path)
  -> std::expected<void, std::error_code> {
  AsyncUnifexAvroIO io_service(io_context_);
  auto result = io_service.Load(mr, file_path);
  if (result) {
    for (auto&& req : *result) {
      arrival_requests_.emplace(std::move(req));
    }
    return {};
  } else {
    return std::unexpected(result.error());
  }
}

auto AxonStorage::save_request(
  ucxx::UcxMemoryResourceManager& mr,
  request_iterator it,
  const std::filesystem::path& base_path) const -> std::error_code {
  AsyncUnifexAvroIO io_service(io_context_);
  std::vector<std::shared_ptr<AxonRequest>> single_request_vec;
  if (it != arrival_requests_.end()) {
    single_request_vec.push_back(*it);
  }
  return io_service.Save(single_request_vec, mr, base_path, "");
}

}  // namespace storage
}  // namespace axon
}  // namespace eux
