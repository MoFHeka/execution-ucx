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

#include <expected>
#include <filesystem>
#include <generator>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <avro/DataFile.hh>
#include <avro/Decoder.hh>
#include <avro/Encoder.hh>

#include <unifex/sync_wait.hpp>

#include "axon/storage/avro_schema.hpp"
#include "axon/storage/avro_serialization.hpp"
#include "axon/storage/avro_unifex_io.hpp"

namespace eux {
namespace axon {
namespace storage {

AsyncUnifexAvroIO::AsyncUnifexAvroIO(io_context& io_context, size_t buffer_size)
  : io_context_(io_context), buffer_size_(buffer_size) {
  async_scope_.emplace();
}

AsyncUnifexAvroIO::~AsyncUnifexAvroIO() {
  // Wait for all tasks to complete. This ensures that any file
  // operations are finished before we try to delete the file.
  unifex::sync_wait(async_scope_.value().join());
}

io_context& AsyncUnifexAvroIO::GetDefaultIOContext() {
  static auto default_io_context = io_context();
  return default_io_context;
}

std::error_code AsyncUnifexAvroIO::Save(
  const std::vector<std::shared_ptr<AxonRequest>>& requests,
  ucxx::UcxMemoryResourceManager& mr, const std::filesystem::path& base_path,
  const std::string& prefix) {
  return Save<std::vector<std::shared_ptr<AxonRequest>>>(
    requests, mr, base_path, prefix);
}

std::expected<std::vector<std::shared_ptr<AxonRequest>>, std::error_code>
AsyncUnifexAvroIO::Load(
  ucxx::UcxMemoryResourceManager& mr, const std::filesystem::path& file_path) {
  if (!std::filesystem::exists(file_path)) {
    std::cerr << "Error: Failed to load file: " << file_path
              << ". No such file or directory" << std::endl;
    return std::unexpected(
      std::make_error_code(std::errc::no_such_file_or_directory));
  }

  std::vector<std::shared_ptr<AxonRequest>> requests;
  try {
    auto in_stream = std::make_unique<UnifexAvroInputStream>(
      async_scope_.value(), io_context_, file_path, buffer_size_);

    avro::ValidSchema schema = avro::compileJsonSchemaFromString(
      AvroSchema::GetArrivalRequestSchema().data());
    avro::DataFileReader<avro::GenericDatum> reader(
      std::move(in_stream), schema);

    avro::GenericDatum datum(schema);
    while (reader.read(datum)) {
      requests.emplace_back(AvroSerialization::Deserialize(datum, mr));
    }
    return requests;
  } catch (const std::exception& e) {
    std::cerr << "Error: Failed to load file: " << file_path << ". " << e.what()
              << std::endl;
    return std::unexpected(std::make_error_code(std::errc::io_error));
  }
}

}  // namespace storage
}  // namespace axon
}  // namespace eux
