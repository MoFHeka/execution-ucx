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

#ifndef AXON_CORE_STORAGE_ASYNC_IO_HPP_
#define AXON_CORE_STORAGE_ASYNC_IO_HPP_

#include <chrono>
#include <concepts>
#include <ctime>
#include <expected>
#include <filesystem>
#include <iostream>
#include <memory>
#include <ranges>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <unifex/sync_wait.hpp>
#include <unifex/task.hpp>
#include <unifex/v2/async_scope.hpp>
#if !UNIFEX_NO_LIBURING
#include <unifex/linux/io_uring_context.hpp>
#else
#error "UNIFEX_NO_LIBURING is true. Compilation is terminated."
#endif

#include <avro/Compiler.hh>
#include <avro/DataFile.hh>
#include <avro/Generic.hh>

#include <unifex/async_scope.hpp>

#include "axon/storage/avro_serialization.hpp"
#include "axon/storage/avro_unifex_io.hpp"
#include "axon/utils/axon_message.hpp"

namespace eux {
namespace axon {
namespace storage {

/**
 * @brief Asynchronous I/O utilities for AxonRequest persistence
 *
 * This class provides async operations for saving and loading AxonRequest
 * objects to/from disk using libunifex for efficient async execution.
 */
class AsyncIO {
 public:
  AsyncIO() = default;
  virtual ~AsyncIO() = default;

  /**
   * @brief Asynchronously save multiple ArrivalRequests to disk
   * @param requests Vector of shared_ptr<AxonRequest> to save
   * @param mr Memory resource manager for buffer allocation
   * @param base_path Base directory path for saving
   * @param prefix Prefix for the filename
   * @return std::error_code
   */
  // template <typename R>
  //   requires std::ranges::range<R>
  //              && std::same_as<
  //                std::remove_cv_t<std::ranges::range_value_t<R>>,
  //                std::shared_ptr<AxonRequest>>
  // virtual std::error_code Save(
  //   const R& requests, const std::filesystem::path& base_path,
  //   ucxx::UcxMemoryResourceManager& mr);
  virtual std::error_code Save(
    const std::vector<std::shared_ptr<AxonRequest>>& requests,
    ucxx::UcxMemoryResourceManager& mr,
    const std::filesystem::path& base_path,
    const std::string& prefix) {
    throw std::runtime_error("Save not implemented");
  }

  /**
   * @brief Asynchronously save multiple ArrivalRequests to disk
   * @param requests Vector of shared_ptr<AxonRequest> to save
   * @param mr Memory resource manager for buffer allocation
   * @param base_path Base directory path for saving
   * @param prefix Prefix for the filename
   * @return Task wrapper
   */
  // template <typename R>
  //   requires std::ranges::range<R>
  //            && std::same_as<
  //              std::remove_cv_t<std::ranges::range_value_t<R>>,
  //              std::shared_ptr<AxonRequest>>
  // virtual unifex::task<std::error_code> SaveAsync(
  //   const R& requests,
  //   ucxx::UcxMemoryResourceManager& mr,
  //   const std::filesystem::path& base_path,
  //   const std::string& prefix);
  virtual std::error_code SaveAsync(
    const std::vector<std::shared_ptr<AxonRequest>>& requests,
    ucxx::UcxMemoryResourceManager& mr,
    const std::filesystem::path& base_path,
    const std::string& prefix = "") {
    throw std::runtime_error("SaveAsync not implemented");
  }

  /**
   * @brief Asynchronously load multiple ArrivalRequests from disk
   * @param mr Memory resource manager for buffer allocation
   * @param file_path Path to the file to load
   * @return std::vector<std::shared_ptr<AxonRequest>>
   */
  virtual std::expected<
    std::vector<std::shared_ptr<AxonRequest>>, std::error_code>
  Load(
    ucxx::UcxMemoryResourceManager& mr,
    const std::filesystem::path& file_path) {
    throw std::runtime_error("Load not implemented");
  }

  /**
   * @brief Asynchronously load ArrivalRequests from disk as a generator
   * @param file_path Path to the file to load
   * @param mr Memory resource manager for buffer allocation
   * @return A generator that yields shared_ptr<AxonRequest>
   */
  // virtual std::generator<
  //   std::expected<std::shared_ptr<AxonRequest>, std::error_code>>
  // LoadAsync(
  //   const std::filesystem::path& file_path, ucxx::UcxMemoryResourceManager&
  //   mr);

 protected:
  /**
   * @brief Generate a unique filename for a request
   * @param session_id Session ID of the request
   * @param timestamp Timestamp string for uniqueness (human-readable with
   * decimal precision)
   * @return Generated filename
   */
  static inline std::string GenerateFilename(
    uint32_t session_id, const std::string& timestamp) {
    return fmt::format("{}_{}.avro", session_id, timestamp);
  }

  /**
   * @brief Generate a unique filename for a request
   * @param prefix Prefix for the filename
   * @param timestamp Timestamp string for uniqueness (human-readable with
   * decimal precision)
   * @return Generated filename
   */
  static inline std::string GenerateFilename(
    const std::string& prefix, const std::string& timestamp) {
    return fmt::format("{}_{}.avro", prefix, timestamp);
  }

  /**
   * @brief Get the current timestamp as a human-readable string with decimal
   * precision
   * @return Current timestamp string in format "seconds.milliseconds"
   */
  static inline std::string get_current_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto time_t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch())
                    % 1000;
    return fmt::format("{}.{:03d}", time_t, ms.count());
  }
};

/**
 * @brief Asynchronous I/O utilities for AxonRequest persistence
 *
 * This class provides async operations for saving and loading AxonRequest
 * objects to/from disk using libunifex for efficient async execution.
 */
class AsyncUnifexAvroIO : public AsyncIO {
 public:
  /**
   * @brief Constructor
   * @param thread_pool Thread pool for async operations
   */
  explicit AsyncUnifexAvroIO(io_context& io_context, size_t buffer_size = 8192);
  ~AsyncUnifexAvroIO() override;

  std::error_code Save(
    const std::vector<std::shared_ptr<AxonRequest>>& requests,
    ucxx::UcxMemoryResourceManager& mr, const std::filesystem::path& base_path,
    const std::string& prefix = "") override;

  /**
   * @brief Asynchronously save multiple ArrivalRequests to disk
   * @param requests Vector of shared_ptr<AxonRequest> to save
   * @param base_path Base directory path for saving
   * @param prefix Prefix for the filename
   * @return std::error_code
   */
  template <typename R>
    requires std::ranges::range<R>
             && std::same_as<
               std::remove_cv_t<std::ranges::range_value_t<R>>,
               std::shared_ptr<AxonRequest>>
  std::error_code Save(
    const R& requests,
    ucxx::UcxMemoryResourceManager& mr,
    const std::filesystem::path& base_path,
    const std::string& prefix) {
    if (requests.empty()) {
      return std::error_code();
    }

    std::string filename;
    auto timestamp = get_current_timestamp();
    if (prefix.empty()) {
      filename = GenerateFilename(
        cista::to_idx((*requests.begin())->header.session_id), timestamp);
    } else {
      filename = GenerateFilename(prefix, timestamp);
    }
    auto file_path = base_path / filename;

    try {
      auto out_stream = std::make_unique<UnifexAvroOutputStream>(
        async_scope_.value(), io_context_, file_path, buffer_size_);

      avro::ValidSchema schema = avro::compileJsonSchemaFromString(
        AvroSchema::GetArrivalRequestSchema().data());
      avro::DataFileWriter<avro::GenericDatum> writer(
        std::move(out_stream), schema);
      for (const auto& req : requests) {
        writer.write(AvroSerialization::Serialize(*req, mr));
      }
      writer.close();
    } catch (const std::exception& e) {
      std::cerr << "Error: Failed to save file: " << file_path << ". "
                << e.what() << std::endl;
      return std::make_error_code(std::errc::io_error);
    }
    return std::error_code();
  }

  /**
   * @brief Asynchronously load multiple ArrivalRequests from disk
   * @param mr Memory resource manager for buffer allocation
   * @param file_path Path to the file to load
   * @return std::vector<std::shared_ptr<AxonRequest>>
   */
  std::expected<std::vector<std::shared_ptr<AxonRequest>>, std::error_code>
  Load(
    ucxx::UcxMemoryResourceManager& mr,
    const std::filesystem::path& file_path) override;

  /**
   * @brief Waits for all pending async operations to complete.
   */
  void sync() {
    unifex::sync_wait(async_scope_->join());
    async_scope_.emplace();
  }

  /**
   * @brief Get the default IO context for async operations
   * @return IO context
   */
  static io_context& GetDefaultIOContext();

 private:
  std::optional<_::async_scope> async_scope_;
  io_context& io_context_;
  size_t buffer_size_;
};

}  // namespace storage
}  // namespace axon
}  // namespace eux

#endif  // AXON_CORE_STORAGE_ASYNC_IO_HPP_
