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

#ifndef AXON_CORE_STORAGE_AXON_STORAGE_HPP_
#define AXON_CORE_STORAGE_AXON_STORAGE_HPP_

#include <cista.h>
#include <plf_hive.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if !UNIFEX_NO_LIBURING
#include <unifex/linux/io_uring_context.hpp>
#else
// using io_context = unifex::linuxos::io_epoll_context;
#error "UNIFEX_NO_LIBURING is true. Compilation is terminated."
#endif

#include "axon/utils/axon_message.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace axon {
namespace storage {

#if !UNIFEX_NO_LIBURING
using io_context = unifex::linuxos::io_uring_context;
#else
// using io_context = unifex::linuxos::io_epoll_context;
#error "UNIFEX_NO_LIBURING is true. Compilation is terminated."
#endif

using AxonRequest = eux::axon::utils::AxonRequest;
using AxonRequestPtr = std::shared_ptr<AxonRequest>;

// Naming conventions in this class imitate those of standard containers (std),
// because AxonStorage is designed to behave similarly to them.

class AxonStorage {
 public:
  using request_iterator = plf::hive<AxonRequestPtr>::iterator;

  /**
   * @brief Constructor
   * @param io_context io_context for async operations (optional)
   */
  explicit AxonStorage(
    std::optional<std::reference_wrapper<io_context>> io_context =
      std::nullopt);
  ~AxonStorage() = default;

  template <typename... Args>
    requires std::is_constructible_v<AxonRequest, Args...>
  request_iterator emplace(Args&&... args) {
    return arrival_requests_.emplace(
      std::make_shared<AxonRequest>(std::forward<Args>(args)...));
  }

  request_iterator emplace(const std::shared_ptr<AxonRequest>& request) {
    return arrival_requests_.emplace(request);  // increase ref count
  }

  request_iterator emplace(std::shared_ptr<AxonRequest>&& request) {
    return arrival_requests_.emplace(std::move(request));
  }

  /**
   * @brief Returns an iterator to the beginning and end.
   * @return An iterator to the beginning and end.
   */
  auto begin() { return arrival_requests_.begin(); }
  auto end() { return arrival_requests_.end(); }
  auto begin() const { return arrival_requests_.begin(); }
  auto end() const { return arrival_requests_.end(); }

  /**
   * @brief Returns a reverse iterator to the beginning and end.
   * @return A reverse iterator to the beginning and end.
   */
  auto rbegin() { return arrival_requests_.rbegin(); }
  auto rend() { return arrival_requests_.rend(); }
  auto rbegin() const { return arrival_requests_.rbegin(); }
  auto rend() const { return arrival_requests_.rend(); }

  /**
   * @brief Returns a reference to the first request.
   * @return Reference to the first stored request.
   */
  auto& front() { return *arrival_requests_.begin(); }
  const auto& front() const { return *arrival_requests_.begin(); }

  /**
   * @brief Returns a reference to the last request.
   * @return Reference to the last stored request.
   */
  auto& back() {
    auto it = arrival_requests_.end();
    --it;
    return *it;
  }
  const auto& back() const {
    auto it = arrival_requests_.end();
    --it;
    return *it;
  }

  /**
   * @brief Checks if the container is empty.
   * @return true if empty, false otherwise.
   */
  bool empty() const { return arrival_requests_.empty(); }

  /**
   * @brief Advances a given iterator by 'n' steps forward.
   * @param it The iterator to advance.
   * @param n Number of steps to advance (can be negative if
   * BidirectionalIterator).
   * @return The advanced iterator.
   */
  request_iterator advance(request_iterator it, std::ptrdiff_t n) {
    std::advance(it, n);
    return it;
  }
  plf::hive<std::shared_ptr<AxonRequest>>::const_iterator advance(
    plf::hive<std::shared_ptr<AxonRequest>>::const_iterator it,
    std::ptrdiff_t n) const {
    std::advance(it, n);
    return it;
  }

  /**
   * @brief Returns a generator to iterate through all stored requests.
   * @return A generator to iterate through all stored requests.
   */
  // std::generator<std::shared_ptr<AxonRequest>> AsGenerator() const {
  //   for (const auto& req : arrival_requests_) {
  //     co_yield req;
  //   }
  // }

  /**
   * @brief Erases a given iterator.
   * @param it The iterator to erase.
   */
  void erase(request_iterator it) { arrival_requests_.erase(it); }

  /**
   * @brief Returns the size of the storage.
   * @return The size of the storage.
   */
  size_t size() const { return arrival_requests_.size(); }

  // Async persistence methods
  std::error_code save(
    ucxx::UcxMemoryResourceManager& mr, const std::filesystem::path& base_path,
    const std::string& prefix = "") const;

  std::expected<void, std::error_code> load(
    ucxx::UcxMemoryResourceManager& mr, const std::filesystem::path& file_path);

  std::error_code save_request(
    ucxx::UcxMemoryResourceManager& mr, request_iterator it,
    const std::filesystem::path& base_path) const;

 private:
  plf::hive<AxonRequestPtr> arrival_requests_;
  io_context& io_context_;
};

}  // namespace storage
}  // namespace axon
}  // namespace eux

#endif  // AXON_CORE_STORAGE_AXON_STORAGE_HPP_
