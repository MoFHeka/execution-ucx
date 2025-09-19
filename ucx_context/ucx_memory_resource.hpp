/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.

Licensed under the Apache License Version 2.0 with LLVM Exceptions
(the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    https://llvm.org/LICENSE.txt

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#ifndef UCX_MEMORY_RESOURCE_HPP_
#define UCX_MEMORY_RESOURCE_HPP_

#include <array>
#include <functional>
#include <memory>
#include <memory_resource>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "ucx_context/ucx_context_def.h"

namespace eux {
namespace ucxx {

/**
 * @brief Base class for managing different types of memory resources in UCX
 * context
 *
 * This class provides an interface for managing memory resources across
 * different memory types (HOST, CUDA, ROCM, etc.) and handling memory
 * operations between them.
 */
class UcxMemoryResourceManager {
 public:
  UcxMemoryResourceManager() = default;
  virtual ~UcxMemoryResourceManager() = default;

  static constexpr const size_t UCX_MEMORY_TYPE_COUNT =
    static_cast<std::underlying_type_t<ucx_memory_type>>(ucx_memory_type::LAST)
    + 1;

  /**
   * @brief Register a memory resource for a specific memory type
   * @param type The memory type to register the resource for
   * @param resource Reference to the memory resource to register
   * @throw std::invalid_argument if memory type is invalid
   */
  void register_memory_resource(
    ucx_memory_type_t type, std::pmr::memory_resource& resource);

  /**
   * @brief Register a memory copy function for specific source and destination
   * types
   * @param dest_type Destination memory type
   * @param src_type Source memory type
   * @param memcpy_fn Function to handle memory copy between these types
   * @throw std::invalid_argument if memory types are invalid
   */
  void register_memcpy_fn(
    ucx_memory_type_t dest_type, ucx_memory_type_t src_type,
    std::function<void*(void*, const void*, size_t)> memcpy_fn);

  /**
   * @brief Get the memory resource for a specific memory type
   * @param type The memory type to get resource for
   * @return Reference to the registered memory resource
   * @throw std::invalid_argument if memory type is invalid
   * @throw std::runtime_error if no resource is registered for the type
   */
  std::pmr::memory_resource& get_memory_resource(ucx_memory_type_t type) const;

  /**
   * @brief Get the memory copy function for specific source and destination
   * types
   * @param dest_type Destination memory type
   * @param src_type Source memory type
   * @return Function to handle memory copy between these types
   * @throw std::invalid_argument if memory types are invalid
   * @throw std::runtime_error if no copy function is registered
   */
  std::function<void*(void*, const void*, size_t)> get_memcpy_fn(
    ucx_memory_type_t dest_type, ucx_memory_type_t src_type) const;

  /**
   * @brief Allocate memory of specified type and size
   * @param type Memory type to allocate
   * @param bytes Size of memory to allocate
   * @param alignment Memory alignment requirement
   * @return Pointer to allocated memory
   */
  virtual void* allocate(
    ucx_memory_type_t type, size_t bytes, size_t alignment = 8) = 0;

  /**
   * @brief Deallocate memory of specified type
   * @param type Memory type of the pointer
   * @param p Pointer to deallocate
   * @param bytes Size of memory to deallocate
   * @param alignment Memory alignment that was used
   */
  virtual void deallocate(
    ucx_memory_type_t type, void* p, size_t bytes, size_t alignment = 8) = 0;

  /**
   * @brief Copy memory between different memory types
   * @param dest_type Destination memory type
   * @param dest Destination pointer
   * @param src_type Source memory type
   * @param src Source pointer
   * @param count Number of bytes to copy
   * @return Pointer to destination
   */
  virtual void* memcpy(
    ucx_memory_type_t dest_type, void* dest, ucx_memory_type_t src_type,
    const void* src, size_t count) = 0;

 protected:
  std::array<
    std::optional<std::reference_wrapper<std::pmr::memory_resource>>,
    UCX_MEMORY_TYPE_COUNT>
    memory_resources_;
  std::array<
    std::array<
      std::function<void*(void*, const void*, size_t)>, UCX_MEMORY_TYPE_COUNT>,
    UCX_MEMORY_TYPE_COUNT>
    memcpy_fns_;  // [src_type][dest_type]
};

/**
 * @brief Default implementation of memory resource manager
 *
 * This class provides a default implementation of the memory resource manager
 * using standard memory allocation and copy operations. It serves as a fallback
 * implementation when no specific memory type handling is required.
 */
class DefaultUcxMemoryResourceManager : public UcxMemoryResourceManager {
 public:
  /**
   * @brief Constructor initializes default memory resources and copy functions
   *
   * Sets up default memory resources and standard memory copy functions
   * for all memory type combinations.
   */
  DefaultUcxMemoryResourceManager();
  ~DefaultUcxMemoryResourceManager() = default;

  /**
   * @brief Allocate memory using default memory resource
   * @param type Memory type to allocate
   * @param bytes Size of memory to allocate
   * @param alignment Memory alignment requirement
   * @return Pointer to allocated memory
   */
  void* allocate(
    ucx_memory_type_t type, size_t bytes, size_t alignment = 8) override;

  /**
   * @brief Deallocate memory using default memory resource
   * @param type Memory type of the pointer
   * @param p Pointer to deallocate
   * @param bytes Size of memory to deallocate
   * @param alignment Memory alignment that was used
   */
  void deallocate(
    ucx_memory_type_t type, void* p, size_t bytes,
    size_t alignment = 8) override;

  /**
   * @brief Copy memory using default memory copy function
   * @param dest_type Destination memory type
   * @param dest Destination pointer
   * @param src_type Source memory type
   * @param src Source pointer
   * @param count Number of bytes to copy
   * @return Pointer to destination
   */
  void* memcpy(
    ucx_memory_type_t dest_type, void* dest, ucx_memory_type_t src_type,
    const void* src, size_t count) override;
};

}  // namespace ucxx
}  // namespace eux

#endif  // UCX_MEMORY_RESOURCE_HPP_
