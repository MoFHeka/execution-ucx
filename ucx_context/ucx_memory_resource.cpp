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

#include "ucx_context/ucx_memory_resource.hpp"

#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>

namespace eux {
namespace ucxx {

// Default constructor initializes all memory resources to default resource
// and memcpy functions to std::memcpy
DefaultUcxMemoryResourceManager::DefaultUcxMemoryResourceManager() {
  // Initialize all memory resources to the default resource
  for (size_t i = 0; i < UCX_MEMORY_TYPE_COUNT; ++i) {
    memory_resources_[i] = std::ref(*std::pmr::get_default_resource());
  }
  // Initialize all memcpy functions to std::memcpy
  for (size_t i = 0; i < UCX_MEMORY_TYPE_COUNT; ++i) {
    for (size_t j = 0; j < UCX_MEMORY_TYPE_COUNT; ++j) {
      memcpy_fns_[i][j] =
        [](void* dest, const void* src, size_t count) -> void* {
        return std::memcpy(dest, src, count);
      };
    }
  }
}

// Register memory resource by reference (more efficient)
void UcxMemoryResourceManager::register_memory_resource(
  ucx_memory_type_t type,
  std::reference_wrapper<std::pmr::memory_resource> resource) {
  if (__builtin_expect(type >= UCX_MEMORY_TYPE_COUNT, 0)) {
    throw std::invalid_argument("Invalid memory type");
  }
  memory_resources_[type] = resource;
}

// Register memcpy function
void UcxMemoryResourceManager::register_memcpy_fn(
  ucx_memory_type_t dest_type, ucx_memory_type_t src_type,
  std::function<void*(void*, const void*, size_t)> memcpy_fn) {
  if (__builtin_expect(
        dest_type >= UCX_MEMORY_TYPE_COUNT || src_type >= UCX_MEMORY_TYPE_COUNT,
        0)) {
    throw std::invalid_argument("Invalid memory type");
  }
  memcpy_fns_[dest_type][src_type] = std::move(memcpy_fn);
}

std::pmr::memory_resource* UcxMemoryResourceManager::get_memory_resource(
  ucx_memory_type_t type) const {
  if (__builtin_expect(type >= UCX_MEMORY_TYPE_COUNT, 0)) {
    throw std::invalid_argument("Invalid memory type");
  }
  try {
    return &memory_resources_[type].value().get();
  } catch (const std::bad_optional_access&) {
    throw std::runtime_error(
      "Memory resource not registered for type: " + std::to_string(type));
  }
}

std::function<void*(void*, const void*, size_t)>
UcxMemoryResourceManager::get_memcpy_fn(
  ucx_memory_type_t dest_type, ucx_memory_type_t src_type) const {
  if (__builtin_expect(
        dest_type >= UCX_MEMORY_TYPE_COUNT || src_type >= UCX_MEMORY_TYPE_COUNT,
        0)) {
    throw std::invalid_argument("Invalid memory type");
  }
  auto fn = memcpy_fns_[dest_type][src_type];
  if (__builtin_expect(!fn, 0)) {
    throw std::runtime_error(
      "Memcpy function not registered for type: " + std::to_string(src_type)
      + " -> " + std::to_string(dest_type));
  }
  return fn;
}

void* DefaultUcxMemoryResourceManager::allocate(
  ucx_memory_type_t type, size_t bytes, size_t alignment) {
  return get_memory_resource(type)->allocate(bytes, alignment);
}

void DefaultUcxMemoryResourceManager::deallocate(
  ucx_memory_type_t type, void* p, size_t bytes, size_t alignment) {
  get_memory_resource(type)->deallocate(p, bytes, alignment);
}

void* DefaultUcxMemoryResourceManager::memcpy(
  ucx_memory_type_t dest_type, void* dest, ucx_memory_type_t src_type,
  const void* src, size_t count) {
  return get_memcpy_fn(dest_type, src_type)(dest, src, count);
}

}  // namespace ucxx
}  // namespace eux
