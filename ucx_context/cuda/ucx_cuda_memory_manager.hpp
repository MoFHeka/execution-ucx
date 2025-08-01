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

#include <cuda.h>
#include <cuda_runtime.h>

#include "ucx_context/cuda/ucx_cuda_macro.h"
#include "ucx_context/ucx_context_def.h"
#include "ucx_context/ucx_memory_resource.hpp"

/*
Be careful, please use your own memory manager instead of this class!
This class is only for testing purpose!
*/

namespace stdexe_ucx_runtime {

/**
 * @brief CUDA-specific memory resource manager implementation
 *
 * This class provides CUDA-specific implementations for memory allocation,
 * deallocation and memory copy operations between different memory types
 * (HOST, CUDA, CUDA_MANAGED).
 */
class UcxCudaMemoryResourceManager : public UcxMemoryResourceManager {
 public:
  /**
   * @brief Constructor initializes CUDA memory copy functions
   *
   * Sets up appropriate memory copy functions for different combinations of
   * memory types (HOST, CUDA, CUDA_MANAGED) using CUDA runtime API.
   */
  UcxCudaMemoryResourceManager() {
    // Initialize memcpy functions for different memory types
    for (int i = 0; i < UCS_MEMORY_TYPE_LAST; i++) {    // dest type
      for (int j = 0; j < UCS_MEMORY_TYPE_LAST; j++) {  // src type
        memcpy_fns_[i][j] = nullptr;
      }
    }

    // Set up CUDA memory copy functions
    memcpy_fns_[UCS_MEMORY_TYPE_HOST][UCS_MEMORY_TYPE_HOST] = std::memcpy;
    // Helper function to create cudaMemcpy lambda with specific kind
    auto createCudaMemcpyFn = [](cudaMemcpyKind kind) {
      return [kind](void* dest, const void* src, size_t count) -> void* {
        auto ret = cudaMemcpy(dest, src, count, kind);
        if (ret != cudaSuccess) {
          throw std::runtime_error("CUDA memcpy failed");
        }
        return dest;
      };
    };
    // Set up CUDA memory copy functions with different transfer directions
    memcpy_fns_[UCS_MEMORY_TYPE_HOST][UCS_MEMORY_TYPE_CUDA] =
      createCudaMemcpyFn(cudaMemcpyDeviceToHost);
    memcpy_fns_[UCS_MEMORY_TYPE_CUDA][UCS_MEMORY_TYPE_HOST] =
      createCudaMemcpyFn(cudaMemcpyHostToDevice);
    memcpy_fns_[UCS_MEMORY_TYPE_CUDA][UCS_MEMORY_TYPE_CUDA] =
      createCudaMemcpyFn(cudaMemcpyDeviceToDevice);
  }

  /**
   * @brief Allocate memory of specified type and size
   * @param type Memory type to allocate (HOST, CUDA, CUDA_MANAGED)
   * @param size Size of memory to allocate
   * @param alignment Memory alignment requirement
   * @return Pointer to allocated memory
   * @throw std::runtime_error if memory type is not supported
   */
  void* allocate(
    ucx_memory_type_t type, size_t size, size_t alignment = 8) override {
    void* ptr = nullptr;
    switch (type) {
      case ucx_memory_type::HOST:
        ptr = malloc(size);
        break;
      case ucx_memory_type::CUDA:
        UCX_CUDA_CHECK(cudaMalloc(&ptr, size));
        break;
      case ucx_memory_type::CUDA_MANAGED:
        UCX_CUDA_CHECK(cudaMallocManaged(&ptr, size));
        break;
      default:
        throw std::runtime_error("Unsupported memory type");
    }
    return ptr;
  }

  /**
   * @brief Deallocate memory of specified type
   * @param type Memory type of the pointer (HOST, CUDA, CUDA_MANAGED)
   * @param ptr Pointer to deallocate
   * @param size Size of memory to deallocate
   * @param alignment Memory alignment that was used
   * @throw std::runtime_error if memory type is not supported
   */
  void deallocate(
    ucx_memory_type_t type, void* ptr, size_t size,
    size_t alignment = 8) override {
    switch (type) {
      case ucx_memory_type::HOST:
        free(ptr);
        break;
      case ucx_memory_type::CUDA:
        UCX_CUDA_CHECK(cudaFree(ptr));
        break;
      case ucx_memory_type::CUDA_MANAGED:
        UCX_CUDA_CHECK(cudaFree(ptr));
        break;
      default:
        throw std::runtime_error("Unsupported memory type");
    }
  }

  /**
   * @brief Copy memory between different memory types using CUDA runtime
   * @param dst_type Destination memory type
   * @param dst Destination pointer
   * @param src_type Source memory type
   * @param src Source pointer
   * @param size Number of bytes to copy
   * @return Pointer to destination
   * @throw std::runtime_error if memory copy combination is not supported
   */
  void* memcpy(
    ucx_memory_type_t dst_type, void* dst, ucx_memory_type_t src_type,
    const void* src, size_t size) override {
    if (memcpy_fns_[dst_type][src_type]) {
      return memcpy_fns_[dst_type][src_type](dst, src, size);
    } else {
      throw std::runtime_error("Unsupported memory copy combination");
    }
    return nullptr;
  }
};

}  // namespace stdexe_ucx_runtime
