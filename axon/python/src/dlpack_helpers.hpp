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

#ifndef AXON_PYTHON_DLPACK_HELPERS_HPP_
#define AXON_PYTHON_DLPACK_HELPERS_HPP_

#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>

#include <atomic>

#include "rpc_core/rpc_types.hpp"
#include "rpc_core/utils/tensor_meta.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;
namespace rpc = eux::rpc;
namespace ucxx = eux::ucxx;

using TensorMetaVec = rpc::TensorMetaVec;

// =============================================================================
// DLPackCapsuleWrapper: A lightweight wrapper around PyCapsule for from_dlpack
// =============================================================================
// This class wraps a raw DLPack PyCapsule and exposes the __dlpack__ and
// __dlpack_device__ methods required by the DLPack protocol. This allows
// frameworks like numpy, PyTorch, JAX, etc. to consume our tensors via
// their standard from_dlpack() functions.
//
// DLPack Protocol Compliance (v2024.12):
// - __dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)
// - __dlpack_device__() -> tuple(device_type, device_id)
//
// Key semantics:
// - Capsule ownership is TRANSFERRED on __dlpack__() call (can only be called
//   once)
// - The capsule name changes from "dltensor" to "used_dltensor" after
//   consumption
//
// Performance notes:
// - Zero-copy: The wrapper just holds a reference to the capsule
// - Minimal overhead: Only creates a small Python object
// - No locks: Uses atomic flag for thread-safe consumption check
// - Compatible with all DLPack-compliant frameworks
class DLPackCapsuleWrapper {
 public:
  DLPackCapsuleWrapper(nb::object capsule, DLDevice device)
    : capsule_(std::move(capsule)), device_(device), consumed_(false) {}

  // Move constructor
  DLPackCapsuleWrapper(DLPackCapsuleWrapper&& other) noexcept
    : capsule_(std::move(other.capsule_)),
      device_(other.device_),
      consumed_(other.consumed_.load(std::memory_order_relaxed)) {
    other.consumed_.store(true, std::memory_order_relaxed);
  }

  // Move assignment
  DLPackCapsuleWrapper& operator=(DLPackCapsuleWrapper&& other) noexcept {
    if (this != &other) {
      capsule_ = std::move(other.capsule_);
      device_ = other.device_;
      consumed_.store(
        other.consumed_.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
      other.consumed_.store(true, std::memory_order_relaxed);
    }
    return *this;
  }

  // Delete copy operations
  DLPackCapsuleWrapper(const DLPackCapsuleWrapper&) = delete;
  DLPackCapsuleWrapper& operator=(const DLPackCapsuleWrapper&) = delete;

  // __dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)
  // -> PyCapsule
  //
  // DLPack protocol: This transfers ownership of the capsule to the consumer.
  // After this call, the capsule is marked as consumed and cannot be reused.
  //
  // Parameters (per DLPack v2024.12 spec):
  // - stream: For CUDA/ROCm, a Python int representing stream pointer.
  //           For CPU, must be None.
  // - max_version: tuple(major, minor) - max DLPack version consumer supports
  // - dl_device: tuple(device_type, device_id) - target device for export
  // - copy: bool or None - whether to copy (True=always, False=never,
  // None=auto)
  nb::object dlpack(
    nb::object stream = nb::none(), nb::object max_version = nb::none(),
    nb::object dl_device = nb::none(), nb::object copy = nb::none()) {
    // Check if already consumed (atomic for thread safety)
    bool expected = false;
    if (!consumed_.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
      throw std::runtime_error(
        "DLPack capsule has already been consumed. "
        "__dlpack__() can only be called once per capsule.");
    }

    // Validate stream for CPU device
    if (device_.device_type == kDLCPU && !stream.is_none()) {
      throw std::invalid_argument(
        "stream must be None for CPU device (kDLCPU)");
    }

    // Handle dl_device parameter - validate target device compatibility
    if (!dl_device.is_none()) {
      auto target = nb::cast<nb::tuple>(dl_device);
      int target_type = nb::cast<int>(target[0]);
      // For now, we only support same-device export (no cross-device copy)
      if (target_type != static_cast<int>(device_.device_type)) {
        // Check copy parameter
        if (!copy.is_none() && !nb::cast<bool>(copy)) {
          throw std::runtime_error(
            "Cross-device export requires copy=True or copy=None, "
            "but copy=False was specified");
        }
        // Cross-device copy not implemented yet
        throw std::runtime_error(
          "Cross-device DLPack export is not yet supported");
      }
    }

    // Handle copy parameter
    if (!copy.is_none() && nb::cast<bool>(copy)) {
      // Explicit copy requested - not supported (we're zero-copy only)
      throw std::runtime_error(
        "copy=True is not supported; this implementation is zero-copy only");
    }

    // max_version is informational - we always return DLPack 0.x compatible
    // capsules. When max_version >= (1, 0), we could return
    // DLManagedTensorVersioned but for now we stick with DLManagedTensor.

    // Transfer ownership by moving the capsule
    return std::move(capsule_);
  }

  // __dlpack_device__() -> tuple(device_type, device_id)
  //
  // Returns the device information without consuming the capsule.
  // This can be called multiple times.
  nb::tuple dlpack_device() const {
    return nb::make_tuple(
      static_cast<int>(device_.device_type), device_.device_id);
  }

 private:
  nb::object capsule_;  // The underlying PyCapsule (mutable for move)
  DLDevice device_;     // Device info
  mutable std::atomic<bool> consumed_;  // Thread-safe consumption flag
};

// Helper to convert ucx_memory_type_t to DLDevice
DLDevice UcxMemoryTypeToDlDevice(ucx_memory_type_t mem_type);

// Helper to convert DLDevice to ucx_memory_type_t
ucx_memory_type_t DlDeviceToUcxMemoryType(DLDevice device);

std::pair<DLManagedTensor*, nb::object> ExtractDlpackTensor(
  nb::object py_dlpack);

ucxx::UcxBuffer DlpackToUcxBuffer(
  nb::object py_dlpack, const rpc::utils::TensorMeta& meta,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr);

ucxx::UcxBuffer DlpackToUcxBuffer(
  nb::object py_dlpack, size_t size, ucx_memory_type_t mem_type,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr);

ucxx::UcxBufferVec DlpackToUcxBufferVec(
  nb::object py_dlpack_list, const rpc::utils::TensorMetaSpan tensor_metas,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr);

ucxx::UcxBufferVec DlpackToUcxBufferVec(
  nb::object py_dlpack_list, std::span<const size_t> sizes,
  ucx_memory_type_t mem_type,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr);

// Convert TensorMeta + UcxBuffer to dlpack object (single tensor)
nb::object TensorMetaToDlpack(
  rpc::utils::TensorMeta&& meta, ucxx::UcxBuffer&& buffer);

// Convert TensorMetaVec + UcxBufferVec to dlpack objects (multiple tensors)
nb::list TensorMetaVecToDlpack(
  cista::offset::vector<rpc::utils::TensorMeta>&& meta_vec,
  ucxx::UcxBufferVec&& buffer_vec);

// Convert UcxBuffer to dlpack object (no copy, uses reference)
nb::object UcxBufferToDLTensor(const ucxx::UcxBuffer& buffer);

// Convert UcxBuffer to dlpack object (transfer ownership)
nb::object UcxBufferToDLTensor(ucxx::UcxBuffer&& buffer);

// Convert UcxBufferVec to list of dlpack objects (transfer ownership)
nb::list UcxBufferVecToDLTensor(ucxx::UcxBufferVec&& buffer_vec);

// Create UcxBuffer from dlpack object
ucxx::UcxBuffer CreateUcxBufferFromPayload(
  nb::object payload_obj,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr);

// Create UcxBufferVec from dlpack object
ucxx::UcxBufferVec CreateUcxBufferVecFromPayload(
  nb::object payload_obj,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr);

// Extract TensorMeta directly from DLManagedTensor* (no Python call)
rpc::utils::TensorMeta ExtractTensorMetaFromDlm(DLManagedTensor* dlm);

// Create UcxBuffer from pre-extracted DLManagedTensor* (no repeated extraction)
ucxx::UcxBuffer DlpackToUcxBufferFromDlm(
  DLManagedTensor* dlm, nb::object owner, const rpc::utils::TensorMeta& meta,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr);

// Create UcxBufferVec from pre-extracted DLManagedTensor* array
ucxx::UcxBufferVec DlpackToUcxBufferVecFromDlm(
  std::span<DLManagedTensor*> dlms, std::span<nb::object> owners,
  const rpc::utils::TensorMetaSpan tensor_metas,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr);

// Extract TensorMeta from dlpack tensor
rpc::utils::TensorMeta ExtractTensorMetaFromDlpack(nb::object py_dlpack);

// Check if Python object is a dlpack tensor
bool IsDlpackTensor(nb::object py_obj);

}  // namespace python
}  // namespace axon
}  // namespace eux

#endif  // AXON_PYTHON_DLPACK_HELPERS_HPP_
