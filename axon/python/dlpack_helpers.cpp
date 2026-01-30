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

#include "axon/python/dlpack_helpers.hpp"

#include <cstddef>
#include <cstdio>
#include <dlpack/dlpack.h>
#include <format>
#include <nanobind/nanobind.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "axon/python/python_helpers.hpp"
#include "axon/utils/tensor.hpp"
#include "rpc_core/utils/tensor_meta.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;
namespace rpc = eux::rpc;
namespace ucxx = eux::ucxx;

// =============================================================================
// DLPack Capsule Destructor
// =============================================================================
// Per DLPack protocol:
// - Capsule name is "dltensor" when created
// - Consumer renames capsule to "used_dltensor" AFTER consuming AND calling
//   the DLManagedTensor->deleter
// - Therefore, if capsule name is "used_dltensor", the deleter was ALREADY
//   called by the consumer, and we must NOT call it again
//
// This is the correct way to implement the PyCapsule destructor according
// to the DLPack specification.
inline void SafeDlpackCapsuleDestructor(PyObject* obj) {
  // Try to get pointer with "dltensor" name (not consumed)
  DLManagedTensor* dlm =
    static_cast<DLManagedTensor*>(PyCapsule_GetPointer(obj, "dltensor"));

  if (dlm == nullptr) {
    // Failed - this means capsule was consumed (renamed to "used_dltensor")
    // Per DLPack protocol, consumer ALREADY called deleter, so we do nothing
    PyErr_Clear();  // Clear the error from PyCapsule_GetPointer
    return;
  }

  // Capsule was NOT consumed (still named "dltensor")
  // We are responsible for calling the deleter to free resources
  if (dlm->deleter) {
    dlm->deleter(dlm);
  }
}

// Helper to mark capsule as used (renaming to "used_dltensor")
inline void MarkDlpackCapsuleUsed(PyObject* capsule) {
  if (PyCapsule_IsValid(capsule, "dltensor")) {
    PyCapsule_SetName(capsule, "used_dltensor");
  }
}

// Helper to convert ucx_memory_type_t to DLDevice
DLDevice UcxMemoryTypeToDlDevice(ucx_memory_type_t mem_type) {
  DLDevice device;
  device.device_id = 0;  // TODO(He Jia): get actual device ID if needed
  switch (mem_type) {
    case ucx_memory_type::CUDA:
      device.device_type = kDLCUDA;
      break;
    case ucx_memory_type::CUDA_MANAGED:
      device.device_type = kDLCUDAManaged;
      break;
    case ucx_memory_type::ROCM:
      device.device_type = kDLROCM;
      break;
    case ucx_memory_type::ROCM_MANAGED:
      device.device_type = kDLROCMHost;
      break;
    case ucx_memory_type::RDMA:
      // RDMA maps to CPU as it's typically host-accessible
      device.device_type = kDLCPU;
      break;
    case ucx_memory_type::ZE_HOST:
    case ucx_memory_type::ZE_DEVICE:
    case ucx_memory_type::ZE_MANAGED:
      device.device_type = kDLOneAPI;
      break;
    case ucx_memory_type::HOST:
    case ucx_memory_type::UNKNOWN:  // UNKNOWN = LAST, so only use one
    default:
      device.device_type = kDLCPU;
      break;
  }
  return device;
}

// Helper to convert DLDevice to ucx_memory_type_t
ucx_memory_type_t DlDeviceToUcxMemoryType(DLDevice device) {
  switch (device.device_type) {
    case kDLCUDA:
      return ucx_memory_type::CUDA;
    case kDLCUDAManaged:
      return ucx_memory_type::CUDA_MANAGED;
    case kDLROCM:
      return ucx_memory_type::ROCM;
    case kDLROCMHost:
      return ucx_memory_type::ROCM_MANAGED;
    case kDLCPU:
      return ucx_memory_type::HOST;
    case kDLOneAPI:
      return ucx_memory_type::UNKNOWN;
    default:
      return ucx_memory_type::UNKNOWN;
  }
}

std::pair<DLManagedTensor*, nb::object> ExtractDlpackTensor(
  nb::object py_dlpack) {
  // Try to get __dlpack__ method first (preferred for writable arrays)
  if (nb::hasattr(py_dlpack, "__dlpack__")) {
    nb::object dlm_obj = py_dlpack.attr("__dlpack__")();
    // Check if it's a capsule
    if (PyCapsule_CheckExact(dlm_obj.ptr())) {
      return {
        static_cast<DLManagedTensor*>(
          PyCapsule_GetPointer(dlm_obj.ptr(), "dltensor")),
        std::move(dlm_obj)};
    } else {
      // Try to cast directly
      return {nb::cast<DLManagedTensor*>(dlm_obj), std::move(dlm_obj)};
    }
  }

  throw std::runtime_error(std::format(
    "Invalid dlpack object {} when extracting: missing __dlpack__ method",
    nb::cast<std::string>(nb::repr(py_dlpack))));

  // // Fallback: use __array_interface__ for readonly arrays (zero-copy)
  // if (nb::hasattr(py_dlpack, "__array_interface__")) {
  //   nb::dict interface =
  //     nb::cast<nb::dict>(py_dlpack.attr("__array_interface__"));

  //   // Extract data pointer from interface['data'] (tuple of (ptr, readonly))
  //   nb::tuple data_info = nb::cast<nb::tuple>(interface["data"]);
  //   uintptr_t data_ptr = nb::cast<uintptr_t>(data_info[0]);

  //   // Extract shape
  //   nb::tuple shape_tuple = nb::cast<nb::tuple>(interface["shape"]);
  //   int ndim = static_cast<int>(nb::len(shape_tuple));

  //   // Create a wrapped DLManagedTensor that keeps py_dlpack alive
  //   struct ArrayInterfaceWrapper {
  //     DLManagedTensor tensor;
  //     nb::object owner;  // Prevent cleanup until we're done
  //     std::vector<int64_t> shape_data;
  //     std::vector<int64_t> strides_data;
  //   };

  //   auto* wrapper = new ArrayInterfaceWrapper();
  //   wrapper->owner = py_dlpack;  // Keep original array alive

  //   // Parse shape
  //   wrapper->shape_data.resize(ndim);
  //   for (int i = 0; i < ndim; ++i) {
  //     wrapper->shape_data[i] = nb::cast<int64_t>(shape_tuple[i]);
  //   }

  //   // Parse strides (convert from bytes to element count)
  //   std::string typestr = nb::cast<std::string>(interface["typestr"]);
  //   int itemsize = 1;
  //   if (typestr.size() >= 3) {
  //     itemsize = std::stoi(typestr.substr(2));
  //   }

  //   wrapper->strides_data.resize(ndim);
  //   if (interface.contains("strides") && !interface["strides"].is_none()) {
  //     nb::tuple strides_tuple = nb::cast<nb::tuple>(interface["strides"]);
  //     for (int i = 0; i < ndim; ++i) {
  //       int64_t byte_stride = nb::cast<int64_t>(strides_tuple[i]);
  //       wrapper->strides_data[i] = byte_stride / itemsize;
  //     }
  //   } else {
  //     // C-contiguous default strides
  //     int64_t stride = 1;
  //     for (int i = ndim - 1; i >= 0; --i) {
  //       wrapper->strides_data[i] = stride;
  //       stride *= wrapper->shape_data[i];
  //     }
  //   }

  //   // Determine dtype from typestr (e.g., "<f4", "<i8")
  //   DLDataType dtype = {kDLFloat, 32, 1};  // Default: float32
  //   if (typestr.size() >= 3) {
  //     char kind = typestr[1];
  //     int bits = itemsize * 8;
  //     switch (kind) {
  //       case 'f':
  //         dtype = {kDLFloat, static_cast<uint8_t>(bits), 1};
  //         break;
  //       case 'i':
  //         dtype = {kDLInt, static_cast<uint8_t>(bits), 1};
  //         break;
  //       case 'u':
  //         dtype = {kDLUInt, static_cast<uint8_t>(bits), 1};
  //         break;
  //       case 'b':
  //         dtype = {kDLBool, 8, 1};
  //         break;
  //       default:
  //         break;
  //     }
  //   }

  //   // Fill DLTensor
  //   wrapper->tensor.dl_tensor.data = reinterpret_cast<void*>(data_ptr);
  //   wrapper->tensor.dl_tensor.device = {kDLCPU, 0};
  //   wrapper->tensor.dl_tensor.ndim = ndim;
  //   wrapper->tensor.dl_tensor.dtype = dtype;
  //   wrapper->tensor.dl_tensor.shape = wrapper->shape_data.data();
  //   wrapper->tensor.dl_tensor.strides = wrapper->strides_data.data();
  //   wrapper->tensor.dl_tensor.byte_offset = 0;
  //   wrapper->tensor.manager_ctx = wrapper;
  //   wrapper->tensor.deleter = [](DLManagedTensor* self) {
  //     delete static_cast<ArrayInterfaceWrapper*>(self->manager_ctx);
  //   };

  //   // Create a capsule that will call our deleter
  //   nb::capsule cap(&wrapper->tensor, "dltensor", [](void* ptr) noexcept {
  //     auto* tensor = static_cast<DLManagedTensor*>(ptr);
  //     if (tensor->deleter) {
  //       tensor->deleter(tensor);
  //     }
  //   });

  //   return {&wrapper->tensor, std::move(cap)};
  // }

  if (PyCapsule_CheckExact(py_dlpack.ptr())) {
    // Direct capsule
    return {
      static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(py_dlpack.ptr(), "dltensor")),
      std::move(py_dlpack)};
  }

  // throw std::runtime_error(
  //   "Invalid dlpack object: missing __dlpack__ and __array_interface__ "
  //   "methods");
}

ucxx::UcxBuffer DlpackToUcxBuffer(
  nb::object py_dlpack, const rpc::utils::TensorMeta& meta,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr) {
  auto [dlm, owner] = ExtractDlpackTensor(py_dlpack);
  if (!dlm) {
    throw std::runtime_error(
      "Failed to extract DLManagedTensor from dlpack object");
  }

  size_t required_size = rpc::utils::CalculateTensorSize(meta);
  void* data_ptr =
    static_cast<char*>(dlm->dl_tensor.data) + dlm->dl_tensor.byte_offset;
  ucx_memory_type_t mem_type = DlDeviceToUcxMemoryType(dlm->dl_tensor.device);

  SharedPyObject owner_guard(std::move(owner));
  return ucxx::UcxBuffer(
    mr, mem_type, data_ptr, required_size, nullptr, true,
    [owner_guard = std::move(owner_guard)](void*) {});
}

ucxx::UcxBuffer DlpackToUcxBuffer(
  nb::object py_dlpack, size_t size, ucx_memory_type_t mem_type,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr) {
  auto [dlm, owner] = ExtractDlpackTensor(py_dlpack);
  if (!dlm) {
    throw std::runtime_error(
      "Failed to extract DLManagedTensor from dlpack object");
  }

  void* data_ptr =
    static_cast<char*>(dlm->dl_tensor.data) + dlm->dl_tensor.byte_offset;

  SharedPyObject owner_guard(std::move(owner));
  return ucxx::UcxBuffer(
    mr, mem_type, data_ptr, size, nullptr, true,
    [owner_guard = std::move(owner_guard)](void*) {});
}

ucxx::UcxBufferVec DlpackToUcxBufferVec(
  nb::object py_dlpack_list, const rpc::utils::TensorMetaSpan tensor_metas,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr) {
  if (!nb::isinstance<nb::list>(py_dlpack_list)) {
    throw std::runtime_error("Expected list of dlpack objects");
  }

  nb::list py_list = nb::cast<nb::list>(py_dlpack_list);
  if (nb::len(py_list) != tensor_metas.size()) {
    throw std::runtime_error(
      "Mismatch between dlpack list size and tensor_metas size");
  }

  // Calculate sizes for each tensor
  std::vector<size_t> sizes;
  sizes.reserve(tensor_metas.size());
  for (const auto& meta : tensor_metas) {
    sizes.push_back(rpc::utils::CalculateTensorSize(meta));
  }

  // Determine memory type (assume all tensors have same device type)
  ucx_memory_type_t mem_type = ucx_memory_type::HOST;
  if (!tensor_metas.empty()) {
    mem_type = DlDeviceToUcxMemoryType(tensor_metas[0].device);
  }

  // Create ucx_buffer_t vector pointing directly to dlpack memory (no copy)
  std::vector<ucx_buffer_t> buffers;
  buffers.reserve(tensor_metas.size());
  std::vector<SharedPyObject> owners;
  owners.reserve(tensor_metas.size());

  for (size_t i = 0; i < tensor_metas.size(); ++i) {
    auto [dlm, owner] = ExtractDlpackTensor(py_list[i]);
    if (!dlm) {
      throw std::runtime_error(
        "Failed to extract DLManagedTensor from dlpack object");
    }
    owners.emplace_back(std::move(owner));
    void* data_ptr =
      static_cast<char*>(dlm->dl_tensor.data) + dlm->dl_tensor.byte_offset;
    // Create ucx_buffer_t pointing directly to dlpack memory (no copy)

    buffers.push_back({data_ptr, sizes[i]});
  }

  // Create UcxBufferVec using external memory pointers with custom deleter
  // that holds the owners list.
  return ucxx::UcxBufferVec(
    mr, mem_type, buffers, nullptr, true,
    [owners = std::move(owners)](void*) {});
}

ucxx::UcxBufferVec DlpackToUcxBufferVec(
  nb::object py_dlpack_list, std::span<const size_t> sizes,
  ucx_memory_type_t mem_type,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr) {
  if (!nb::isinstance<nb::list>(py_dlpack_list)) {
    throw std::runtime_error("Expected list of dlpack objects");
  }

  nb::list py_list = nb::cast<nb::list>(py_dlpack_list);
  if (nb::len(py_list) != sizes.size()) {
    throw std::runtime_error(
      "Mismatch between dlpack list size and sizes size");
  }

  std::vector<ucx_buffer_t> buffers;
  buffers.reserve(sizes.size());
  std::vector<SharedPyObject> owners;
  owners.reserve(sizes.size());

  for (size_t i = 0; i < sizes.size(); ++i) {
    auto [dlm, owner] = ExtractDlpackTensor(py_list[i]);
    if (!dlm) {
      throw std::runtime_error(
        "Failed to extract DLManagedTensor from dlpack object");
    }
    owners.emplace_back(std::move(owner));
    void* data_ptr =
      static_cast<char*>(dlm->dl_tensor.data) + dlm->dl_tensor.byte_offset;
    buffers.push_back({data_ptr, sizes[i]});
  }

  return ucxx::UcxBufferVec(
    mr, mem_type, buffers, nullptr, true,
    [owners = std::move(owners)](void*) {});
}

// Context struct for TensorMetaToDlpack ownership management
struct TensorDlpackContext {
  std::unique_ptr<axon::utils::TensorBase> tensor;
  std::unique_ptr<ucxx::UcxBuffer> buffer;

  TensorDlpackContext(
    std::unique_ptr<axon::utils::TensorBase> t,
    std::unique_ptr<ucxx::UcxBuffer> b)
    : tensor(std::move(t)), buffer(std::move(b)) {}
};

nb::object TensorMetaToDlpack(
  rpc::utils::TensorMeta&& meta, ucxx::UcxBuffer&& buffer) {
  // Create TensorBase from TensorMeta and UcxBuffer
  auto tensor = std::make_unique<axon::utils::TensorBase>();
  tensor->assign(std::move(meta), buffer.data());

  DLDevice device = tensor->device;

  // Create DLManagedTensor for Python
  DLManagedTensor* dlm_tensor = new DLManagedTensor;
  dlm_tensor->dl_tensor.data = tensor->data;
  dlm_tensor->dl_tensor.device = tensor->device;
  dlm_tensor->dl_tensor.ndim = tensor->ndim;
  dlm_tensor->dl_tensor.dtype = tensor->dtype;
  dlm_tensor->dl_tensor.shape = tensor->shape;
  dlm_tensor->dl_tensor.strides = tensor->strides;
  dlm_tensor->dl_tensor.byte_offset = tensor->byte_offset;

  // Use unique_ptr-based context for ownership management
  // IMPORTANT: Set own_buffer = false to prevent UcxBuffer::~UcxBuffer
  // from calling deallocate() on UcxMemoryResourceManager, which may have been
  // destroyed before this DLPack capsule (lifetime is controlled by Python GC).
  auto owned_buffer =
    std::make_unique<ucxx::UcxBuffer>(std::move(buffer), false);

  dlm_tensor->manager_ctx =
    new TensorDlpackContext(std::move(tensor), std::move(owned_buffer));
  dlm_tensor->deleter = [](DLManagedTensor* self) {
    delete static_cast<TensorDlpackContext*>(self->manager_ctx);
    delete self;
  };

  // Create PyCapsule with safe destructor that handles consumed capsules
  nb::object capsule = nb::steal<nb::object>(
    PyCapsule_New(dlm_tensor, "dltensor", SafeDlpackCapsuleDestructor));

  // Return wrapped in DLPackCapsuleWrapper for from_dlpack compatibility
  return nb::cast(DLPackCapsuleWrapper(std::move(capsule), device));
}

// Context struct for TensorMetaVecToDlpack ownership management
// Uses shared_ptr for UcxBufferVec because:
// 1. UcxBufferVec may have a single backing_buffer_ that's a contiguous
// allocation
// 2. Individual buffers are views into the backing buffer, can't be freed
// independently
// 3. All DLPack objects must keep the backing memory alive until ALL are done
struct TensorVecDlpackContext {
  std::unique_ptr<axon::utils::TensorBase> tensor;
  std::shared_ptr<ucxx::UcxBufferVec> buffer_vec;

  TensorVecDlpackContext(
    std::unique_ptr<axon::utils::TensorBase> t,
    std::shared_ptr<ucxx::UcxBufferVec> bv)
    : tensor(std::move(t)), buffer_vec(std::move(bv)) {}
};

nb::list TensorMetaVecToDlpack(
  cista::offset::vector<rpc::utils::TensorMeta>&& meta_vec,
  ucxx::UcxBufferVec&& buffer_vec) {
  nb::list result;
  if (meta_vec.size() != buffer_vec.size()) {
    throw std::runtime_error("TensorMetaVec and UcxBufferVec size mismatch");
  }

  // Use shared_ptr to allow multiple capsules to share the same buffer_vec
  // IMPORTANT: Set own_buffer = false to prevent UcxBufferVec::~UcxBufferVec
  // from calling deallocate() on UcxMemoryResourceManager, which may have been
  // destroyed before this DLPack capsule (lifetime is controlled by Python GC).
  // The actual buffer memory will be freed when the unifex sender chain
  // completes.
  auto shared_buffer_vec =
    std::make_shared<ucxx::UcxBufferVec>(std::move(buffer_vec), false);

  // Get buffers from UcxBufferVec
  const auto& buffers = shared_buffer_vec->buffers();
  for (size_t i = 0; i < meta_vec.size(); ++i) {
    auto& meta = meta_vec[i];
    const auto& buffer = buffers[i];

    // Create TensorBase from TensorMeta and buffer data
    auto tensor = std::make_unique<axon::utils::TensorBase>();
    tensor->assign(std::move(meta), buffer.data);

    // Save device before moving tensor (for DLPackCapsuleWrapper)
    DLDevice device = tensor->device;

    // Create DLManagedTensor for Python (zero-copy)
    DLManagedTensor* dlm_tensor = new DLManagedTensor;
    dlm_tensor->dl_tensor.data = tensor->data;
    dlm_tensor->dl_tensor.device = device;
    dlm_tensor->dl_tensor.ndim = tensor->ndim;
    dlm_tensor->dl_tensor.dtype = tensor->dtype;
    dlm_tensor->dl_tensor.shape = tensor->shape;
    dlm_tensor->dl_tensor.strides = tensor->strides;
    dlm_tensor->dl_tensor.byte_offset = tensor->byte_offset;

    // Use named context struct for ownership management
    dlm_tensor->manager_ctx =
      new TensorVecDlpackContext(std::move(tensor), shared_buffer_vec);
    dlm_tensor->deleter = [](DLManagedTensor* self) {
      delete static_cast<TensorVecDlpackContext*>(self->manager_ctx);
      delete self;
    };

    // Wrap in DLPackCapsuleWrapper for from_dlpack compatibility
    nb::object capsule = nb::steal<nb::object>(
      PyCapsule_New(dlm_tensor, "dltensor", SafeDlpackCapsuleDestructor));
    result.append(nb::cast(DLPackCapsuleWrapper(std::move(capsule), device)));
  }
  return result;
}

nb::object UcxBufferToDLTensor(const ucxx::UcxBuffer& buffer) {
  DLManagedTensor* dlm_tensor = new DLManagedTensor;
  dlm_tensor->dl_tensor.data = const_cast<void*>(buffer.data());
  // Derive device from UcxBuffer's memory type
  dlm_tensor->dl_tensor.device = UcxMemoryTypeToDlDevice(buffer.type());
  dlm_tensor->dl_tensor.ndim = 1;
  dlm_tensor->dl_tensor.dtype = {kDLUInt, 8, 1};
  dlm_tensor->dl_tensor.shape =
    new int64_t[1]{static_cast<int64_t>(buffer.size())};
  dlm_tensor->dl_tensor.strides = nullptr;
  dlm_tensor->dl_tensor.byte_offset = 0;
  dlm_tensor->manager_ctx = const_cast<ucxx::UcxBuffer*>(&buffer);
  dlm_tensor->deleter = [](DLManagedTensor* self) {
    // UcxBuffer owns the memory, so we don't delete it here
    delete[] self->dl_tensor.shape;
    delete self;
  };
  // Return wrapped in DLPackCapsuleWrapper with safe destructor
  return nb::cast(DLPackCapsuleWrapper(
    nb::steal<nb::object>(
      PyCapsule_New(dlm_tensor, "dltensor", SafeDlpackCapsuleDestructor)),
    UcxMemoryTypeToDlDevice(buffer.type())));
}

// Context struct for UcxBufferToDLTensor ownership management (rvalue version)
struct BufferDlpackContext {
  std::unique_ptr<ucxx::UcxBuffer> buffer;
  std::unique_ptr<int64_t[]> shape;

  BufferDlpackContext(
    std::unique_ptr<ucxx::UcxBuffer> b, std::unique_ptr<int64_t[]> s)
    : buffer(std::move(b)), shape(std::move(s)) {}
};

nb::object UcxBufferToDLTensor(ucxx::UcxBuffer&& buffer) {
  // Create unique_ptr for ownership transfer
  auto owned_buffer = std::make_unique<ucxx::UcxBuffer>(std::move(buffer));
  auto shape = std::make_unique<int64_t[]>(1);
  shape[0] = static_cast<int64_t>(owned_buffer->size());

  DLManagedTensor* dlm_tensor = new DLManagedTensor;
  dlm_tensor->dl_tensor.data = owned_buffer->data();
  // Derive device from UcxBuffer's memory type
  dlm_tensor->dl_tensor.device = UcxMemoryTypeToDlDevice(owned_buffer->type());
  dlm_tensor->dl_tensor.ndim = 1;
  dlm_tensor->dl_tensor.dtype = {kDLUInt, 8, 1};
  dlm_tensor->dl_tensor.shape = shape.get();
  dlm_tensor->dl_tensor.strides = nullptr;
  dlm_tensor->dl_tensor.byte_offset = 0;

  // Use unique_ptr-based context for ownership management
  dlm_tensor->manager_ctx =
    new BufferDlpackContext(std::move(owned_buffer), std::move(shape));
  dlm_tensor->deleter = [](DLManagedTensor* self) {
    delete static_cast<BufferDlpackContext*>(self->manager_ctx);
    delete self;
  };
  // Return wrapped in DLPackCapsuleWrapper with safe destructor
  return nb::cast(DLPackCapsuleWrapper(
    nb::steal<nb::object>(
      PyCapsule_New(dlm_tensor, "dltensor", SafeDlpackCapsuleDestructor)),
    dlm_tensor->dl_tensor.device));
}

// Context struct for UcxBufferVecToDLTensor ownership management
// Uses shared_ptr for UcxBufferVec because:
// 1. UcxBufferVec may have a single backing_buffer_ that's a contiguous
// allocation
// 2. Individual buffers are views into the backing buffer, can't be freed
// independently
// 3. All DLPack objects must keep the backing memory alive until ALL are done
struct BufferVecDlpackContext {
  std::shared_ptr<ucxx::UcxBufferVec> buffer_vec;
  std::unique_ptr<int64_t[]> shape;

  BufferVecDlpackContext(
    std::shared_ptr<ucxx::UcxBufferVec> bv, std::unique_ptr<int64_t[]> s)
    : buffer_vec(std::move(bv)), shape(std::move(s)) {}
};

nb::list UcxBufferVecToDLTensor(ucxx::UcxBufferVec&& buffer_vec) {
  nb::list result;
  if (buffer_vec.size() == 0) {
    return result;
  }

  // Use shared_ptr to allow multiple capsules to share the same buffer_vec
  auto shared_buffer_vec =
    std::make_shared<ucxx::UcxBufferVec>(std::move(buffer_vec));

  const auto& buffers = shared_buffer_vec->buffers();
  ucx_memory_type_t mem_type = shared_buffer_vec->type();
  DLDevice dl_device = UcxMemoryTypeToDlDevice(mem_type);

  for (size_t i = 0; i < buffers.size(); ++i) {
    const auto& buffer = buffers[i];

    // Create unique_ptr for shape
    auto shape = std::make_unique<int64_t[]>(1);
    shape[0] = static_cast<int64_t>(buffer.size);

    DLManagedTensor* dlm_tensor = new DLManagedTensor;
    dlm_tensor->dl_tensor.data = buffer.data;
    // Derive device from UcxBufferVec's memory type
    dlm_tensor->dl_tensor.device = dl_device;
    dlm_tensor->dl_tensor.ndim = 1;
    dlm_tensor->dl_tensor.dtype = {kDLUInt, 8, 1};
    dlm_tensor->dl_tensor.shape = shape.get();
    dlm_tensor->dl_tensor.strides = nullptr;
    dlm_tensor->dl_tensor.byte_offset = 0;

    // Use named context struct for ownership management
    dlm_tensor->manager_ctx =
      new BufferVecDlpackContext(shared_buffer_vec, std::move(shape));
    dlm_tensor->deleter = [](DLManagedTensor* self) {
      delete static_cast<BufferVecDlpackContext*>(self->manager_ctx);
      delete self;
    };
    // Append wrapped in DLPackCapsuleWrapper with safe destructor
    result.append(nb::cast(DLPackCapsuleWrapper(
      nb::steal<nb::object>(
        PyCapsule_New(dlm_tensor, "dltensor", SafeDlpackCapsuleDestructor)),
      dl_device)));
  }
  return result;
}

ucxx::UcxBuffer CreateUcxBufferFromPayload(
  nb::object payload_obj,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr) {
  // payload_obj should be a dlpack object (UcxBuffer/UcxBufferVec are only
  // exposed through dlpack in Python)
  auto [dlm, owner] = ExtractDlpackTensor(payload_obj);
  if (!dlm) {
    throw std::runtime_error(
      "Failed to extract DLManagedTensor from dlpack object");
  }
  void* data_ptr =
    static_cast<char*>(dlm->dl_tensor.data) + dlm->dl_tensor.byte_offset;
  // Calculate size from shape
  size_t size = 1;
  for (int i = 0; i < dlm->dl_tensor.ndim; ++i) {
    size *= dlm->dl_tensor.shape[i];
  }
  // Calculate bytes from dtype
  size_t dtype_bytes = (dlm->dl_tensor.dtype.bits + 7) / 8;
  size *= dtype_bytes;
  ucx_memory_type_t mem_type = DlDeviceToUcxMemoryType(dlm->dl_tensor.device);
  // Create UcxBuffer with deleter keeping owner alive
  SharedPyObject owner_guard(std::move(owner));
  return ucxx::UcxBuffer(
    mr, mem_type, data_ptr, size, nullptr, true,
    [owner_guard = std::move(owner_guard)](void*) {});
}

ucxx::UcxBufferVec CreateUcxBufferVecFromPayload(
  nb::object payload_obj,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr) {
  // payload_obj should be a list of dlpack objects
  if (!nb::isinstance<nb::list>(payload_obj)) {
    throw std::runtime_error(
      "UcxBufferVec payload must be a list of dlpack objects");
  }

  nb::list py_list = nb::cast<nb::list>(payload_obj);
  size_t list_size = nb::len(py_list);

  // Calculate sizes for each tensor from dlpack objects
  std::vector<size_t> sizes;
  sizes.reserve(list_size);
  ucx_memory_type_t mem_type = ucx_memory_type::HOST;
  bool mem_type_set = false;

  // Create ucx_buffer_t vector pointing directly to dlpack memory (no copy)
  std::vector<ucx_buffer_t> buffers;
  buffers.reserve(list_size);
  std::vector<SharedPyObject> owners;
  owners.reserve(list_size);

  for (size_t i = 0; i < list_size; ++i) {
    auto [dlm, owner] = ExtractDlpackTensor(py_list[i]);
    if (!dlm) {
      throw std::runtime_error(
        "Failed to extract DLManagedTensor from dlpack object");
    }
    owners.emplace_back(std::move(owner));

    // Calculate size from shape and dtype
    size_t size = 1;
    for (int j = 0; j < dlm->dl_tensor.ndim; ++j) {
      size *= dlm->dl_tensor.shape[j];
    }
    size_t dtype_bytes = (dlm->dl_tensor.dtype.bits + 7) / 8;
    size *= dtype_bytes;
    sizes.push_back(size);

    // Determine memory type from first tensor
    if (!mem_type_set) {
      mem_type = DlDeviceToUcxMemoryType(dlm->dl_tensor.device);
      mem_type_set = true;
    }

    void* data_ptr =
      static_cast<char*>(dlm->dl_tensor.data) + dlm->dl_tensor.byte_offset;
    // Create ucx_buffer_t pointing directly to dlpack memory (no copy)
    buffers.push_back({data_ptr, size});
  }

  // Create UcxBufferVec with deleter keeping owners alive
  return ucxx::UcxBufferVec(
    mr, mem_type, buffers, nullptr, true,
    [owners = std::move(owners)](void*) {});
}

rpc::utils::TensorMeta ExtractTensorMetaFromDlm(DLManagedTensor* dlm) {
  if (!dlm) {
    throw std::runtime_error(
      "ExtractTensorMetaFromDlm: DLManagedTensor pointer is null");
  }

  rpc::utils::TensorMeta meta;
  meta.device = dlm->dl_tensor.device;
  meta.dtype = dlm->dl_tensor.dtype;
  meta.ndim = dlm->dl_tensor.ndim;

  meta.shape.clear();
  meta.shape.reserve(dlm->dl_tensor.ndim);
  for (int i = 0; i < dlm->dl_tensor.ndim; ++i) {
    meta.shape.push_back(dlm->dl_tensor.shape[i]);
  }

  if (dlm->dl_tensor.strides != nullptr) {
    meta.strides.clear();
    meta.strides.reserve(dlm->dl_tensor.ndim);
    for (int i = 0; i < dlm->dl_tensor.ndim; ++i) {
      meta.strides.push_back(dlm->dl_tensor.strides[i]);
    }
  }

  meta.byte_offset = dlm->dl_tensor.byte_offset;

  return meta;
}

ucxx::UcxBuffer DlpackToUcxBufferFromDlm(
  DLManagedTensor* dlm, nb::object owner, const rpc::utils::TensorMeta& meta,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr) {
  if (!dlm) {
    throw std::runtime_error(
      "DlpackToUcxBufferFromDlm: DLManagedTensor pointer is null");
  }

  size_t required_size = rpc::utils::CalculateTensorSize(meta);
  void* data_ptr =
    static_cast<char*>(dlm->dl_tensor.data) + dlm->dl_tensor.byte_offset;
  ucx_memory_type_t mem_type = DlDeviceToUcxMemoryType(dlm->dl_tensor.device);

  SharedPyObject owner_guard(std::move(owner));
  return ucxx::UcxBuffer(
    mr, mem_type, data_ptr, required_size, nullptr, true,
    [owner_guard = std::move(owner_guard)](void*) {});
}

ucxx::UcxBufferVec DlpackToUcxBufferVecFromDlm(
  std::span<DLManagedTensor*> dlms, std::span<nb::object> owners,
  const rpc::utils::TensorMetaSpan tensor_metas,
  std::reference_wrapper<ucxx::UcxMemoryResourceManager> mr) {
  if (dlms.size() != owners.size() || dlms.size() != tensor_metas.size()) {
    throw std::runtime_error(
      "DlpackToUcxBufferVecFromDlm: size mismatch between dlms, owners, and "
      "tensor_metas");
  }

  if (dlms.empty()) {
    return ucxx::UcxBufferVec(mr, ucx_memory_type::HOST, std::vector<size_t>{});
  }

  // Determine memory type from first tensor
  ucx_memory_type_t mem_type =
    DlDeviceToUcxMemoryType(dlms[0]->dl_tensor.device);

  std::vector<ucx_buffer_t> buffers;
  buffers.reserve(dlms.size());
  std::vector<SharedPyObject> owner_guards;
  owner_guards.reserve(owners.size());

  for (size_t i = 0; i < dlms.size(); ++i) {
    if (!dlms[i]) {
      throw std::runtime_error(
        "DlpackToUcxBufferVecFromDlm: DLManagedTensor pointer is null at "
        "index "
        + std::to_string(i));
    }

    size_t size = rpc::utils::CalculateTensorSize(tensor_metas[i]);
    void* data_ptr = static_cast<char*>(dlms[i]->dl_tensor.data)
                     + dlms[i]->dl_tensor.byte_offset;

    buffers.push_back({data_ptr, size});
    owner_guards.emplace_back(std::move(owners[i]));
  }

  return ucxx::UcxBufferVec(
    mr, mem_type, buffers, nullptr, true,
    [owner_guards = std::move(owner_guards)](void*) {});
}

rpc::utils::TensorMeta ExtractTensorMetaFromDlpack(nb::object py_dlpack) {
  auto [dlm, _] = ExtractDlpackTensor(py_dlpack);
  return ExtractTensorMetaFromDlm(dlm);
}

bool IsDlpackTensor(nb::object py_obj) {
  nb::gil_scoped_acquire acquire;
  // Check if object has __dlpack__ method
  return nb::hasattr(py_obj, "__dlpack__");
}

}  // namespace python
}  // namespace axon
}  // namespace eux
