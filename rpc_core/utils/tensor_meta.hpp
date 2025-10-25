/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.
 *
 *Licensed under the Apache License Version 2.0 with LLVM Exceptions
 *(the "License"); you may not use this file except in compliance with
 *the License. You may obtain a copy of the License at
 *
 *    https://llvm.org/LICENSE.txt
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 *==============================================================================*/

#pragma once

#ifndef RPC_CORE_UTILS_TENSOR_META_HPP_
#define RPC_CORE_UTILS_TENSOR_META_HPP_

#include <cista.h>
#include <dlpack/dlpack.h>  // from @dlpack

#include <cstdint>

namespace eux {
namespace rpc {
namespace utils {
// A Cista-serializable representation of DLTensor's metadata.
// It captures all information from a DLTensor except for the raw data pointer.
struct TensorMeta {
  auto cista_members() {
    return std::tie(device, ndim, dtype, byte_offset, shape, strides);
  }

  // Default constructor
  TensorMeta() = default;

  // Constructor from an existing DLTensor.
  // Performs a deep copy of the metadata (shape and strides) to ensure safety.
  explicit TensorMeta(const DLTensor& source)
    : device(source.device),
      ndim(source.ndim),
      dtype(source.dtype),
      byte_offset(source.byte_offset) {
    if (source.ndim > 0) {
      if (source.shape == nullptr) {
        throw std::runtime_error(
          "Cannot construct TensorMeta: source DLTensor shape is null for "
          "non-zero dimensions.");
      }
      shape.resize(source.ndim);
      for (int i = 0; i < source.ndim; ++i) {
        shape[i] = source.shape[i];
      }

      if (source.strides != nullptr) {
        strides.resize(source.ndim);
        for (int i = 0; i < source.ndim; ++i) {
          strides[i] = source.strides[i];
        }
      } else {
        // DLPack spec allows strides to be null for contiguous 1D tensors.
        if (source.ndim == 1) {
          strides.resize(1);
          strides[0] = 1;
        } else {
          throw std::runtime_error(
            "Cannot construct TensorMeta: source DLTensor strides is null "
            "for dimensions > 1.");
        }
      }
    }
  }

  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  uint64_t byte_offset;
  cista::offset::vector<int64_t> shape;
  cista::offset::vector<int64_t> strides;
};

}  // namespace utils
}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_UTILS_TENSOR_META_HPP_
