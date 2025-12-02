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

#ifndef AXON_CORE_UTILS_TENSOR_HPP_
#define AXON_CORE_UTILS_TENSOR_HPP_

#include <cista.h>
#include <dlpack/dlpack.h>  // from @dlpack

#include <memory>
#include <utility>
#if __STDCPP_FLOAT16_T__ == 1 && __STDCPP_BFLOAT16_T__ == 1
#include <stdfloat>
#endif

#include "rpc_core/utils/tensor_meta.hpp"

namespace eux {
namespace axon {
namespace utils {

using eux::rpc::utils::TensorMeta;

// A base class that encapsulates the ownership of tensor metadata to ensure
// lifetime safety. It can operate in two modes:
// 1. Owning mode: Manages the lifetime of metadata via `metadata_owner_`.
// 2. View mode: Acts as a non-owning view of an external DLTensor.
struct TensorBase : public DLTensor {
  // If this tensor owns its metadata, this pointer will be valid.
  std::unique_ptr<TensorMeta> metadata_owner_;

  // Default constructor: creates an empty, non-owning tensor view. Zero-cost.
  TensorBase() noexcept {
    this->data = nullptr;
    this->device = {kDLCPU, 0};
    this->ndim = 0;
    this->dtype = {kDLFloat, 32, 1};
    this->shape = nullptr;
    this->strides = nullptr;
    this->byte_offset = 0;
  }

  // Non-owning view constructor. No allocations, high performance.
  explicit TensorBase(const DLTensor& view_source) noexcept {
    static_cast<DLTensor&>(*this) = view_source;
  }

  // Copy constructor: deep-copies if owning, shallow-copies if a view.
  TensorBase(const TensorBase& other) {
    static_cast<DLTensor&>(*this) = other;
    if (other.metadata_owner_) {
      metadata_owner_ = std::make_unique<TensorMeta>(*other.metadata_owner_);
      rewire_pointers_to_owner();
    }
  }

  // Move constructor: ownership is transferred.
  TensorBase(TensorBase&& other) noexcept : DLTensor(other) {
    metadata_owner_ = std::move(other.metadata_owner_);
    other.data = nullptr;
    other.shape = nullptr;
    other.strides = nullptr;
  }

  // Copy assignment
  TensorBase& operator=(const TensorBase& other) {
    if (this != &other) {
      static_cast<DLTensor&>(*this) = other;
      if (other.metadata_owner_) {
        metadata_owner_ = std::make_unique<TensorMeta>(*other.metadata_owner_);
        rewire_pointers_to_owner();
      } else {
        metadata_owner_.reset();
      }
    }
    return *this;
  }

  // Move assignment
  TensorBase& operator=(TensorBase&& other) noexcept {
    if (this != &other) {
      static_cast<DLTensor&>(*this) = other;
      metadata_owner_ = std::move(other.metadata_owner_);
      other.data = nullptr;
      other.shape = nullptr;
      other.strides = nullptr;
    }
    return *this;
  }

  virtual ~TensorBase() = default;

  // Reconstructs this tensor from deserialized metadata, taking ownership.
  void assign(TensorMeta&& meta, void* data_ptr) {
    if (
      meta.ndim != 0
      && (meta.shape.empty() || (meta.strides.empty() && meta.ndim > 1))) {
      throw std::runtime_error(
        "Invalid metadata: shape or strides are empty for non-zero "
        "dimensions.");
    }
    metadata_owner_ = std::make_unique<TensorMeta>(std::move(meta));
    this->data = data_ptr;
    this->device = metadata_owner_->device;
    this->ndim = metadata_owner_->ndim;
    this->dtype = metadata_owner_->dtype;
    this->byte_offset = metadata_owner_->byte_offset;
    rewire_pointers_to_owner();
  }

  void assign_owned(const DLTensor& source) {
    // Validate the source DLTensor.
    if (source.ndim > 0 && source.shape == nullptr) {
      throw std::runtime_error(
        "Invalid source DLTensor: shape is null for non-zero dimensions.");
    }
    if (source.ndim > 1 && source.strides == nullptr) {
      throw std::runtime_error(
        "Invalid source DLTensor: strides is null for dimensions > 1.");
    }

    // Copy the non-pointer metadata fields.
    this->data = source.data;
    this->device = source.device;
    this->ndim = source.ndim;
    this->dtype = source.dtype;
    this->byte_offset = source.byte_offset;

    // Deep copy shape and strides into our owned vectors.
    metadata_owner_ = std::make_unique<TensorMeta>(source);

    // Rewire internal pointers to our owned metadata.
    rewire_pointers_to_owner();
  }

  // Makes this tensor a non-owning view of an external DLTensor. Zero-cost.
  void assign_view(const DLTensor& source) noexcept {
    metadata_owner_.reset();
    static_cast<DLTensor&>(*this) = source;
  }

  // Returns a pointer to the underlying DLTensor object.
  DLTensor* dltensor() noexcept { return this; }

  // Returns a const pointer to the underlying DLTensor object.
  const DLTensor* dltensor() const noexcept { return this; }

 private:
  // Ensures the DLTensor's pointers always point to the internal owned
  // metadata.
  void rewire_pointers_to_owner() noexcept {
    // This should only be called when metadata_owner_ is valid.
    this->shape =
      metadata_owner_->shape.empty() ? nullptr : metadata_owner_->shape.data();
    this->strides = metadata_owner_->strides.empty()
                      ? nullptr
                      : metadata_owner_->strides.data();
  }
};

template <typename DType = std::nullptr_t>
struct Tensor : TensorBase {};

template <typename DType>
  requires std::integral<DType> && std::is_signed_v<DType>
struct Tensor<DType> : TensorBase {
  static constexpr DLDataType dtype{
    DLDataTypeCode::kDLInt, sizeof(DType) * 8, 1};
  DType std_dtype;
};

template <typename DType>
concept ValidUintType = std::integral<DType> && std::is_unsigned_v<DType>
                        && !std::same_as<DType, bool>;

template <typename DType>
  requires ValidUintType<DType>
struct Tensor<DType> : TensorBase {
  static constexpr DLDataType dtype{
    DLDataTypeCode::kDLUInt, sizeof(DType) * 8, 1};
  DType std_dtype;
};

template <typename DType>
  requires std::floating_point<DType>
struct Tensor<DType> : TensorBase {
  static constexpr DLDataType dtype{
    DLDataTypeCode::kDLFloat, sizeof(DType) * 8, 1};
  DType std_dtype;
};

#if __STDCPP_FLOAT16_T__ == 1
template <typename DType>
  requires std::floating_point<DType> && std::same_as<DType, std::float16_t>
struct Tensor<DType> : TensorBase {
  static constexpr DLDataType dtype{
    DLDataTypeCode::kDLFloat, sizeof(DType) * 8, 1};
  DType std_dtype;
};
#endif

#if __STDCPP_BFLOAT16_T__ == 1
template <typename DType>
  requires std::floating_point<DType> && std::same_as<DType, std::bfloat16_t>
struct Tensor<DType> : TensorBase {
  static constexpr DLDataType dtype{
    DLDataTypeCode::kDLBfloat, sizeof(DType) * 8, 1};
  DType std_dtype;
};
#endif

template <typename DType>
  requires std::same_as<DType, bool>
struct Tensor<DType> : TensorBase {
  static constexpr DLDataType dtype{
    DLDataTypeCode::kDLBool, sizeof(DType) * 8, 1};
  DType std_dtype;
};

}  // namespace utils
}  // namespace axon
}  // namespace eux

#endif  // AXON_CORE_UTILS_TENSOR_HPP_
