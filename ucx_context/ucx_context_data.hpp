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

#ifndef UCX_CONTEXT_UCX_CONTEXT_DATA_HPP_
#define UCX_CONTEXT_UCX_CONTEXT_DATA_HPP_

#include <atomic>
#include <utility>
#include <vector>

#include "ucx_context/ucx_context_def.h"
#include "ucx_context/ucx_memory_resource.hpp"

namespace eux {
namespace ucxx {

/**
 * @class UcxBuffer
 * @brief RAII wrapper for a ucx_buffer_t that manages its memory.
 *
 * This class allocates and deallocates memory for a ucx_buffer_t's data
 * pointer using a provided UcxMemoryResourceManager.
 */
class UcxBuffer {
  friend class UcxBufferVec;

 public:
  /**
   * @brief Constructs a UcxBuffer.
   * @param mr The memory resource manager to use for allocations.
   * @param type The memory type of the buffer.
   * @param size The size of the buffer in bytes.
   * @param own_buffer Whether to own the buffer.
   */
  UcxBuffer(
    UcxMemoryResourceManager& mr, ucx_memory_type_t type, size_t size,
    void* mem_h = nullptr, bool own_buffer = true)
    : mr_(mr),
      type_(type),
      buffer_{nullptr, 0},
      mem_h_(mem_h),
      own_buffer_(own_buffer) {
    if (size > 0) {
      buffer_.data = mr_.get().allocate(type_, size);
      buffer_.size = size;
    }
  }

  UcxBuffer(
    UcxMemoryResourceManager& mr, ucx_memory_type_t type, ucx_buffer_t&& buffer,
    void* mem_h = nullptr, bool own_buffer = true)
    : mr_(mr),
      type_(type),
      buffer_(std::move(buffer)),
      mem_h_(mem_h),
      own_buffer_(own_buffer) {}

  UcxBuffer(
    UcxMemoryResourceManager& mr, ucx_memory_type_t type,
    const ucx_buffer_t& buffer, void* mem_h = nullptr, bool own_buffer = false)
    : mr_(mr),
      type_(type),
      buffer_(buffer),
      mem_h_(mem_h),
      own_buffer_(own_buffer) {}

  UcxBuffer(
    UcxMemoryResourceManager& mr, ucx_memory_type_t type, const void* buffer,
    size_t size, void* mem_h = nullptr, bool own_buffer = false)
    : mr_(mr),
      type_(type),
      buffer_({const_cast<void*>(buffer), size}),
      mem_h_(mem_h),
      own_buffer_(own_buffer) {}

  UcxBuffer(UcxBuffer&& other, bool own_buffer)
    : mr_(other.mr_),
      type_(other.type_),
      buffer_(other.buffer_),
      mem_h_(other.mem_h_),
      own_buffer_(own_buffer) {
    other.buffer_ = {nullptr, 0};
    other.own_buffer_.store(false, std::memory_order_relaxed);
  }

  /**
   * @brief Destroys the UcxBuffer, deallocating its memory.
   */
  ~UcxBuffer() {
    if (buffer_.data && own_buffer_) {
      mr_.get().deallocate(type_, buffer_.data, buffer_.size);
    }
  }

  UcxBuffer(const UcxBuffer&) = delete;
  UcxBuffer& operator=(const UcxBuffer&) = delete;

  UcxBuffer(UcxBuffer&& other) noexcept
    : mr_(other.mr_),
      type_(other.type_),
      buffer_(other.buffer_),
      mem_h_(other.mem_h_),
      own_buffer_(other.own_buffer_.load(std::memory_order_relaxed)) {
    other.buffer_ = {nullptr, 0};
    other.own_buffer_.store(false, std::memory_order_relaxed);
  }

  UcxBuffer& operator=(UcxBuffer&& other) noexcept {
    if (this != &other) {
      if (buffer_.data && own_buffer_) {
        mr_.get().deallocate(type_, buffer_.data, buffer_.size);
      }
      mr_ = other.mr_;
      type_ = other.type_;
      buffer_ = other.buffer_;
      mem_h_ = other.mem_h_;
      own_buffer_ = other.own_buffer_.load(std::memory_order_relaxed);
      other.buffer_ = {nullptr, 0};
      other.own_buffer_.store(false, std::memory_order_relaxed);
    }
    return *this;
  }

  /**
   * @brief Gets a pointer to the underlying ucx_buffer_t.
   * @return A pointer to the ucx_buffer_t.
   */
  ucx_buffer_t* get() { return &buffer_; }
  /**
   * @brief Gets a const pointer to the underlying ucx_buffer_t.
   * @return A const pointer to the ucx_buffer_t.
   */
  const ucx_buffer_t* get() const { return &buffer_; }

  /**
   * @brief Gets the data pointer.
   */
  void* data() { return buffer_.data; }
  /**
   * @brief Gets the const data pointer.
   */
  const void* data() const { return buffer_.data; }
  /**
   * @brief Gets the size of the buffer.
   */
  size_t size() const { return buffer_.size; }
  /**
   * @brief Gets the memory type of the buffer.
   */
  ucx_memory_type_t type() const { return type_; }
  /**
   * @brief Gets the memory handle of the buffer.
   */
  void* mem_h() const { return mem_h_; }
  /**
   * @brief Gets whether the buffer is owned by the UcxBuffer.
   */
  bool own_buffer() const { return own_buffer_; }

 private:
  std::reference_wrapper<UcxMemoryResourceManager> mr_;
  ucx_memory_type_t type_;
  ucx_buffer_t buffer_;
  void* mem_h_;
  std::atomic<bool> own_buffer_;
};

using UcxHeader = UcxBuffer;

/**
 * @class UcxBufferVec
 * @brief RAII wrapper for a vector of ucx_buffer_t that manages its memory.
 *
 * This class allocates and deallocates memory for the buffers of a
 * ucx_buffer_t using a provided UcxMemoryResourceManager.
 */
class UcxBufferVec {
  friend class UcxBuffer;

 public:
  /**
   * @brief Constructs a UcxBufferVec.
   * @param mr The memory resource manager.
   * @param type The memory type of the buffers.
   * @param sizes The sizes of the buffers.
   * @param mem_h The memory handle of the buffers.
   * @param own_buffer Whether to own the buffers.
   */
  UcxBufferVec(
    UcxMemoryResourceManager& mr, ucx_memory_type_t type,
    const std::vector<size_t>& sizes, void* mem_h = nullptr,
    bool own_buffer = true)
    : mr_(mr),
      type_(type),
      buffers_(std::vector<ucx_buffer_t>(sizes.size())),
      mem_h_(mem_h),
      own_buffer_(own_buffer) {
    if (sizes.size() > 0) {
      for (size_t i = 0; i < sizes.size(); ++i) {
        buffers_[i].data = mr_.get().allocate(type, sizes[i]);
        buffers_[i].size = sizes[i];
      }
    }
  }

  UcxBufferVec(UcxBufferVec&& other, bool own_buffer)
    : mr_(other.mr_),
      type_(other.type_),
      buffers_(std::move(other.buffers_)),
      mem_h_(other.mem_h_),
      own_buffer_(own_buffer) {
    other.buffers_ = {};
    other.own_buffer_.store(false, std::memory_order_relaxed);
  }

  ~UcxBufferVec() {
    if (own_buffer_) {
      for (auto& buf : buffers_) {
        mr_.get().deallocate(type_, buf.data, buf.size);
      }
    }
  }

  UcxBufferVec(const UcxBufferVec&) = delete;

  UcxBufferVec(UcxBufferVec&& other) noexcept
    : mr_(other.mr_),
      type_(other.type_),
      buffers_(std::move(other.buffers_)),
      mem_h_(other.mem_h_),
      own_buffer_(other.own_buffer_.load(std::memory_order_relaxed)) {
    other.buffers_ = {};
    other.own_buffer_.store(false, std::memory_order_relaxed);
  }

  UcxBufferVec& operator=(UcxBufferVec&& other) noexcept {
    if (this != &other) {
      if (own_buffer_) {
        for (auto& buf : buffers_) {
          mr_.get().deallocate(type_, buf.data, buf.size);
        }
      }
      mr_ = other.mr_;
      type_ = other.type_;
      buffers_ = std::move(other.buffers_);
      mem_h_ = other.mem_h_;
      own_buffer_ = other.own_buffer_.load(std::memory_order_relaxed);
      other.buffers_ = {};
      other.own_buffer_.store(false, std::memory_order_relaxed);
    }
    return *this;
  }

  UcxBufferVec& operator=(const UcxBufferVec&) = delete;

  UcxBufferVec(
    UcxMemoryResourceManager& mr, ucx_memory_type_t type,
    std::vector<ucx_buffer_t>&& buffers, void* mem_h = nullptr,
    bool own_buffer = false)
    : mr_(mr),
      type_(type),
      buffers_(std::move(buffers)),
      mem_h_(mem_h),
      own_buffer_(own_buffer) {}

  UcxBufferVec(
    UcxMemoryResourceManager& mr, ucx_memory_type_t type,
    const std::vector<ucx_buffer_t>& buffers, void* mem_h = nullptr,
    bool own_buffer = false)
    : mr_(mr),
      type_(type),
      buffers_(buffers),
      mem_h_(mem_h),
      own_buffer_(own_buffer) {}

  explicit UcxBufferVec(std::vector<UcxBuffer>&& buffers, void* mem_h = nullptr)
    : mr_(buffers[0].mr_),
      type_(buffers[0].type()),
      mem_h_(buffers[0].mem_h()),
      own_buffer_(buffers[0].own_buffer()) {
    for (auto& buf : buffers) {
      buffers_.push_back(*buf.get());
    }
    buffers[0].own_buffer_.store(false, std::memory_order_relaxed);
  }

  /**
   * @brief Gets the data pointers.
   */
  std::vector<void*> data_ptrs() {
    std::vector<void*> ptrs;
    ptrs.reserve(buffers_.size());
    for (auto& buf : buffers_) {
      ptrs.push_back(buf.data);
    }
    return ptrs;
  }
  /**
   * @brief Gets the const data pointers.
   */
  const std::vector<const void*> data_ptrs() const { return data_ptrs(); }
  /**
   * @brief Gets the sizes of the buffer.
   */
  std::vector<size_t> data_sizes() const {
    std::vector<size_t> sizes;
    sizes.reserve(buffers_.size());
    for (auto& buf : buffers_) {
      sizes.push_back(buf.size);
    }
    return sizes;
  }

  void* data() { return buffers_.data(); }
  const void* data() const { return buffers_.data(); }

  size_t size() const { return buffers_.size(); }

  /**
   * @brief Returns an iterator to the beginning of the buffer vector.
   */
  std::vector<ucx_buffer_t>::iterator begin() { return buffers_.begin(); }
  /**
   * @brief Returns a const iterator to the beginning of the buffer vector.
   */
  std::vector<ucx_buffer_t>::const_iterator begin() const {
    return buffers_.begin();
  }
  /**
   * @brief Returns an iterator to the end of the buffer vector.
   */
  std::vector<ucx_buffer_t>::iterator end() { return buffers_.end(); }
  /**
   * @brief Returns a const iterator to the end of the buffer vector.
   */
  std::vector<ucx_buffer_t>::const_iterator end() const {
    return buffers_.end();
  }

  /**
   * @brief Accesses the buffer at the given index.
   * @param idx The index of the buffer.
   * @return Reference to the ucx_buffer_t at the given index.
   */
  ucx_buffer_t& operator[](size_t idx) { return buffers_[idx]; }
  /**
   * @brief Accesses the buffer at the given index (const).
   * @param idx The index of the buffer.
   * @return Const reference to the ucx_buffer_t at the given index.
   */
  const ucx_buffer_t& operator[](size_t idx) const { return buffers_[idx]; }

  /**
   * @brief Returns the underlying buffer vector.
   * @return Reference to the std::vector<ucx_buffer_t>.
   */
  std::vector<ucx_buffer_t>& buffers() { return buffers_; }
  /**
   * @brief Returns the underlying buffer vector (const).
   * @return Const reference to the std::vector<ucx_buffer_t>.
   */
  const std::vector<ucx_buffer_t>& buffers() const { return buffers_; }

  /**
   * @brief Gets the memory type of the buffer.
   */
  ucx_memory_type_t type() const { return type_; }
  /**
   * @brief Gets the memory handle of the buffer.
   */
  void* mem_h() const { return mem_h_; }
  /**
   * @brief Gets whether the buffer is owned by the UcxBufferVec.
   */
  bool own_buffer() const { return own_buffer_; }

 private:
  std::reference_wrapper<UcxMemoryResourceManager> mr_;
  ucx_memory_type_t type_;
  std::vector<ucx_buffer_t> buffers_;
  void* mem_h_;
  std::atomic<bool> own_buffer_;
};

/**
 * @class UcxAmData
 * @brief RAII wrapper for a ucx_am_data_t that manages its memory.
 *
 * This class allocates and deallocates memory for the header and data buffers
 * of a ucx_am_data_t using a provided UcxMemoryResourceManager.
 */
class UcxAmData {
 public:
  /**
   * @brief Constructs a UcxAmData.
   * @param mr The memory resource manager.
   * @param header_size The size of the header buffer.
   * @param buffer_size The size of the data buffer.
   * @param buffer_type The memory type of the data buffer.
   * @param own_header Whether to own the header.
   * @param own_buffer Whether to own the buffer.
   */
  UcxAmData(
    UcxMemoryResourceManager& mr, size_t header_size, size_t buffer_size,
    ucx_memory_type_t buffer_type, bool own_header = true,
    bool own_buffer = true, std::function<void(void*)> ucp_release_fn = nullptr)
    : mr_(mr),
      data_{},
      own_header_(own_header),
      own_buffer_(own_buffer),
      ucp_release_fn_(ucp_release_fn) {
    data_.buffer_type = buffer_type;

    if (header_size > 0 && own_header) {
      data_.header.data =
        mr_.get().allocate(ucx_memory_type::HOST, header_size);
      data_.header.size = header_size;
    }

    if (buffer_size > 0 && own_buffer) {
      data_.buffer.data = mr_.get().allocate(buffer_type, buffer_size);
      data_.buffer.size = buffer_size;
    }
  }

  UcxAmData(
    UcxMemoryResourceManager& mr, ucx_am_data_t&& data, bool own_header = true,
    bool own_buffer = true, std::function<void(void*)> ucp_release_fn = nullptr)
    : mr_(mr),
      data_(std::move(data)),
      own_header_(own_header),
      own_buffer_(own_buffer),
      ucp_release_fn_(std::move(ucp_release_fn)) {}

  UcxAmData(
    UcxMemoryResourceManager& mr, const ucx_am_data_t& data,
    bool own_header = false, bool own_buffer = false)
    : mr_(mr), data_(data), own_header_(own_header), own_buffer_(own_buffer) {}

  UcxAmData(UcxAmData&& other, bool own_header, bool own_buffer)
    : mr_(other.mr_),
      data_(std::move(other.data_)),
      own_header_(own_header),
      own_buffer_(own_buffer),
      ucp_release_fn_(std::move(other.ucp_release_fn_)) {
    other.data_ = {};
    other.own_header_.store(false, std::memory_order_relaxed);
    other.own_buffer_.store(false, std::memory_order_relaxed);
    other.ucp_release_fn_ = nullptr;
  }

  UcxAmData(const UcxAmData&) = delete;
  UcxAmData& operator=(const UcxAmData&) = delete;

  UcxAmData(UcxAmData&& other) noexcept
    : mr_(other.mr_),
      data_(other.data_),
      own_header_(other.own_header_.load(std::memory_order_relaxed)),
      own_buffer_(other.own_buffer_.load(std::memory_order_relaxed)),
      ucp_release_fn_(std::move(other.ucp_release_fn_)) {
    other.data_ = {};
    other.own_header_.store(false, std::memory_order_relaxed);
    other.own_buffer_.store(false, std::memory_order_relaxed);
    other.ucp_release_fn_ = nullptr;
  }

  UcxAmData& operator=(UcxAmData&& other) noexcept {
    if (this != &other) {
      if (data_.header.data && own_header_) {
        mr_.get().deallocate(
          ucx_memory_type::HOST, data_.header.data, data_.header.size);
      }
      if (data_.buffer.data && own_buffer_) {
        if (ucp_release_fn_) {
          ucp_release_fn_(data_.buffer.data);
        } else {
          mr_.get().deallocate(
            data_.buffer_type, data_.buffer.data, data_.buffer.size);
        }
      }
      mr_ = other.mr_;
      data_ = other.data_;
      own_header_ = other.own_header_.load(std::memory_order_relaxed);
      own_buffer_ = other.own_buffer_.load(std::memory_order_relaxed);
      other.data_ = {};
      other.own_header_.store(false, std::memory_order_relaxed);
      other.own_buffer_.store(false, std::memory_order_relaxed);
      ucp_release_fn_ = std::move(other.ucp_release_fn_);
    }
    return *this;
  }

  /**
   * @brief Destroys the UcxAmData, deallocating its memory.
   */
  ~UcxAmData() {
    if (data_.header.data && own_header_) {
      mr_.get().deallocate(
        ucx_memory_type::HOST, data_.header.data, data_.header.size);
    }
    if (data_.buffer.data && own_buffer_) {
      if (ucp_release_fn_) {
        ucp_release_fn_(data_.buffer.data);
      } else {
        mr_.get().deallocate(
          data_.buffer_type, data_.buffer.data, data_.buffer.size);
      }
    }
  }

  /**
   * @brief Gets a pointer to the underlying ucx_am_data_t.
   * @return A pointer to the ucx_am_data_t.
   */
  ucx_am_data_t* get() { return &data_; }
  /**
   * @brief Gets a const pointer to the underlying ucx_am_data_t.
   * @return A const pointer to the ucx_am_data_t.
   */
  const ucx_am_data_t* get() const { return &data_; }

  /**
   * @brief Gets whether this object owns the header.
   */
  bool owns_header() const {
    return own_header_.load(std::memory_order_relaxed);
  }
  /**
   * @brief Gets whether this object owns the buffer.
   */
  bool own_buffer() const {
    return own_buffer_.load(std::memory_order_relaxed);
  }

  bool set_ucp_release_fn(std::function<void(void*)>&& ucp_release_fn) {
    if (ucp_release_fn_) {
      return false;
    }
    ucp_release_fn_ = std::move(ucp_release_fn);
    return true;
  }

 private:
  std::reference_wrapper<UcxMemoryResourceManager> mr_;
  ucx_am_data_t data_;
  std::atomic<bool> own_header_;
  std::atomic<bool> own_buffer_;
  std::function<void(void*)> ucp_release_fn_;
};

/**
 * @class UcxAmIovec
 * @brief RAII wrapper for a ucx_am_iovec_t that manages its memory.
 *
 * This class allocates and deallocates memory for the header, the buffer
 * vector, and each buffer within the vector of a ucx_am_iovec_t.
 */
class UcxAmIovec {
 public:
  /**
   * @brief Constructs a UcxAmIovec.
   * @param mr The memory resource manager.
   * @param header_size The size of the header buffer.
   * @param buffer_sizes A vector of sizes for each buffer in the iovec.
   * @param buffer_type The memory type of the data buffers.
   * @param own_header Whether to own the header.
   * @param own_buffer Whether to own the buffer.
   */
  UcxAmIovec(
    UcxMemoryResourceManager& mr, size_t header_size,
    const std::vector<size_t>& buffer_sizes, ucx_memory_type_t buffer_type,
    bool own_header = true, bool own_buffer = true)
    : mr_(mr), iovec_{}, own_header_(own_header), own_buffer_(own_buffer) {
    iovec_.buffer_type = buffer_type;
    iovec_.buffer_count = buffer_sizes.size();

    if (header_size > 0 && own_header) {
      iovec_.header.data =
        mr_.get().allocate(ucx_memory_type::HOST, header_size);
      iovec_.header.size = header_size;
    }

    if (iovec_.buffer_count > 0) {
      // Allocate the buffer_vec array
      iovec_.buffer_vec = static_cast<ucx_buffer_t*>(mr_.get().allocate(
        ucx_memory_type::HOST, sizeof(ucx_buffer_t) * iovec_.buffer_count));

      // Allocate data for each buffer in the vector
      for (size_t i = 0; i < iovec_.buffer_count; ++i) {
        if (buffer_sizes[i] > 0) {
          iovec_.buffer_vec[i].data =
            mr_.get().allocate(buffer_type, buffer_sizes[i]);
          iovec_.buffer_vec[i].size = buffer_sizes[i];
        } else {
          iovec_.buffer_vec[i] = {nullptr, 0};
        }
      }
    }
  }

  UcxAmIovec(
    UcxMemoryResourceManager& mr, ucx_am_iovec_t&& iovec,
    bool own_header = true, bool own_buffer = true)
    : mr_(mr),
      iovec_(std::move(iovec)),
      own_header_(own_header),
      own_buffer_(own_buffer) {}

  UcxAmIovec(
    UcxMemoryResourceManager& mr, const ucx_am_iovec_t& iovec,
    bool own_header = false, bool own_buffer = false)
    : mr_(mr),
      iovec_(iovec),
      own_header_(own_header),
      own_buffer_(own_buffer) {}

  UcxAmIovec(UcxAmIovec&& other, bool own_header, bool own_buffer)
    : mr_(other.mr_),
      iovec_(other.iovec_),
      own_header_(own_header),
      own_buffer_(own_buffer) {
    other.iovec_ = {};
    other.own_header_.store(false, std::memory_order_relaxed);
    other.own_buffer_.store(false, std::memory_order_relaxed);
  }

  /**
   * @brief Destroys the UcxAmIovec, deallocating its memory.
   */
  ~UcxAmIovec() {
    if (iovec_.header.data && own_header_) {
      mr_.get().deallocate(
        ucx_memory_type::HOST, iovec_.header.data, iovec_.header.size);
    }
    if (iovec_.buffer_vec && own_buffer_) {
      for (size_t i = 0; i < iovec_.buffer_count; ++i) {
        if (iovec_.buffer_vec[i].data) {
          mr_.get().deallocate(
            iovec_.buffer_type, iovec_.buffer_vec[i].data,
            iovec_.buffer_vec[i].size);
        }
      }
      mr_.get().deallocate(
        ucx_memory_type::HOST, iovec_.buffer_vec,
        sizeof(ucx_buffer_t) * iovec_.buffer_count);
    }
  }

  UcxAmIovec(const UcxAmIovec&) = delete;
  UcxAmIovec& operator=(const UcxAmIovec&) = delete;

  UcxAmIovec(UcxAmIovec&& other) noexcept
    : mr_(other.mr_),
      iovec_(other.iovec_),
      own_header_(other.own_header_.load(std::memory_order_relaxed)),
      own_buffer_(other.own_buffer_.load(std::memory_order_relaxed)) {
    other.iovec_ = {};
    other.own_header_.store(false, std::memory_order_relaxed);
    other.own_buffer_.store(false, std::memory_order_relaxed);
  }

  UcxAmIovec& operator=(UcxAmIovec&& other) noexcept {
    if (this != &other) {
      if (iovec_.header.data && own_header_) {
        mr_.get().deallocate(
          ucx_memory_type::HOST, iovec_.header.data, iovec_.header.size);
      }
      if (iovec_.buffer_vec && own_buffer_) {
        for (size_t i = 0; i < iovec_.buffer_count; ++i) {
          if (iovec_.buffer_vec[i].data) {
            mr_.get().deallocate(
              iovec_.buffer_type, iovec_.buffer_vec[i].data,
              iovec_.buffer_vec[i].size);
          }
        }
        mr_.get().deallocate(
          ucx_memory_type::HOST, iovec_.buffer_vec,
          sizeof(ucx_buffer_t) * iovec_.buffer_count);
      }
      mr_ = other.mr_;
      iovec_ = other.iovec_;
      own_header_ = other.own_header_.load(std::memory_order_relaxed);
      own_buffer_ = other.own_buffer_.load(std::memory_order_relaxed);
      other.iovec_ = {};
      other.own_header_.store(false, std::memory_order_relaxed);
      other.own_buffer_.store(false, std::memory_order_relaxed);
    }
    return *this;
  }

  /**
   * @brief Gets a pointer to the underlying ucx_am_iovec_t.
   * @return A pointer to the ucx_am_iovec_t.
   */
  ucx_am_iovec_t* get() { return &iovec_; }
  /**
   * @brief Gets a const pointer to the underlying ucx_am_iovec_t.
   * @return A const pointer to the ucx_am_iovec_t.
   */
  const ucx_am_iovec_t* get() const { return &iovec_; }

  /**
   * @brief Gets whether this object owns the header.
   */
  bool owns_header() const {
    return own_header_.load(std::memory_order_relaxed);
  }
  /**
   * @brief Gets whether this object owns the buffer.
   */
  bool own_buffer() const {
    return own_buffer_.load(std::memory_order_relaxed);
  }

 private:
  std::reference_wrapper<UcxMemoryResourceManager> mr_;
  ucx_am_iovec_t iovec_;
  std::atomic<bool> own_header_;
  std::atomic<bool> own_buffer_;
};

}  // namespace ucxx
}  // namespace eux

#endif  // UCX_CONTEXT_UCX_CONTEXT_DATA_HPP_
