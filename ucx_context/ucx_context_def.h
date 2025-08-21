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

#ifndef UCX_CONTEXT_DEF_H_
#define UCX_CONTEXT_DEF_H_

#include <stddef.h>

/**
 * @enum ucx_memory_type
 * @brief Enumerates memory types supported by UCX.
 */
typedef enum ucx_memory_type {
  HOST,         /**< Default system memory */
  CUDA,         /**< NVIDIA CUDA memory */
  CUDA_MANAGED, /**< NVIDIA CUDA managed (or unified) memory */
  ROCM,         /**< AMD ROCM memory */
  ROCM_MANAGED, /**< AMD ROCM managed system memory */
  RDMA,         /**< RDMA device memory */
  ZE_HOST,      /**< Intel ZE memory (USM host) */
  ZE_DEVICE,    /**< Intel ZE memory (USM device) */
  ZE_MANAGED,   /**< Intel ZE managed memory (USM shared) */
  LAST,
  UNKNOWN = LAST
} ucx_memory_type_t;

/**
 * @struct ucx_buffer_t
 * @brief Represents a single buffer element for continuous or scatter-gather
 * I/O. Also used for header buffer.
 *
 * This structure describes a buffer and its length, used in continuous or
 * scatter-gather operations. It is functionally similar to @ref ucp_dt_iov_t in
 * UCX.
 *
 * @note If @a size is zero, the memory pointed to by @a data is not
 * accessed. Otherwise, @a data must point to valid memory.
 *
 * @var ucx_buffer_t::data
 *   Pointer to the data.
 * @var ucx_buffer_t::size
 *   Length of the data in bytes.
 */
typedef struct ucx_buffer {
  void* data;   ///< Pointer to a buffer.
  size_t size;  ///< Length of the @a data in bytes.
} ucx_buffer_t;
typedef ucx_buffer_t ucx_header_t;

/**
 * @struct ucx_am_data_t
 * @brief Describes the payload for a UCX Active Message.
 *
 * This structure contains pointers to the header and data buffers, their
 * lengths, the memory type of the data buffer, and an optional memory handle.
 *
 * @var ucx_am_data_t::header
 *   Header buffer payload.
 * @var ucx_am_data_t::buffer
 *   Data buffer payload.
 * @var ucx_am_data_t::buffer_type
 *   Memory type of the data buffer (see @ref ucx_memory_type_t).
 * @var ucx_am_data_t::mem_h
 *   Memory handle for the data buffer (can be reinterpret_cast to ucp_mem_h).
 */
typedef struct ucx_am_data {
  ucx_buffer_t header;            ///< Header buffer payload.
  ucx_buffer_t buffer;            ///< Data buffer payload.
  ucx_memory_type_t buffer_type;  ///< Memory type of the data buffer.
  void* mem_h;  ///< Memory handle for the data buffer. Could be
                ///< reinterpret_cast to ucp_mem_h.
  /* Release function for data, especially for AM Eager DATA */
  // void (*data_release_fn)(void*, size_t);  // NOLINT
  /* Atomic variable maybe useless here. Because even though if the cancel
   * operation was called somewhere else, the UCX worker may still be running
   * the memcpy. So a sudden memory freeing may cause a crash. The memory
   * sweeping should be done after the whole operation is finished. */
  // _Atomic(unsigned int) data_allocated;  // NOLINT
} ucx_am_data_t;

/**
 * @struct ucx_am_iovec_t
 * @brief Describes a scatter-gather list for UCX Active Message transfer.
 *
 * This structure specifies a list of buffers (scatter-gather array) for a
 * single Active Message data transfer. The array and all referenced buffers
 * must remain valid until the transfer is complete.
 *
 * @var ucx_am_iovec_t::header
 *   Header buffer payload.
 * @var ucx_am_iovec_t::buffer_vec
 *   Pointer to an array of @ref ucx_buffer_t elements, each describing a data
 * buffer. This array will be converted to a @ref ucp_dt_iov_t list for UCX data
 * transfer.
 * @var ucx_am_iovec_t::buffer_count
 *   Number of elements in the @a buffer_vec array.
 * @var ucx_am_iovec_t::data_type
 *   Memory type of the data buffers (see @ref ucx_memory_type_t).
 * @var ucx_am_iovec_t::mem_h
 *   Memory handle for the data buffers (can be reinterpret_cast to @ref
 * ucp_mem_h).
 *
 * @note The @a buffer_vec array and all referenced buffers must remain valid
 *       until the data transfer operation is complete.
 * @note Currently, only the RNDV protocol is supported, as the data length for
 *       each segment is determined from the user-defined header.
 */
typedef struct ucx_am_iovec {
  ucx_buffer_t header;            ///< Header buffer payload.
  ucx_buffer_t* buffer_vec;       ///< Pointer to array of data vector elements.
  size_t buffer_count;            ///< Number of elements in the data vector.
  ucx_memory_type_t buffer_type;  ///< Memory type of the data buffers.
  void* mem_h;  ///< Memory handle for the data buffers. Could be
                ///< reinterpret_cast to ucp_mem_h.
} ucx_am_iovec_t;

// /* Set data_allocated flag to true */
// static inline void ucx_am_data_set_data_allocated(ucx_am_data_t* data) {
//   __atomic_store_n(&data->data_allocated, 1, __ATOMIC_SEQ_CST);
// }

// /* Set data_allocated flag to false */
// static inline void ucx_am_data_clear_data_allocated(ucx_am_data_t* data) {
//   __atomic_store_n(&data->data_allocated, 0, __ATOMIC_SEQ_CST);
// }

// /* Check if buffer has been allocated */
// static inline int ucx_am_data_is_data_allocated(const ucx_am_data_t* data)
// {
//   return __atomic_load_n(&data->data_allocated, __ATOMIC_SEQ_CST);
// }

/**
 * @struct ucx_am_cqe
 * @brief Represents an entry in the UCX Active Message Completion Queue.
 *
 * This structure holds user data, the result of the operation, and flags.
 */
typedef struct ucx_am_cqe {
  size_t user_data;  ///< Arbitrary user-data from submission, e.g., a pointer
                     ///< to a completion state.
  int res;           ///< Result code of the completed operation.
  int flags;         ///< Metadata flags (currently unused).
} ucx_am_cqe_t;

#endif  // UCX_CONTEXT_DEF_H_
