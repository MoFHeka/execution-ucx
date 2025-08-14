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
 * @struct ucx_am_data
 * @brief Represents the data payload for a UCX Active Message.
 *
 * This structure holds pointers to the header and data buffers, along with
 * their lengths and the memory type of the data buffer.
 */
typedef struct ucx_am_data {
  void* header;                 ///< Pointer to the message header.
  size_t header_length;         ///< Length of the message header.
  void* data;                   ///< Pointer to the message data payload.
  size_t data_length;           ///< Length of the message data payload.
  ucx_memory_type_t data_type;  ///< Memory type of the data buffer.
  void* mem_h;                  ///< Memory handle for the data buffer. Could be
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
 * @struct ucx_iovec_data_t
 * @brief Structure for a single scatter-gather I/O vector element.
 *
 * This structure is used to specify a single buffer and its length for
 * scatter-gather I/O operations. It is analogous to @ref ucp_dt_iov_t in UCX.
 *
 * @note If @a data_length is zero, the memory pointed to by @a data will not be
 *       accessed. Otherwise, @a data must point to valid memory.
 */
typedef struct ucx_iovec_data {
  void* data;          ///< Pointer to a data buffer.
  size_t data_length;  ///< Length of the @a data buffer in bytes.
} ucx_iovec_data_t;

/**
 * @struct ucx_am_iovec_t
 * @brief Structure for Active Message scatter-gather I/O.
 *
 * This structure is used to specify a list of buffers (scatter-gather list)
 * for a single Active Message data transfer operation. The list should remain
 * valid until the data transfer request is completed.
 *
 * @var ucx_am_iovec_t::header
 *   Pointer to the message header buffer.
 * @var ucx_am_iovec_t::header_length
 *   Length of the message header buffer in bytes.
 * @var ucx_am_iovec_t::data_vec
 *   Pointer to an array of @ref ucx_iovec_data_t elements, each describing a
 *   data buffer and its length. This array will be converted to a
 *   @ref ucp_dt_iov_t list for UCX data transfer.
 * @var ucx_am_iovec_t::data_count
 *   Number of elements in the @a data_vec array.
 * @var ucx_am_iovec_t::data_type
 *   Memory type of the data buffers (see @ref ucx_memory_type_t).
 * @var ucx_am_iovec_t::mem_h
 *   Memory handle for the data buffers. This can be reinterpret_cast to
 *   @ref ucp_mem_h as required by UCX.
 *
 * @note The @a data_vec array and all referenced buffers must remain valid
 *       until the data transfer operation is complete.
 *
 * @note For now, we only support RNDV protocol. Because we need to get each
 *       data length from the header buffer, and header is defined by user.
 */
typedef struct ucx_am_iovec {
  void* header;                 ///< Pointer to the message header buffer.
  size_t header_length;         ///< Length of the message header buffer.
  ucx_iovec_data_t* data_vec;   ///< Pointer to array of data vector elements.
  size_t data_count;            ///< Number of elements in the data vector.
  ucx_memory_type_t data_type;  ///< Memory type of the data buffers.
  void* mem_h;                  ///< Memory handle for the data buffers.
                                ///< Could be reinterpret_cast to ucp_mem_h.
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
