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

#ifndef UCX_AM_CONTEXT_TEST_CUDA_HELPER_CUH_
#define UCX_AM_CONTEXT_TEST_CUDA_HELPER_CUH_

#include "ucx_context/ucx_am_context/ucx_am_context_test_helper.h"

#include "ucx_context/ucx_context_def.h"

#include <cstdio>
#include <cuda_runtime.h>

// CUDA kernel to divide each element of a float array by 2.0f
__global__ void divide_by_two_kernel_cuh(float* data, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] /= 2.0f;
  }
}

/**
 * @brief Processes received data on the CUDA device.
 *
 * This function takes a pointer to ucx_am_data structure. It assumes that
 * recvData->data points to CUDA device memory and recvData->data_type is
 * ucx_memory_type::CUDA. The function divides each float element in
 * recvData->data by 2.0f.
 *
 * @param recvData Pointer to the ucx_am_data structure containing the data to
 * be processed. The data within must be float type and reside in CUDA device
 * memory.
 */
__attribute__((visibility("default"))) void processRecvDataCuda(
  ucx_am_data_t& recvData) {
  if (recvData.data == nullptr || recvData.data_length == 0) {
    fprintf(stderr, "processRecvDataCuda: Received null or empty data.\n");
    return;
  }

  // This function assumes that recvData->data points to CUDA device memory
  // and recvData->data_type is ucx_memory_type::CUDA.
  float* d_data = static_cast<float*>(recvData.data);
  size_t num_elements = recvData.data_length / sizeof(float);

  if (num_elements == 0) {
    return;  // No elements to process
  }

  // Configure kernel launch parameters
  int threads_per_block = 256;  // Common default, can be tuned
  int blocks_per_grid =
    (num_elements + threads_per_block - 1) / threads_per_block;

  // Launch the kernel
  divide_by_two_kernel_cuh<<<blocks_per_grid, threads_per_block>>>(
    d_data, num_elements);

  cudaError_t lastError = cudaGetLastError();
  if (lastError != cudaSuccess) {
    fprintf(
      stderr, "CUDA error in processRecvDataCuda kernel launch: %s\n",
      cudaGetErrorString(lastError));
  }
  cudaDeviceSynchronize();
}

#endif  // UCX_AM_CONTEXT_TEST_CUDA_HELPER_CUH_
