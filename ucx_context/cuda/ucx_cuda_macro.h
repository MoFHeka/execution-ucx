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

#ifndef UCX_CONTEXT_CUDA_UCX_CUDA_MACRO_H_
#define UCX_CONTEXT_CUDA_UCX_CUDA_MACRO_H_

#if CUDA_ENABLED

#include <cuda_runtime.h>

#include <iostream>

#include "ucx_context/ucx_context_logger.hpp"

// cuda driver and/or runtime entry points
#define CUDA_DRIVER_HAS_FNPTR(name) ((name##_fnptr) != nullptr)
#define CUDA_DRIVER_FNPTR(name) (assert(name##_fnptr != nullptr), name##_fnptr)

#define UCX_CUDA_API_CHECK(expr)               \
  do {                                         \
    CUresult err = (expr);                     \
    if (err != CUDA_SUCCESS) {                 \
      fprintf(stderr, "CUDA error %d\n", err); \
      abort();                                 \
    }                                          \
  } while (0)

#define UCX_CUDA_CHECK(expr)                                                \
  do {                                                                      \
    cudaError_t err = (expr);                                               \
    if (err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %d: %s\n", err, cudaGetErrorString(err)); \
      abort();                                                              \
    }                                                                       \
  } while (0)

#define UCX_CUDA_CHECK_LAST(expr) UCX_CUDA_CHECK((expr) == cudaSuccess)

#define UCX_CUDA_CHECK_LAST_MSG(expr, msg) \
  UCX_CUDA_CHECK((expr) == cudaSuccess, msg)

#define UCX_CUDA_CHECK_LAST_MSG_AND_ABORT(expr, msg) \
  UCX_CUDA_CHECK((expr) == cudaSuccess, msg);        \
  abort();

#define UCX_CUDA_CHECK_LAST_MSG_AND_ABORT_WITH_PREFIX(expr, msg, prefix)      \
  UCX_CUDA_CHECK((expr) == cudaSuccess, msg);                                 \
  fprintf(                                                                    \
    stderr, "%s: CUDA error %d: %s\n", prefix, err, cudaGetErrorString(err)); \
  abort();

#define UCX_CUDA_CHECK_LAST_MSG_AND_ABORT_WITH_PREFIX_AND_LINE(               \
  expr, msg, prefix)                                                          \
  UCX_CUDA_CHECK((expr) == cudaSuccess, msg);                                 \
  fprintf(                                                                    \
    stderr, "%s: CUDA error %d: %s\n", prefix, err, cudaGetErrorString(err)); \
  abort();

#define UCX_CUDA_CHECK_LAST_MSG_AND_ABORT_WITH_PREFIX_AND_LINE_FILE(          \
  expr, msg, prefix, line, file)                                              \
  UCX_CUDA_CHECK((expr) == cudaSuccess, msg);                                 \
  fprintf(                                                                    \
    stderr, "%s: CUDA error %d: %s\n", prefix, err, cudaGetErrorString(err)); \
  fprintf(stderr, "File: %s, Line: %d\n", file, line);                        \
  abort();

#define UCX_CUDA_CHECK_LAST_MSG_AND_ABORT_WITH_PREFIX_AND_LINE_FILE_FUNCTION(  \
  expr, msg, prefix, line, file, function)                                     \
  UCX_CUDA_CHECK((expr) == cudaSuccess, msg);                                  \
  fprintf(                                                                     \
    stderr, "%s: CUDA error %d: %s\n", prefix, err, cudaGetErrorString(err));  \
  fprintf(stderr, "File: %s, Line: %d, Function: %s\n", file, line, function); \
  abort();

#endif  // CUDA_ENABLED
#endif  // UCX_CONTEXT_CUDA_UCX_CUDA_MACRO_H_
