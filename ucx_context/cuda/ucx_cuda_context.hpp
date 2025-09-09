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

#ifndef UCX_CONTEXT_CUDA_UCX_CUDA_CONTEXT_HPP_
#define UCX_CONTEXT_CUDA_UCX_CUDA_CONTEXT_HPP_

#include <cuda.h>

#include <memory>

#include "ucx_context/cuda/ucx_cuda_macro.h"
#include "ucx_context/ucx_device_context.hpp"

class UcxAutoCudaDeviceContext : public UcxAutoDeviceContext {
 public:
  explicit UcxAutoCudaDeviceContext(CUcontext cuda_context)
    : cuda_context_(cuda_context) {}
  ~UcxAutoCudaDeviceContext() override = default;

  class CudaOperation;

  friend class CudaOperation;

  /**
   * @class CudaOperation
   * @brief The specific operation class for the CUDA context.
   *
   * Its lifetime manages the CUDA context on the current thread using RAII.
   * It pushes the context on activation and pops it on deactivation.
   */
  class CudaOperation : public OperationRAII {
   public:
    using OperationRAII::OperationRAII;

    // The destructor must be overridden to call specific deactivate.
    ~CudaOperation() override { deactivate(); }

    void activate() override {
      auto* owner = static_cast<UcxAutoCudaDeviceContext*>(context_);
      UCX_CUDA_API_CHECK(cuCtxPushCurrent(owner->cuda_context_));
    }

    void deactivate() override {
      auto* owner = static_cast<UcxAutoCudaDeviceContext*>(context_);
      UCX_CUDA_API_CHECK(cuCtxPopCurrent(&(owner->cuda_context_)));
    }
  };

  std::unique_ptr<OperationRAII> operator()(
    const ucp_context_h ucp_context, const ucp_worker_h ucp_worker) override {
    auto op = std::make_unique<CudaOperation>(this, ucp_context, ucp_worker);
    op->activate();  // Activate after the object is fully constructed.
    return op;
  }

 private:
  CUcontext cuda_context_;
};

#endif  // UCX_CONTEXT_CUDA_UCX_CUDA_CONTEXT_HPP_
