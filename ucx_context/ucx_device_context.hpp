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

#ifndef UCX_CONTEXT_UCX_DEVICE_CONTEXT_HPP_
#define UCX_CONTEXT_UCX_DEVICE_CONTEXT_HPP_

#include <ucp/api/ucp.h>

#include <iostream>
#include <memory>

// Forward declaration
class UcxAutoDeviceContext;

/**
 * @class OperationRAII
 * @brief Base class for the RAII operation object.
 *
 * Its lifetime controls the activation/deactivation of a resource.
 * Deactivation is automatically handled by the destructor.
 */
class OperationRAII {
 public:
  // Constructor stores context information.
  OperationRAII(
    UcxAutoDeviceContext* context, const ucp_context_h ucp_context,
    const ucp_worker_h ucp_worker)
    : context_(context), ucp_context_(ucp_context), ucp_worker_(ucp_worker) {}

  // The virtual destructor is essential for polymorphic deletion.
  virtual ~OperationRAII() = default;

  // Pure virtual functions for derived classes to implement logic.
  virtual void activate() = 0;
  virtual void deactivate() = 0;

  // Disable copy/move to enforce RAII uniqueness and clear ownership.
  OperationRAII(const OperationRAII&) = delete;
  OperationRAII& operator=(const OperationRAII&) = delete;
  OperationRAII(OperationRAII&&) = delete;
  OperationRAII& operator=(OperationRAII&&) = delete;

 protected:
  UcxAutoDeviceContext* context_;
  const ucp_context_h ucp_context_;
  const ucp_worker_h ucp_worker_;
};

/**
 * @class UcxAutoDeviceContext
 * @brief Abstract base class for all device contexts.
 *
 * This is the interface other classes will use, hiding implementation details.
 */
class UcxAutoDeviceContext {
 public:
  virtual ~UcxAutoDeviceContext() = default;

  // Factory method to create the specific operation object.
  virtual std::unique_ptr<OperationRAII> operator()(
    const ucp_context_h ucp_context, const ucp_worker_h ucp_worker) = 0;
};

/**
 * @class UcxAutoDefaultDeviceContext
 * @brief A default example implementation of the device context.
 */
class UcxAutoDefaultDeviceContext : public UcxAutoDeviceContext {
 private:
  // Example of a private member specific to this subclass.
  int private_state_ = 123;

 public:
  UcxAutoDefaultDeviceContext() = default;
  ~UcxAutoDefaultDeviceContext() override = default;

  // Forward declare the inner class
  class DefaultOperation;

  // Make the inner class a friend to allow access to private members.
  friend class DefaultOperation;

  /**
   * @class DefaultOperation
   * @brief The specific operation class for this context.
   *
   * It's an inner class, tightly coupled with its outer context.
   * Its destructor automatically calls deactivate(), ensuring RAII.
   */
  class DefaultOperation : public OperationRAII {
   public:
    // Inherit constructor from the base class.
    using OperationRAII::OperationRAII;

    // The destructor must be overridden to call our specific deactivate.
    ~DefaultOperation() override { deactivate(); }

    [[maybe_unused]] void activate() override {
      /*
        // Safely cast the base context pointer to our concrete type.
        // This is safe because DefaultOperation is only created by
        // UcxAutoDefaultDeviceContext's factory method.
        auto* owner = static_cast<UcxAutoDefaultDeviceContext*>(context_);

        // Now we can access private members of the owner.
        std::cout << "Activating with private state: " << owner->private_state_
                  << std::endl;
      */
    }

    [[maybe_unused]] void deactivate() override {
      /*
        auto* owner = static_cast<UcxAutoDefaultDeviceContext*>(context_);
        std::cout << "Deactivating with private state: " <<
        owner->private_state_
                  << std::endl;
      */
    }
  };

  // Override the factory method to return our specific operation type.
  [[maybe_unused]] std::unique_ptr<OperationRAII> operator()(
    const ucp_context_h ucp_context, const ucp_worker_h ucp_worker) override {
    auto op = std::make_unique<DefaultOperation>(this, ucp_context, ucp_worker);
    op->activate();  // Activate after the object is fully constructed.
    return op;
  }
};

#endif  // UCX_CONTEXT_UCX_DEVICE_CONTEXT_HPP_
