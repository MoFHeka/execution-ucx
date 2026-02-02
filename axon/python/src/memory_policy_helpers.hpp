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

#pragma once

#ifndef AXON_PYTHON_MEMORY_POLICY_HELPERS_HPP_
#define AXON_PYTHON_MEMORY_POLICY_HELPERS_HPP_

#include <nanobind/nanobind.h>

#include "axon/axon_runtime.hpp"
#include "axon/memory_policy.hpp"
#include "axon/message_lifecycle_policy.hpp"

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;

// Helper function template to create custom memory policy from Python callable
template <typename BufferType>
axon::CustomMemoryPolicy<BufferType> create_custom_memory_policy(
  nb::object py_callable, AxonRuntime& runtime);

// Helper function to create retention policy from Python callable
axon::RetentionPolicy create_retention_policy(nb::object py_callable);

// Register a Python object associated with a buffer pointer
void RegisterCustomBuffer(const void* ptr, nb::object obj);

// Try to retrieve a Python object associated with a buffer pointer
// Returns nb::none() if not found
nb::object TryGetCustomBuffer(const void* ptr);

// Get cached memory policy from Python object (None or callable)
// Returns variant of AlwaysOnHostPolicy or CustomMemoryPolicy
template <typename BufferType>
axon::ReceiverMemoryPolicy<BufferType> create_memory_policy(
  nb::object py_obj, AxonRuntime& runtime);

}  // namespace python
}  // namespace axon
}  // namespace eux

#endif  // AXON_PYTHON_MEMORY_POLICY_HELPERS_HPP_
