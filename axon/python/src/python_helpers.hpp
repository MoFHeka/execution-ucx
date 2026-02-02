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

#ifndef AXON_PYTHON_PYTHON_HELPERS_HPP_
#define AXON_PYTHON_PYTHON_HELPERS_HPP_

#include <nanobind/nanobind.h>

#include <vector>

#include "rpc_core/rpc_types.hpp"

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;
namespace rpc = eux::rpc;

// Thread-safe wrapper for PyObject* using shared_ptr to manage lifetime.
// Performance optimization:
// 1. atomic refcounting avoids GIL for copy/move.
// 2. stores PyObject* directly, avoiding extra heap allocation for nb::object.
// GIL is only acquired when the last reference is dropped (deletion).
struct SharedPyObject {
  std::shared_ptr<PyObject> ptr_;

  static void Deleter(PyObject* p) {
    if (p) {
      // Ensure GIL is held for destruction (Py_DECREF)
      nb::gil_scoped_acquire guard;
      Py_XDECREF(p);
    }
  }

  SharedPyObject() = default;

  // Implicit construction from nb::object
  SharedPyObject(nb::object obj) {
    if (obj.is_valid()) {
      // Increase refcount because shared_ptr will hold a strong reference
      // and manage its release via Deleter.
      // Note: using obj.ptr() returns borrowed reference, so we must manually
      // INCREF.
      PyObject* raw_ptr = obj.ptr();
      Py_XINCREF(raw_ptr);
      ptr_ = std::shared_ptr<PyObject>(raw_ptr, Deleter);
    }
  }

  // Copy Constructor
  SharedPyObject(const SharedPyObject& other) : ptr_(other.ptr_) {}

  // Move Constructor - COPY to ensure source remains valid (Copy-on-Move)
  SharedPyObject(SharedPyObject&& other) noexcept : ptr_(other.ptr_) {}

  // Copy Assignment
  SharedPyObject& operator=(const SharedPyObject& other) {
    if (this != &other) {
      ptr_ = other.ptr_;
    }
    return *this;
  }

  // Move Assignment - COPY to ensure source remains valid
  SharedPyObject& operator=(SharedPyObject&& other) noexcept {
    if (this != &other) {
      ptr_ = other.ptr_;
    }
    return *this;
  }

  // Access underlying object - caller must ensure GIL if manipulating it
  // Returns a borrowed reference wrapper (cheap stack object)
  nb::object get() const { return nb::borrow(ptr_.get()); }

  // Check validity
  explicit operator bool() const { return ptr_ && ptr_.get() != nullptr; }
};

// Check if Python object is an async function/coroutine
bool IsAsyncFunction(nb::object py_obj);

// Result extraction modes - precomputed at registration time
enum class ResultExtractionMode : uint8_t {
  VOID,                    // No return value
  SINGLE_NON_TENSOR,       // Single non-tensor (int, str, etc.)
  SINGLE_TENSOR,           // Single tensor (np.ndarray)
  LIST_TENSOR,             // List[Tensor]
  TUPLE_NON_TENSORS_ONLY,  // Tuple with only non-tensors
  TUPLE_WITH_TENSORS       // Tuple with mixed types
};

// Helper function to parse Python function signature and extract types
struct FunctionSignatureInfo {
  std::vector<rpc::ParamType> param_types;
  std::vector<rpc::ParamType> return_types;
  rpc::PayloadType input_payload_type = rpc::PayloadType::NO_PAYLOAD;
  rpc::PayloadType return_payload_type = rpc::PayloadType::NO_PAYLOAD;
  std::vector<size_t> tensor_param_indices;   // Indices of tensor parameters
  std::vector<size_t> tensor_return_indices;  // Indices of tensor return values
  bool has_tensor_return = false;

  // from_dlpack_fn callables for tensor parameters (indexed by tensor index)
  // These are the type's from_dlpack methods saved during registration,
  // used to convert dltensor capsules to the correct user-expected types.
  std::vector<SharedPyObject> tensor_param_from_dlpack;
  // from_dlpack_fn callables for tensor returns
  std::vector<SharedPyObject> tensor_return_from_dlpack;

  // ===== Precomputed for zero-overhead result extraction =====
  ResultExtractionMode extraction_mode = ResultExtractionMode::VOID;
  std::vector<size_t> non_tensor_indices;  // Positions of non-tensor returns
};

FunctionSignatureInfo ParseFunctionSignature(nb::object py_func);

}  // namespace python
}  // namespace axon
}  // namespace eux

#endif  // AXON_PYTHON_PYTHON_HELPERS_HPP_
