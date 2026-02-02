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

#include "axon/python/src/python_helpers.hpp"

#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <utility>

#include "axon/python/src/python_module.hpp"
#include "rpc_core/rpc_types.hpp"

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;
namespace rpc = eux::rpc;

bool IsAsyncFunction(nb::object py_obj) {
  // Ensure we have a valid Python object before proceeding
  if (py_obj.is_none()) {
    return false;
  }

  // Create a new reference to ensure the object stays alive
  // This is critical: nanobind objects may not hold a reference
  nb::object py_obj_ref = py_obj;

  nb::gil_scoped_acquire acquire;

  // Use Python C API directly to avoid potential issues with inspect module
  // This is more reliable and avoids the segfault in
  // inspect.iscoroutinefunction
  try {
    // Get the underlying PyObject* from nanobind
    // Use the reference we created to ensure object is valid
    PyObject* obj = py_obj_ref.ptr();
    if (obj == nullptr) {
      return false;
    }

    // Increment reference count to ensure object stays alive during our
    // operations
    Py_INCREF(obj);

    // Check if it's a function or method
    if (!PyCallable_Check(obj)) {
      Py_DECREF(obj);
      return false;
    }

    // Check object type to ensure it's valid
    PyTypeObject* obj_type = Py_TYPE(obj);
    if (obj_type == nullptr) {
      Py_DECREF(obj);
      return false;
    }

    // Try to get __code__ attribute using a safer approach
    // Use PyObject_GetAttr instead of PyObject_GetAttrString for better error
    // handling
    PyObject* code_str = PyUnicode_FromString("__code__");
    if (code_str == nullptr) {
      Py_DECREF(obj);
      PyErr_Clear();
      return false;
    }

    PyObject* code_attr = PyObject_GetAttr(obj, code_str);
    Py_DECREF(code_str);
    Py_DECREF(obj);  // Release our reference to obj

    if (code_attr == nullptr) {
      PyErr_Clear();  // Clear the error
      return false;
    }

    // Check if __code__ has CO_COROUTINE flag
    if (!PyObject_HasAttrString(code_attr, "co_flags")) {
      Py_DECREF(code_attr);
      PyErr_Clear();
      return false;
    }

    PyObject* flags_attr = PyObject_GetAttrString(code_attr, "co_flags");
    Py_DECREF(code_attr);

    if (flags_attr == nullptr) {
      PyErr_Clear();  // Clear the error
      return false;
    }

    // Check if CO_COROUTINE flag is set
    int64_t flags = PyLong_AsLong(flags_attr);
    Py_DECREF(flags_attr);

    if (flags == -1 && PyErr_Occurred()) {
      PyErr_Clear();  // Clear the error
      return false;
    }

    // CO_COROUTINE = 0x80 (128)
    bool is_async = (flags & 0x80) != 0;

    return is_async;
  } catch (const nb::python_error& e) {
    PyErr_Clear();  // Clear any Python errors
    return false;
  } catch (const std::exception& e) {
    PyErr_Clear();  // Clear any Python errors
    return false;
  }
}

// Helper function to convert Python type annotation to ParamType
rpc::ParamType AnnotationToParamType(nb::object annotation) {
  if (annotation.is_none()) {
    return rpc::ParamType::UNKNOWN;
  }

  nb::module_ builtins = GetBuiltinsModule();
  nb::module_ typing = GetTypingModule();

  // Check for common builtin types
  if (nb::isinstance(annotation, builtins.attr("int"))) {
    return rpc::ParamType::PRIMITIVE_INT64;
  }
  if (nb::isinstance(annotation, builtins.attr("float"))) {
    return rpc::ParamType::PRIMITIVE_FLOAT64;
  }
  if (nb::isinstance(annotation, builtins.attr("bool"))) {
    return rpc::ParamType::PRIMITIVE_BOOL;
  }
  if (nb::isinstance(annotation, builtins.attr("str"))) {
    return rpc::ParamType::STRING;
  }

  // Check for __dlpack__ attribute (standard DLPack protocol)
  const bool has_dlpack = nb::hasattr(annotation, "__dlpack__");
  if (has_dlpack) {
    return rpc::ParamType::TENSOR_META;
  }

  // Check for typing types (List, Tuple, etc.)
  try {
    // Check if it's a List type
    if (nb::hasattr(typing, "List")) {
      nb::object list_type = typing.attr("List");
      nb::object builtin_list = builtins.attr("list");

      bool is_list = false;
      try {
        is_list = nb::isinstance(annotation, list_type);
      } catch (...) {
      }

      if (!is_list && nb::hasattr(annotation, "__origin__")) {
        nb::object origin = nb::cast<nb::object>(annotation.attr("__origin__"));
        is_list = origin.is(list_type) || origin.is(builtin_list);
      }

      if (is_list) {
        // Try to get element type
        if (nb::hasattr(annotation, "__args__")) {
          nb::tuple args = nb::cast<nb::tuple>(annotation.attr("__args__"));
          if (nb::len(args) > 0) {
            nb::object elem_type = args[0];
            std::string elem_str = nb::cast<std::string>(nb::str(elem_type));

            if (nb::isinstance(elem_type, builtins.attr("int"))) {
              return rpc::ParamType::VECTOR_INT64;
            }
            if (nb::isinstance(elem_type, builtins.attr("float"))) {
              return rpc::ParamType::VECTOR_FLOAT64;
            }
            if (nb::isinstance(elem_type, builtins.attr("bool"))) {
              return rpc::ParamType::VECTOR_BOOL;
            }
            // Check if element type has __dlpack__ for TENSOR_META_VEC
            // OR if it matches known tensor type names (string fallback)
            if (
              nb::hasattr(elem_type, "__dlpack__")
              || elem_str.find("numpy.ndarray") != std::string::npos
              || elem_str.find("torch.Tensor") != std::string::npos
              || elem_str.find("jax.Array") != std::string::npos
              || elem_str.find("jaxlib.xla_extension.ArrayImpl")
                   != std::string::npos) {
              return rpc::ParamType::TENSOR_META_VEC;
            }
          }
        }
        // Default to UNKNOWN for untyped List or unsupported element types
        return rpc::ParamType::UNKNOWN;
      }
    }

    // Check if it's a Tuple type (for return types)
    if (nb::hasattr(typing, "Tuple")) {
      nb::object tuple_type = typing.attr("Tuple");
      if (nb::isinstance(annotation, tuple_type) ||
          (nb::hasattr(annotation, "__origin__") &&
           nb::cast<nb::object>(
            annotation.attr("__origin__")).is(tuple_type))) {
        // Tuple is handled separately in ParseFunctionSignature
        return rpc::ParamType::UNKNOWN;
      }
    }
  } catch (const nb::python_error&) {
    // Fall through to string-based matching
  }

  // String-based matching as fallback
  auto annotation_str = nb::cast<std::string>(nb::str(annotation));
  if (annotation_str.find("int") != std::string::npos) {
    return rpc::ParamType::PRIMITIVE_INT64;
  }
  if (annotation_str.find("float") != std::string::npos) {
    return rpc::ParamType::PRIMITIVE_FLOAT64;
  }
  if (annotation_str.find("bool") != std::string::npos) {
    return rpc::ParamType::PRIMITIVE_BOOL;
  }
  if (annotation_str.find("str") != std::string::npos) {
    return rpc::ParamType::STRING;
  }
  if (
    annotation_str.find("List") != std::string::npos
    || annotation_str.find("list") != std::string::npos) {
    if (annotation_str.find("int") != std::string::npos) {
      return rpc::ParamType::VECTOR_INT64;
    }
    if (annotation_str.find("float") != std::string::npos) {
      return rpc::ParamType::VECTOR_FLOAT64;
    }
    return rpc::ParamType::UNKNOWN;  // Default
  }

  return rpc::ParamType::UNKNOWN;
}

// Helper to extract from_dlpack method from a type annotation.
// Returns the from_dlpack callable if the type supports DLPack, otherwise
// returns None.
nb::object ExtractFromDlpack(nb::object annotation) {
  if (annotation.is_none()) {
    return nb::none();
  }

  // The annotation is a type (class), check if it has from_dlpack as a class
  // method or static method.
  if (nb::hasattr(annotation, "from_dlpack")) {
    return annotation.attr("from_dlpack");
  }

  // // Fallback: check if numpy module's from_dlpack can be used.
  // // This handles cases where the user annotates with numpy.ndarray but
  // // from_dlpack is a module-level function.
  // try {
  //   std::string type_str = nb::cast<std::string>(nb::str(annotation));
  //   if (type_str.find("numpy") != std::string::npos) {
  //     nb::module_ numpy = nb::module_::import_("numpy");
  //     if (nb::hasattr(numpy, "from_dlpack")) {
  //       return numpy.attr("from_dlpack");
  //     }
  //   }
  // } catch (...) {
  //   // Ignore import errors
  // }

  return nb::none();
}

FunctionSignatureInfo ParseFunctionSignature(nb::object py_func) {
  nb::gil_scoped_acquire acquire;
  FunctionSignatureInfo info;

  // Import inspect module
  nb::module_ inspect = GetInspectModule();
  nb::module_ builtins = GetBuiltinsModule();
  nb::module_ typing = GetTypingModule();

  // Get function signature
  nb::object sig = inspect.attr("signature")(py_func);
  nb::object parameters = sig.attr("parameters");
  nb::object return_annotation = sig.attr("return_annotation");

  // Parse parameters
  // parameters is a mappingproxy, we need to convert it to a dict first
  nb::object dict_type = builtins.attr("dict");
  nb::dict params_dict = nb::cast<nb::dict>(dict_type(parameters));

  for (auto item : params_dict) {
    nb::object param_obj = nb::cast<nb::object>(item.second);
    nb::object param_annotation = param_obj.attr("annotation");

    // Check if annotation is not empty
    nb::object empty_param = inspect.attr("Parameter").attr("empty");
    bool has_annotation = !empty_param.is(param_annotation);

    if (has_annotation) {
      rpc::ParamType param_type = AnnotationToParamType(param_annotation);
      info.param_types.push_back(param_type);

      // Track tensor parameter indices and save from_dlpack method
      if (param_type == rpc::ParamType::TENSOR_META) {
        info.tensor_param_indices.push_back(info.param_types.size() - 1);

        // Extract and save from_dlpack method for this tensor type
        nb::object from_dlpack_fn = ExtractFromDlpack(param_annotation);
        if (!from_dlpack_fn.is_none()) {
          info.tensor_param_from_dlpack.push_back(
            SharedPyObject(std::move(from_dlpack_fn)));
        } else {
          // Push a placeholder to maintain index alignment
          info.tensor_param_from_dlpack.push_back(SharedPyObject());
        }
      }
    } else {
      // No annotation, will be determined at runtime
      info.param_types.push_back(rpc::ParamType::UNKNOWN);
    }
  }

  // Parse return type
  nb::object empty_sig = inspect.attr("Signature").attr("empty");
  // Check if return type is a tuple (multiple returns)
  // Use .is() directly which returns bool in C++ for nanobind
  bool has_return_annotation = !empty_sig.is(return_annotation);

  if (has_return_annotation) {
    // Staging vectors for classification
    std::vector<rpc::ParamType> non_tensor_types;
    std::vector<size_t>
      tensor_original_indices;  // indices in original annotation
    std::vector<SharedPyObject> tensor_from_dlpacks;
    bool has_tensor_meta_vec = false;  // True if any element is TENSOR_META_VEC

    auto classify_return_type = [&](nb::object elem_type, size_t original_idx) {
      rpc::ParamType return_type = AnnotationToParamType(elem_type);
      if (return_type == rpc::ParamType::TENSOR_META) {
        tensor_original_indices.push_back(original_idx);
        nb::object from_dlpack_fn = ExtractFromDlpack(elem_type);
        tensor_from_dlpacks.push_back(
          SharedPyObject(std::move(from_dlpack_fn)));
      } else if (return_type == rpc::ParamType::TENSOR_META_VEC) {
        has_tensor_meta_vec = true;
        tensor_original_indices.push_back(original_idx);
        // For List[Tensor], extract from element type
        nb::object from_dlpack_fn = nb::none();
        if (nb::hasattr(elem_type, "__args__")) {
          nb::tuple args_vec = nb::cast<nb::tuple>(elem_type.attr("__args__"));
          if (nb::len(args_vec) > 0) {
            from_dlpack_fn = ExtractFromDlpack(args_vec[0]);
          }
        }
        tensor_from_dlpacks.push_back(
          SharedPyObject(std::move(from_dlpack_fn)));
      } else {
        non_tensor_types.push_back(return_type);
      }
    };

    try {
      nb::object tuple_type = builtins.attr("tuple");
      bool is_tuple = false;

      // Check if it has __origin__ attribute (generic type like Tuple[...] or
      // tuple[...])
      if (nb::hasattr(return_annotation, "__origin__")) {
        nb::object origin = return_annotation.attr("__origin__");
        // In Python 3.9+, typing.Tuple[...].__origin__ is builtins.tuple
        // In Python 3.8, typing.Tuple[...].__origin__ is typing.Tuple
        if (
          origin.is(tuple_type)
          || (nb::hasattr(typing, "Tuple") && origin.is(typing.attr("Tuple")))) {
          is_tuple = true;
          // Extract tuple element types
          if (nb::hasattr(return_annotation, "__args__")) {
            nb::tuple args =
              nb::cast<nb::tuple>(return_annotation.attr("__args__"));
            for (size_t i = 0; i < nb::len(args); ++i) {
              nb::object elem_type = args[i];
              classify_return_type(elem_type, i);
            }
          }
        }
      }

      // Check if it's builtins.tuple (non-generic)
      if (!is_tuple && nb::isinstance(return_annotation, tuple_type)) {
        is_tuple = true;
        // For builtins.tuple, we can't determine element types, use UNKNOWN
        non_tensor_types.push_back(rpc::ParamType::UNKNOWN);
      }

      // Check for None return type (void)
      if (return_annotation.is_none()) {
        info.return_types.push_back(rpc::ParamType::VOID);
      } else if (!is_tuple) {
        // Single return type
        classify_return_type(return_annotation, 0);
      }

      // === Build final return_types: non-tensors first, then grouped tensor at
      // end ===
      for (auto t : non_tensor_types) {
        info.return_types.push_back(t);
      }

      size_t total_tensor_count = tensor_original_indices.size();
      if (total_tensor_count > 0) {
        // tensor_return_indices now contains the ORIGINAL indices (in
        // py_result) so we can extract them correctly on the receiving side.
        info.tensor_return_indices = std::move(tensor_original_indices);
        info.tensor_return_from_dlpack = std::move(tensor_from_dlpacks);
        if (total_tensor_count == 1) {
          info.return_types.push_back(rpc::ParamType::TENSOR_META);
          info.return_payload_type = rpc::PayloadType::UCX_BUFFER;
        } else if (total_tensor_count > 1) {
          info.return_types.push_back(rpc::ParamType::TENSOR_META_VEC);
          info.return_payload_type = rpc::PayloadType::UCX_BUFFER_VEC;
        }
      } else {
        info.return_payload_type = rpc::PayloadType::NO_PAYLOAD;
      }

      // ===== Compute extraction_mode for zero-overhead result processing =====
      size_t num_return_types = info.return_types.size();
      bool has_tensors = !info.tensor_return_indices.empty();

      if (
        num_return_types == 0
        || (num_return_types == 1 && info.return_types[0] == rpc::ParamType::VOID)) {
        info.extraction_mode = ResultExtractionMode::VOID;
      } else if (num_return_types == 1) {
        if (has_tensors) {
          if (has_tensor_meta_vec) {
            info.extraction_mode = ResultExtractionMode::LIST_TENSOR;
          } else {
            info.extraction_mode = ResultExtractionMode::SINGLE_TENSOR;
          }
        } else {
          info.extraction_mode = ResultExtractionMode::SINGLE_NON_TENSOR;
        }
      } else {
        // Multiple return types = Tuple
        if (has_tensors) {
          info.extraction_mode = ResultExtractionMode::TUPLE_WITH_TENSORS;
        } else {
          info.extraction_mode = ResultExtractionMode::TUPLE_NON_TENSORS_ONLY;
        }
      }

      // Precompute non-tensor indices for fast extraction
      size_t original_idx = 0;
      for (size_t i = 0; i < non_tensor_types.size(); ++i) {
        // Skip tensor indices
        while (std::find(
                 info.tensor_return_indices.begin(),
                 info.tensor_return_indices.end(), original_idx)
               != info.tensor_return_indices.end()) {
          ++original_idx;
        }
        info.non_tensor_indices.push_back(original_idx);
        ++original_idx;
      }

    } catch (const nb::python_error&) {
      // Fallback: try to infer from string representation
      std::string return_str =
        nb::cast<std::string>(nb::str(return_annotation));
      if (
        return_str.find("tuple") != std::string::npos
        || return_str.find("Tuple") != std::string::npos) {
        // Multiple returns, but can't determine types
        info.return_types.push_back(rpc::ParamType::UNKNOWN);
      } else {
        rpc::ParamType return_type = AnnotationToParamType(return_annotation);
        info.return_types.push_back(return_type);
      }
      info.return_payload_type = rpc::PayloadType::NO_PAYLOAD;
    }
  } else {
    // No return annotation - default to UNKNOWN
    info.return_types.push_back(rpc::ParamType::UNKNOWN);
    info.return_payload_type = rpc::PayloadType::NO_PAYLOAD;
  }

  // Determine input payload type based on tensor parameters
  if (info.tensor_param_indices.size() == 1) {
    info.input_payload_type = rpc::PayloadType::UCX_BUFFER;
  } else if (info.tensor_param_indices.size() > 1) {
    info.input_payload_type = rpc::PayloadType::UCX_BUFFER_VEC;
  } else {
    info.input_payload_type = rpc::PayloadType::NO_PAYLOAD;
  }

  return info;
}

}  // namespace python
}  // namespace axon
}  // namespace eux
