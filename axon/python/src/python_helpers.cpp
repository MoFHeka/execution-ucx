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

#include <optional>
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
  if (py_obj.is_none()) {
    throw nb::type_error("Expected a callable, got None");
  }

  nb::gil_scoped_acquire acquire;

  PyObject* obj = py_obj.ptr();
  if (!PyCallable_Check(obj)) {
    throw nb::type_error(
      "Expected a callable object (async function or async callable instance)");
  }

  // Resolve the code object: for plain functions / methods, __code__ lives
  // directly on the object; for callable class instances it lives on
  // __call__.
  auto get_code = [](PyObject* target) -> PyObject* {
    PyObject* code_str = PyUnicode_FromString("__code__");
    if (!code_str) {
      PyErr_Clear();
      return nullptr;
    }
    PyObject* code = PyObject_GetAttr(target, code_str);
    Py_DECREF(code_str);
    if (!code) PyErr_Clear();
    return code;  // caller owns the reference
  };

  PyObject* code_attr = get_code(obj);

  if (!code_attr) {
    // Not a plain function: try obj.__call__
    PyObject* call_str = PyUnicode_FromString("__call__");
    if (!call_str) {
      PyErr_Clear();
      throw nb::type_error(
        "IsAsyncFunction: failed to intern '__call__' string");
    }
    PyObject* call_method = PyObject_GetAttr(obj, call_str);
    Py_DECREF(call_str);
    if (!call_method) {
      PyErr_Clear();
      throw nb::type_error(
        "Expected a callable with an accessible __call__ method");
    }
    code_attr = get_code(call_method);
    Py_DECREF(call_method);
    if (!code_attr) {
      throw nb::type_error(
        "Cannot determine if the callable is async: no __code__ attribute "
        "found on the object or its __call__ method");
    }
  }

  if (!PyObject_HasAttrString(code_attr, "co_flags")) {
    Py_DECREF(code_attr);
    throw nb::type_error(
      "IsAsyncFunction: code object has no co_flags attribute");
  }

  PyObject* flags_attr = PyObject_GetAttrString(code_attr, "co_flags");
  Py_DECREF(code_attr);

  if (!flags_attr) {
    PyErr_Clear();
    throw nb::type_error(
      "IsAsyncFunction: failed to read co_flags from code object");
  }

  int64_t flags = PyLong_AsLong(flags_attr);
  Py_DECREF(flags_attr);

  if (flags == -1 && PyErr_Occurred()) {
    PyErr_Clear();
    throw nb::type_error(
      "IsAsyncFunction: co_flags value could not be converted to integer");
  }

  // CO_COROUTINE = 0x100 (Python 3.12+), 0x80 on older versions.
  // Using both masks ensures compatibility across CPython versions.
  return (flags & 0x100) != 0 || (flags & 0x80) != 0;
}

// Helper to check if a type is a subclass of a specific class from a module.
// Uses fast C-API path, avoiding overhead of creating nanobind objects or
// triggering imports. This is safe because if the type annotation is a class
// object, its module MUST be loaded.
static inline bool IsSubclassOf(
  nb::handle type, const char* module_name, const char* class_name) {
  PyObject* modules = PyImport_GetModuleDict();
  if (!modules) return false;

  // Check if module is loaded
  PyObject* module = PyDict_GetItemString(modules, module_name);
  if (!module) return false;

  PyObject* cls = PyObject_GetAttrString(module, class_name);
  if (!cls) {
    PyErr_Clear();  // Attribute might not exist
    return false;
  }

  // Check subclass relationship
  int res = PyObject_IsSubclass(type.ptr(), cls);
  Py_DECREF(cls);  // Release reference to class

  if (res == -1) {
    PyErr_Clear();
    return false;
  }
  return res == 1;
}

static bool IsListType(
  nb::object type, nb::object typing_list, nb::object builtin_list) {
  try {
    if (nb::isinstance(type, typing_list)) return true;
  } catch (...) {
  }
  if (nb::hasattr(type, "__origin__")) {
    nb::object origin = nb::cast<nb::object>(type.attr("__origin__"));
    return origin.is(typing_list) || origin.is(builtin_list);
  }
  return false;
}

static bool IsTensorType(nb::object type) {
  // Reject generic container types (e.g. List[np.ndarray]) that happen to
  // contain tensor type names in their string representation. Only bare types
  // or types with __dlpack__ are actual tensor types.
  if (nb::hasattr(type, "__origin__")) return false;
  if (nb::hasattr(type, "__dlpack__")) return true;
  std::string type_str = nb::cast<std::string>(nb::str(type));
  return type_str.find("numpy.ndarray") != std::string::npos
         || type_str.find("torch.Tensor") != std::string::npos
         || type_str.find("jax.Array") != std::string::npos
         || type_str.find("jaxlib.xla_extension.ArrayImpl")
              != std::string::npos;
}

struct AnnotationTypeInfo {
  rpc::ParamType param_type;
  std::optional<FunctionSignatureInfo::EncodedElementType> encoded_element_type;
  nb::object nested_tensor_type;
};

static AnnotationTypeInfo AnnotationToTypeInfo(nb::object annotation) {
  using NL = FunctionSignatureInfo::EncodedElementType;

  if (annotation.is_none()) {
    return {rpc::ParamType::UNKNOWN, std::nullopt, nb::none()};
  }

  nb::module_ builtins = GetBuiltinsModule();
  nb::module_ typing = GetTypingModule();

  if (annotation.is(builtins.attr("int"))) {
    return {rpc::ParamType::PRIMITIVE_INT64, std::nullopt, nb::none()};
  }
  if (annotation.is(builtins.attr("float"))) {
    return {rpc::ParamType::PRIMITIVE_FLOAT64, std::nullopt, nb::none()};
  }
  if (annotation.is(builtins.attr("bool"))) {
    return {rpc::ParamType::PRIMITIVE_BOOL, std::nullopt, nb::none()};
  }
  if (annotation.is(builtins.attr("str"))) {
    return {rpc::ParamType::STRING, std::nullopt, nb::none()};
  }

  if (nb::hasattr(annotation, "__dlpack__")) {
    return {rpc::ParamType::TENSOR_META, std::nullopt, nb::none()};
  }

  try {
    if (nb::hasattr(typing, "List")) {
      nb::object list_type = typing.attr("List");
      nb::object builtin_list = builtins.attr("list");

      if (IsListType(annotation, list_type, builtin_list)) {
        if (nb::hasattr(annotation, "__args__")) {
          nb::tuple args = nb::cast<nb::tuple>(annotation.attr("__args__"));
          if (nb::len(args) > 0) {
            nb::object elem_type = args[0];

            if (elem_type.is(builtins.attr("int"))) {
              return {rpc::ParamType::VECTOR_INT64, std::nullopt, nb::none()};
            }
            if (elem_type.is(builtins.attr("float"))) {
              return {rpc::ParamType::VECTOR_FLOAT64, std::nullopt, nb::none()};
            }
            if (elem_type.is(builtins.attr("bool"))) {
              return {rpc::ParamType::VECTOR_BOOL, std::nullopt, nb::none()};
            }
            if (elem_type.is(builtins.attr("str"))) {
              return {rpc::ParamType::STRING, NL::STRING_ELEM, nb::none()};
            }
            if (IsTensorType(elem_type)) {
              return {
                rpc::ParamType::TENSOR_META_VEC, std::nullopt, nb::none()};
            }

            if (IsListType(elem_type, list_type, builtin_list)) {
              if (nb::hasattr(elem_type, "__args__")) {
                nb::tuple inner_args =
                  nb::cast<nb::tuple>(elem_type.attr("__args__"));
                if (nb::len(inner_args) > 0) {
                  nb::object inner_type = inner_args[0];
                  if (inner_type.is(builtins.attr("int"))) {
                    return {rpc::ParamType::STRING, NL::INT64, nb::none()};
                  }
                  if (inner_type.is(builtins.attr("float"))) {
                    return {rpc::ParamType::STRING, NL::FLOAT64, nb::none()};
                  }
                  if (inner_type.is(builtins.attr("bool"))) {
                    return {rpc::ParamType::STRING, NL::BOOL, nb::none()};
                  }
                  if (IsTensorType(inner_type)) {
                    return {
                      rpc::ParamType::STRING, NL::NESTED_TENSOR_LIST,
                      inner_type};
                  }
                  std::string inner_str =
                    nb::cast<std::string>(nb::str(inner_type));
                  throw nb::type_error(
                    ("Unsupported inner element type in List[List[...]]:"
                     " '"
                     + inner_str + "'. Supported: int, float, bool, Tensor")
                      .c_str());
                }
              }
              throw nb::type_error(
                "List[List[...]] requires explicit element type annotation"
                " (e.g. List[List[int]])");
            }
            std::string elem_str = nb::cast<std::string>(nb::str(elem_type));
            throw nb::type_error(
              ("Unsupported List element type: '" + elem_str
               + "'. Supported: int, float, bool, str, Tensor, List[int],"
                 " List[float], List[bool], List[Tensor]")
                .c_str());
          }
        }
        throw nb::type_error(
          "List without type argument is not supported."
          " Use List[int], List[float], List[str], List[bool], List[Tensor],"
          " List[List[int]], List[List[float]], List[List[bool]], "
          "List[List[Tensor]]"
          " instead.");
      }
    }

    if (nb::hasattr(typing, "Tuple")) {
      nb::object tuple_type = typing.attr("Tuple");
      if (nb::isinstance(annotation, tuple_type)
          || (nb::hasattr(annotation, "__origin__")
              && nb::cast<nb::object>(annotation.attr("__origin__"))
                   .is(tuple_type))) {
        throw nb::type_error(
          "Tuple is not supported as a parameter type annotation."
          " Tuple is only allowed as a return type.");
      }
    }
  } catch (const nb::python_error&) {
    throw;
  }

  if (
    IsSubclassOf(annotation, "jax", "Array")
    || IsSubclassOf(annotation, "tensorflow", "Tensor")
    || IsSubclassOf(annotation, "torch", "Tensor")
    || IsSubclassOf(annotation, "numpy", "ndarray")) {
    return {rpc::ParamType::TENSOR_META, std::nullopt, nb::none()};
  }

  return {rpc::ParamType::UNKNOWN, std::nullopt, nb::none()};
}

rpc::ParamType AnnotationToParamType(nb::object annotation) {
  return AnnotationToTypeInfo(std::move(annotation)).param_type;
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
      auto type_info = AnnotationToTypeInfo(param_annotation);
      size_t current_idx = info.param_types.size();
      info.param_types.push_back(type_info.param_type);

      if (type_info.param_type == rpc::ParamType::TENSOR_META) {
        info.tensor_param_indices.push_back(current_idx);
        nb::object from_dlpack_fn = ExtractFromDlpack(param_annotation);
        if (!from_dlpack_fn.is_none()) {
          info.tensor_param_from_dlpack.push_back(
            SharedPyObject(std::move(from_dlpack_fn)));
        } else {
          info.tensor_param_from_dlpack.push_back(SharedPyObject());
        }
      } else if (type_info.param_type == rpc::ParamType::TENSOR_META_VEC) {
        info.tensor_param_indices.push_back(current_idx);
        nb::object from_dlpack_fn = nb::none();
        if (nb::hasattr(param_annotation, "__args__")) {
          nb::tuple pargs =
            nb::cast<nb::tuple>(param_annotation.attr("__args__"));
          if (nb::len(pargs) > 0) {
            from_dlpack_fn = ExtractFromDlpack(pargs[0]);
          }
        }
        info.tensor_param_from_dlpack.push_back(
          SharedPyObject(std::move(from_dlpack_fn)));
      }

      if (type_info.encoded_element_type.has_value()) {
        auto elem_type = *type_info.encoded_element_type;
        info.encoded_params.push_back({current_idx, elem_type});
        if (
          elem_type
          == FunctionSignatureInfo::EncodedElementType::NESTED_TENSOR_LIST) {
          info.tensor_param_indices.push_back(current_idx);
          nb::object from_dlpack_fn =
            ExtractFromDlpack(type_info.nested_tensor_type);
          info.tensor_param_from_dlpack.push_back(
            SharedPyObject(std::move(from_dlpack_fn)));
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
      throw;
    }
  } else {
    // No return annotation - default to UNKNOWN
    info.return_types.push_back(rpc::ParamType::UNKNOWN);
    info.return_payload_type = rpc::PayloadType::NO_PAYLOAD;
  }

  // Determine input payload type based on tensor parameters.
  // - NESTED_TENSOR_LIST (List[List[Tensor]]): flattens into multiple buffers →
  // VEC
  // - TENSOR_META_VEC (List[Tensor]): multiple tensor buffers in one param →
  // VEC
  // - TENSOR_META (Tensor): one buffer → single UCX_BUFFER
  // - Multiple Tensor params: all share one TENSOR_META_VEC in the header → VEC
  bool has_multi_buffer_param = false;
  for (const auto& ep : info.encoded_params) {
    if (
      ep.element_type
      == FunctionSignatureInfo::EncodedElementType::NESTED_TENSOR_LIST) {
      has_multi_buffer_param = true;
      break;
    }
  }
  if (!has_multi_buffer_param) {
    for (size_t idx : info.tensor_param_indices) {
      if (info.param_types[idx] == rpc::ParamType::TENSOR_META_VEC) {
        has_multi_buffer_param = true;
        break;
      }
    }
  }
  if (has_multi_buffer_param || info.tensor_param_indices.size() > 1) {
    info.input_payload_type = rpc::PayloadType::UCX_BUFFER_VEC;
  } else if (info.tensor_param_indices.size() == 1) {
    info.input_payload_type = rpc::PayloadType::UCX_BUFFER;
  } else {
    info.input_payload_type = rpc::PayloadType::NO_PAYLOAD;
  }

  return info;
}

}  // namespace python
}  // namespace axon
}  // namespace eux
