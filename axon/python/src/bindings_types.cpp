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

#include "axon/python/src/bindings_types.hpp"

#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "axon/python/src/dlpack_helpers.hpp"
#include "axon/python/src/param_conversion.hpp"
#include "rpc_core/rpc_types.hpp"
#include "rpc_core/utils/cista_serialize_helper.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace nb = nanobind;
namespace rpc = eux::rpc;
namespace ucxx = eux::ucxx;

void BindDltensorMetaTypes(nb::module_& m) {
  nb::enum_<DLDeviceType>(m, "DLDeviceType")
    .value("kDLCPU", kDLCPU)
    .value("kDLCUDA", kDLCUDA)
    .value("kDLCUDAManaged", kDLCUDAManaged)
    .value("kDLROCM", kDLROCM)
    .value("kDLROCMHost", kDLROCMHost)
    .value("kDLOpenCL", kDLOpenCL)
    .value("kDLVulkan", kDLVulkan)
    .value("kDLMetal", kDLMetal)
    .value("kDLVPI", kDLVPI)
    .value("kDLOneAPI", kDLOneAPI);

  nb::class_<DLDevice>(m, "DLDevice")
    .def_rw("device_type", &DLDevice::device_type)
    .def_rw("device_id", &DLDevice::device_id)
    .def("__repr__", [](const DLDevice& d) {
      return std::format(
        "DLDevice(type={}, id={})", static_cast<int>(d.device_type),
        d.device_id);
    });

  nb::enum_<DLDataTypeCode>(m, "DLDataTypeCode")
    .value("kDLInt", kDLInt)
    .value("kDLUInt", kDLUInt)
    .value("kDLFloat", kDLFloat)
    .value("kDLBfloat", kDLBfloat)
    .value("kDLComplex", kDLComplex);

  nb::class_<DLDataType>(m, "DLDataType")
    .def_rw("code", &DLDataType::code)
    .def_rw("bits", &DLDataType::bits)
    .def_rw("lanes", &DLDataType::lanes)
    .def("__repr__", [](const DLDataType& d) {
      return std::format(
        "DLDataType(code={}, bits={}, lanes={})", d.code, d.bits, d.lanes);
    });

  nb::class_<rpc::utils::TensorMeta>(m, "TensorMeta")
    .def(nb::init<>())
    .def_rw("device", &rpc::utils::TensorMeta::device)
    .def_rw("ndim", &rpc::utils::TensorMeta::ndim)
    .def_rw("dtype", &rpc::utils::TensorMeta::dtype)
    .def_rw("byte_offset", &rpc::utils::TensorMeta::byte_offset)
    .def_prop_ro(
      "shape",
      [](const rpc::utils::TensorMeta& m) {
        nb::list l;
        for (auto v : m.shape) l.append(v);
        return l;
      })
    .def_prop_ro(
      "strides",
      [](const rpc::utils::TensorMeta& m) {
        nb::list l;
        for (auto v : m.strides) l.append(v);
        return l;
      })
    .def("__repr__", [](const rpc::utils::TensorMeta& m) {
      return std::format(
        "TensorMeta(ndim={}, dtype={{c={}, b={}, l={}}}, offset={})", m.ndim,
        m.dtype.code, m.dtype.bits, m.dtype.lanes, m.byte_offset);
    });
}

void RegisterTypes(nb::module_& m) {
  // ============================================================================
  // DLPackCapsuleWrapper: Wrapper class for from_dlpack compatibility
  // ============================================================================
  // This wrapper exposes __dlpack__ and __dlpack_device__ methods,
  // allowing frameworks like numpy, PyTorch, JAX to use from_dlpack()
  //
  // DLPack v2024.12 API:
  // - __dlpack__(*, stream=None, max_version=None, dl_device=None, copy=None)
  // - __dlpack_device__() -> tuple(device_type, device_id)
  nb::class_<eux::axon::python::DLPackCapsuleWrapper>(m, "DLPackCapsuleWrapper")
    .def(
      "__dlpack__", &eux::axon::python::DLPackCapsuleWrapper::dlpack,
      nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none(),
      nb::arg("dl_device") = nb::none(), nb::arg("copy") = nb::none(),
      "Export array data as a DLPack capsule.\n\n"
      "Parameters\n"
      "----------\n"
      "stream : int or None\n"
      "    For CUDA/ROCm, a pointer to a stream. For CPU, must be None.\n"
      "max_version : tuple(int, int) or None\n"
      "    Maximum DLPack version the consumer supports.\n"
      "dl_device : tuple(int, int) or None\n"
      "    Target device for export (device_type, device_id).\n"
      "copy : bool or None\n"
      "    Whether to copy: True=always, False=never, None=auto.\n\n"
      "Returns\n"
      "-------\n"
      "PyCapsule\n"
      "    A DLPack capsule containing the tensor data.\n\n"
      "Notes\n"
      "-----\n"
      "This method transfers ownership of the underlying data. It can only be\n"
      "called once per wrapper instance.")
    .def(
      "__dlpack_device__",
      &eux::axon::python::DLPackCapsuleWrapper::dlpack_device,
      "Return the device type and device ID in DLPack format.\n\n"
      "Returns\n"
      "-------\n"
      "tuple(int, int)\n"
      "    A tuple (device_type, device_id).");

  // ============================================================================
  // UcxBuffer class with DLPack support
  // ============================================================================

  nb::class_<ucxx::UcxBuffer>(m, "UcxBuffer")
    .def(
      "data",
      [](ucxx::UcxBuffer& self) -> nb::ndarray<nb::numpy, uint8_t> {
        // Return as bytes array - UcxBuffer contains raw bytes
        size_t shape_arr[] = {self.size()};
        return nb::ndarray<nb::numpy, uint8_t>(
          static_cast<uint8_t*>(self.data()), 1, shape_arr, nb::handle(),
          nullptr);
      },
      nb::rv_policy::reference_internal)
    .def("size", &ucxx::UcxBuffer::size)
    .def("type", &ucxx::UcxBuffer::type)
    .def(
      "__dlpack__",
      [](ucxx::UcxBuffer& self, nb::object stream) {
        // Create DLTensor from UcxBuffer
        DLManagedTensor* dlm_tensor = new DLManagedTensor;
        dlm_tensor->dl_tensor.data = self.data();
        dlm_tensor->dl_tensor.device = {kDLCPU, 0};
        dlm_tensor->dl_tensor.ndim = 1;
        dlm_tensor->dl_tensor.dtype = {kDLUInt, 8, 1};
        dlm_tensor->dl_tensor.shape =
          new int64_t[1]{static_cast<int64_t>(self.size())};
        dlm_tensor->dl_tensor.strides = nullptr;
        dlm_tensor->dl_tensor.byte_offset = 0;
        dlm_tensor->manager_ctx = &self;
        dlm_tensor->deleter = [](DLManagedTensor* self) {
          // UcxBuffer owns the memory, so we don't delete it here
          delete[] self->dl_tensor.shape;
          delete self;
        };
        return nb::cast(dlm_tensor);
      },
      nb::arg("stream") = nb::none())
    .def("__dlpack_device__", [](ucxx::UcxBuffer& self) {
      return std::make_pair(static_cast<int>(kDLCPU), 0);
    });

  // ============================================================================
  // UcxBufferVec class
  // ============================================================================

  nb::class_<ucxx::UcxBufferVec>(m, "UcxBufferVec")
    .def("__len__", [](ucxx::UcxBufferVec& self) { return self.size(); })
    .def(
      "__getitem__",
      [](ucxx::UcxBufferVec& self, size_t idx) {
        if (idx >= self.size()) {
          throw std::out_of_range("Index out of range");
        }
        // Return ucx_buffer_t reference - Python will need to wrap it
        return nb::cast(&self.buffers()[idx]);
      },
      nb::rv_policy::reference_internal)
    .def("size", &ucxx::UcxBufferVec::size)
    .def("type", &ucxx::UcxBufferVec::type)
    .def(
      "buffers",
      [](ucxx::UcxBufferVec& self) { return nb::cast(self.buffers()); },
      nb::rv_policy::reference_internal);

  // ============================================================================
  // RpcRequestHeader class
  // ============================================================================

  nb::class_<rpc::RpcRequestHeader>(m, "RpcRequestHeader")
    .def(nb::init<>())
    // Custom getters/setters for strong types to allow Python int conversion
    .def_prop_rw(
      "session_id",
      [](const rpc::RpcRequestHeader& self) {
        return cista::to_idx(self.session_id);
      },
      [](rpc::RpcRequestHeader& self, uint32_t value) {
        self.session_id = rpc::session_id_t{value};
      })
    .def_prop_rw(
      "request_id",
      [](const rpc::RpcRequestHeader& self) {
        return cista::to_idx(self.request_id);
      },
      [](rpc::RpcRequestHeader& self, uint32_t value) {
        self.request_id = rpc::request_id_t{value};
      })
    .def_prop_rw(
      "function_id",
      [](const rpc::RpcRequestHeader& self) {
        return cista::to_idx(self.function_id);
      },
      [](rpc::RpcRequestHeader& self, uint32_t value) {
        self.function_id = rpc::function_id_t{value};
      })
    .def_prop_rw(
      "workflow_id",
      [](const rpc::RpcRequestHeader& self) {
        return cista::to_idx(self.workflow_id);
      },
      [](rpc::RpcRequestHeader& self, uint64_t value) {
        self.workflow_id =
          rpc::utils::workflow_id_t{static_cast<uint32_t>(value)};
      })
    .def_prop_ro(
      "params",
      [](const rpc::RpcRequestHeader& self) {
        namespace python = eux::axon::python;
        // TODO(He Jia): support different payload types
        return python::ResultsToPython(self.params);
      })
    .def(
      "AddParam",
      [](rpc::RpcRequestHeader& self, nb::object value) {
        namespace python = eux::axon::python;
        self.AddParam(python::InferParamMeta(value));
      },
      nb::arg("value"),
      "Add a parameter to the request header. The parameter type is "
      "automatically inferred from the Python value.");

  // ============================================================================
  // RpcResponseHeader class
  // ============================================================================

  nb::class_<rpc::RpcResponseHeader>(m, "RpcResponseHeader")
    .def(nb::init<>())
    // Custom getters for strong types to allow Python int conversion
    .def_prop_ro(
      "session_id",
      [](const rpc::RpcResponseHeader& self) {
        return cista::to_idx(self.session_id);
      })
    .def_prop_ro(
      "request_id",
      [](const rpc::RpcResponseHeader& self) {
        return cista::to_idx(self.request_id);
      })
    .def_prop_ro(
      "workflow_id",
      [](const rpc::RpcResponseHeader& self) {
        return cista::to_idx(self.workflow_id);
      })
    .def_prop_ro(
      "status",
      [](const rpc::RpcResponseHeader& self) {
        // Convert RpcStatus to a Python dict with value and category_name
        nb::dict status_dict;
        status_dict["value"] = nb::cast(self.status.value);
        namespace data = cista::offset;
        const data::string& category_name = self.status.category_name;
        status_dict["category_name"] =
          nb::str(category_name.data(), category_name.size());
        return status_dict;
      })
    .def_prop_ro("results", [](const rpc::RpcResponseHeader& self) {
      namespace python = eux::axon::python;
      return python::ResultsToPython(self.results);
    });

  // ============================================================================
  // RpcFunctionSignature class
  // ============================================================================

  nb::class_<rpc::RpcFunctionSignature>(m, "RpcFunctionSignature")
    .def(nb::init<>())
    .def_prop_ro(
      "instance_name",
      [](const rpc::RpcFunctionSignature& self) {
        namespace data = cista::offset;
        const data::string& str_ref = self.instance_name;
        return nb::str(str_ref.data(), str_ref.size());
      })
    .def_prop_ro(
      "id",
      [](const rpc::RpcFunctionSignature& self) {
        return cista::to_idx(self.id);
      })
    .def_prop_ro(
      "function_name",
      [](const rpc::RpcFunctionSignature& self) {
        namespace data = cista::offset;
        const data::string& str_ref = self.function_name;
        return nb::str(str_ref.data(), str_ref.size());
      })
    .def_prop_ro(
      "param_types",
      [](const rpc::RpcFunctionSignature& self) {
        // Convert cista::offset::vector<ParamType> to Python list
        nb::list result;
        for (const auto& type : self.param_types) {
          result.append(nb::cast(static_cast<int>(type)));
        }
        return result;
      })
    .def_prop_ro(
      "return_types",
      [](const rpc::RpcFunctionSignature& self) {
        // Convert cista::offset::vector<ParamType> to Python list
        nb::list result;
        for (const auto& type : self.return_types) {
          result.append(nb::cast(static_cast<int>(type)));
        }
        return result;
      })
    .def_prop_ro(
      "input_payload_type",
      [](const rpc::RpcFunctionSignature& self) {
        return nb::cast(static_cast<int>(self.input_payload_type));
      })
    .def_prop_ro(
      "return_payload_type", [](const rpc::RpcFunctionSignature& self) {
        return nb::cast(static_cast<int>(self.return_payload_type));
      });

  // ============================================================================
  // TensorMeta class
  // ============================================================================
  BindDltensorMetaTypes(m);

  // ============================================================================
  // deserialize_signatures function
  // ============================================================================

  m.def(
    "deserialize_signatures",
    [](nb::bytes signatures_bytes) {
      namespace data = cista::offset;

      const char* sig_data = signatures_bytes.c_str();
      size_t sig_size = signatures_bytes.size();
      std::string_view sig_view(sig_data, sig_size);

      const data::vector<rpc::RpcFunctionSignature>* signatures = nullptr;
      try {
        signatures = cista::deserialize<
          data::vector<rpc::RpcFunctionSignature>, rpc::utils::SerializerMode>(
          sig_view);
      } catch (const std::exception& e) {
        throw std::runtime_error(
          std::string("Failed to deserialize signatures: ") + e.what());
      }

      if (signatures == nullptr) {
        throw std::runtime_error(
          "Failed to deserialize signatures: null result");
      }

      // Convert to Python list
      nb::list result;
      for (const auto& sig : *signatures) {
        result.append(nb::cast(sig));
      }
      return result;
    },
    nb::arg("signatures_bytes"),
    "Deserialize function signatures from bytes. Returns a list of "
    "RpcFunctionSignature objects.");
}
