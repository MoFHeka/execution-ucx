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

#include "axon/python/src/bindings_runtime.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <expected>

#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unifex/on.hpp>
#include <unifex/then.hpp>
#include <unifex/upon_error.hpp>

#include "axon/axon_runtime.hpp"
#include "axon/python/src/bindings_runtime_helpers.hpp"
#include "axon/python/src/bindings_runtime_wrapper.hpp"
#include "axon/python/src/dlpack_helpers.hpp"
#include "axon/python/src/memory_policy_helpers.hpp"
#include "axon/python/src/python_helpers.hpp"
#include "axon/python/src/python_module.hpp"
#include "axon/python/src/python_wake_manager.hpp"

#include "ucx_context/ucx_device_context.hpp"

#if CUDA_ENABLED
#include <cuda.h>
#include "ucx_context/cuda/ucx_cuda_context.hpp"
#endif

namespace nb = nanobind;
namespace rpc = eux::rpc;
namespace ucxx = eux::ucxx;
namespace axon = eux::axon;
namespace python = eux::axon::python;

#if CUDA_ENABLED
std::optional<uint64_t> GetDeviceContextHandle(const std::string& device_type) {
  if (device_type == "cuda") {
    CUcontext ctx;
    CUresult res = cuCtxGetCurrent(&ctx);
    if (res != CUDA_SUCCESS || ctx == nullptr) {
      return std::nullopt;
    }
    return reinterpret_cast<uint64_t>(ctx);
  }
  return std::nullopt;
}
#else
std::optional<uint64_t> GetDeviceContextHandle(
  const std::string& /*device_type*/) {
  return std::nullopt;
}
#endif

// Helper to compute function ID from name and worker name
inline uint32_t ComputeFunctionId(
  const std::string& function_name, const std::string& worker_name) {
  // Use MHash from Axon utils to compute a stable hash
  // We use the lower 32 bits of the 128-bit hash
  axon::utils::hash_t h = axon::utils::MHash(function_name, worker_name);
  return static_cast<uint32_t>(h.low());
}

void RegisterRuntime(nb::module_& m) {
  m.def(
    "get_device_context_handle", &GetDeviceContextHandle,
    nb::arg("device_type") = "cuda",
    "Get the current device context handle as an integer.",
    nb::call_guard<nb::gil_scoped_release>());

  nb::class_<axon::AxonRuntime> cls(m, "AxonRuntime");

  // Custom constructor that accepts Device object
  // Supports: CpuDevice, CudaDevice, RocmDevice, SyclDevice
  // or None (defaults to CPU)
  // Use __init__ with placement new
  cls.def(
    "__init__",
    [](
      axon::AxonRuntime* self, const std::string& worker_name,
      size_t thread_pool_size, nb::object timeout_obj, nb::object device_obj) {
      std::unique_ptr<ucxx::UcxAutoDeviceContext> device_context = nullptr;

      if (!device_obj.is_none()) {
        // Get device type string
        nb::object get_type_string = device_obj.attr("get_type_string");
        std::string device_type = nb::cast<std::string>(get_type_string());

        // Get context handle
        nb::object get_context_handle = device_obj.attr("get_context_handle");
        nb::object handle_obj = get_context_handle();

        if (!handle_obj.is_none()) {
          uint64_t handle = nb::cast<uint64_t>(handle_obj);

          if (device_type == "cuda") {
#if CUDA_ENABLED
            CUcontext cuda_ctx = reinterpret_cast<CUcontext>(handle);
            device_context =
              std::make_unique<ucxx::UcxAutoCudaDeviceContext>(cuda_ctx);
#else
            throw std::runtime_error(
              "CUDA device requested but CUDA support is not enabled in this "
              "build.");
#endif
          } else if (device_type == "rocm") {
#if ROCM_ENABLED
            // TODO: Add ROCm support when available
            throw std::runtime_error(
              "ROCm device requested but ROCm support is not yet "
              "implemented.");
#else
            throw std::runtime_error(
              "ROCm device requested but ROCm support is not enabled in this "
              "build.");
#endif
          } else if (device_type == "sycl") {
#if SYCL_ENABLED
            // TODO: Add SYCL support when available
            throw std::runtime_error(
              "SYCL device requested but SYCL support is not yet "
              "implemented.");
#else
            throw std::runtime_error(
              "SYCL device requested but SYCL support is not enabled in this "
              "build.");
#endif
          } else if (device_type == "cpu") {
            // CPU device doesn't need a context
            device_context = nullptr;
          } else {
            throw std::runtime_error("Unsupported device type: " + device_type);
          }
        }
      }

      new (self) axon::AxonRuntime(
        worker_name, thread_pool_size, python::ConvertTimeout(timeout_obj),
        std::move(device_context));
    },
    nb::arg("worker_name"),
    nb::arg("thread_pool_size") =
      (std::thread::hardware_concurrency() < 16 ? 4 : 16),
    nb::arg("timeout") = nb::none(), nb::arg("device") = nb::none());

  cls.def("__del__", [](axon::AxonRuntime& self) {
    nb::gil_scoped_release release;
    self.Stop();
  });

  cls.def("start", [](axon::AxonRuntime& self) {
    // Auto-register wake manager before starting
    auto& wake_manager = python::GetPythonWakeManager();
    wake_manager.RegisterAsyncioReader();
    auto result = self.Start();
    if (!result) {
      std::string error_msg =
        python::BuildErrorMessageFromContext(result.error(), "Start failed");
      python::ThrowRuntimeError(error_msg);
    }
  });

  cls.def("start_server", [](axon::AxonRuntime& self) {
    auto& wake_manager = python::GetPythonWakeManager();
    wake_manager.RegisterAsyncioReader();
    auto result = self.StartServer();
    if (!result) {
      std::string error_msg = python::BuildErrorMessageFromContext(
        result.error(), "StartServer failed");
      python::ThrowRuntimeError(error_msg);
    }
  });

  cls.def("start_client", [](axon::AxonRuntime& self) {
    auto& wake_manager = python::GetPythonWakeManager();
    wake_manager.RegisterAsyncioReader();
    auto result = self.StartClient();
    if (!result) {
      std::string error_msg = python::BuildErrorMessageFromContext(
        result.error(), "StartClient failed");
      python::ThrowRuntimeError(error_msg);
    }
  });

  cls
    .def(
      "stop",
      [](axon::AxonRuntime& self) {
        self.Stop();
        // Unregister asyncio reader after stopping to allow proper cleanup
        auto& wake_manager = python::GetPythonWakeManager();
        wake_manager.UnregisterAsyncioReader();
        // After stopping, ensure all Python objects are released
        // by acquiring GIL and forcing garbage collection
        {
          nb::gil_scoped_acquire acquire;
          try {
            nb::module_ gc = nb::module_::import_("gc");
            gc.attr("collect")();
          } catch (...) {
            // Ignore GC errors
          }
        }
      },
      nb::call_guard<nb::gil_scoped_release>())
    .def(
      "stop_server",
      [](axon::AxonRuntime& self) {
        self.StopServer();
        // Unregister asyncio reader after stopping server
        auto& wake_manager = python::GetPythonWakeManager();
        wake_manager.UnregisterAsyncioReader();
      },
      nb::call_guard<nb::gil_scoped_release>())
    .def(
      "stop_client",
      [](axon::AxonRuntime& self) {
        self.StopClient();

        // Unregister asyncio reader after stopping client
        auto& wake_manager = python::GetPythonWakeManager();
        wake_manager.UnregisterAsyncioReader();
      },
      nb::call_guard<nb::gil_scoped_release>());

  cls.def(
    "get_local_address",
    [](axon::AxonRuntime& self) {
      auto addr = self.GetLocalAddress();
      return nb::bytes(reinterpret_cast<const char*>(addr.data()), addr.size());
    },
    nb::call_guard<nb::gil_scoped_release>());

  cls.def(
    "get_local_signatures",
    [](axon::AxonRuntime& self) {
      auto sigs = self.GetLocalSignatures();
      return nb::bytes(reinterpret_cast<const char*>(sigs.data()), sigs.size());
    },
    nb::call_guard<nb::gil_scoped_release>());

  cls.def(
    "connect_endpoint_async",
    [](
      axon::AxonRuntime& self, nb::object ucp_address_obj,
      std::string_view worker_name) {
      // Convert Python bytes or sequence to std::vector<std::byte>
      std::vector<std::byte> ucp_address;

      if (nb::isinstance<nb::bytes>(ucp_address_obj)) {
        nb::bytes b = nb::cast<nb::bytes>(ucp_address_obj);
        const char* data = b.c_str();
        size_t size = b.size();
        if (size == 0) {
          throw std::runtime_error("ucp_address bytes is empty");
        }
        ucp_address.resize(size);
        for (size_t i = 0; i < size; ++i) {
          ucp_address[i] = static_cast<std::byte>(data[i]);
        }
      } else {
        try {
          std::vector<uint8_t> uint8_vec =
            nb::cast<std::vector<uint8_t>>(ucp_address_obj);
          ucp_address.reserve(uint8_vec.size());
          for (uint8_t byte : uint8_vec) {
            ucp_address.push_back(static_cast<std::byte>(byte));
          }
        } catch (const nb::cast_error&) {
          throw nb::type_error(
            "ucp_address must be bytes or a sequence of integers (0-255)");
        }
      }

      // Import asyncio
      nb::module_ asyncio = python::GetAsyncioModule();

      nb::object future = asyncio.attr("Future")();

      // Incref future to keep it alive across threads
      PyObject* future_ptr = future.ptr();
      Py_INCREF(future_ptr);

      auto& wake_manager = python::GetPythonWakeManager();

      // Call ConnectEndpointAsync directly
      auto sender =
        self.ConnectEndpointAsync(std::move(ucp_address), worker_name);

      // Spawn task on the client async scope
      self.SpawnClientTask(unifex::on(
        self.GetTimeContextScheduler(),
        std::move(sender)
          | unifex::then(
            [future_ptr, &wake_manager](
              std::expected<uint64_t, axon::errors::AxonErrorContext>
                result) mutable {
              if (result.has_value()) {
                wake_manager.Enqueue(pro::make_proxy<python::TaskFacade>(
                  [future_ptr, value = result.value()]() mutable {
                    nb::object future = nb::steal<nb::object>(future_ptr);
                    future.attr("set_result")(nb::cast(value));
                  }));
              } else {
                const auto& error_ctx = result.error();
                std::string error_msg = python::BuildErrorMessageFromContext(
                  error_ctx, "Connection failed");
                wake_manager.Enqueue(pro::make_proxy<python::TaskFacade>(
                  [future_ptr, error_msg = std::move(error_msg)]() mutable {
                    nb::object future = nb::steal<nb::object>(future_ptr);
                    python::SetFutureException(future, error_msg);
                  }));
              }
            })
          | unifex::upon_error([future_ptr, &wake_manager](auto&& error) {
              using ErrorType = std::decay_t<decltype(error)>;
              constexpr bool is_axon_error_context =
                std::is_same_v<ErrorType, axon::errors::AxonErrorContext>;
              constexpr bool is_exception_ptr =
                std::is_same_v<ErrorType, std::exception_ptr>;

              std::string error_msg = "Unknown error in connection";

              if constexpr (is_axon_error_context) {
                error_msg = python::BuildErrorMessageFromContext(
                  error, "Connection failed");
              } else if constexpr (is_exception_ptr) {
                try {
                  std::rethrow_exception(error);
                } catch (const axon::errors::AxonErrorException& e) {
                  error_msg = python::BuildErrorMessageFromContext(
                    e.context(), "Connection failed");
                } catch (const rpc::RpcException& e) {
                  std::string_view error_msg_view = e.what();
                  if (!error_msg_view.empty()) {
                    error_msg = error_msg_view;
                  } else {
                    error_msg = "RPC error: " + e.code().message();
                  }
                } catch (const std::exception& e) {
                  error_msg = e.what();
                }
              }

              wake_manager.Enqueue(pro::make_proxy<python::TaskFacade>(
                [future_ptr, error_msg = std::move(error_msg)]() mutable {
                  nb::object future = nb::steal<nb::object>(future_ptr);
                  python::SetFutureException(future, error_msg);
                }));
            })));
      return future;
    },
    nb::arg("ucp_address"), nb::arg("worker_name"));

  cls.def(
    "register_function_raw",
    [](
      axon::AxonRuntime& self, uint32_t function_id,
      const std::string& function_name, std::vector<rpc::ParamType> param_types,
      std::vector<rpc::ParamType> return_types,
      rpc::PayloadType input_payload_type, rpc::PayloadType return_payload_type,
      nb::object py_callable, nb::object memory_policy_factory,
      nb::object lifecycle_policy_obj, nb::object from_dlpack_fn) {
      // Check if function is async
      bool is_async = axon::python::IsAsyncFunction(py_callable);
      if (!is_async) {
        throw std::runtime_error(
          "register_function_raw only accepts async functions. "
          "Please use 'async def' to define your function.");
      }

      // Detect tensor parameters and returns
      std::vector<size_t> tensor_param_indices;
      std::vector<size_t> tensor_return_indices;
      for (size_t i = 0; i < param_types.size(); ++i) {
        if (param_types[i] == rpc::ParamType::TENSOR_META) {
          tensor_param_indices.push_back(i);
        }
      }
      for (size_t i = 0; i < return_types.size(); ++i) {
        if (return_types[i] == rpc::ParamType::TENSOR_META) {
          tensor_return_indices.push_back(i);
        }
      }

      // Determine payload types based on tensor parameters
      rpc::PayloadType actual_input_payload = input_payload_type;
      rpc::PayloadType actual_return_payload = return_payload_type;

      if (tensor_param_indices.size() == 1) {
        actual_input_payload = rpc::PayloadType::UCX_BUFFER;
      } else if (tensor_param_indices.size() > 1) {
        actual_input_payload = rpc::PayloadType::UCX_BUFFER_VEC;
      }

      if (tensor_return_indices.size() == 1) {
        actual_return_payload = rpc::PayloadType::UCX_BUFFER;
      } else if (tensor_return_indices.size() > 1) {
        actual_return_payload = rpc::PayloadType::UCX_BUFFER_VEC;
      }

      // Convert std::vector to cista::offset::vector
      cista::offset::vector<rpc::ParamType> cista_param_types;
      cista_param_types.set(param_types.begin(), param_types.end());
      cista::offset::vector<rpc::ParamType> cista_return_types;
      cista_return_types.set(return_types.begin(), return_types.end());

      // Validate: if memory_policy is set, from_dlpack_fn must also be set
      // for tensor parameters. This is because custom_memory_policy only works
      // in rendezvous path; eager path requires from_dlpack_fn to convert
      // received data to Python tensors.
      if (
        !memory_policy_factory.is_none() && !tensor_param_indices.empty()
        && from_dlpack_fn.is_none()) {
        throw std::runtime_error(std::format(
          "Function '{}' has tensor parameters and custom memory_policy is "
          "set, but from_dlpack_fn is not provided. "
          "from_dlpack_fn is required to handle UCX eager path where data "
          "arrives pre-allocated in host memory and custom_memory_policy "
          "is not invoked. Please provide from_dlpack_fn (e.g., "
          "numpy.from_dlpack, torch.from_dlpack, jax.dlpack.from_dlpack).",
          function_name));
      }

      // Build from_dlpack_fn vectors if user provided one
      std::vector<python::SharedPyObject> tensor_param_from_dlpack;
      std::vector<python::SharedPyObject> tensor_return_from_dlpack;

      if (!from_dlpack_fn.is_none()) {
        // Use user-provided from_dlpack_fn for all tensor parameters
        tensor_param_from_dlpack.reserve(tensor_param_indices.size());
        for (size_t i = 0; i < tensor_param_indices.size(); ++i) {
          tensor_param_from_dlpack.push_back(
            python::SharedPyObject(nb::borrow(from_dlpack_fn)));
        }
        // Also use it for all tensor returns
        tensor_return_from_dlpack.reserve(tensor_return_indices.size());
        for (size_t i = 0; i < tensor_return_indices.size(); ++i) {
          tensor_return_from_dlpack.push_back(
            python::SharedPyObject(nb::borrow(from_dlpack_fn)));
        }
      }

      // Create function object
      python::PythonAsyncFunctionWrapper wrapped_func{
        py_callable,
        std::move(param_types),
        std::move(return_types),
        std::move(tensor_param_indices),
        std::move(tensor_return_indices),
        self,
        std::move(tensor_param_from_dlpack),
        std::move(tensor_return_from_dlpack),
        python::ResultExtractionMode::VOID,  // extraction_mode (unknown)
        {}  // non_tensor_indices - empty for raw register
      };

      // Create DynamicAsyncRpcFunction
      auto dynamic_func = pro::make_proxy<rpc::DynamicAsyncRpcFunctionFacade>(
        std::move(wrapped_func));

      // Create lifecycle policy from Python object
      axon::MessageLifecyclePolicy lc_policy;
      if (lifecycle_policy_obj.is_none()) {
        lc_policy = axon::TransientPolicy{};
      } else {
        lc_policy = axon::python::create_retention_policy(lifecycle_policy_obj);
      }

      // Register function with policies
      python::RegisterFunctionWithPolicies(
        self, rpc::function_id_t{function_id},
        cista::offset::string(function_name), cista_param_types,
        cista_return_types, actual_input_payload, actual_return_payload,
        std::move(dynamic_func), memory_policy_factory, std::move(lc_policy),
        self);
    },
    nb::arg("function_id"), nb::arg("function_name"), nb::arg("param_types"),
    nb::arg("return_types"), nb::arg("input_payload_type"),
    nb::arg("return_payload_type"), nb::arg("callable"),
    nb::arg("memory_policy") = nb::none(),
    nb::arg("lifecycle_policy") = nb::none(),
    nb::arg("from_dlpack_fn") = nb::none());

  // Simplified register_function_raw overload with automatic type inference
  cls.def(
    "register_function_raw",
    [](
      axon::AxonRuntime& self, uint32_t function_id, nb::object py_callable,
      nb::object function_name_obj, nb::object memory_policy_factory,
      nb::object lifecycle_policy_obj) {
      // Check if function is async
      bool is_async = axon::python::IsAsyncFunction(py_callable);
      if (!is_async) {
        throw std::runtime_error(
          "register_function_raw only accepts async functions. "
          "Please use 'async def' to define your function.");
      }

      // Get function name
      std::string function_name;
      if (function_name_obj.is_none()) {
        try {
          nb::object name_attr = py_callable.attr("__name__");
          function_name = nb::cast<std::string>(name_attr);
        } catch (const nb::python_error&) {
          function_name = std::format("function_{}", function_id);
        }
      } else {
        function_name = nb::cast<std::string>(function_name_obj);
      }

      // Parse function signature to infer types
      axon::python::FunctionSignatureInfo sig_info =
        axon::python::ParseFunctionSignature(py_callable);

      // Use inferred types
      std::vector<rpc::ParamType> param_types = std::move(sig_info.param_types);
      std::vector<rpc::ParamType> return_types =
        std::move(sig_info.return_types);

      // If no return types inferred, default to UNKNOWN
      if (return_types.empty()) {
        return_types.push_back(rpc::ParamType::UNKNOWN);
      }

      // Use inferred payload types
      rpc::PayloadType input_payload_type = sig_info.input_payload_type;
      rpc::PayloadType return_payload_type = sig_info.return_payload_type;

      // Convert std::vector to cista::offset::vector
      cista::offset::vector<rpc::ParamType> cista_param_types;
      cista_param_types.set(param_types.begin(), param_types.end());
      cista::offset::vector<rpc::ParamType> cista_return_types;
      cista_return_types.set(return_types.begin(), return_types.end());

      // Create function object
      python::PythonAsyncFunctionWrapper wrapped_func{
        py_callable,
        std::move(param_types),
        std::move(return_types),
        std::move(sig_info.tensor_param_indices),
        std::move(sig_info.tensor_return_indices),
        self,
        std::move(sig_info.tensor_param_from_dlpack),
        std::move(sig_info.tensor_return_from_dlpack),
        sig_info.extraction_mode,
        std::move(sig_info.non_tensor_indices)};

      // Create DynamicAsyncRpcFunction
      auto dynamic_func = pro::make_proxy<rpc::DynamicAsyncRpcFunctionFacade>(
        std::move(wrapped_func));

      // Create lifecycle policy from Python object
      axon::MessageLifecyclePolicy lc_policy;
      if (lifecycle_policy_obj.is_none()) {
        lc_policy = axon::TransientPolicy{};
      } else {
        lc_policy = axon::python::create_retention_policy(lifecycle_policy_obj);
      }

      // Register function with policies
      python::RegisterFunctionWithPolicies(
        self, rpc::function_id_t{function_id},
        cista::offset::string(function_name), cista_param_types,
        cista_return_types, input_payload_type, return_payload_type,
        std::move(dynamic_func), memory_policy_factory, std::move(lc_policy),
        self);
    },
    nb::arg("function_id"), nb::arg("callable"),
    nb::arg("function_name") = nb::none(),
    nb::arg("memory_policy") = nb::none(),
    nb::arg("lifecycle_policy") = nb::none());

  // High-level register_function with strict type checking
  cls.def(
    "register_function",
    [](
      axon::AxonRuntime& self, nb::object py_callable,
      std::optional<uint32_t> function_id, nb::object function_name_obj,
      nb::object memory_policy_factory, nb::object lifecycle_policy_obj,
      nb::object from_dlpack_fn) {
      // Check if function is async
      bool is_async = axon::python::IsAsyncFunction(py_callable);
      if (!is_async) {
        throw std::runtime_error(
          "register_function only accepts async functions. "
          "Please use 'async def' to define your function.");
      }

      // Get function name
      std::string function_name;
      if (function_name_obj.is_none()) {
        try {
          nb::object name_attr = py_callable.attr("__name__");
          function_name = nb::cast<std::string>(name_attr);
        } catch (const nb::python_error&) {
          function_name = std::format("function_{}", function_id.value_or(0));
        }
      } else {
        function_name = nb::cast<std::string>(function_name_obj);
      }

      // Use provided ID or compute from name
      uint32_t final_function_id;
      if (function_id.has_value() && function_id.value() != 0) {
        final_function_id = function_id.value();
      } else {
        final_function_id =
          ComputeFunctionId(function_name, self.GetWorkerName());
      }

      // Parse function signature to infer types
      axon::python::FunctionSignatureInfo sig_info =
        axon::python::ParseFunctionSignature(py_callable);

      // Strict type checking: Verify no UNKNOWN types in parameters
      for (size_t i = 0; i < sig_info.param_types.size(); ++i) {
        if (sig_info.param_types[i] == rpc::ParamType::UNKNOWN) {
          throw nb::type_error(
            std::format(
              "Unsupported or missing type annotation for parameter {} in "
              "function '{}'. "
              "Supported types: int, float, bool, str, List[T], Tuple[...], "
              "Tensor/Array.",
              i, function_name)
              .c_str());
        }
      }

      // Strict type checking: Verify no UNKNOWN types in return
      for (size_t i = 0; i < sig_info.return_types.size(); ++i) {
        if (sig_info.return_types[i] == rpc::ParamType::UNKNOWN) {
          // If return type is UNKNOWN, it might mean missing annotation or
          // unsupported type However, if void/None, it should be VOID
          throw nb::type_error(
            std::format(
              "Unsupported or missing return type annotation for function "
              "'{}'. "
              "Supported types: int, float, bool, str, List[T], Tuple[...], "
              "Tensor/Array, None.",
              function_name)
              .c_str());
        }
      }

      // Use inferred types
      std::vector<rpc::ParamType> param_types = std::move(sig_info.param_types);
      std::vector<rpc::ParamType> return_types =
        std::move(sig_info.return_types);

      // Use inferred payload types
      rpc::PayloadType input_payload_type = sig_info.input_payload_type;
      rpc::PayloadType return_payload_type = sig_info.return_payload_type;

      // Convert std::vector to cista::offset::vector
      cista::offset::vector<rpc::ParamType> cista_param_types;
      cista_param_types.set(param_types.begin(), param_types.end());
      cista::offset::vector<rpc::ParamType> cista_return_types;
      cista_return_types.set(return_types.begin(), return_types.end());

      // Validate: if memory_policy is set, from_dlpack_fn must also be set
      // for tensor parameters. This is because custom_memory_policy only works
      // in rendezvous path; eager path requires from_dlpack_fn to convert
      // received data to Python tensors.
      if (
        !memory_policy_factory.is_none()
        && !sig_info.tensor_param_indices.empty() && from_dlpack_fn.is_none()) {
        throw std::runtime_error(std::format(
          "Function '{}' has tensor parameters and custom memory_policy is "
          "set, but from_dlpack_fn is not provided. "
          "from_dlpack_fn is required to handle UCX eager path where data "
          "arrives pre-allocated in host memory and custom_memory_policy "
          "is not invoked. Please provide from_dlpack_fn (e.g., "
          "numpy.from_dlpack, torch.from_dlpack, jax.dlpack.from_dlpack).",
          function_name));
      }

      // Override from_dlpack_fn callables if user provided one
      std::vector<python::SharedPyObject> tensor_param_from_dlpack;
      std::vector<python::SharedPyObject> tensor_return_from_dlpack;

      if (!from_dlpack_fn.is_none()) {
        // User provided from_dlpack_fn: use it for all tensor parameters
        tensor_param_from_dlpack.reserve(sig_info.tensor_param_indices.size());
        for (size_t i = 0; i < sig_info.tensor_param_indices.size(); ++i) {
          tensor_param_from_dlpack.push_back(
            python::SharedPyObject(nb::borrow(from_dlpack_fn)));
        }
        // Also use it for all tensor returns
        tensor_return_from_dlpack.reserve(
          sig_info.tensor_return_indices.size());
        for (size_t i = 0; i < sig_info.tensor_return_indices.size(); ++i) {
          tensor_return_from_dlpack.push_back(
            python::SharedPyObject(nb::borrow(from_dlpack_fn)));
        }
      } else {
        // Use auto-detected from_dlpack_fn callables
        tensor_param_from_dlpack = std::move(sig_info.tensor_param_from_dlpack);
        tensor_return_from_dlpack =
          std::move(sig_info.tensor_return_from_dlpack);
      }

      // Create function object
      python::PythonAsyncFunctionWrapper wrapped_func{
        py_callable,
        std::move(param_types),
        std::move(return_types),
        std::move(sig_info.tensor_param_indices),
        std::move(sig_info.tensor_return_indices),
        self,
        std::move(tensor_param_from_dlpack),
        std::move(tensor_return_from_dlpack),
        sig_info.extraction_mode,
        std::move(sig_info.non_tensor_indices)};

      // Create DynamicAsyncRpcFunction
      auto dynamic_func = pro::make_proxy<rpc::DynamicAsyncRpcFunctionFacade>(
        std::move(wrapped_func));

      // Create lifecycle policy from Python object
      axon::MessageLifecyclePolicy lc_policy;
      if (lifecycle_policy_obj.is_none()) {
        lc_policy = axon::TransientPolicy{};
      } else {
        lc_policy = axon::python::create_retention_policy(lifecycle_policy_obj);
      }

      // Register function with policies
      python::RegisterFunctionWithPolicies(
        self, rpc::function_id_t{final_function_id},
        cista::offset::string(function_name), cista_param_types,
        cista_return_types, input_payload_type, return_payload_type,
        std::move(dynamic_func), memory_policy_factory, std::move(lc_policy),
        self);
    },
    nb::arg("callable"), nb::arg("function_id") = nb::none(),
    nb::arg("function_name") = nb::none(),
    nb::arg("memory_policy") = nb::none(),
    nb::arg("lifecycle_policy") = nb::none(),
    nb::arg("from_dlpack_fn") = nb::none());

  cls.def(
    "invoke_raw",
    [](
      axon::AxonRuntime& self, std::string_view worker_name,
      nb::object request_header_obj, nb::object payload_obj,
      nb::object memory_policy_factory) {
      // Import asyncio and create future
      nb::module_ asyncio = python::GetAsyncioModule();
      nb::object future = asyncio.attr("Future")();

      if (request_header_obj.is_none()) {
        throw std::runtime_error("request_header cannot be None");
      }

      // Create result handler
      auto result_handler =
        python::CreateRpcResultHandler(future, python::GetPythonWakeManager());

      // Determine if using custom memory policy
      bool use_custom_memory = !memory_policy_factory.is_none();

      // Detect payload type
      python::PayloadTypeInfo payload_info =
        python::DetectPayloadType(payload_obj);

      // Define dispatch logic
      auto dispatch_rpc = [&](auto&& handler) {
        if (payload_info.type == python::PayloadTypeInfo::NONE) {
          handler(std::monostate{});
        } else {
          try {
            auto mr = self.GetMemoryResourceManager();
            if (payload_info.type == python::PayloadTypeInfo::UCX_BUFFER) {
              ucxx::UcxBuffer buffer =
                python::CreateUcxBufferFromPayload(payload_obj, mr);
              handler(std::move(buffer));
            } else if (
              payload_info.type == python::PayloadTypeInfo::UCX_BUFFER_VEC) {
              ucxx::UcxBufferVec buffer_vec =
                python::CreateUcxBufferVecFromPayload(payload_obj, mr);
              handler(std::move(buffer_vec));
            } else {
              handler(std::monostate{});
            }
          } catch (const nb::python_error& e) {
            handler(std::monostate{});
          }
        }
      };

      if (use_custom_memory) {
        auto handler =
          [&self, &memory_policy_factory, worker_name, request_header_obj,
           result_handler = std::move(result_handler)](auto&& payload) mutable {
            rpc::RpcRequestHeader& header_ref =
              nb::cast<rpc::RpcRequestHeader&>(request_header_obj);
            self.SpawnClientTask(python::InvokeRpcWithCustomMemory(
              self, worker_name, std::move(header_ref), std::move(payload),
              memory_policy_factory, std::move(result_handler)));
          };
        dispatch_rpc(std::move(handler));
      } else {
        auto handler = [&self, worker_name, request_header_obj,
                        result_handler =
                          std::move(result_handler)](auto&& payload) mutable {
          rpc::RpcRequestHeader& header_ref =
            nb::cast<rpc::RpcRequestHeader&>(request_header_obj);
          self.SpawnClientTask(python::InvokeRpcWithHostPolicy(
            self, worker_name, std::move(header_ref), std::move(payload),
            std::move(result_handler)));
        };
        dispatch_rpc(std::move(handler));
      }
      return future;
    },
    nb::arg("worker_name"), nb::arg("request_header"),
    nb::arg("payload") = nb::none(), nb::arg("memory_policy") = nb::none(),
    "Invoke RPC natively with strictly structured request header.\n");

  cls.def(
    "invoke",
    [](
      axon::AxonRuntime& self, nb::args args, const std::string& worker_name,
      uint32_t session_id, nb::object function, uint32_t workflow_id,
      nb::object memory_policy_factory,
      nb::object from_dlpack_fn) {  // Removed kwargs
      // Import asyncio
      nb::module_ asyncio = python::GetAsyncioModule();
      nb::object future = asyncio.attr("Future")();

      // Resolve function ID
      rpc::function_id_t function_id;
      if (nb::isinstance<nb::int_>(function)) {
        function_id = rpc::function_id_t{nb::cast<uint32_t>(function)};
      } else if (nb::isinstance<nb::str>(function)) {
        std::string fname = nb::cast<std::string>(function);
        function_id = rpc::function_id_t{ComputeFunctionId(fname, worker_name)};
      } else {
        throw std::invalid_argument(
          "function_id must be int (ID) or str (name)");
      }

      // Build header
      rpc::RpcRequestHeader request_header;
      request_header.session_id = rpc::session_id_t{session_id};
      request_header.function_id = function_id;
      request_header.workflow_id = rpc::utils::workflow_id_t{workflow_id};

      // Use modular InvokeContext for single-pass argument processing
      python::InvokeContext ctx(request_header);

      // Single pass: classify and collect all arguments
      for (size_t i = 0; i < args.size(); ++i) {
        nb::object obj = nb::cast<nb::object>(args[i]);
        if (axon::python::IsDlpackTensor(obj)) {
          ctx.add_tensor(std::move(obj));
        } else {
          ctx.add_non_tensor(std::move(obj));
        }
      }

      // Finalize header with tensor metadata (added at end for O(1) lookup)
      ctx.finalize_header();

      // Prepare for dispatch
      auto mr = self.GetMemoryResourceManager();
      bool use_custom_memory = !memory_policy_factory.is_none();
      auto result_handler = python::CreateRpcResultHandler(
        future, python::GetPythonWakeManager(), std::move(from_dlpack_fn));

      // Dispatch based on tensor count
      if (ctx.tensor_count() == 0) {
        if (use_custom_memory) {
          self.SpawnClientTask(python::InvokeRpcWithCustomMemory(
            self, worker_name, std::move(request_header), std::monostate{},
            memory_policy_factory, std::move(result_handler)));
        } else {
          self.SpawnClientTask(python::InvokeRpcWithHostPolicy(
            self, worker_name, std::move(request_header), std::monostate{},
            std::move(result_handler)));
        }
      } else if (ctx.tensor_count() == 1) {
        // Use cached data to create buffer (no repeated ExtractDlpackTensor)
        ucxx::UcxBuffer buffer = ctx.to_ucx_buffer(mr);
        if (use_custom_memory) {
          self.SpawnClientTask(python::InvokeRpcWithCustomMemory(
            self, worker_name, std::move(request_header), std::move(buffer),
            memory_policy_factory, std::move(result_handler)));
        } else {
          self.SpawnClientTask(python::InvokeRpcWithHostPolicy(
            self, worker_name, std::move(request_header), std::move(buffer),
            std::move(result_handler)));
        }
      } else {
        // Vector path
        ucxx::UcxBufferVec buffer_vec = ctx.to_ucx_buffer_vec(mr);
        if (use_custom_memory) {
          self.SpawnClientTask(python::InvokeRpcWithCustomMemory(
            self, worker_name, std::move(request_header), std::move(buffer_vec),
            memory_policy_factory, std::move(result_handler)));
        } else {
          self.SpawnClientTask(python::InvokeRpcWithHostPolicy(
            self, worker_name, std::move(request_header), std::move(buffer_vec),
            std::move(result_handler)));
        }
      }

      return future;
    },
    nb::rv_policy::none,  // Don't apply any rv_policy to prevent copy of self
    nb::arg("args"), nb::arg("worker_name"), nb::arg("session_id"),
    nb::arg("function"), nb::arg("workflow_id") = 0,
    nb::arg("memory_policy") = nb::none(),
    nb::arg("from_dlpack_fn") = nb::none(),
    "Invoke RPC with natural Python arguments (automatically handles Tensor "
    "payloads).\n");
}
