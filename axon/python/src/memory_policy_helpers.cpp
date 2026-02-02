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

#include "axon/python/src/memory_policy_helpers.hpp"

#include <nanobind/nanobind.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "axon/axon_runtime.hpp"
#include "axon/memory_policy.hpp"
#include "axon/message_lifecycle_policy.hpp"
#include "axon/python/src/dlpack_helpers.hpp"
#include "axon/python/src/python_helpers.hpp"
#include "axon/python/src/python_module.hpp"
#include "axon/utils/axon_message.hpp"
#include "rpc_core/rpc_payload_types.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;
namespace ucxx = eux::ucxx;
namespace rpc = eux::rpc;

namespace {
std::unordered_map<const void*, nb::object> g_custom_buffer_map;

void UnregisterCustomBuffer(const void* ptr) { g_custom_buffer_map.erase(ptr); }
}  // namespace

void RegisterCustomBuffer(const void* ptr, nb::object obj) {
  if (!ptr) {
    return;
  }

  auto weakref_mod = GetWeakRefModule();
  auto callback = nb::cpp_function(
    [ptr](nb::object /*ref*/) { UnregisterCustomBuffer(ptr); });
  nb::object ref = weakref_mod.attr("ref")(obj, callback);

  g_custom_buffer_map[ptr] = ref;
}

nb::object TryGetCustomBuffer(const void* ptr) {
  if (!ptr) {
    return nb::none();
  }
  auto it = g_custom_buffer_map.find(ptr);
  if (it != g_custom_buffer_map.end()) {
    // it->second is the weakref. Call it to get the object.
    return it->second();
  }
  return nb::none();
}

template <typename BufferType>
axon::CustomMemoryPolicy<BufferType> create_custom_memory_policy(
  nb::object py_callable, AxonRuntime& runtime) {
  SharedPyObject safe_callable(std::move(py_callable));

  auto provider = [safe_callable = std::move(safe_callable),
                   &runtime](rpc::utils::TensorMetaSpan meta) -> BufferType {
    nb::gil_scoped_acquire acquire;
    nb::object py_callable = safe_callable.get();

    // Pass meta as a list of TensorMeta objects (requires TensorMeta binding)
    nb::list py_meta;
    for (const auto& m : meta) {
      py_meta.append(nb::cast(m));
    }
    nb::object py_result = py_callable(py_meta);

    if constexpr (std::is_same_v<BufferType, ucxx::UcxBuffer>) {
      if (meta.size() != 1) {
        throw std::runtime_error(
          "Expected single size for UcxBuffer, got "
          + std::to_string(meta.size()));
      }
      auto mr = runtime.GetMemoryResourceManager();
      auto buffer = DlpackToUcxBuffer(py_result, meta[0], mr);
      RegisterCustomBuffer(buffer.data(), py_result);
      return buffer;
    } else if constexpr (std::is_same_v<BufferType, ucxx::UcxBufferVec>) {
      auto mr = runtime.GetMemoryResourceManager();
      auto buffer_vec = DlpackToUcxBufferVec(py_result, meta, mr);
      // Register each buffer in the vector
      const auto& buffers = buffer_vec.buffers();
      nb::list py_list = nb::cast<nb::list>(py_result);
      for (size_t i = 0; i < buffers.size(); ++i) {
        RegisterCustomBuffer(buffers[i].data, py_list[i]);
      }
      return buffer_vec;
    } else if constexpr (std::is_same_v<BufferType, rpc::PayloadVariant>) {
      auto mr = runtime.GetMemoryResourceManager();
      if (meta.size() == 1) {
        auto buffer = DlpackToUcxBuffer(py_result, meta[0], mr);
        RegisterCustomBuffer(buffer.data(), py_result);
        return rpc::PayloadVariant(
          std::in_place_type<ucxx::UcxBuffer>, std::move(buffer));
      } else {
        auto buffer_vec = DlpackToUcxBufferVec(py_result, meta, mr);
        const auto& buffers = buffer_vec.buffers();
        nb::list py_list = nb::cast<nb::list>(py_result);
        for (size_t i = 0; i < buffers.size(); ++i) {
          RegisterCustomBuffer(buffers[i].data, py_list[i]);
        }
        return rpc::PayloadVariant(
          std::in_place_type<ucxx::UcxBufferVec>, std::move(buffer_vec));
      }
    } else {
      static_assert(
        std::is_same_v<BufferType, ucxx::UcxBuffer>
          || std::is_same_v<BufferType, ucxx::UcxBufferVec>
          || std::is_same_v<BufferType, rpc::PayloadVariant>,
        "Unsupported BufferType");
    }
  };
  return CustomMemoryPolicy<BufferType>(
    ::pro::make_proxy<axon::BufferProviderFacade<BufferType>>(
      std::move(provider)));
}

// Explicit template instantiations
template axon::CustomMemoryPolicy<ucxx::UcxBuffer> create_custom_memory_policy<
  ucxx::UcxBuffer>(nb::object py_callable, AxonRuntime& runtime);

template axon::CustomMemoryPolicy<ucxx::UcxBufferVec>
create_custom_memory_policy<ucxx::UcxBufferVec>(
  nb::object py_callable, AxonRuntime& runtime);

template axon::CustomMemoryPolicy<rpc::PayloadVariant>
create_custom_memory_policy<rpc::PayloadVariant>(
  nb::object py_callable, AxonRuntime& runtime);

axon::RetentionPolicy create_retention_policy(nb::object py_callable) {
  // Wrap Python callable as LifecycleStatusHandlerFacade
  // Wrap Python callable in SharedPyObject to ensure thread-safe destruction
  SharedPyObject safe_callable(std::move(py_callable));

  auto handler =
    [safe_callable = std::move(safe_callable)](
      std::shared_ptr<axon::utils::AxonRequest> request,
      axon::utils::AxonMessageID message_id) -> axon::LifecycleStatus {
    nb::gil_scoped_acquire acquire;
    nb::object py_callable = safe_callable.get();

    // Call Python function with request and message_id
    // Python function should return an integer representing LifecycleStatus
    nb::object py_result = py_callable(nb::cast(request), nb::cast(message_id));
    // Convert Python result to LifecycleStatus enum
    int status_value = nb::cast<int>(py_result);
    return static_cast<axon::LifecycleStatus>(status_value);
  };
  return axon::RetentionPolicy(
    ::pro::make_proxy<axon::LifecycleStatusHandlerFacade>(std::move(handler)));
}

template <typename BufferType>
axon::ReceiverMemoryPolicy<BufferType> create_memory_policy(
  nb::object py_obj, AxonRuntime& runtime) {
  if (py_obj.is_none()) {
    // Return AlwaysOnHostPolicy
    return axon::AlwaysOnHostPolicy{};
  } else {
    // Create CustomMemoryPolicy from Python callable
    auto custom_policy =
      create_custom_memory_policy<BufferType>(py_obj, runtime);
    return custom_policy;
  }
}

// Explicit template instantiations for create_memory_policy
template axon::ReceiverMemoryPolicy<ucxx::UcxBuffer>
create_memory_policy<ucxx::UcxBuffer>(nb::object py_obj, AxonRuntime& runtime);

template axon::ReceiverMemoryPolicy<ucxx::UcxBufferVec> create_memory_policy<
  ucxx::UcxBufferVec>(nb::object py_obj, AxonRuntime& runtime);

template axon::ReceiverMemoryPolicy<rpc::PayloadVariant> create_memory_policy<
  rpc::PayloadVariant>(nb::object py_obj, AxonRuntime& runtime);

}  // namespace python
}  // namespace axon
}  // namespace eux
