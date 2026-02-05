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

#include "axon/python/src/bindings_runtime_wrapper.hpp"

#include <cstdio>
#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>

#include <utility>

#include <unifex/create.hpp>
#include <unifex/then.hpp>

#include "axon/axon_runtime.hpp"
#include "axon/python/src/dlpack_helpers.hpp"
#include "axon/python/src/memory_policy_helpers.hpp"
#include "axon/python/src/param_conversion.hpp"
#include "axon/python/src/python_module.hpp"
#include "axon/python/src/python_wake_manager.hpp"
#include "rpc_core/rpc_types.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace axon {
namespace python {

// PythonAsyncFunctionWrapper implementation
unifex::any_sender_of<
  std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
PythonAsyncFunctionWrapper::operator()(
  const data::vector<rpc::ParamMeta>& params) const {
  return FunctionImpl(params, std::monostate{});
}

unifex::any_sender_of<
  std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
PythonAsyncFunctionWrapper::operator()(
  const data::vector<rpc::ParamMeta>& params,
  const std::monostate& payload) const {
  return FunctionImpl(params, payload);
}

unifex::any_sender_of<
  std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
PythonAsyncFunctionWrapper::operator()(
  const data::vector<rpc::ParamMeta>& params,
  const ucxx::UcxBuffer& payload) const {
  return FunctionImpl(params, payload);
}

unifex::any_sender_of<
  std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
PythonAsyncFunctionWrapper::operator()(
  const data::vector<rpc::ParamMeta>& params,
  const ucxx::UcxBufferVec& payload) const {
  return FunctionImpl(params, payload);
}

template <typename PayloadT>
nb::object PythonAsyncFunctionWrapper::ConvertSingleParamToPython(
  size_t tensor_idx, rpc::utils::TensorMeta&& meta,
  const PayloadT& payload) const {
  // Check if we have a cached Python object from custom memory policy
  // This avoids reconstructing the object from UcxBuffer/UcxBufferVec
  if constexpr (std::is_same_v<PayloadT, ucxx::UcxBuffer>) {
    auto cached_obj = TryGetCustomBuffer(payload.data());
    if (!cached_obj.is_none()) {
      return cached_obj;
    }
  } else if constexpr (std::is_same_v<PayloadT, ucxx::UcxBufferVec>) {
    const auto& buffers = payload.buffers();
    if (tensor_idx < buffers.size()) {
      auto cached_obj = TryGetCustomBuffer(buffers[tensor_idx].data);
      if (!cached_obj.is_none()) {
        return cached_obj;
      }
    }
  }

  nb::object dltensor_capsule;

  if constexpr (std::is_same_v<PayloadT, ucxx::UcxBuffer>) {
    // Create a non-owning UcxBuffer view for this tensor
    ucxx::UcxBuffer non_owning_buffer(
      self.GetMemoryResourceManager(), payload.type(), payload.data(),
      payload.size(), payload.mem_h(), false);
    dltensor_capsule =
      TensorMetaToDlpack(std::move(meta), std::move(non_owning_buffer));
  } else if constexpr (std::is_same_v<PayloadT, ucxx::UcxBufferVec>) {
    const auto& buffers = payload.buffers();
    if (tensor_idx >= buffers.size()) {
      throw std::runtime_error("Tensor index out of range in UcxBufferVec");
    }

    // Create a non-owning UcxBuffer view for this specific tensor
    // instead of passing the entire UcxBufferVec
    const auto& buffer = buffers[tensor_idx];
    ucxx::UcxBuffer non_owning_buffer(
      self.GetMemoryResourceManager(), payload.type(), buffer.data, buffer.size,
      payload.mem_h(), false);

    dltensor_capsule =
      TensorMetaToDlpack(std::move(meta), std::move(non_owning_buffer));
  } else {
    throw std::runtime_error(
      "TensorMeta parameter requires UcxBuffer or UcxBufferVec payload");
  }

  // Use saved from_dlpack_fn method to convert to user-expected type
  if (tensor_idx < tensor_param_from_dlpack.size()) {
    const auto& from_dlpack_ref = tensor_param_from_dlpack[tensor_idx];
    if (from_dlpack_ref) {
      nb::object from_dlpack_fn = from_dlpack_ref.get();
      if (!from_dlpack_fn.is_none()) {
        // Call type.from_dlpack_fn(dltensor_capsule) to get the correct type
        return from_dlpack_fn(dltensor_capsule);
      }
    }
  }

  // Fallback: return raw dltensor capsule if no from_dlpack_fn is available
  return dltensor_capsule;
}

template <typename PayloadT>
nb::list PythonAsyncFunctionWrapper::ConvertParamsToPython(
  data::vector<rpc::ParamMeta>& params, const PayloadT& payload) const {
  nb::list py_args;

  const size_t num_params = param_types.size();
  if (num_params == 0) {
    return py_args;
  }

  // ============================================================
  // Small Buffer Optimization: Use stack for typical RPC cases
  // ============================================================
  // Most RPC functions have <= 16 parameters, avoid heap allocation
  static constexpr size_t kMaxStackParams = 16;

  // Stack-allocated buffer for non-tensor param pointers
  std::array<const rpc::ParamMeta*, kMaxStackParams> stack_non_tensor_params;
  std::vector<const rpc::ParamMeta*> heap_non_tensor_params;

  // Choose stack or heap based on param count
  const rpc::ParamMeta** non_tensor_params_ptr;
  if (params.size() <= kMaxStackParams) {
    non_tensor_params_ptr = stack_non_tensor_params.data();
  } else {
    heap_non_tensor_params.reserve(params.size());
    non_tensor_params_ptr = nullptr;  // Will use vector directly
  }

  // ============================================================
  // Single pass: Find TENSOR_META_VEC + collect non-tensor params
  // ============================================================
  const rpc::TensorMetaVec* tensor_meta_vec_ptr = nullptr;
  const rpc::utils::TensorMeta* single_tensor_meta_ptr = nullptr;
  size_t non_tensor_param_count = 0;

  for (const auto& param : params) {
    if (param.type == rpc::ParamType::TENSOR_META_VEC) {
      tensor_meta_vec_ptr = &cista::get<rpc::TensorMetaVec>(param.value);
    } else if (param.type == rpc::ParamType::TENSOR_META) {
      single_tensor_meta_ptr = &cista::get<rpc::utils::TensorMeta>(param.value);
    } else {
      if (non_tensor_params_ptr) {
        non_tensor_params_ptr[non_tensor_param_count++] = &param;
      } else {
        heap_non_tensor_params.push_back(&param);
      }
    }
  }

  // If we used heap, update the count
  if (!non_tensor_params_ptr) {
    non_tensor_param_count = heap_non_tensor_params.size();
    non_tensor_params_ptr = heap_non_tensor_params.data();
  }

  size_t tensor_idx = 0;
  size_t non_tensor_idx = 0;

  for (size_t i = 0; i < num_params; ++i) {
    if (param_types[i] == rpc::ParamType::TENSOR_META) {
      // Try TENSOR_META_VEC first (common case for multiple tensors)
      if (tensor_meta_vec_ptr && tensor_idx < tensor_meta_vec_ptr->size()) {
        // Move meta from vector (zero-copy)
        auto py_dlpack = ConvertSingleParamToPython(
          tensor_idx,
          std::move(const_cast<rpc::utils::TensorMeta&>(
            (*tensor_meta_vec_ptr)[tensor_idx])),
          payload);
        py_args.append(py_dlpack);
        ++tensor_idx;
        continue;
      }
      // Fallback to direct param if no TENSOR_META_VEC found
      else if (single_tensor_meta_ptr && tensor_idx == 0) {
        // Move meta from single_tensor_meta_ptr (zero-copy)
        auto py_dlpack = ConvertSingleParamToPython(
          tensor_idx,
          std::move(
            const_cast<rpc::utils::TensorMeta&>(*single_tensor_meta_ptr)),
          payload);
        py_args.append(py_dlpack);
        ++tensor_idx;
        continue;
      }

      throw std::runtime_error(
        "Tensor meta not found for parameter " + std::to_string(i));
    } else {
      // Non-tensor parameter: direct O(1) lookup
      if (non_tensor_idx < non_tensor_param_count) {
        py_args.append(
          ResultMetaToPython(*non_tensor_params_ptr[non_tensor_idx]));
        ++non_tensor_idx;
      }
    }
  }

  return py_args;
}

template <typename PayloadT>
nb::object PythonAsyncFunctionWrapper::ConvertPayloadToPython(
  PayloadT&& payload) const {
  if constexpr (std::is_same_v<PayloadT, std::monostate>) {
    return nb::none();
  } else if constexpr (std::is_same_v<PayloadT, ucxx::UcxBuffer>) {
    if (tensor_param_indices.empty()) {
      return UcxBufferToDLTensor(std::move(payload));
    }
    return nb::none();
  } else if constexpr (std::is_same_v<PayloadT, ucxx::UcxBufferVec>) {
    if (tensor_param_indices.empty()) {
      return UcxBufferVecToDLTensor(std::move(payload));
    }
    return nb::none();
  }
  return nb::none();
}

std::pair<rpc::ParamMeta, rpc::ReturnedPayload>
PythonAsyncFunctionWrapper::ConvertTensorReturns(
  nb::object dlpack_return) const {
  auto [dlm, owner] = ExtractDlpackTensor(dlpack_return);
  if (!dlm) {
    throw std::runtime_error(
      "Failed to extract DLManagedTensor from dlpack object");
  }
  // Extract tensor meta from DLManagedTensor
  rpc::utils::TensorMeta meta;
  meta.device = dlm->dl_tensor.device;
  meta.dtype = dlm->dl_tensor.dtype;
  meta.ndim = dlm->dl_tensor.ndim;
  meta.shape.clear();
  meta.shape.reserve(dlm->dl_tensor.ndim);
  for (int i = 0; i < dlm->dl_tensor.ndim; ++i) {
    meta.shape.push_back(dlm->dl_tensor.shape[i]);
  }
  if (dlm->dl_tensor.strides != nullptr) {
    meta.strides.clear();
    meta.strides.reserve(dlm->dl_tensor.ndim);
    for (int i = 0; i < dlm->dl_tensor.ndim; ++i) {
      meta.strides.push_back(dlm->dl_tensor.strides[i]);
    }
  }
  meta.byte_offset = dlm->dl_tensor.byte_offset;

  // Get memory type from first tensor
  auto mem_type = DlDeviceToUcxMemoryType(dlm->dl_tensor.device);

  // Calculate buffer size and data pointer
  size_t size = rpc::utils::CalculateTensorSize(meta);
  void* data_ptr =
    static_cast<char*>(dlm->dl_tensor.data) + dlm->dl_tensor.byte_offset;

  // Create UcxBufferVec with owners to keep data alive
  auto mr = self.GetMemoryResourceManager();
  ucxx::UcxBuffer buffer(
    mr, mem_type, {data_ptr, size}, nullptr, true,
    [owner = SharedPyObject(std::move(owner))](void*) {});

  rpc::ParamMeta meta_param;
  meta_param.type = rpc::ParamType::TENSOR_META;
  meta_param.value.template emplace<rpc::TensorMeta>(std::move(meta));

  return {std::move(meta_param), std::move(buffer)};
}

// Template implementation for tensor sequence conversion
template <typename Container>
std::pair<rpc::ParamMeta, rpc::ReturnedPayload> ConvertTensorReturnsImpl(
  const Container& dlpack_returns, axon::AxonRuntime& self) {
  // Build TensorMetaVec and UcxBufferVec in a SINGLE PASS
  // to avoid calling ExtractDlpackTensor twice per tensor
  // (which would cause use-after-free for readonly arrays using
  // __array_interface__)

  rpc::TensorMetaVec meta_vec_value;
  std::vector<ucx_buffer_t> buffers;
  std::vector<SharedPyObject> owners;

  size_t count = 0;
  if constexpr (std::is_convertible_v<Container, nb::handle>) {
    count = nb::len(dlpack_returns);
  } else {
    count = dlpack_returns.size();
  }
  meta_vec_value.reserve(count);
  buffers.reserve(count);
  owners.reserve(count);

  ucx_memory_type_t mem_type = ucx_memory_type::HOST;

  for (auto&& item : dlpack_returns) {
    // Handle both nb::object (from span) and nb::handle (from sequence
    // iterator)
    nb::object dlpack = nb::borrow<nb::object>(item);

    auto [dlm, owner] = ExtractDlpackTensor(dlpack);
    if (!dlm) {
      throw std::runtime_error(
        "Failed to extract DLManagedTensor from dlpack object");
    }

    // Extract tensor meta from DLManagedTensor
    rpc::utils::TensorMeta meta;
    meta.device = dlm->dl_tensor.device;
    meta.dtype = dlm->dl_tensor.dtype;
    meta.ndim = dlm->dl_tensor.ndim;
    meta.shape.clear();
    meta.shape.reserve(dlm->dl_tensor.ndim);
    for (int i = 0; i < dlm->dl_tensor.ndim; ++i) {
      meta.shape.push_back(dlm->dl_tensor.shape[i]);
    }
    if (dlm->dl_tensor.strides != nullptr) {
      meta.strides.clear();
      meta.strides.reserve(dlm->dl_tensor.ndim);
      for (int i = 0; i < dlm->dl_tensor.ndim; ++i) {
        meta.strides.push_back(dlm->dl_tensor.strides[i]);
      }
    }
    meta.byte_offset = dlm->dl_tensor.byte_offset;

    // Get memory type from first tensor
    if (meta_vec_value.empty()) {
      mem_type = DlDeviceToUcxMemoryType(dlm->dl_tensor.device);
    }

    // Calculate buffer size and data pointer
    size_t size = rpc::utils::CalculateTensorSize(meta);
    void* data_ptr =
      static_cast<char*>(dlm->dl_tensor.data) + dlm->dl_tensor.byte_offset;

    meta_vec_value.push_back(std::move(meta));
    buffers.push_back({data_ptr, size});
    owners.emplace_back(std::move(owner));
  }

  // Create UcxBufferVec with owners to keep data alive
  auto mr = self.GetMemoryResourceManager();
  ucxx::UcxBufferVec buffer_vec(
    mr, mem_type, buffers, nullptr, true,
    [owners = std::move(owners)](void*) {});

  rpc::ParamMeta meta_param;
  meta_param.type = rpc::ParamType::TENSOR_META_VEC;
  meta_param.value.template emplace<rpc::TensorMetaVec>(
    std::move(meta_vec_value));

  return {std::move(meta_param), std::move(buffer_vec)};
}

std::pair<rpc::ParamMeta, rpc::ReturnedPayload>
PythonAsyncFunctionWrapper::ConvertTensorReturns(
  std::span<nb::object> dlpack_returns) const {
  return ConvertTensorReturnsImpl(dlpack_returns, self);
}

std::pair<rpc::ParamMeta, rpc::ReturnedPayload>
PythonAsyncFunctionWrapper::ConvertTensorReturns(
  nb::list dlpack_returns) const {
  return ConvertTensorReturnsImpl(dlpack_returns, self);
}

template <typename Receiver>
auto PythonAsyncFunctionWrapper::CreatePythonResultCallback(
  Receiver& receiver) const {
  return [&receiver, this](nb::object fut) {
    try {
      nb::object py_result = fut.attr("result")();

      data::vector<rpc::ParamMeta> result_params;
      rpc::ReturnedPayload result_payload = std::monostate{};

      // ===== Zero-overhead dispatch using precomputed extraction_mode =====
      switch (extraction_mode) {
        case ResultExtractionMode::VOID:
          // No return value, nothing to do
          break;

        case ResultExtractionMode::SINGLE_NON_TENSOR: {
          // Single non-tensor return - direct conversion
          result_params.emplace_back(
            PythonToParamMeta(py_result, return_types[0]));
          break;
        }

        case ResultExtractionMode::SINGLE_TENSOR: {
          // Single tensor return - direct extraction
          auto [meta_param, payload] = ConvertTensorReturns(py_result);
          result_params.push_back(std::move(meta_param));
          result_payload = std::move(payload);
          break;
        }

        case ResultExtractionMode::LIST_TENSOR: {
          // List[Tensor] return - iterate list
          nb::list py_list = nb::cast<nb::list>(py_result);
          // Zero-copy: pass nb::list directly to sequence overload
          auto [meta_param, payload] = ConvertTensorReturns(py_list);
          result_params.push_back(std::move(meta_param));
          result_payload = std::move(payload);
          break;
        }

        case ResultExtractionMode::TUPLE_NON_TENSORS_ONLY: {
          // Tuple with only non-tensors - use precomputed indices
          nb::tuple py_tuple = nb::cast<nb::tuple>(py_result);
          for (size_t i = 0; i < non_tensor_indices.size(); ++i) {
            result_params.emplace_back(PythonToParamMeta(
              py_tuple[non_tensor_indices[i]], return_types[i]));
          }
          break;
        }

        case ResultExtractionMode::TUPLE_WITH_TENSORS: {
          // Mixed tuple - extract non-tensors and tensors separately
          nb::tuple py_tuple = nb::cast<nb::tuple>(py_result);

          // Non-tensors first (using precomputed indices)
          for (size_t i = 0; i < non_tensor_indices.size(); ++i) {
            result_params.emplace_back(PythonToParamMeta(
              py_tuple[non_tensor_indices[i]], return_types[i]));
          }

          // Then tensors (using precomputed tensor_return_indices)
          std::vector<nb::object> tensors;
          tensors.reserve(tensor_return_indices.size());
          for (size_t idx : tensor_return_indices) {
            nb::object elem = py_tuple[idx];
            // Handle nested List[Tensor] inside tuple
            if (nb::isinstance<nb::list>(elem)) {
              nb::list inner_list = nb::cast<nb::list>(elem);
              for (size_t j = 0; j < nb::len(inner_list); ++j) {
                tensors.push_back(inner_list[j]);
              }
            } else {
              tensors.push_back(elem);
            }
          }
          auto [meta_param, payload] = ConvertTensorReturns(tensors);
          result_params.push_back(std::move(meta_param));
          result_payload = std::move(payload);
          break;
        }
      }

      receiver.set_value(
        std::make_pair(std::move(result_params), std::move(result_payload)));
    } catch (const nb::python_error& e) {
      receiver.set_error(rpc::MakeRpcExceptionPtr(
        rpc::RpcErrc::INTERNAL,
        std::format(
          "Python server async function raised exception: {}", e.what())));
    } catch (const std::exception& e) {
      receiver.set_error(rpc::MakeRpcExceptionPtr(
        rpc::RpcErrc::INTERNAL,
        std::format("Python server binding runtime error: {}", e.what())));
    }
  };
}

template <typename PayloadT>
unifex::any_sender_of<
  std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
PythonAsyncFunctionWrapper::FunctionImpl(
  const data::vector<rpc::ParamMeta>& params, const PayloadT& payload) const {
  // Take ownership of params and payload by moving them into the sender
  // chain. This uses const_cast + std::move, which is SAFE because the
  // dispatcher (DynamicAsyncFuncWrapper in async_rpc_dispatcher.hpp) doesn't
  // use the params/payload after calling the wrapper. This pattern is
  // required due to DynamicAsyncRpcFunctionFacade constraining operator() to
  // take const refs.
  auto owned_params =
    std::move(const_cast<data::vector<rpc::ParamMeta>&>(params));
  auto owned_payload = std::move(const_cast<PayloadT&>(payload));

  return unifex::create<
    std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>(
    [this, owned_params = std::move(owned_params),
     owned_payload = std::move(owned_payload)](auto& receiver) mutable {
      auto& wake_manager = GetPythonWakeManager();
      wake_manager.Enqueue(pro::make_proxy<TaskFacade>(
        [this, owned_params = std::move(owned_params),
         owned_payload = std::move(owned_payload), &receiver]() mutable {
          try {
            // Use non-const refs to allow moving TensorMeta for zero-copy
            auto& params = owned_params;
            const auto& payload = owned_payload;
            auto py_args = ConvertParamsToPython(params, payload);

            nb::object py_coro;
            if (!tensor_param_indices.empty()) {
              // Normal mode: tensor params are already parsed and in py_args
              // via ConvertParamsToPython which uses
              // ConvertSingleParamToPython
              py_coro = this->py_callable(*py_args);
            } else {
              // Raw mode (register_function_raw): no tensor params parsed,
              // pass the payload directly as UcxBuffer/UcxBufferVec
              if constexpr (std::is_same_v<PayloadT, std::monostate>) {
                py_coro = this->py_callable(*py_args);
              } else {
                // Pass UcxBuffer or UcxBufferVec directly - they are already
                // registered with nanobind in bindings_types.cpp
                py_coro = this->py_callable(*py_args, nb::cast(payload));
              }
            }

            nb::module_ asyncio = GetAsyncioModule();
            nb::object task = asyncio.attr("create_task")(py_coro);
            nb::object future = asyncio.attr("ensure_future")(task);

            auto py_callback = CreatePythonResultCallback(receiver);

            // CRITICAL FIX: Use shared_ptr to extend payload lifetime until
            // the Python async task completes. Without this, the payload
            // would be destroyed when the TaskFacade lambda exits, but the
            // Python task may not have executed yet. We use shared_ptr
            // because nanobind::cpp_function requires a const callable, and
            // we need to move the payload into shared ownership.
            auto payload_keeper =
              std::make_shared<PayloadT>(std::move(owned_payload));

            future.attr("add_done_callback")(nb::cpp_function(
              [callback = std::move(py_callback),
               payload_keeper](nb::object fut) { callback(fut); }));
          } catch (const nb::python_error& e) {
            receiver.set_error(std::make_exception_ptr(std::runtime_error(
              "Failed to call Python async function: "
              + std::string(e.what()))));
          } catch (const std::exception& e) {
            receiver.set_error(std::make_exception_ptr(e));
          }
        }));
    });
}

// Explicit template instantiations
template nb::object
PythonAsyncFunctionWrapper::ConvertSingleParamToPython<ucxx::UcxBuffer>(
  size_t, rpc::utils::TensorMeta&&, const ucxx::UcxBuffer&) const;
template nb::object
PythonAsyncFunctionWrapper::ConvertSingleParamToPython<ucxx::UcxBufferVec>(
  size_t, rpc::utils::TensorMeta&&, const ucxx::UcxBufferVec&) const;

template nb::list PythonAsyncFunctionWrapper::ConvertParamsToPython<
  std::monostate>(data::vector<rpc::ParamMeta>&, const std::monostate&) const;
template nb::list PythonAsyncFunctionWrapper::ConvertParamsToPython<
  ucxx::UcxBuffer>(data::vector<rpc::ParamMeta>&, const ucxx::UcxBuffer&) const;
template nb::list
PythonAsyncFunctionWrapper::ConvertParamsToPython<ucxx::UcxBufferVec>(
  data::vector<rpc::ParamMeta>&, const ucxx::UcxBufferVec&) const;

template nb::object PythonAsyncFunctionWrapper::ConvertPayloadToPython<
  std::monostate>(std::monostate&&) const;
template nb::object PythonAsyncFunctionWrapper::ConvertPayloadToPython<
  ucxx::UcxBuffer>(ucxx::UcxBuffer&&) const;
template nb::object PythonAsyncFunctionWrapper::ConvertPayloadToPython<
  ucxx::UcxBufferVec>(ucxx::UcxBufferVec&&) const;

template unifex::any_sender_of<
  std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
PythonAsyncFunctionWrapper::FunctionImpl<std::monostate>(
  const data::vector<rpc::ParamMeta>&, const std::monostate&) const;
template unifex::any_sender_of<
  std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
PythonAsyncFunctionWrapper::FunctionImpl<ucxx::UcxBuffer>(
  const data::vector<rpc::ParamMeta>&, const ucxx::UcxBuffer&) const;
template unifex::any_sender_of<
  std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
PythonAsyncFunctionWrapper::FunctionImpl<ucxx::UcxBufferVec>(
  const data::vector<rpc::ParamMeta>&, const ucxx::UcxBufferVec&) const;

}  // namespace python
}  // namespace axon
}  // namespace eux
