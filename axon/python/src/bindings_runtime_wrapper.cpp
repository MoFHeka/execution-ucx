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
#include <cstring>
#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>

#include <optional>
#include <span>
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

nb::object PythonAsyncFunctionWrapper::ConvertSingleParamToPython(
  size_t tensor_param_idx, rpc::utils::TensorMeta&& meta,
  ucxx::UcxBuffer&& buffer) const {
  auto cached_obj = TryGetCustomBuffer(buffer.data());
  if (!cached_obj.is_none()) {
    return cached_obj;
  }

  nb::object dltensor_capsule =
    TensorMetaToDlpack(std::move(meta), std::move(buffer));

  if (tensor_param_idx < tensor_param_from_dlpack.size()) {
    const auto& from_dlpack_ref = tensor_param_from_dlpack[tensor_param_idx];
    if (from_dlpack_ref) {
      nb::object from_dlpack_fn = from_dlpack_ref.get();
      if (!from_dlpack_fn.is_none()) {
        return from_dlpack_fn(dltensor_capsule);
      }
    }
  }

  return dltensor_capsule;
}

template <typename PayloadT>
nb::list PythonAsyncFunctionWrapper::ConvertParamsToPython(
  data::vector<rpc::ParamMeta>& params, PayloadT& payload) const {
  nb::list py_args;

  const size_t num_params = param_types.size();
  if (num_params == 0) {
    return py_args;
  }

  // ============================================================
  // Small Buffer Optimization: Use stack for typical RPC cases
  // ============================================================
  static constexpr size_t kMaxStackParams = 16;

  std::array<const rpc::ParamMeta*, kMaxStackParams> stack_non_tensor_params;
  std::vector<const rpc::ParamMeta*> heap_non_tensor_params;

  const rpc::ParamMeta** non_tensor_params_ptr;
  if (params.size() <= kMaxStackParams) {
    non_tensor_params_ptr = stack_non_tensor_params.data();
  } else {
    heap_non_tensor_params.reserve(params.size());
    non_tensor_params_ptr = nullptr;
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

  if (!non_tensor_params_ptr) {
    non_tensor_param_count = heap_non_tensor_params.size();
    non_tensor_params_ptr = heap_non_tensor_params.data();
  }

  // Pre-build O(1) encoded param lookup indexed by param position
  using NL = FunctionSignatureInfo::EncodedElementType;
  std::vector<std::optional<NL>> encoded_lookup(num_params, std::nullopt);
  for (const auto& ep : encoded_params) {
    if (ep.param_index < num_params) {
      encoded_lookup[ep.param_index] = ep.element_type;
    }
  }

  // flat_tensor_idx: position in the flattened tensor array (UcxBufferVec
  //   buffers and TensorMetaVec). Increments once per individual tensor.
  // tensor_param_idx: index into tensor_param_from_dlpack[], increments once
  // per
  //   tensor-typed *parameter*. Multiple tensors in the same parameter
  //   (List[Tensor] / List[List[Tensor]]) share one tensor_param_idx.
  size_t flat_tensor_idx = 0;
  size_t tensor_param_idx = 0;
  size_t non_tensor_idx = 0;

  std::vector<ucxx::UcxBuffer> extracted_buffers;
  if constexpr (std::is_same_v<PayloadT, ucxx::UcxBufferVec>) {
    if (payload.size() > 0) {
      // By calling ExtractBuffers(), we take ownership of the individual
      // UcxBuffer instances. This is crucial for avoiding shared lifetime
      // constraints: each resulting DLPack capsule effectively owns its
      // independent underlying component, transferring memory management
      // natively to the Python Garbage Collector.
      extracted_buffers = std::move(payload).ExtractBuffers();
    }
  }

  for (size_t i = 0; i < num_params; ++i) {
    const auto& enc = encoded_lookup[i];
    if (enc == NL::NESTED_TENSOR_LIST) {
      // List[List[Tensor]]: group sizes stored as VECTOR_UINT32
      // (data::vector<uint32_t>)
      if (non_tensor_idx >= non_tensor_param_count) {
        throw std::runtime_error(
          "Missing group-sizes param for nested tensor list at index "
          + std::to_string(i));
      }
      const auto& group_sizes_vec =
        cista::get<data::vector<uint32_t>>(cista::get<rpc::VectorValue>(
          non_tensor_params_ptr[non_tensor_idx]->value));
      nb::list outer;
      for (uint32_t gsz : group_sizes_vec) {
        nb::list inner;
        for (uint32_t ti = 0; ti < gsz; ++ti) {
          if (
            !tensor_meta_vec_ptr
            || flat_tensor_idx >= tensor_meta_vec_ptr->size()) {
            throw std::runtime_error(
              "Not enough tensors for nested tensor list");
          }
          if constexpr (std::is_same_v<PayloadT, ucxx::UcxBuffer>) {
            auto py_dlpack = ConvertSingleParamToPython(
              tensor_param_idx,
              std::move(const_cast<rpc::utils::TensorMeta&>(
                (*tensor_meta_vec_ptr)[flat_tensor_idx])),
              std::move(payload));
            inner.append(std::move(py_dlpack));
          } else if constexpr (std::is_same_v<PayloadT, ucxx::UcxBufferVec>) {
            auto py_dlpack = ConvertSingleParamToPython(
              tensor_param_idx,
              std::move(const_cast<rpc::utils::TensorMeta&>(
                (*tensor_meta_vec_ptr)[flat_tensor_idx])),
              std::move(extracted_buffers[flat_tensor_idx]));
            inner.append(std::move(py_dlpack));
          } else {
            throw std::runtime_error("Unexpected payload type for tensor");
          }
          ++flat_tensor_idx;
        }
        outer.append(std::move(inner));
      }
      py_args.append(std::move(outer));
      ++non_tensor_idx;
      ++tensor_param_idx;
    } else if (enc == NL::STRING_ELEM) {
      if (non_tensor_idx >= non_tensor_param_count) {
        throw std::runtime_error(
          "Missing param for vector-string at index " + std::to_string(i));
      }
      py_args.append(
        DecodeVectorStringBlob(*non_tensor_params_ptr[non_tensor_idx++]));
    } else if (enc.has_value()) {
      // List[List[int/float/bool]]
      if (non_tensor_idx >= non_tensor_param_count) {
        throw std::runtime_error(
          "Missing param for nested list at index " + std::to_string(i));
      }
      py_args.append(
        DecodeNestedListBlob(*non_tensor_params_ptr[non_tensor_idx++]));
    } else if (param_types[i] == rpc::ParamType::TENSOR_META) {
      // Single Tensor parameter. Note: the header always uses TENSOR_META_VEC
      // when multiple Tensor params exist; each TENSOR_META branch takes the
      // next tensor from the shared stream sequentially.
      if (
        tensor_meta_vec_ptr && flat_tensor_idx < tensor_meta_vec_ptr->size()) {
        if constexpr (std::is_same_v<PayloadT, ucxx::UcxBuffer>) {
          auto py_dlpack = ConvertSingleParamToPython(
            tensor_param_idx,
            std::move(const_cast<rpc::utils::TensorMeta&>(
              (*tensor_meta_vec_ptr)[flat_tensor_idx])),
            std::move(payload));
          py_args.append(std::move(py_dlpack));
        } else if constexpr (std::is_same_v<PayloadT, ucxx::UcxBufferVec>) {
          auto py_dlpack = ConvertSingleParamToPython(
            tensor_param_idx,
            std::move(const_cast<rpc::utils::TensorMeta&>(
              (*tensor_meta_vec_ptr)[flat_tensor_idx])),
            std::move(extracted_buffers[flat_tensor_idx]));
          py_args.append(std::move(py_dlpack));
        } else {
          throw std::runtime_error("Unexpected payload type for tensor");
        }
        ++flat_tensor_idx;
      } else if (single_tensor_meta_ptr && flat_tensor_idx == 0) {
        if constexpr (std::is_same_v<PayloadT, ucxx::UcxBuffer>) {
          auto py_dlpack = ConvertSingleParamToPython(
            tensor_param_idx,
            std::move(
              const_cast<rpc::utils::TensorMeta&>(*single_tensor_meta_ptr)),
            std::move(payload));
          py_args.append(std::move(py_dlpack));
        } else if constexpr (std::is_same_v<PayloadT, ucxx::UcxBufferVec>) {
          auto py_dlpack = ConvertSingleParamToPython(
            tensor_param_idx,
            std::move(
              const_cast<rpc::utils::TensorMeta&>(*single_tensor_meta_ptr)),
            std::move(extracted_buffers[flat_tensor_idx]));
          py_args.append(std::move(py_dlpack));
        } else {
          throw std::runtime_error("Unexpected payload type for tensor");
        }
        ++flat_tensor_idx;
      } else {
        throw std::runtime_error(
          "Tensor meta not found for parameter " + std::to_string(i));
      }
      ++tensor_param_idx;
    } else if (param_types[i] == rpc::ParamType::TENSOR_META_VEC) {
      // List[Tensor] parameter (or the implicit vec created for multiple Tensor
      // params). Consumes all remaining tensors in the stream for this param.
      nb::list tensor_list;
      if (tensor_meta_vec_ptr) {
        while (flat_tensor_idx < tensor_meta_vec_ptr->size()) {
          if constexpr (std::is_same_v<PayloadT, ucxx::UcxBuffer>) {
            auto py_dlpack = ConvertSingleParamToPython(
              tensor_param_idx,
              std::move(const_cast<rpc::utils::TensorMeta&>(
                (*tensor_meta_vec_ptr)[flat_tensor_idx])),
              std::move(payload));
            tensor_list.append(std::move(py_dlpack));
          } else if constexpr (std::is_same_v<PayloadT, ucxx::UcxBufferVec>) {
            auto py_dlpack = ConvertSingleParamToPython(
              tensor_param_idx,
              std::move(const_cast<rpc::utils::TensorMeta&>(
                (*tensor_meta_vec_ptr)[flat_tensor_idx])),
              std::move(extracted_buffers[flat_tensor_idx]));
            tensor_list.append(std::move(py_dlpack));
          } else {
            throw std::runtime_error("Unexpected payload type for tensor");
          }
          ++flat_tensor_idx;
        }
      }
      py_args.append(std::move(tensor_list));
      ++tensor_param_idx;
    } else {
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
            auto& params = owned_params;
            nb::object py_coro;

            if (!tensor_param_indices.empty()) {
              auto py_args = ConvertParamsToPython(params, owned_payload);
              py_coro = this->py_callable(*py_args);
            } else {
              auto py_args = ConvertParamsToPython(params, owned_payload);
              if constexpr (std::is_same_v<PayloadT, std::monostate>) {
                py_coro = this->py_callable(*py_args);
              } else {
                py_coro = this->py_callable(
                  *py_args, nb::cast(std::move(owned_payload)));
              }
            }

            nb::module_ asyncio = GetAsyncioModule();
            nb::object task = asyncio.attr("create_task")(py_coro);
            nb::object future = asyncio.attr("ensure_future")(task);

            auto py_callback = CreatePythonResultCallback(receiver);
            future.attr("add_done_callback")(
              nb::cpp_function([callback = std::move(py_callback)](
                                 nb::object fut) { callback(fut); }));
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
template nb::list PythonAsyncFunctionWrapper::ConvertParamsToPython<
  std::monostate>(data::vector<rpc::ParamMeta>&, std::monostate&) const;
template nb::list PythonAsyncFunctionWrapper::ConvertParamsToPython<
  ucxx::UcxBuffer>(data::vector<rpc::ParamMeta>&, ucxx::UcxBuffer&) const;
template nb::list PythonAsyncFunctionWrapper::ConvertParamsToPython<
  ucxx::UcxBufferVec>(data::vector<rpc::ParamMeta>&, ucxx::UcxBufferVec&) const;

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
