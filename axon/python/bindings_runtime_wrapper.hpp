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

#ifndef AXON_PYTHON_BINDINGS_RUNTIME_WRAPPER_HPP_
#define AXON_PYTHON_BINDINGS_RUNTIME_WRAPPER_HPP_

#include <unifex/any_sender_of.hpp>

#include "axon/axon_runtime.hpp"
#include "axon/python/python_helpers.hpp"
#include "axon/python/python_wake_manager.hpp"
#include "rpc_core/rpc_types.hpp"
#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace axon {
namespace python {

namespace nb = nanobind;
namespace rpc = eux::rpc;
namespace ucxx = eux::ucxx;
namespace axon = eux::axon;
namespace data = cista::offset;

// Use ResultExtractionMode from python_helpers.hpp
using python::ResultExtractionMode;

// Function object wrapper for Python async RPC functions
struct __attribute__((visibility("hidden"))) PythonAsyncFunctionWrapper {
  mutable nb::object py_callable;
  std::vector<rpc::ParamType> param_types;
  std::vector<rpc::ParamType> return_types;
  std::vector<size_t> tensor_param_indices;
  std::vector<size_t> tensor_return_indices;
  axon::AxonRuntime& self;

  // from_dlpack_fn callables for tensor parameters and returns
  std::vector<SharedPyObject> tensor_param_from_dlpack;
  std::vector<SharedPyObject> tensor_return_from_dlpack;

  // ===== Precomputed at registration time for zero-overhead extraction =====
  ResultExtractionMode extraction_mode = ResultExtractionMode::VOID;
  std::vector<size_t> non_tensor_indices;  // Positions of non-tensor returns

  // Overload for no payload
  unifex::any_sender_of<
    std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
  operator()(const data::vector<rpc::ParamMeta>& params) const;

  // Overload for monostate
  unifex::any_sender_of<
    std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
  operator()(
    const data::vector<rpc::ParamMeta>& params,
    const std::monostate& payload) const;

  // Overload for UcxBuffer
  unifex::any_sender_of<
    std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
  operator()(
    const data::vector<rpc::ParamMeta>& params,
    const ucxx::UcxBuffer& payload) const;

  // Overload for UcxBufferVec
  unifex::any_sender_of<
    std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
  operator()(
    const data::vector<rpc::ParamMeta>& params,
    const ucxx::UcxBufferVec& payload) const;

 private:
  // Convert single tensor parameter to Python dlpack object
  template <typename PayloadT>
  nb::object ConvertSingleParamToPython(
    size_t tensor_idx, rpc::utils::TensorMeta&& meta,
    const PayloadT& payload) const;

  // Convert all parameters to Python objects
  template <typename PayloadT>
  nb::list ConvertParamsToPython(
    data::vector<rpc::ParamMeta>& params, const PayloadT& payload) const;

  // Convert payload to Python object (for non-tensor payloads)
  template <typename PayloadT>
  nb::object ConvertPayloadToPython(PayloadT&& payload) const;

  // Convert multiple tensor returns to RPC format
  std::pair<rpc::ParamMeta, rpc::ReturnedPayload> ConvertTensorReturns(
    nb::object dlpack_return) const;

  std::pair<rpc::ParamMeta, rpc::ReturnedPayload> ConvertTensorReturns(
    std::span<nb::object> dlpack_returns) const;

  std::pair<rpc::ParamMeta, rpc::ReturnedPayload> ConvertTensorReturns(
    nb::list dlpack_returns) const;

  // Create callback to handle Python async function result
  template <typename Receiver>
  auto CreatePythonResultCallback(Receiver& receiver) const;

  // FunctionImpl takes const refs due to Facade constraints, but internally
  // uses const_cast + std::move to transfer ownership (safe because dispatcher
  // doesn't use params/payload after calling wrapper)
  template <typename PayloadT>
  unifex::any_sender_of<
    std::pair<data::vector<rpc::ParamMeta>, rpc::ReturnedPayload>>
  FunctionImpl(
    const data::vector<rpc::ParamMeta>& params, const PayloadT& payload) const;
};

}  // namespace python
}  // namespace axon
}  // namespace eux

#endif  // AXON_PYTHON_BINDINGS_RUNTIME_WRAPPER_HPP_
