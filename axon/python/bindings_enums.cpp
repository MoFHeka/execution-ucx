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

#include "axon/python/bindings_enums.hpp"

#include <nanobind/nanobind.h>

#include "rpc_core/rpc_types.hpp"

namespace nb = nanobind;
namespace rpc = eux::rpc;

void RegisterEnums(nb::module_& m) {
  // ParamType enum
  nb::enum_<rpc::ParamType>(m, "ParamType")
    .value("PRIMITIVE_BOOL", rpc::ParamType::PRIMITIVE_BOOL)
    .value("PRIMITIVE_INT8", rpc::ParamType::PRIMITIVE_INT8)
    .value("PRIMITIVE_INT16", rpc::ParamType::PRIMITIVE_INT16)
    .value("PRIMITIVE_INT32", rpc::ParamType::PRIMITIVE_INT32)
    .value("PRIMITIVE_INT64", rpc::ParamType::PRIMITIVE_INT64)
    .value("PRIMITIVE_UINT8", rpc::ParamType::PRIMITIVE_UINT8)
    .value("PRIMITIVE_UINT16", rpc::ParamType::PRIMITIVE_UINT16)
    .value("PRIMITIVE_UINT32", rpc::ParamType::PRIMITIVE_UINT32)
    .value("PRIMITIVE_UINT64", rpc::ParamType::PRIMITIVE_UINT64)
    .value("PRIMITIVE_FLOAT32", rpc::ParamType::PRIMITIVE_FLOAT32)
    .value("PRIMITIVE_FLOAT64", rpc::ParamType::PRIMITIVE_FLOAT64)
    .value("VECTOR_BOOL", rpc::ParamType::VECTOR_BOOL)
    .value("VECTOR_INT8", rpc::ParamType::VECTOR_INT8)
    .value("VECTOR_INT16", rpc::ParamType::VECTOR_INT16)
    .value("VECTOR_INT32", rpc::ParamType::VECTOR_INT32)
    .value("VECTOR_INT64", rpc::ParamType::VECTOR_INT64)
    .value("VECTOR_UINT8", rpc::ParamType::VECTOR_UINT8)
    .value("VECTOR_UINT16", rpc::ParamType::VECTOR_UINT16)
    .value("VECTOR_UINT32", rpc::ParamType::VECTOR_UINT32)
    .value("VECTOR_UINT64", rpc::ParamType::VECTOR_UINT64)
    .value("VECTOR_FLOAT32", rpc::ParamType::VECTOR_FLOAT32)
    .value("VECTOR_FLOAT64", rpc::ParamType::VECTOR_FLOAT64)
    .value("STRING", rpc::ParamType::STRING)
    .value("VOID", rpc::ParamType::VOID)
    .value("TENSOR_META", rpc::ParamType::TENSOR_META)
    .value("UNKNOWN", rpc::ParamType::UNKNOWN);

  // PayloadType enum
  nb::enum_<rpc::PayloadType>(m, "PayloadType")
    .value("UCX_BUFFER", rpc::PayloadType::UCX_BUFFER)
    .value("UCX_BUFFER_VEC", rpc::PayloadType::UCX_BUFFER_VEC)
    .value("NO_PAYLOAD", rpc::PayloadType::NO_PAYLOAD)
    .value("MONOSTATE", rpc::PayloadType::MONOSTATE);

  // RpcErrc enum
  nb::enum_<rpc::RpcErrc>(m, "RpcErrc")
    .value("OK", rpc::RpcErrc::OK)
    .value("CANCELLED", rpc::RpcErrc::CANCELLED)
    .value("UNKNOWN", rpc::RpcErrc::UNKNOWN)
    .value("INVALID_ARGUMENT", rpc::RpcErrc::INVALID_ARGUMENT)
    .value("DEADLINE_EXCEEDED", rpc::RpcErrc::DEADLINE_EXCEEDED)
    .value("NOT_FOUND", rpc::RpcErrc::NOT_FOUND)
    .value("ALREADY_EXISTS", rpc::RpcErrc::ALREADY_EXISTS)
    .value("PERMISSION_DENIED", rpc::RpcErrc::PERMISSION_DENIED)
    .value("RESOURCE_EXHAUSTED", rpc::RpcErrc::RESOURCE_EXHAUSTED)
    .value("FAILED_PRECONDITION", rpc::RpcErrc::FAILED_PRECONDITION)
    .value("ABORTED", rpc::RpcErrc::ABORTED)
    .value("OUT_OF_RANGE", rpc::RpcErrc::OUT_OF_RANGE)
    .value("UNIMPLEMENTED", rpc::RpcErrc::UNIMPLEMENTED)
    .value("INTERNAL", rpc::RpcErrc::INTERNAL)
    .value("UNAVAILABLE", rpc::RpcErrc::UNAVAILABLE)
    .value("DATA_LOSS", rpc::RpcErrc::DATA_LOSS)
    .value("UNAUTHENTICATED", rpc::RpcErrc::UNAUTHENTICATED);
}
