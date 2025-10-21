/*Copyright 2025 He Jia <mofhejia@163.com>. All Rights Reserved.
 *
 *Licensed under the Apache License Version 2.0 with LLVM Exceptions
 *(the "License"); you may not use this file except in compliance with
 *the License. You may obtain a copy of the License at
 *
 *    https://llvm.org/LICENSE.txt
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 *==============================================================================*/

#pragma once

#ifndef RPC_CORE_RPC_PAYLOAD_TYPES_HPP_
#define RPC_CORE_RPC_PAYLOAD_TYPES_HPP_

#include <variant>

#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace rpc {

// Describes the type of context object returned by an RPC function.
enum class PayloadType : uint8_t {
  MONOSTATE,
  UCX_BUFFER,
  UCX_BUFFER_VEC,
};

// The set of possible payload types that can be part of an RPC call.
using PayloadVariant =
  std::variant<std::monostate, ucxx::UcxBuffer, ucxx::UcxBufferVec>;

}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_RPC_PAYLOAD_TYPES_HPP_
