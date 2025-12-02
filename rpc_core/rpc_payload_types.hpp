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

#include <optional>
#include <utility>
#include <variant>

#include "ucx_context/ucx_context_data.hpp"

namespace eux {
namespace rpc {

template <typename T>
  requires std::is_same_v<T, ucxx::UcxBuffer>
           || std::is_same_v<T, ucxx::UcxBufferVec>
           || std::is_same_v<T, ucxx::UcxHeader>
struct UcxDataDeleter {
  UcxDataDeleter() = default;
  explicit UcxDataDeleter(T&& buffer) : buffer_(std::move(buffer)) {}

  template <typename P>
  void operator()(P* /*ignore ptr*/) {
    buffer_.reset();
  }

 private:
  std::optional<T> buffer_;
};

using UcxHeaderDeleter = UcxDataDeleter<ucxx::UcxHeader>;
using UcxBufferDeleter = UcxDataDeleter<ucxx::UcxBuffer>;
using UcxBufferVecDeleter = UcxDataDeleter<ucxx::UcxBufferVec>;

// Describes the type of context object returned by an RPC function.
enum class PayloadType : uint8_t {
  UCX_BUFFER,
  UCX_BUFFER_VEC,
  NO_PAYLOAD,              // no payload monostate
  MONOSTATE = NO_PAYLOAD,  // monostate
};

// The set of possible payload types that can be part of an RPC call.
using PayloadVariant =
  std::variant<std::monostate, ucxx::UcxBuffer, ucxx::UcxBufferVec>;

// Because same signature can only return same tpye in C++, so we use
// PayloadVariant to represent the returned payload.
using ReturnedPayload = PayloadVariant;

}  // namespace rpc
}  // namespace eux

#endif  // RPC_CORE_RPC_PAYLOAD_TYPES_HPP_
