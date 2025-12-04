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

#ifndef AXON_CORE_MEMORY_POLICY_HPP_
#define AXON_CORE_MEMORY_POLICY_HPP_

#include <proxy/proxy.h>

#include <vector>

#include "rpc_core/utils/tensor_meta.hpp"

namespace eux {
namespace axon {

using TensorMetaRef = std::reference_wrapper<const rpc::utils::TensorMeta>;
using TensorMetaRefVec = std::vector<TensorMetaRef>;

// Facade for user to get the memory resource for the request.
template <typename ReceivedBufferT>
struct BufferProviderFacade
  : pro::facade_builder::add_convention<
      pro::operator_dispatch<"()">,
      ReceivedBufferT(const TensorMetaRefVec&)>::build {};

template <typename ReceivedBufferT>
using BufferProvider = pro::proxy<BufferProviderFacade<ReceivedBufferT>>;

/**
 * @brief Memory policy type representing that allocation is always on host
 * memory.
 *
 * AlwaysOnHostPolicy is an empty marker class indicating host memory should be
 * used for all buffer allocations. It is not called nor does it provide any
 * allocation interface itself.
 *
 * Actual buffer allocation is performed by the UCX context internally when this
 * policy is applied.
 *
 * @note
 * AlwaysOnHostPolicy and CustomMemoryPolicy are different types:
 *   - AlwaysOnHostPolicy is a tag-only type; it will never be invoked or called
 * as a BufferProviderFacade implementation.
 *   - CustomMemoryPolicy is a proxy_view for BufferProviderFacade, enabling
 * users to implement custom buffer allocation.
 */
struct AlwaysOnHostPolicy {};

/**
 * @brief Customizable memory policy for user-defined buffer allocation.
 *
 * CustomMemoryPolicy is a proxy_view for BufferProviderFacade. The user
 * provides an implementation of the allocation logic capable of taking a @c
 * TensorMetaRefVec and returning a buffer (such as ucxx::UcxBuffer,
 * ucxx::UcxBufferVec, or PayloadVariant).
 *
 * @tparam ReceivedBufferT The buffer type to return.
 *
 * @note
 * CustomMemoryPolicy enables user customization, while AlwaysOnHostPolicy is
 * only a tag.
 */
template <typename ReceivedBufferT>
using CustomMemoryPolicy = BufferProvider<ReceivedBufferT>;

template <typename ReceivedBufferT>
using ReceiverMemoryPolicy =
  std::variant<AlwaysOnHostPolicy, CustomMemoryPolicy<ReceivedBufferT>>;

template <typename T, typename ReceivedBufferT, typename VariantT>
struct is_variant_alternative_of_receiver_memory_policy;

template <typename T, typename ReceivedBufferT, typename... Alts>
struct is_variant_alternative_of_receiver_memory_policy<
  T, ReceivedBufferT, std::variant<Alts...>>
  : std::disjunction<std::is_same<std::decay_t<T>, std::decay_t<Alts>>...> {};

template <typename T, typename ReceivedBufferT>
constexpr bool is_receiver_memory_policy_v =
  std::is_same_v<std::decay_t<T>, ReceiverMemoryPolicy<ReceivedBufferT>>
  || is_variant_alternative_of_receiver_memory_policy<
    T, ReceivedBufferT, ReceiverMemoryPolicy<ReceivedBufferT>>::value;

}  // namespace axon
}  // namespace eux

#endif  // AXON_CORE_MEMORY_POLICY_HPP_
